#!/usr/bin/env python3
"""
mink_score.py -- Membership-inference scoring for the Memorisation Probe.

For each item in a JSONL manifest, scores it under a given open-weight model
with two token-probability detectors, both averaging the k% LEAST confident
per-token log-probs (higher = more member-like: the model wasn't surprised
even by the "hard" tokens):
  - Min-K% Prob   (k configurable, default 20)   [Shi et al., ICLR'24]
  - Min-K%++      (k configurable, default 20)   [Zhang et al., ICLR'25]
                  (Min-K% Prob z-normalised per-token before averaging)
Also always reports token count -- length is the #1 confounder.

Two scoring modes per item:
  * whole-file : score every token of `text`
  * span       : score ONLY the tokens of `target`, conditioned on `prefix`.
                 Use this to score the *fix hunk* rather than the whole file --
                 whole-file scores are dominated by unchanged public code and
                 wash out the signal you actually care about.

Manifest format (one JSON object per line):
  {"id": "...", "group": "member|nonmember|bump", "text": "<full text>"}
  or
  {"id": "...", "group": "...", "prefix": "<context>", "target": "<hunk>"}
  Any extra keys (build_ok, n_files, patch_size, bc_type, ...) are passed through.

  Also accepts the project's SFT training rows directly (e.g.
  data/sft/sft_data_train.jsonl: {"original_code": ..., "prompt": ...,
  "label": ..., "accepted": ..., "breakingCommit": ..., ...}). These are
  auto-detected (presence of "original_code") and normalised to whole-file
  scoring of `original_code` -- the real pre-existing client source file
  (verbatim-embedded inside `prompt`), which is what we're actually probing
  for pretraining contamination. `label` is a synthetic patch generated
  during distillation, not something that could have been memorised from
  pretraining, so it is NOT scored here. `accepted: false` rows are dropped,
  mirroring the same filter pipeline/sft/backends.build_dataset applies
  before training, so this scores exactly the source files behind the rows
  the model was actually fine-tuned on.

Usage:
  python mink_score.py \
      --model Qwen/Qwen2.5-Coder-14B \
      --manifest data/items.jsonl \
      --out results/scores_qwen25c14b.csv \
      --k 20 --max-len 7000 --dtype bfloat16

  python mink_score.py \
      --model unsloth/qwen3.5-9b \
      --manifest data/sft/sft_data_train.jsonl \
      --out results/mink_score/qwen3.5-9b.csv --max-len 16000

Notes:
  - Score with NO chat template and NO system
    prompt: template tokens shift every conditional probability.

"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


# --------------------------------------------------------------------------- #
# core math
# --------------------------------------------------------------------------- #
def token_stats(logits: torch.Tensor, target_ids: torch.Tensor):
    """
    logits: [T, V] logits predicting target_ids (already shifted by caller).

    Returns (logp, mu, sigma), each [T], float64 on CPU:
      logp  = log p(x_t | x_<t)         -- the actual token's log-prob
      mu    = E_p[log p]  (= -entropy)  -- Min-K%++ z-score mean
      sigma = std of log p under the model's own predicted distribution
    Min-K%++ scores each token as (logp - mu) / sigma before averaging.
    """
    logits = logits.float()
    logprobs = F.log_softmax(logits, dim=-1)           # [T, V]
    probs = logprobs.exp()

    logp = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)   # [T]
    mu = (probs * logprobs).sum(dim=-1)                                # [T]
    second = (probs * logprobs.pow(2)).sum(dim=-1)                     # [T]
    var = (second - mu.pow(2)).clamp_min(1e-12)
    sigma = var.sqrt()
    return logp.double().cpu(), mu.double().cpu(), sigma.double().cpu()


def aggregate(logp: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
              k: float) -> dict:
    """Reduce per-token log-probs to the two Min-K% scores: each is the mean
    of the k% least-confident tokens (raw for Min-K% Prob, z-normalised for
    Min-K%++). Higher = more member-like."""
    n = logp.numel()
    if n == 0:
        return {}

    n_low = max(1, int(round(n * k / 100.0)))

    mink = torch.topk(logp, n_low, largest=False).values.mean().item()

    z = (logp - mu) / sigma
    minkpp = torch.topk(z, n_low, largest=False).values.mean().item()

    return {
        "n_tokens": n,
        f"mink_{int(k)}": mink,
        f"minkpp_{int(k)}": minkpp,
    }


_SFT_PASSTHROUGH_KEYS = (
    "project", "libraryName", "libraryGroupID",
    "newVersion", "previousVersion", "breakingCommit", "accepted",
)


def normalize_items(raw_items: list[dict]) -> list[dict]:
    """Normalise SFT training rows (`original_code`/`prompt`/`label`/`accepted`)
    into the whole-file manifest schema score_item expects, scoring
    `original_code` rather than `label` .
    """
    items = []
    for i, it in enumerate(raw_items):
        if "original_code" in it:
            if not it.get("accepted", True):
                continue
            item = {
                "id": it.get("breakingCommit", i),
                "group": "sft_train_original_code",
                "text": it["original_code"],
            }
            for key in _SFT_PASSTHROUGH_KEYS:
                if key in it:
                    item[key] = it[key]
            if "model" in it:
                item["label_gen_model"] = it["model"]
            items.append(item)
        else:
            items.append(it)
    return items


# --------------------------------------------------------------------------- #
# per-item scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_item(model, tok, item: dict, k: float, max_len: int, device,
                chunk_size: int = 2048) -> dict:
    """Score one manifest item. Returns {} if the item has no scorable tokens."""
    if "target" in item:
        prefix = item.get("prefix", "")
        target = item["target"]
        pre_ids = tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
        tgt_ids = tok(target, add_special_tokens=False)["input_ids"]
        if len(tgt_ids) == 0:
            return {}
        # left-truncate the CONTEXT, never the target
        budget = max_len - len(tgt_ids)
        if budget < 0:
            tgt_ids = tgt_ids[:max_len]
            pre_ids = []
        else:
            pre_ids = pre_ids[-budget:]
        ids = pre_ids + tgt_ids
        n_score = len(tgt_ids)
    else:
        ids = tok(item["text"], add_special_tokens=False)["input_ids"][:max_len]
        n_score = max(0, len(ids) - 1)

    # the very first token of the sequence has no context and cannot be scored
    n_score = min(n_score, len(ids) - 1)
    if len(ids) < 2 or n_score < 1:
        return {}

    input_ids = torch.tensor([ids], device=device)
    logits = model(input_ids).logits[0]            # [L, V], model dtype (bf16/fp16)

    # logits[i] predicts ids[i+1]; we want the last n_score targets
    tgt = torch.tensor(ids[-n_score:], device=device)
    lg = logits[-n_score - 1:-1, :]

    # token_stats() upcasts to float32 over the full vocab; for whole-file
    # mode n_score is ~the entire sequence, so doing that in one shot OOM's
    # on long files with large-vocab models. Chunk along the token dimension
    # so peak memory is bounded by chunk_size * vocab, not file length.
    logp_parts, mu_parts, sigma_parts = [], [], []
    for start in range(0, n_score, chunk_size):
        end = min(start + chunk_size, n_score)
        logp_c, mu_c, sigma_c = token_stats(lg[start:end], tgt[start:end])
        logp_parts.append(logp_c)
        mu_parts.append(mu_c)
        sigma_parts.append(sigma_c)
    logp = torch.cat(logp_parts)
    mu = torch.cat(mu_parts)
    sigma = torch.cat(sigma_parts)

    return aggregate(logp, mu, sigma, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=20.0)
    ap.add_argument("--max-len", type=int, default=7000)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=2048,
                    help="Tokens scored per token_stats() call; bounds peak "
                         "memory to chunk_size*vocab_size regardless of file "
                         "length. Lower if still OOMing on very long files.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code)

    model_kwargs = dict(
        torch_dtype=getattr(torch, args.dtype),
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except ValueError:
        # Some families (e.g. Mistral-3) are decoder-only text models whose
        # architecture is only registered under the image-text-to-text
        # mapping. We never pass pixel_values, so this is a plain causal LM
        # forward pass in practice.
        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    model = model.eval()

    items = normalize_items([json.loads(l) for l in open(args.manifest) if l.strip()])
    rows, skipped = [], 0
    for i, it in enumerate(items):
        s = score_item(model, tok, it, args.k, args.max_len, args.device,
                       chunk_size=args.chunk_size)
        if not s:
            skipped += 1
            continue
        passthrough = {kk: vv for kk, vv in it.items()
                       if kk not in ("text", "prefix", "target")}
        rows.append({"model": args.model, **passthrough, **s})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)

    import pandas as pd
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(rows)} rows ({skipped} skipped)")


if __name__ == "__main__":
    main()
