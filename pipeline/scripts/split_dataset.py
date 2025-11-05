
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import json, random
from datasets import load_dataset, Dataset

    
def get_commit(ex: dict) -> str:
    return ex.get("breakingCommit") or ex.get("data", {}).get("breakingCommit")

def get_errors_len(ex: dict) -> int:
    errors = ex.get("errors")
    if errors is None:
        errors = ex.get("data", {}).get("errors", [])
    return len(errors or [])

def get_prompt_text(ex: dict) -> str:
    p = ex.get("prompt")
    if p is None:
        p = ex.get("data", {}).get("prompt", "")
    return p or ""

def main(out_dir, prompts_path, min_bucket=10, train_ratio=0.7, seed=42):
    rng = random.Random(seed)
    ds = load_dataset("json", data_files=str(prompts_path), split="train")
    print(f"[LOAD] 样本数: {ds.num_rows}")

    # === 按 breakingCommit 分组 ===
    # commit -> list[样本索引]
    groups = defaultdict(list)
    for i, ex in enumerate(ds):
        commit = get_commit(ex)
        if not commit:
            raise ValueError(f"样本[{i}] 缺少 breakingCommit")
        groups[commit].append(i)

    # === 组的 N 为该 commit 下“所有样本 errors 的总和” ===
    # 生成 (commit, total_errors_N, idx_list)
    grouped = []
    for commit, idxs in groups.items():
        total_N = 0
        for i in idxs:
            total_N += get_errors_len(ds[i])
        grouped.append((commit, total_N, idxs))

    print(f"[INFO] breakingCommit 组数: {len(grouped)}")

    # 以 N 分桶：N -> list[(commit, N, idxs)]
    buckets = defaultdict(list)
    for commit, N, idxs in grouped:
        buckets[N].append((commit, N, idxs))

    # 按 commit 排序；再按 N 升序遍历
    for N in buckets:
        buckets[N].sort(key=lambda t: t[0])
    Ns_sorted = sorted(buckets.keys())

    # 合桶 & 划分
    train_commits, test_commits = set(), set()
    pending = []  # list[(commit, N, idxs)]
    for N in Ns_sorted:
        pending.extend(buckets[N])
        if len(pending) >= min_bucket:
            rng.shuffle(pending)
            k = int(round(len(pending) * train_ratio))
            train, test = pending[:k], pending[k:]
            train_commits.update(c for c, _, _ in train)
            test_commits.update(c for c, _, _ in test)

            n_dist = Counter(n for _, n, _ in pending)
            print(f"[SPLIT] 合并到 N>={N}: total_groups={len(pending)} -> "
                  f"train_groups={len(train)} test_groups={len(test)} ratio≈{train_ratio:.2f}")
            print(f"        该桶组内 N 分布: {dict(sorted(n_dist.items()))}")
            pending = []

    if pending:
        # 规则 4：末尾残留 < min_bucket 全进 test；否则仍按比例
        n_dist = Counter(n for _, n, _ in pending)
        if len(pending) < min_bucket:
            test_commits.update(c for c, _, _ in pending)
            print(f"[TAIL] 残留 {len(pending)}(<{min_bucket}) 组 -> 全部分配 test")
            print(f"       残留 N 分布: {dict(sorted(n_dist.items()))}")
        else:
            rng.shuffle(pending)
            k = int(round(len(pending) * train_ratio))
            train_commits.update(c for c, _, _ in pending[:k])
            test_commits.update(c for c, _, _ in pending[k:])
            print(f"[TAIL] 末尾桶 {len(pending)} 组 -> 按 {train_ratio:.2f} 比例划分")
            print(f"       N 分布: {dict(sorted(n_dist.items()))}")

    assert not (train_commits & test_commits), "同一 breakingCommit 组不应同时出现在 train 与 test"

    # 输出 JSONL
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    test_path  = out / "test.jsonl"

    def write_jsonl(path: Path, commits: set):
        with path.open("w", encoding="utf-8") as f:
            for ex in ds:
                if get_commit(ex) in commits:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    write_jsonl(train_path, train_commits)
    write_jsonl(test_path,  test_commits)
    
    # train 长度信息
    train_prompt_lengths = []
    for ex in ds:
        if get_commit(ex) in train_commits:
            train_prompt_lengths.append(len(get_prompt_text(ex)))

    if train_prompt_lengths:
        min_len = min(train_prompt_lengths)
        max_len = max(train_prompt_lengths)
        print(f"[STATS] train prompt 长度列表（字符数）：{train_prompt_lengths}")
        print(f"[STATS] train prompt 最短长度：{min_len}")
        print(f"[STATS] train prompt 最长长度：{max_len}")
    else:
        print("[STATS] train 为空，无法统计 prompt 长度。")

    train_rows = sum(1 for _ in open(train_path, "r", encoding="utf-8"))
    test_rows  = sum(1 for _ in open(test_path,  "r", encoding="utf-8"))
    print(f"[RESULT] groups -> train: {len(train_commits)} | test: {len(test_commits)}")
    print(f"[RESULT] rows   -> train: {train_rows} | test: {test_rows}")
    print(f"[SAVE] JSONL 写入：{train_path}")
    print(f"[SAVE] JSONL 写入：{test_path}")
    print("[LOAD] 下次读取示例：")
    print(f"       load_dataset('json', data_files={{'train': '{train_path.as_posix()}', "
          f"'test': '{test_path.as_posix()}'}})")

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent.parent / "data" / "prompts_diff"
    prompts_path = Path(__file__).parent.parent.parent / "data" / "prompts_diff" / "dataset.json"
    main(out_dir=str(out_dir), prompts_path=prompts_path, min_bucket=10, train_ratio=0.7, seed=42)


