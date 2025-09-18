import argparse
from pathlib import Path
import sys
import logging

from pipeline.constants.constants import LOGGING_FORMAT, DEBUG_LEVEL
from pipeline.engine.runner import Runner, RunConfig
from pipeline.agents.llm_policy import LLMPolicy
from pipeline.envs.repair_env import RepairEnv
from pipeline.dataloader.bu_loader import as_iterable


logging.basicConfig(level=DEBUG_LEVEL, format=LOGGING_FORMAT)


def main(argv: list[str] | None = None) -> None:
    """Unified entry point for both training and evaluation.

    Training and evaluation share the same runner/loop. The only
    difference is controlled by the `--mode` flag.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Train/Eval LLM on breaking updates dataset")
    parser.add_argument("--input", "-i", type=Path,
                        default=Path(__file__).parent.parent / "data" / "dataset",
                        help="Path to dataset root containing context.json files")
    parser.add_argument("--model", "-m", type=str, default="codellama/CodeLlama-7b-hf",
                        help="Model name or path")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval",
                        help="Run mode: train or eval (default: eval)")
    parser.add_argument("--level", choices=["file", "project"], default="file",
                        help="Episode level: file or project (default: file)")
    parser.add_argument("--deterministic-eval", action="store_true",
                        help="Use deterministic generation during eval (default true)")
    args = parser.parse_args(argv)

    # Build components
    policy = LLMPolicy(model_name=args.model)
    env = RepairEnv()
    dataset_iter = as_iterable(args.input, level=args.level)

    run_cfg = RunConfig(
        max_steps_per_episode=1,
        deterministic_eval=True if args.mode == "eval" else False,
        save_every_n_episodes=None,
    )

    runner = Runner(policy=policy, env=env, dataset=dataset_iter, run_cfg=run_cfg)
    runner.run(mode=args.mode)


if __name__ == "__main__":
    main()
    
            
            
