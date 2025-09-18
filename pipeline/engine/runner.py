from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

from pipeline.agents.llm_policy import LLMPolicy, PolicyOutput
from pipeline.envs.repair_env import RepairEnv, Observation
from pipeline.types.episode import EpisodeSample
from pipeline.callbacks.base import Callback, DefaultLoggingCallback


@dataclass
class RunConfig:
    """
    Configuration for running training/evaluation in a unified loop.
    """
    max_steps_per_episode: int = 1  # single-step patch per file
    deterministic_eval: bool = True
    save_every_n_episodes: Optional[int] = None


class Runner:
    """Unified runner for both training and evaluation.

    The runner drives a single rollout-per-sample loop. The only
    differences between train and eval are controlled via `mode`
    and `run_cfg` without duplicating logic.
    """

    def __init__(
        self,
        policy: LLMPolicy,
        env: RepairEnv,
        dataset: Iterable[EpisodeSample],
        callbacks: Optional[list[Callback]] = None,
        run_cfg: Optional[RunConfig] = None,
    ) -> None:
        self.policy = policy
        self.env = env
        self.dataset = dataset
        self.callbacks = callbacks or [DefaultLoggingCallback()]
        self.run_cfg = run_cfg or RunConfig()

    def run(self, mode: Literal["train", "eval"]) -> None:
        """Run the unified loop in either training or evaluation mode.

        - Both modes: reset env with sample, policy.act once, env.step once.
        - Train mode: optionally call policy.learn on collected data.
        - Eval mode: deterministic generation (if configured) and metrics only.
        """
        episode_idx = 0

        for sample in self.dataset:
            episode_idx += 1
            for callback in self.callbacks:
                callback.on_episode_begin(episode_idx, mode)
            obs = self.env.reset(sample)

            # Generation config toggles between train/eval behavior.
            deterministic = (
                self.run_cfg.deterministic_eval if mode == "eval" else False
            )

            # Single-step episode by default (patch once per file)
            steps = 0
            done = False

            while not done and steps < self.run_cfg.max_steps_per_episode:
                steps += 1
                # Policy act
                policy_out: PolicyOutput = self.policy.act(
                    obs=obs, deterministic=deterministic
                )
                # Env step
                obs, reward, done, info = self.env.step(policy_out)

                for cb in self.callbacks:
                    cb.on_step(episode_idx, steps, reward, info)

            # Optional learning step (kept unified; a no-op in eval)
            if mode == "train":
                # In a real setup, we would pass trajectories/batch to learn().
                # Here we invoke a placeholder to preserve a single flow.
                learn_stats = self.policy.learn()
                for cb in self.callbacks:
                    cb.on_learn_step(episode_idx, learn_stats)

            for cb in self.callbacks:
                cb.on_episode_end(episode_idx, self.env.metrics())

            # Optional checkpointing
            if (
                self.run_cfg.save_every_n_episodes
                and episode_idx % self.run_cfg.save_every_n_episodes == 0
            ):
                self.policy.save_checkpoint()
                for cb in self.callbacks:
                    cb.on_checkpoint(episode_idx)
