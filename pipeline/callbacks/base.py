from __future__ import annotations

from typing import Any, Dict


class Callback:
    """Base callback with no-ops for all hooks."""

    def on_episode_begin(self, episode_idx: int, mode: str) -> None:
        return None

    def on_step(self, episode_idx: int, step_idx: int, reward: float, info: Dict[str, Any]) -> None:
        return None

    def on_episode_end(self, episode_idx: int, metrics: Dict[str, Any]) -> None:
        return None

    def on_learn_step(self, episode_idx: int, learn_stats: Dict[str, Any]) -> None:
        return None

    def on_checkpoint(self, episode_idx: int) -> None:
        return None


class DefaultLoggingCallback(Callback):
    """Console logger for progress visibility."""

    def on_episode_begin(self, episode_idx: int, mode: str) -> None:
        print(f"[EP {episode_idx}] Mode={mode}")

    def on_step(self, episode_idx: int, step_idx: int, reward: float, info: Dict[str, Any]) -> None:
        print(f"  step={step_idx} reward={reward:.4f} info={info}")

    def on_episode_end(self, episode_idx: int, metrics: Dict[str, Any]) -> None:
        print(f"[EP {episode_idx}] metrics={metrics}")

    def on_learn_step(self, episode_idx: int, learn_stats: Dict[str, Any]) -> None:
        print(f"[EP {episode_idx}] learn={learn_stats}")

    def on_checkpoint(self, episode_idx: int) -> None:
        print(f"[EP {episode_idx}] checkpoint saved")

