from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    """Load YAML config file. Return empty dict when path is None."""
    if path is None:
        return {}
    with path.open() as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def get_cfg(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Read config value with dot path, e.g. get_cfg(cfg, "generation.max_new_tokens").
    """
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
