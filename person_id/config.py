from __future__ import annotations

import yaml

DEFAULT_PATH = "config/default.yaml"


def load_config(path: str = DEFAULT_PATH, **overrides) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["source"] = str(cfg.get("source", "0"))
    return cfg
