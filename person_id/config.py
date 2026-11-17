from __future__ import annotations

import yaml

DEFAULT_PATH = "config/default.yaml"
PER_MODEL_KEYS = ("gallery_path", "recognize_threshold", "unknown_threshold")


def model_settings(cfg: dict, identifier: str) -> dict:
    """gallery_path and thresholds for one backend: its `models:` entry, else the top-level values."""
    entry = (cfg.get("models") or {}).get(identifier) or {}
    fallback = cfg.get("fallback") or cfg
    return {k: entry.get(k, fallback.get(k)) for k in PER_MODEL_KEYS}


def load_config(path: str = DEFAULT_PATH, **overrides) -> dict:
    """Top-level gallery_path / thresholds end up as those of the selected identifier."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["source"] = str(cfg.get("source", "0"))
    cfg.setdefault("models", {})
    cfg["fallback"] = {k: cfg.get(k) for k in PER_MODEL_KEYS}
    cfg.update(model_settings(cfg, cfg.get("identifier", "dummy")))
    return cfg
