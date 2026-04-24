"""Shared runtime path helpers."""

from __future__ import annotations

from pathlib import Path


def get_app_config_dir(app_name: str) -> Path:
    """Return and create the per-app config directory under ~/.config."""
    config_dir = Path.home() / ".config" / app_name
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
