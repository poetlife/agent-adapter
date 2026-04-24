"""Preset loader for model provider configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelEntry:
    """A single model definition within a preset."""

    name: str
    litellm_model: str
    api_base: str
    max_tokens: int = 8192
    context_length: int = 128000
    description: str = ""
    supports_thinking: bool = False
    default_thinking: str = "disabled"   # "enabled" or "disabled"
    reasoning_effort: str = "high"       # "high" or "max"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Preset:
    """A provider preset containing one or more model definitions."""

    provider: str
    models: list[ModelEntry]
    env_key: str
    description: str = ""

    def resolve_model(self, name: str | None = None) -> ModelEntry | None:
        """Resolve a public model name to a preset model entry.

        Falls back to the first configured model when ``name`` is empty or
        unknown so the CLI keeps a sensible default behavior.
        """
        if name:
            for model in self.models:
                if model.name == name:
                    return model
        return self.models[0] if self.models else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Preset:
        _known_model_keys = {
            "name", "litellm_model", "api_base", "max_tokens",
            "context_length", "description",
            "supports_thinking", "default_thinking", "reasoning_effort",
        }
        models = [
            ModelEntry(
                name=m["name"],
                litellm_model=m["litellm_model"],
                api_base=m.get("api_base", data.get("api_base", "")),
                max_tokens=m.get("max_tokens", 8192),
                context_length=m.get("context_length", 128000),
                description=m.get("description", ""),
                supports_thinking=m.get("supports_thinking", False),
                default_thinking=m.get("default_thinking", "disabled"),
                reasoning_effort=m.get("reasoning_effort", "high"),
                extra={k: v for k, v in m.items() if k not in _known_model_keys},
            )
            for m in data["models"]
        ]
        return cls(
            provider=data["provider"],
            models=models,
            env_key=data["env_key"],
            description=data.get("description", ""),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> Preset:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def get_builtin_presets_dir() -> Path:
    """Return the path to the built-in presets directory."""
    return Path(__file__).parent / "presets"


def load_preset(name: str, custom_dir: Path | None = None) -> Preset:
    """Load a preset by name.

    Searches custom_dir first (if provided), then built-in presets.
    """
    # Try custom directory first
    if custom_dir:
        custom_path = custom_dir / f"{name}.yaml"
        if custom_path.exists():
            return Preset.from_yaml(custom_path)

    # Try built-in presets
    builtin_path = get_builtin_presets_dir() / f"{name}.yaml"
    if builtin_path.exists():
        return Preset.from_yaml(builtin_path)

    available = list_presets(custom_dir)
    raise FileNotFoundError(
        f"Preset '{name}' not found. Available presets: {', '.join(available)}"
    )


def list_presets(custom_dir: Path | None = None) -> list[str]:
    """List all available preset names."""
    presets: set[str] = set()

    # Built-in presets
    builtin_dir = get_builtin_presets_dir()
    if builtin_dir.exists():
        for p in builtin_dir.glob("*.yaml"):
            presets.add(p.stem)

    # Custom presets
    if custom_dir and custom_dir.exists():
        for p in custom_dir.glob("*.yaml"):
            presets.add(p.stem)

    return sorted(presets)


def get_user_config_dir() -> Path:
    """Return the user config directory (~/.config/codex-adapter/)."""
    config_dir = Path.home() / ".config" / "codex-adapter"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_user_presets_dir() -> Path:
    """Return the user custom presets directory."""
    presets_dir = get_user_config_dir() / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir
