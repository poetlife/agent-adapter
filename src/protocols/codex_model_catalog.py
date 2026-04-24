"""Generate Codex-compatible model catalog payloads."""

from __future__ import annotations

from typing import Any, Sequence

from providers.catalog import ModelEntry


def generate_codex_model_catalog(
    models: Sequence[ModelEntry],
    *,
    default_description: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build the Codex CLI model catalog wire format from provider models."""
    catalog_models: list[dict[str, Any]] = []
    for model in models:
        if model.supports_thinking:
            reasoning_levels = [
                {"effort": "low", "description": "Fast responses with lighter reasoning"},
                {"effort": "medium", "description": "Balanced speed and reasoning depth"},
                {"effort": "high", "description": "Greater reasoning depth"},
            ]
            default_reasoning = "medium"
        else:
            reasoning_levels = []
            default_reasoning = None

        catalog_models.append({
            "slug": model.name,
            "display_name": model.name,
            "description": model.description or default_description or f"Model: {model.name}",
            "default_reasoning_level": default_reasoning,
            "supported_reasoning_levels": reasoning_levels,
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": True,
            "priority": 1,
            "additional_speed_tiers": [],
            "availability_nux": None,
            "upgrade": None,
            "base_instructions": "",
            "supports_reasoning_summaries": model.supports_thinking,
            "default_reasoning_summary": "none",
            "support_verbosity": False,
            "default_verbosity": None,
            "apply_patch_tool_type": "freeform",
            "web_search_tool_type": "text",
            "truncation_policy": {"mode": "tokens", "limit": 10000},
            "supports_parallel_tool_calls": True,
            "supports_image_detail_original": False,
            "context_window": model.context_length,
            "max_context_window": model.context_length,
            "effective_context_window_percent": 90,
            "experimental_supported_tools": [],
            "input_modalities": ["text"],
        })

    return {"models": catalog_models}
