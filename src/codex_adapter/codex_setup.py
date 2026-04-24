"""Auto-configure Codex CLI to use the adapter proxy."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from codex_adapter.config import ModelEntry

console = Console()

CODEX_CONFIG_DIR = Path.home() / ".codex"
MODEL_CATALOG_PATH = CODEX_CONFIG_DIR / "model-catalog.json"


def detect_codex_cli() -> str | None:
    """Detect if Codex CLI is installed and return its path."""
    return shutil.which("codex")


def generate_shell_exports(port: int = 4000, model: str | None = None) -> str:
    """Generate shell export commands for Codex CLI configuration."""
    lines = [
        f'export OPENAI_BASE_URL="http://localhost:{port}/v1"',
        'export OPENAI_API_KEY="sk-placeholder"',
    ]
    return "\n".join(lines)


def generate_codex_config_toml(
    port: int = 4000,
    model: str | None = None,
    all_models: list[str] | None = None,
) -> str:
    """Generate Codex CLI config.toml content for the adapter proxy.

    Codex CLI talks Responses API to our proxy (wire_api = "responses"),
    the proxy translates to Chat Completions before forwarding to the
    actual model provider.

    Generates a [profiles.<name>] section for each model so users can
    switch with `codex -p <name>` (e.g. `codex -p flash`, `codex -p pro`).
    """
    model_line = f'model = "{model}"' if model else '# model = "deepseek-v4-flash"'
    catalog_path = str(MODEL_CATALOG_PATH)

    lines = [
        "# Codex Adapter - Auto-generated config",
        "# Place this in ~/.codex/config.toml",
        "#",
        "# Switch models with:  codex -p flash  /  codex -p pro",
        "",
        model_line,
        'model_provider = "codex-adapter"',
        f'model_catalog_json = "{catalog_path}"',
        "",
        "[model_providers.codex-adapter]",
        'name = "Codex Adapter Proxy"',
        f'base_url = "http://localhost:{port}/v1"',
        'env_key = "OPENAI_API_KEY"',
        'wire_api = "responses"',
    ]

    # Generate a profile for each model
    if all_models:
        lines.append("")
        for m in all_models:
            short = _short_profile_name(m)
            lines.append(f"[profiles.{short}]")
            lines.append(f'model = "{m}"')
            lines.append("")

    return "\n".join(lines) + "\n"


def _short_profile_name(model_name: str) -> str:
    """Derive a short profile name from a model slug.

    'deepseek-v4-flash' → 'flash'
    'deepseek-v4-pro'   → 'pro'
    'my-model'          → 'my-model'
    """
    parts = model_name.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) >= 2:
        return parts[1]
    return model_name


def generate_model_catalog(models: list[ModelEntry]) -> dict:
    """Generate a Codex CLI model catalog (ModelsResponse format).

    This is the JSON structure Codex CLI deserializes as ModelsResponse{models: [ModelInfo]}.
    When loaded via model_catalog_json, it replaces the remote OpenAI catalog,
    so Codex TUI recognizes our custom models without the fallback metadata warning.
    """
    catalog_models = []
    for m in models:
        if m.supports_thinking:
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
            "slug": m.name,
            "display_name": m.name,
            "description": m.description or f"Model: {m.name}",
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
            "supports_reasoning_summaries": m.supports_thinking,
            "default_reasoning_summary": "none",
            "support_verbosity": False,
            "default_verbosity": None,
            "apply_patch_tool_type": "freeform",
            "web_search_tool_type": "text",
            "truncation_policy": {"mode": "tokens", "limit": 10000},
            "supports_parallel_tool_calls": True,
            "supports_image_detail_original": False,
            "context_window": m.context_length,
            "max_context_window": m.context_length,
            "effective_context_window_percent": 90,
            "experimental_supported_tools": [],
            "input_modalities": ["text"],
        })

    return {"models": catalog_models}


def write_model_catalog(models: list[ModelEntry]) -> Path:
    """Write the model catalog JSON file for Codex CLI.

    Returns the path written.
    """
    CODEX_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    catalog = generate_model_catalog(models)
    MODEL_CATALOG_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    console.print(f"[green]Model catalog written:[/] {MODEL_CATALOG_PATH}")
    return MODEL_CATALOG_PATH


def print_setup_instructions(
    port: int = 4000,
    model: str | None = None,
    provider: str = "deepseek",
) -> None:
    """Print setup instructions for the user."""
    codex_path = detect_codex_cli()

    console.print()
    if codex_path:
        console.print(f"[bold green]Codex CLI detected:[/] {codex_path}")
    else:
        console.print(
            "[bold yellow]Codex CLI not detected.[/] "
            "Install it with: [cyan]npm install -g @openai/codex[/]"
        )

    console.print()

    # Shell exports
    exports = generate_shell_exports(port, model)
    console.print(Panel(
        Syntax(exports, "bash", theme="monokai"),
        title="[bold]Step 1: Set Environment Variables[/]",
        subtitle="Add to your ~/.bashrc or ~/.zshrc",
        border_style="green",
    ))

    console.print()

    # Codex config
    config_toml = generate_codex_config_toml(port, model)
    console.print(Panel(
        Syntax(config_toml, "toml", theme="monokai"),
        title="[bold]Step 2 (Optional): Codex Config File[/]",
        subtitle="~/.codex/config.toml",
        border_style="blue",
    ))

    console.print()

    # Usage example
    model_flag = f" --model {model}" if model else ""
    usage = f"""\
# Start the adapter proxy (in terminal 1):
codex-adapter start --preset {provider}

# Use Codex CLI (in terminal 2):
codex{model_flag} "help me fix this bug"

# Switch models with profiles:
codex -p flash "your prompt"    # uses flash model
codex -p pro "your prompt"      # uses pro model
"""
    console.print(Panel(
        Syntax(usage, "bash", theme="monokai"),
        title="[bold]Step 3: Usage[/]",
        border_style="cyan",
    ))


def write_codex_config(port: int = 4000, model: str | None = None) -> Path | None:
    """Write Codex config.toml if the user confirms.

    Returns the path written, or None if skipped.
    """
    codex_config_dir = Path.home() / ".codex"
    codex_config_path = codex_config_dir / "config.toml"

    if codex_config_path.exists():
        console.print(
            f"[bold yellow]Warning:[/] {codex_config_path} already exists. "
            "Skipping auto-write to avoid overwriting your config."
        )
        console.print(
            "You can manually add the [model_providers.codex-adapter] section."
        )
        return None

    codex_config_dir.mkdir(parents=True, exist_ok=True)
    content = generate_codex_config_toml(port, model)
    codex_config_path.write_text(content)
    console.print(f"[bold green]Wrote config to:[/] {codex_config_path}")
    return codex_config_path
