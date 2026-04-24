"""Auto-configure Codex CLI to use the adapter proxy."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


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


def generate_codex_config_toml(port: int = 4000, model: str | None = None) -> str:
    """Generate Codex CLI config.toml content for the adapter proxy.

    Codex CLI talks Responses API to our proxy (wire_api = "responses"),
    the proxy translates to Chat Completions before forwarding to the
    actual model provider.
    """
    model_line = f'model = "{model}"' if model else '# model = "deepseek-v4-flash"'
    return f"""\
# Codex Adapter - Auto-generated config
# Place this in ~/.codex/config.toml

{model_line}
model_provider = "codex-adapter"

[model_providers.codex-adapter]
name = "Codex Adapter Proxy"
base_url = "http://localhost:{port}/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
"""


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
