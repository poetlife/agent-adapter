"""CLI interface for codex-adapter."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from codex_adapter import __version__
from codex_adapter.codex_setup import print_setup_instructions, write_codex_config
from codex_adapter.config import (
    Preset,
    get_builtin_presets_dir,
    get_user_presets_dir,
    list_presets,
    load_preset,
)
from codex_adapter.proxy import start_proxy

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="codex-adapter")
def main() -> None:
    """Adapt OpenAI Codex CLI to work with DeepSeek and other models.

    Uses LiteLLM as a local proxy to translate between OpenAI-compatible
    API calls and various model provider APIs.
    """


@main.command()
@click.option(
    "--preset", "-p",
    required=True,
    help="Model provider preset to use (e.g., deepseek).",
)
@click.option(
    "--port",
    default=4000,
    show_default=True,
    help="Port for the LiteLLM proxy server.",
)
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Host to bind the proxy server.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging for LiteLLM.",
)
def start(preset: str, port: int, host: str, debug: bool) -> None:
    """Start the LiteLLM proxy server with a model preset."""
    try:
        custom_dir = get_user_presets_dir()
        preset_obj = load_preset(preset, custom_dir)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)

    start_proxy(preset_obj, port=port, host=host, debug=debug)


@main.command(name="list")
def list_cmd() -> None:
    """List all available model presets."""
    custom_dir = get_user_presets_dir()
    preset_names = list_presets(custom_dir)

    if not preset_names:
        console.print("[yellow]No presets found.[/]")
        return

    table = Table(title="Available Model Presets")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Provider", style="green")
    table.add_column("Models", style="white")
    table.add_column("Description", style="dim")

    for name in preset_names:
        try:
            p = load_preset(name, custom_dir)
            models = ", ".join(m.name for m in p.models)
            table.add_row(name, p.provider, models, p.description)
        except Exception:
            table.add_row(name, "?", "?", "[red]Failed to load[/]")

    console.print(table)

    console.print()
    console.print(f"[dim]Built-in presets: {get_builtin_presets_dir()}[/]")
    console.print(f"[dim]Custom presets:   {custom_dir}[/]")


@main.command()
@click.option(
    "--preset", "-p",
    default="deepseek",
    show_default=True,
    help="Model provider preset.",
)
@click.option(
    "--port",
    default=4000,
    show_default=True,
    help="Proxy port.",
)
@click.option(
    "--write-config",
    is_flag=True,
    default=False,
    help="Write Codex CLI config.toml automatically.",
)
def setup(preset: str, port: int, write_config: bool) -> None:
    """Show setup instructions for Codex CLI integration."""
    try:
        custom_dir = get_user_presets_dir()
        preset_obj = load_preset(preset, custom_dir)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)

    default_model = preset_obj.models[0].name if preset_obj.models else None

    print_setup_instructions(
        port=port,
        model=default_model,
        provider=preset_obj.provider,
    )

    if write_config:
        write_codex_config(port=port, model=default_model)


if __name__ == "__main__":
    main()
