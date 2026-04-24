"""CLI interface for codex-adapter."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from codex_adapter import __version__
from codex_adapter.codex_setup import print_setup_instructions, write_codex_config
from providers.catalog import (
    Preset,
    get_builtin_presets_dir,
    get_user_presets_dir,
    list_presets,
    load_preset,
)
from entrypoints.responses_proxy import start_proxy

console = Console()


# ===========================================================================
# Root group
# ===========================================================================

@click.group()
@click.version_option(version=__version__, prog_name="codex-adapter")
def main() -> None:
    """Adapt OpenAI Codex CLI to work with DeepSeek and other models.

    Uses a local proxy to translate between OpenAI-compatible
    API calls and various model provider APIs.
    """


# ===========================================================================
# start — foreground proxy (existing)
# ===========================================================================

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
    help="Port for the proxy server.",
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
    help="Enable debug logging.",
)
def start(preset: str, port: int, host: str, debug: bool) -> None:
    """Start the proxy server in the foreground."""
    try:
        custom_dir = get_user_presets_dir()
        preset_obj = load_preset(preset, custom_dir)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)

    start_proxy(preset_obj, port=port, host=host, debug=debug)


# ===========================================================================
# list — list presets (existing)
# ===========================================================================

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


# ===========================================================================
# setup — configure everything with one command
# ===========================================================================

@main.command()
@click.option(
    "--preset", "-p",
    default="deepseek",
    show_default=True,
    help="Model provider preset (e.g. deepseek).",
)
@click.option(
    "--api-key", "-k",
    default=None,
    help="API key for the model provider. If omitted, reads from env or prompts.",
)
@click.option(
    "--port",
    default=4000,
    show_default=True,
    help="Proxy port.",
)
@click.option(
    "--show-only",
    is_flag=True,
    default=False,
    help="Only print instructions, don't write any files.",
)
def setup(preset: str, api_key: str | None, port: int, show_only: bool) -> None:
    """Configure codex-adapter for a preset: env file, Codex CLI config, shell profile.

    \b
    Examples:
      # Provide token directly — everything else is automatic:
      codex-adapter setup -p deepseek -k sk-your-key

      # Token already in env — just pick a preset:
      export DEEPSEEK_API_KEY=sk-xxx
      codex-adapter setup -p deepseek

      # Interactive — will prompt for key:
      codex-adapter setup

      # Just show instructions without writing files:
      codex-adapter setup --show-only
    """
    import os
    from codex_adapter.deploy.configurator import configure_all

    try:
        custom_dir = get_user_presets_dir()
        preset_obj = load_preset(preset, custom_dir)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)

    default_model = preset_obj.models[0].name if preset_obj.models else None

    # --- Show-only mode: just print instructions, no writes ---
    if show_only:
        print_setup_instructions(
            port=port,
            model=default_model,
            provider=preset_obj.provider,
        )
        return

    # --- Resolve API key ---
    if api_key:
        resolved_key = api_key
    else:
        # Try environment variable
        resolved_key = os.environ.get(preset_obj.env_key, "")

    if not resolved_key:
        # Interactive prompt
        from rich.prompt import Prompt
        resolved_key = Prompt.ask(
            f"Enter [bold]{preset_obj.env_key}[/]",
            password=True,
        )

    if not resolved_key:
        console.print("[bold red]Error:[/] API key is required.")
        raise SystemExit(1)

    # --- Do everything ---
    project_dir = _get_project_dir()

    console.print()
    console.print("[bold]Codex Adapter Setup[/]")
    console.print(f"  Preset:  [cyan]{preset}[/]")
    console.print(f"  Port:    [cyan]{port}[/]")
    if default_model:
        console.print(f"  Model:   [cyan]{default_model}[/]")
    console.print()

    paths = configure_all(
        api_key=resolved_key,
        preset_name=preset,
        port=port,
        project_dir=project_dir,
    )

    # --- Summary ---
    console.print()
    console.print("[bold green]Setup complete![/]")
    console.print()
    console.print("[bold]What was configured:[/]")
    for label, p in [
        ("Environment file", paths.get("env_file")),
        ("Codex CLI config", paths.get("codex_config")),
        ("Shell profile", paths.get("shell_profile")),
    ]:
        if p:
            console.print(f"  [green]✓[/] {label}: [dim]{p}[/]")
        else:
            console.print(f"  [yellow]-[/] {label}: skipped")

    console.print()
    console.print("[bold]Next steps:[/]")
    console.print(f"  [cyan]source ~/.codex-adapter.env[/]")
    console.print(f"  [cyan]codex-adapter service start[/]          # start proxy in background")
    model_flag = f" --model {default_model}" if default_model else ""
    console.print(f"  [cyan]codex{model_flag} \"your prompt\"[/]   # use codex")


# ===========================================================================
# deploy — one-shot install + configure + start
# ===========================================================================

def _get_project_dir() -> Path:
    """Get the project root directory (where pyproject.toml lives)."""
    # Walk up from this file's location
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d
        d = d.parent
    return Path.cwd()


@main.command()
@click.option("--api-key", default=None, help="Model provider API key.")
@click.option("--preset", "-p", default="deepseek", show_default=True, help="Model preset.")
@click.option("--port", default=4000, show_default=True, help="Proxy port.")
@click.option("--skip-install", is_flag=True, default=False, help="Skip dependency installation.")
@click.option("--use-systemd", is_flag=True, default=False, help="Install and use systemd service.")
@click.option(
    "--non-interactive", is_flag=True, default=False,
    help="Non-interactive mode (requires --api-key or env var).",
)
def deploy(
    api_key: str | None,
    preset: str,
    port: int,
    skip_install: bool,
    use_systemd: bool,
    non_interactive: bool,
) -> None:
    """One-command deployment: install deps, configure, and start service."""
    from codex_adapter.deploy.installer import install_all
    from codex_adapter.deploy.configurator import configure_all, prompt_interactive
    from codex_adapter.deploy import service_manager
    from codex_adapter.deploy import systemd

    project_dir = _get_project_dir()

    console.print()
    console.print("[bold]Codex Adapter — Deploy[/]")
    console.print(f"  Preset:  [cyan]{preset}[/]")
    console.print(f"  Port:    [cyan]{port}[/]")
    console.print(f"  Project: [dim]{project_dir}[/]")
    console.print()

    # --- Step 1: Install dependencies ---
    if skip_install:
        console.print("[yellow]Skipping dependency installation[/]\n")
    else:
        console.print("[bold]Step 1: Install dependencies[/]")
        install_all(project_dir)
        console.print()

    # --- Step 2: Configure ---
    console.print("[bold]Step 2: Configure[/]")

    if api_key:
        pass  # use provided key
    elif non_interactive:
        # Try to read from env
        import os
        custom_dir = get_user_presets_dir()
        preset_obj = load_preset(preset, custom_dir)
        api_key = os.environ.get(preset_obj.env_key, "")
        if not api_key:
            console.print(
                f"[bold red]Error:[/] --non-interactive requires --api-key or "
                f"the {preset_obj.env_key} env var"
            )
            raise SystemExit(1)
    else:
        # Interactive prompt
        config = prompt_interactive()
        api_key = config["api_key"]
        preset = config.get("preset", preset)
        port = config.get("port", port)

    configure_all(
        api_key=api_key,
        preset_name=preset,
        port=port,
        project_dir=project_dir,
    )
    console.print()

    # --- Step 3: Optionally install systemd ---
    if use_systemd:
        console.print("[bold]Step 3: Install systemd service[/]")
        systemd.install_unit(project_dir, preset=preset, port=port)
        console.print()

    # --- Step 4: Start service ---
    console.print("[bold]Step 4: Start service[/]")
    service_manager.start(preset=preset, port=port, project_dir=project_dir)
    console.print()

    # --- Step 5: Print usage ---
    console.print("[bold green]Deployment complete![/]\n")
    console.print("Usage:")
    console.print("  [cyan]source ~/.codex-adapter.env[/]")
    console.print(f"  [cyan]codex --model deepseek-v4-flash \"your prompt\"[/]")
    console.print()
    console.print("Service management:")
    console.print("  [cyan]codex-adapter service status[/]")
    console.print("  [cyan]codex-adapter service logs -f[/]")
    console.print("  [cyan]codex-adapter service restart[/]")
    console.print("  [cyan]codex-adapter service stop[/]")
    console.print()


# ===========================================================================
# service — service lifecycle management
# ===========================================================================

@main.group()
def service() -> None:
    """Manage the codex-adapter background service."""


@service.command(name="start")
@click.option("--preset", "-p", default=None, help="Override preset.")
@click.option("--port", default=None, type=int, help="Override port.")
def service_start(preset: str | None, port: int | None) -> None:
    """Start the adapter proxy in the background."""
    from codex_adapter.deploy import service_manager

    service_manager.load_env()
    service_manager.start(preset=preset, port=port, project_dir=_get_project_dir())


@service.command(name="stop")
def service_stop() -> None:
    """Stop the background adapter proxy."""
    from codex_adapter.deploy import service_manager

    service_manager.stop()


@service.command(name="restart")
@click.option("--preset", "-p", default=None, help="Override preset.")
@click.option("--port", default=None, type=int, help="Override port.")
def service_restart(preset: str | None, port: int | None) -> None:
    """Restart the background adapter proxy."""
    from codex_adapter.deploy import service_manager

    service_manager.load_env()
    service_manager.restart(preset=preset, port=port, project_dir=_get_project_dir())


@service.command(name="status")
def service_status() -> None:
    """Show the service status and health check."""
    from codex_adapter.deploy import service_manager

    service_manager.load_env()
    service_manager.print_status()


@service.command(name="logs")
@click.option("-f", "--follow", is_flag=True, default=False, help="Follow log output.")
@click.option("-n", "--lines", default=50, show_default=True, help="Number of lines to show.")
def service_logs(follow: bool, lines: int) -> None:
    """Show service logs."""
    from codex_adapter.deploy import service_manager

    service_manager.logs(follow=follow, lines=lines)


@service.command(name="install-systemd")
@click.option("--preset", "-p", default="deepseek", show_default=True)
@click.option("--port", default=4000, show_default=True)
def service_install_systemd(preset: str, port: int) -> None:
    """Install the systemd service unit."""
    from codex_adapter.deploy import systemd

    systemd.install_unit(_get_project_dir(), preset=preset, port=port)


@service.command(name="uninstall-systemd")
def service_uninstall_systemd() -> None:
    """Uninstall the systemd service unit."""
    from codex_adapter.deploy import systemd

    systemd.uninstall_unit()


if __name__ == "__main__":
    main()
