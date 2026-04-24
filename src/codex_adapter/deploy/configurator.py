"""Configuration generation for codex-adapter deployment.

Handles: environment file, Codex CLI config, shell profile injection.
Reuses existing logic from codex_setup.py where possible.
"""

from __future__ import annotations

import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Confirm, Prompt

from codex_adapter.codex_setup import generate_codex_config_toml
from codex_adapter.config import list_presets, load_preset, get_user_presets_dir

console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ENV_FILE = Path.home() / ".codex-adapter.env"
CODEX_CONFIG_DIR = Path.home() / ".codex"
CODEX_CONFIG_FILE = CODEX_CONFIG_DIR / "config.toml"
LOG_DIR = Path.home() / ".codex-adapter" / "logs"

SHELL_MARKER_BEGIN = "# >>> codex-adapter >>>"
SHELL_MARKER_END = "# <<< codex-adapter <<<"


# ---------------------------------------------------------------------------
# Environment file
# ---------------------------------------------------------------------------

def generate_env_content(
    api_key: str,
    env_key: str,
    preset_name: str,
    port: int,
    project_dir: str | Path,
) -> str:
    """Generate the content of the environment file.

    This is a pure function — no side effects, easy to test.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""\
# codex-adapter environment configuration
# Auto-generated at {now}

# Model provider API key
{env_key}={api_key}

# Codex CLI points to local proxy
OPENAI_BASE_URL=http://localhost:{port}/v1
OPENAI_API_KEY=sk-placeholder

# Adapter settings
CODEX_ADAPTER_PRESET={preset_name}
CODEX_ADAPTER_PORT={port}
CODEX_ADAPTER_PROJECT_DIR={project_dir}
"""


def write_env_file(
    api_key: str,
    env_key: str,
    preset_name: str,
    port: int,
    project_dir: str | Path,
    path: Path | None = None,
) -> Path:
    """Write the environment file with restricted permissions (0600).

    Returns the path written.
    """
    path = path or ENV_FILE
    content = generate_env_content(api_key, env_key, preset_name, port, project_dir)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    console.print(f"[green]Environment file written:[/] {path} (mode 0600)")
    return path


# ---------------------------------------------------------------------------
# Codex CLI config
# ---------------------------------------------------------------------------

def write_codex_config_file(
    port: int = 4000,
    model: str | None = None,
    all_models: list[str] | None = None,
) -> Path | None:
    """Generate and write Codex CLI config.toml.

    Strategy:
    - If no file exists: create from scratch.
    - If file exists with our provider and correct wire_api: skip.
    - If file exists but has issues (missing wire_api, old wire_api="chat",
      or no codex-adapter section): back up and rewrite.

    Returns the path written, or None if skipped.
    """
    CODEX_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    section = generate_codex_config_toml(port=port, model=model, all_models=all_models)

    if CODEX_CONFIG_FILE.exists():
        existing = CODEX_CONFIG_FILE.read_text()

        # Check if already correctly configured
        has_adapter = "codex-adapter" in existing
        has_correct_wire = 'wire_api = "responses"' in existing
        has_bad_wire = 'wire_api = "chat"' in existing
        has_profiles = "[profiles." in existing

        if has_adapter and has_correct_wire and not has_bad_wire and has_profiles:
            console.print(
                f"[green]Codex config already correctly configured:[/] {CODEX_CONFIG_FILE}"
            )
            return CODEX_CONFIG_FILE

        # Needs fixing — back up and rewrite
        from datetime import datetime
        backup_name = f"config.toml.bak.{datetime.now():%Y%m%d%H%M%S}"
        backup_path = CODEX_CONFIG_DIR / backup_name
        backup_path.write_text(existing)
        console.print(f"[yellow]Backed up existing config to:[/] {backup_path}")

        if has_bad_wire:
            console.print(
                '[yellow]Found wire_api = "chat" (no longer supported by Codex CLI).[/] '
                "Replacing with correct configuration."
            )

        CODEX_CONFIG_FILE.write_text(section)
        console.print(f"[green]Codex config rewritten:[/] {CODEX_CONFIG_FILE}")
    else:
        CODEX_CONFIG_FILE.write_text(section)
        console.print(f"[green]Codex config written:[/] {CODEX_CONFIG_FILE}")

    return CODEX_CONFIG_FILE


# ---------------------------------------------------------------------------
# Shell profile injection
# ---------------------------------------------------------------------------

def _find_shell_profile() -> Path | None:
    """Find the user's shell profile file."""
    candidates = [
        Path.home() / ".zshrc",
        Path.home() / ".bashrc",
        Path.home() / ".profile",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def inject_shell_profile(env_file_path: Path | None = None) -> Path | None:
    """Add source line for the env file into the user's shell profile.

    Returns the profile path if injected, None if skipped.
    """
    env_file_path = env_file_path or ENV_FILE
    profile = _find_shell_profile()

    if profile is None:
        console.print(
            "[yellow]No shell profile found.[/] "
            f"Manually add: [cyan]source {env_file_path}[/]"
        )
        return None

    # Check if already injected
    content = profile.read_text()
    if SHELL_MARKER_BEGIN in content:
        console.print(f"[green]Shell profile already configured:[/] {profile}")
        return profile

    snippet = f"""
{SHELL_MARKER_BEGIN}
# Auto-load codex-adapter environment
if [ -f "{env_file_path}" ]; then
    set -a
    . "{env_file_path}"
    set +a
fi
{SHELL_MARKER_END}
"""
    with open(profile, "a") as f:
        f.write(snippet)

    console.print(f"[green]Injected into:[/] {profile}")
    console.print(f"[dim]Run 'source {profile}' or re-login to activate[/]")
    return profile


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def prompt_interactive() -> dict[str, Any]:
    """Interactively collect deployment configuration.

    Returns a dict with keys: api_key, preset, port.
    """
    custom_dir = get_user_presets_dir()
    preset_names = list_presets(custom_dir)

    # Preset selection
    console.print("\n[bold]Available presets:[/]")
    for i, name in enumerate(preset_names, 1):
        console.print(f"  [cyan]{i})[/] {name}")
    console.print()

    preset_name = Prompt.ask(
        "Select preset",
        default="deepseek",
    )

    # Port
    port_str = Prompt.ask("Proxy port", default="4000")
    try:
        port = int(port_str)
    except ValueError:
        console.print("[yellow]Invalid port, using 4000[/]")
        port = 4000

    # API Key
    preset_obj = load_preset(preset_name, custom_dir)
    env_key = preset_obj.env_key

    # Check if env var already set
    existing_key = os.environ.get(env_key, "")
    if existing_key:
        masked = f"{existing_key[:6]}...{existing_key[-4:]}" if len(existing_key) > 10 else "***"
        use_existing = Confirm.ask(
            f"Found {env_key}={masked}, use it?",
            default=True,
        )
        if use_existing:
            api_key = existing_key
        else:
            api_key = Prompt.ask(f"Enter {env_key}", password=True)
    else:
        api_key = Prompt.ask(f"Enter {env_key}", password=True)

    if not api_key:
        raise ValueError("API key cannot be empty")

    return {
        "api_key": api_key,
        "preset": preset_name,
        "port": port,
    }


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def configure_all(
    api_key: str,
    preset_name: str,
    port: int,
    project_dir: str | Path,
) -> dict[str, Path | None]:
    """Run all configuration steps.

    Returns a dict of paths written (env_file, codex_config, shell_profile).
    """
    custom_dir = get_user_presets_dir()
    preset_obj = load_preset(preset_name, custom_dir)

    # Ensure log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Log directory:[/] {LOG_DIR}")

    # Write env file
    env_path = write_env_file(
        api_key=api_key,
        env_key=preset_obj.env_key,
        preset_name=preset_name,
        port=port,
        project_dir=project_dir,
    )

    # Write Codex config
    default_model = preset_obj.models[0].name if preset_obj.models else None
    all_model_names = [m.name for m in preset_obj.models]
    codex_path = write_codex_config_file(
        port=port, model=default_model, all_models=all_model_names,
    )

    # Inject shell profile
    profile_path = inject_shell_profile(env_path)

    return {
        "env_file": env_path,
        "codex_config": codex_path,
        "shell_profile": profile_path,
    }
