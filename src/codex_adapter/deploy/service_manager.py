"""Service lifecycle management for codex-adapter.

Manages the adapter proxy as a background process:
start / stop / restart / status / logs / health_check.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

PID_FILE = Path.home() / ".codex-adapter.pid"
ENV_FILE = Path.home() / ".codex-adapter.env"
LOG_DIR = Path.home() / ".codex-adapter" / "logs"

DEFAULT_PORT = 4000
DEFAULT_PRESET = "deepseek"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ServiceStatus:
    """Snapshot of the service state."""

    running: bool
    pid: int | None = None
    port: int = DEFAULT_PORT
    health: dict | None = None  # response from /health, or None


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_env(path: Path | None = None) -> dict[str, str]:
    """Load the env file and return the key=value pairs.

    Also sets them in os.environ so child processes inherit them.
    """
    path = path or ENV_FILE
    env_vars: dict[str, str] = {}
    if not path.exists():
        return env_vars

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            env_vars[key] = value
            os.environ[key] = value

    return env_vars


def _get_config() -> tuple[str, int, Path]:
    """Read preset/port/project_dir from env, with defaults."""
    preset = os.environ.get("CODEX_ADAPTER_PRESET", DEFAULT_PRESET)
    port = int(os.environ.get("CODEX_ADAPTER_PORT", str(DEFAULT_PORT)))
    project_dir = Path(
        os.environ.get("CODEX_ADAPTER_PROJECT_DIR", Path(__file__).parents[2])
    )
    return preset, port, project_dir


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def is_running() -> tuple[bool, int | None]:
    """Check if the adapter process is alive via the PID file.

    Returns (alive: bool, pid: int | None).
    """
    if not PID_FILE.exists():
        return False, None

    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return False, None

    try:
        os.kill(pid, 0)  # signal 0: check existence
        return True, pid
    except OSError:
        # Stale PID file
        PID_FILE.unlink(missing_ok=True)
        return False, None


def health_check(port: int | None = None, retries: int = 10, interval: float = 1.0) -> dict | None:
    """Probe the /health endpoint.

    Returns the JSON response dict if healthy, None otherwise.
    """
    if port is None:
        port = int(os.environ.get("CODEX_ADAPTER_PORT", str(DEFAULT_PORT)))

    url = f"http://localhost:{port}/health"
    for _ in range(retries):
        try:
            resp = httpx.get(url, timeout=3)
            if resp.status_code == 200:
                return resp.json()
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
            pass
        time.sleep(interval)
    return None


def start(
    preset: str | None = None,
    port: int | None = None,
    project_dir: Path | str | None = None,
) -> int:
    """Start the adapter proxy as a background process.

    Returns the PID of the spawned process.
    Raises RuntimeError if the service is already running.
    """
    alive, existing_pid = is_running()
    if alive:
        console.print(f"[yellow]Service already running (PID {existing_pid})[/]")
        return existing_pid  # type: ignore[return-value]

    # Load env config
    load_env()
    env_preset, env_port, env_project_dir = _get_config()
    preset = preset or env_preset
    port = port or env_port
    project_dir = Path(project_dir) if project_dir else env_project_dir

    # Prepare log file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"codex-adapter-{datetime.now():%Y%m%d}.log"

    cmd = _build_start_command(preset, port)

    console.print(f"[blue]Starting codex-adapter (preset={preset}, port={port}) ...[/]")

    # Open log file for stdout/stderr redirection
    log_fh = open(log_file, "a")

    proc = subprocess.Popen(
        cmd,
        cwd=project_dir,
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,  # detach from terminal
        env=os.environ.copy(),
    )

    # Write PID
    PID_FILE.write_text(str(proc.pid))
    console.print(f"[blue]Process started (PID {proc.pid}), log: {log_file}[/]")

    # Wait for health
    console.print("[blue]Waiting for service to become healthy ...[/]")
    health = health_check(port, retries=15)
    if health:
        console.print(f"[bold green]Service is up![/] {health}")
    else:
        console.print(
            "[bold yellow]Service started but health check timed out.[/] "
            f"Check logs: {log_file}"
        )

    return proc.pid


def stop() -> bool:
    """Stop the adapter process gracefully.

    Returns True if the process was stopped, False if it wasn't running.
    """
    alive, pid = is_running()
    if not alive:
        console.print("[yellow]Service is not running[/]")
        PID_FILE.unlink(missing_ok=True)
        return False

    console.print(f"[blue]Stopping process (PID {pid}) ...[/]")

    # SIGTERM first
    os.kill(pid, signal.SIGTERM)  # type: ignore[arg-type]

    # Wait up to 10 seconds
    for _ in range(10):
        try:
            os.kill(pid, 0)  # type: ignore[arg-type]
        except OSError:
            break  # process exited
        time.sleep(1)
    else:
        # Still alive, force kill
        console.print("[yellow]Process did not exit, sending SIGKILL ...[/]")
        try:
            os.kill(pid, signal.SIGKILL)  # type: ignore[arg-type]
        except OSError:
            pass

    PID_FILE.unlink(missing_ok=True)
    console.print("[green]Service stopped[/]")
    return True


def restart(
    preset: str | None = None,
    port: int | None = None,
    project_dir: Path | str | None = None,
) -> int:
    """Stop then start the service. Returns the new PID."""
    stop()
    time.sleep(1)
    return start(preset=preset, port=port, project_dir=project_dir)


def status() -> ServiceStatus:
    """Get the current service status."""
    load_env()
    _, port, _ = _get_config()

    alive, pid = is_running()
    health = None
    if alive:
        health = health_check(port, retries=2, interval=0.5)

    return ServiceStatus(running=alive, pid=pid, port=port, health=health)


def print_status() -> None:
    """Pretty-print the service status."""
    s = status()

    console.print()
    console.print("[bold]codex-adapter service status[/]")
    console.print(f"  Process: ", end="")
    if s.running:
        console.print(f"[bold green]running[/] (PID {s.pid})")
    else:
        console.print("[bold red]not running[/]")

    console.print(f"  Port:    [cyan]{s.port}[/]")

    if s.health:
        console.print(f"  Health:  [green]{s.health}[/]")
    elif s.running:
        console.print("  Health:  [yellow]no response[/]")

    console.print()


def logs(follow: bool = False, lines: int = 50) -> None:
    """Display service logs.

    If follow=True, tails the log file continuously (blocking).
    """
    log_file = _find_latest_log()
    if log_file is None:
        console.print("[yellow]No log files found[/]")
        return

    console.print(f"[dim]Log file: {log_file}[/]\n")

    if follow:
        # Use tail -f for real-time following
        try:
            subprocess.run(["tail", "-f", str(log_file)])
        except KeyboardInterrupt:
            pass
    else:
        # Read last N lines
        try:
            all_lines = log_file.read_text().splitlines()
            for line in all_lines[-lines:]:
                console.print(line)
        except OSError as e:
            console.print(f"[red]Error reading log: {e}[/]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_uv() -> Path | None:
    """Locate the uv binary."""
    import shutil

    path = shutil.which("uv")
    if path:
        return Path(path)

    # Common installation paths
    for p in [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
    ]:
        if p.exists() and os.access(p, os.X_OK):
            return p

    return None


def _build_start_command(preset: str, port: int) -> list[str]:
    """Build the background service command in the current Python environment."""
    return [
        sys.executable,
        "-m",
        "codex_adapter.cli",
        "start",
        "--preset",
        preset,
        "--port",
        str(port),
    ]


def _find_latest_log() -> Path | None:
    """Find the most recent log file."""
    if not LOG_DIR.exists():
        return None

    logs = sorted(LOG_DIR.glob("codex-adapter-*.log"), reverse=True)
    return logs[0] if logs else None
