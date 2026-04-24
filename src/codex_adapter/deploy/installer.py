"""Dependency detection and installation for codex-adapter.

Checks and installs: Python 3.12+, uv, Node.js 20+, Codex CLI,
and project Python dependencies.

All functions are importable for programmatic use.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single dependency check."""

    name: str
    ok: bool
    version: str = ""
    message: str = ""


@dataclass
class OSInfo:
    """Detected operating system information."""

    family: str  # "debian" | "rhel" | "unknown"
    id: str = ""
    pretty_name: str = ""


# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

def detect_os() -> OSInfo:
    """Detect the Linux distribution family."""
    if platform.system() != "Linux":
        return OSInfo(family="unknown", pretty_name=platform.system())

    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return OSInfo(family="unknown", pretty_name="Linux (unknown distro)")

    data: dict[str, str] = {}
    for line in os_release.read_text().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            data[k.strip()] = v.strip().strip('"')

    distro_id = data.get("ID", "").lower()
    pretty = data.get("PRETTY_NAME", distro_id)

    debian_ids = {"ubuntu", "debian", "linuxmint", "pop"}
    rhel_ids = {"centos", "rhel", "rocky", "almalinux", "fedora"}

    if distro_id in debian_ids:
        family = "debian"
    elif distro_id in rhel_ids:
        family = "rhel"
    else:
        family = "debian"  # best-effort fallback

    return OSInfo(family=family, id=distro_id, pretty_name=pretty)


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return the result (non-raising)."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        **kwargs,
    )


def _get_version(cmd: list[str]) -> str | None:
    """Run a command and extract a version string from its output."""
    try:
        r = _run(cmd)
        output = (r.stdout + r.stderr).strip()
        # Match common version patterns: X.Y.Z, vX.Y.Z, etc.
        m = re.search(r"(\d+\.\d+(?:\.\d+)?)", output)
        return m.group(1) if m else output[:40]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _version_gte(version: str, major_min: int, minor_min: int = 0) -> bool:
    """Check if a version string is >= major_min.minor_min."""
    parts = version.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor) >= (major_min, minor_min)
    except (ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def check_python() -> CheckResult:
    """Check if Python 3.12+ is available."""
    for cmd in ["python3", "python3.12", "python"]:
        path = shutil.which(cmd)
        if path:
            ver = _get_version([cmd, "--version"])
            if ver and _version_gte(ver, 3, 12):
                return CheckResult("Python 3.12+", ok=True, version=ver)
    return CheckResult("Python 3.12+", ok=False, message="not found or version < 3.12")


def check_uv() -> CheckResult:
    """Check if uv is available."""
    path = shutil.which("uv")
    if path:
        ver = _get_version(["uv", "--version"]) or "unknown"
        return CheckResult("uv", ok=True, version=ver)
    return CheckResult("uv", ok=False, message="not found")


def check_node() -> CheckResult:
    """Check if Node.js 20+ is available."""
    path = shutil.which("node")
    if path:
        ver = _get_version(["node", "--version"])
        if ver and _version_gte(ver, 20):
            return CheckResult("Node.js 20+", ok=True, version=ver)
        return CheckResult("Node.js 20+", ok=False, version=ver or "", message="version < 20")
    return CheckResult("Node.js 20+", ok=False, message="not found")


def check_codex_cli() -> CheckResult:
    """Check if Codex CLI is installed."""
    path = shutil.which("codex")
    if path:
        ver = _get_version(["codex", "--version"]) or "installed"
        return CheckResult("Codex CLI", ok=True, version=ver)
    return CheckResult("Codex CLI", ok=False, message="not found")


def check_all() -> list[CheckResult]:
    """Run all dependency checks and return results."""
    return [
        check_python(),
        check_uv(),
        check_node(),
        check_codex_cli(),
    ]


def print_check_results(results: list[CheckResult]) -> None:
    """Pretty-print dependency check results as a table."""
    table = Table(title="Dependency Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Version", style="dim")
    table.add_column("Note", style="dim")

    for r in results:
        status = "[bold green]OK[/]" if r.ok else "[bold red]MISSING[/]"
        table.add_row(r.name, status, r.version, r.message)

    console.print(table)


# ---------------------------------------------------------------------------
# Installers
# ---------------------------------------------------------------------------

def install_uv() -> None:
    """Install uv via the official installer script."""
    console.print("[blue]Installing uv ...[/]")
    subprocess.run(
        ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        check=True,
    )
    # Ensure it's on PATH for subsequent calls in this process
    for p in [Path.home() / ".local" / "bin", Path.home() / ".cargo" / "bin"]:
        if (p / "uv").exists():
            os.environ["PATH"] = f"{p}:{os.environ.get('PATH', '')}"
            break
    console.print("[green]uv installed successfully[/]")


def install_node(os_info: OSInfo | None = None) -> None:
    """Install Node.js 20 via NodeSource."""
    if os_info is None:
        os_info = detect_os()

    console.print("[blue]Installing Node.js 20 ...[/]")

    if os_info.family == "debian":
        subprocess.run(
            ["bash", "-c", "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"],
            check=True,
        )
        subprocess.run(["sudo", "apt-get", "install", "-y", "nodejs"], check=True)
    elif os_info.family == "rhel":
        subprocess.run(
            ["bash", "-c", "curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo -E bash -"],
            check=True,
        )
        pkg_cmd = "dnf" if shutil.which("dnf") else "yum"
        subprocess.run(["sudo", pkg_cmd, "install", "-y", "nodejs"], check=True)
    else:
        raise RuntimeError(f"Unsupported OS family: {os_info.family}")

    console.print("[green]Node.js installed successfully[/]")


def install_codex_cli() -> None:
    """Install Codex CLI via npm."""
    console.print("[blue]Installing Codex CLI ...[/]")
    subprocess.run(["npm", "install", "-g", "@openai/codex"], check=True)
    console.print("[green]Codex CLI installed successfully[/]")


def install_adapter_deps(project_dir: Path | str) -> None:
    """Install codex-adapter Python dependencies via uv sync."""
    project_dir = Path(project_dir)
    if not (project_dir / "pyproject.toml").exists():
        raise FileNotFoundError(f"pyproject.toml not found in {project_dir}")

    console.print("[blue]Installing codex-adapter dependencies ...[/]")
    subprocess.run(["uv", "sync"], cwd=project_dir, check=True)
    console.print("[green]Dependencies installed successfully[/]")


def install_all(project_dir: Path | str) -> list[CheckResult]:
    """Run all checks and install missing dependencies.

    Returns the final check results after installation attempts.
    """
    project_dir = Path(project_dir)
    os_info = detect_os()

    console.print(f"\n[bold]System:[/] {os_info.pretty_name} (family={os_info.family})\n")

    # --- uv ---
    r = check_uv()
    if r.ok:
        console.print(f"[green]uv already installed:[/] {r.version}")
    else:
        install_uv()

    # --- Node.js ---
    r = check_node()
    if r.ok:
        console.print(f"[green]Node.js already installed:[/] {r.version}")
    else:
        install_node(os_info)

    # --- Codex CLI ---
    r = check_codex_cli()
    if r.ok:
        console.print(f"[green]Codex CLI already installed:[/] {r.version}")
    else:
        install_codex_cli()

    # --- Python deps ---
    install_adapter_deps(project_dir)

    # Final check
    results = check_all()
    console.print()
    print_check_results(results)
    return results
