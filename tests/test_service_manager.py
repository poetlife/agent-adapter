"""Tests for background service management helpers."""

import sys

from codex_adapter.deploy import service_manager


def test_build_start_command_uses_current_python_module():
    cmd = service_manager._build_start_command("deepseek", 4000)

    assert cmd == [
        sys.executable,
        "-m",
        "codex_adapter.cli",
        "start",
        "--preset",
        "deepseek",
        "--port",
        "4000",
    ]
