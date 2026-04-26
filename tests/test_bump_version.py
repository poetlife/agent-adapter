"""Tests for scripts/bump_version.py core functions."""

from __future__ import annotations

import textwrap

import pytest

# The script lives outside the package tree, so we import via importlib.
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bump_version.py"
_spec = importlib.util.spec_from_file_location("bump_version", _SCRIPT)
assert _spec and _spec.loader
bump_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bump_version)

read_current_version = bump_version.read_current_version
replace_in_pyproject = bump_version.replace_in_pyproject
replace_in_doc = bump_version.replace_in_doc


# ---------------------------------------------------------------------------
# read_current_version
# ---------------------------------------------------------------------------


class TestReadCurrentVersion:
    def test_extracts_version(self):
        content = textwrap.dedent("""\
            [project]
            name = "codex-adapter"
            version = "0.1.3"
            description = "Some description"
        """)
        assert read_current_version(content) == "0.1.3"

    def test_extracts_higher_version(self):
        content = 'version = "1.23.456"\n'
        assert read_current_version(content) == "1.23.456"

    def test_raises_when_missing(self):
        content = "[project]\nname = 'foo'\n"
        with pytest.raises(ValueError, match="Could not find version"):
            read_current_version(content)


# ---------------------------------------------------------------------------
# replace_in_pyproject
# ---------------------------------------------------------------------------


class TestReplaceInPyproject:
    SAMPLE = textwrap.dedent("""\
        [project]
        name = "codex-adapter"
        version = "0.1.3"
        dependencies = [
            "click==8.1.8",
            "httpx==0.28.1",
            "pytest==9.0.3",
        ]
    """)

    def test_replaces_project_version(self):
        result = replace_in_pyproject(self.SAMPLE, "0.2.0")
        assert 'version = "0.2.0"' in result

    def test_preserves_dependency_pins(self):
        result = replace_in_pyproject(self.SAMPLE, "0.2.0")
        assert "click==8.1.8" in result
        assert "httpx==0.28.1" in result
        assert "pytest==9.0.3" in result

    def test_only_replaces_first_occurrence(self):
        """Even if there were somehow two version lines, only the first changes."""
        content = 'version = "1.0.0"\nversion = "1.0.0"\n'
        result = replace_in_pyproject(content, "2.0.0")
        assert result.count('version = "2.0.0"') == 1
        assert result.count('version = "1.0.0"') == 1


# ---------------------------------------------------------------------------
# replace_in_doc
# ---------------------------------------------------------------------------


class TestReplaceInDoc:
    SAMPLE_URL = (
        "https://github.com/poetlife/agent-adapter/releases/download/"
        "v0.1.3/codex_adapter-0.1.3-py3-none-any.whl"
    )

    def test_replaces_version_in_urls(self):
        result = replace_in_doc(self.SAMPLE_URL, "0.1.3", "0.2.0")
        assert "v0.2.0" in result
        assert "codex_adapter-0.2.0" in result
        assert "0.1.3" not in result

    def test_ignores_angle_bracket_templates(self):
        content = "python scripts/bump_version.py <new-version>"
        result = replace_in_doc(content, "0.1.3", "0.2.0")
        assert "<new-version>" in result

    def test_no_change_when_old_version_absent(self):
        content = "no version here\n"
        result = replace_in_doc(content, "0.1.3", "0.2.0")
        assert result == content

    def test_multiple_occurrences(self):
        content = "v0.1.3 and codex_adapter-0.1.3.whl and 0.1.3.tar.gz"
        result = replace_in_doc(content, "0.1.3", "0.2.0")
        assert result == "v0.2.0 and codex_adapter-0.2.0.whl and 0.2.0.tar.gz"
