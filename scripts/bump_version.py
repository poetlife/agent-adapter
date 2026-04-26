#!/usr/bin/env python3
"""Bump the project version across all files that contain it.

Updates: pyproject.toml, README.md, docs/release.md
Optionally commits and tags the change.

Usage:
    python scripts/bump_version.py <new-version>
    python scripts/bump_version.py <new-version> --dry-run
    python scripts/bump_version.py <new-version> --no-git
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

# Files containing version strings that need updating.
# pyproject.toml is always first (it's the SSOT we read the old version from).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_DOC_FILES = [
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "docs" / "release.md",
]

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_PYPROJECT_VERSION_RE = re.compile(
    r'^(version\s*=\s*")[\d]+\.[\d]+\.[\d]+(")', re.MULTILINE
)


# ---------------------------------------------------------------------------
# Core helpers (importable for testing)
# ---------------------------------------------------------------------------


def read_current_version(pyproject_text: str) -> str:
    """Extract the ``project.version`` value from *pyproject.toml* content."""
    m = _PYPROJECT_VERSION_RE.search(pyproject_text)
    if not m:
        raise ValueError("Could not find version = \"X.Y.Z\" in pyproject.toml")
    return m.group(0).split('"')[1]


def replace_in_pyproject(content: str, new_version: str) -> str:
    """Replace **only** the ``version = "..."`` line in *pyproject.toml*.

    Dependency pins like ``click==8.1.8`` are never touched.
    """
    result, n = _PYPROJECT_VERSION_RE.subn(rf'\g<1>{new_version}\2', content, count=1)
    if n == 0:
        raise ValueError("version = line not found in pyproject.toml content")
    return result


def replace_in_doc(content: str, old_version: str, new_version: str) -> str:
    """Replace every occurrence of *old_version* with *new_version*."""
    return content.replace(old_version, new_version)


# ---------------------------------------------------------------------------
# Diff display
# ---------------------------------------------------------------------------


def _unified_diff(path: Path, old: str, new: str) -> str:
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(a, b, fromfile=str(path), tofile=str(path))
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bump project version.")
    parser.add_argument("new_version", help="Target version (e.g. 0.2.0)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show diffs without writing files.",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Update files but skip git commit/tag.",
    )
    args = parser.parse_args(argv)

    new_version: str = args.new_version
    if not _VERSION_RE.match(new_version):
        print(
            f"Error: invalid version '{new_version}'. Expected format: X.Y.Z",
            file=sys.stderr,
        )
        return 1

    # 1. Read old version from SSOT.
    pyproject_text = _PYPROJECT.read_text(encoding="utf-8")
    try:
        old_version = read_current_version(pyproject_text)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if old_version == new_version:
        print(f"Already at version {old_version}, nothing to do.")
        return 0

    print(f"Bumping version: {old_version} → {new_version}")

    # 2. Compute new contents.
    changes: list[tuple[Path, str, str]] = []  # (path, old, new)

    new_pyproject = replace_in_pyproject(pyproject_text, new_version)
    changes.append((_PYPROJECT, pyproject_text, new_pyproject))

    for doc_path in _DOC_FILES:
        if not doc_path.exists():
            continue
        old_text = doc_path.read_text(encoding="utf-8")
        new_text = replace_in_doc(old_text, old_version, new_version)
        if new_text != old_text:
            changes.append((doc_path, old_text, new_text))

    # 3. Show diffs / write files.
    if args.dry_run:
        for path, old, new in changes:
            diff = _unified_diff(path, old, new)
            if diff:
                print(diff)
        print("\n(dry run — no files written)")
        return 0

    for path, _old, new in changes:
        path.write_text(new, encoding="utf-8")
        print(f"  Updated {path.relative_to(_REPO_ROOT)}")

    # 4. Refresh uv.lock so CI's `uv sync --locked` passes.
    print("  Running uv lock ...")
    subprocess.run(["uv", "lock"], cwd=_REPO_ROOT, check=True)
    print("  Updated uv.lock")

    # 5. Git commit + tag.
    if not args.no_git:
        files_to_add = [str(p.relative_to(_REPO_ROOT)) for p, _, _ in changes]
        files_to_add.append("uv.lock")
        subprocess.run(
            ["git", "add", *files_to_add],
            cwd=_REPO_ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"chore: bump version to {new_version}"],
            cwd=_REPO_ROOT,
            check=True,
        )
        tag = f"v{new_version}"
        subprocess.run(
            ["git", "tag", tag],
            cwd=_REPO_ROOT,
            check=True,
        )
        print(f"\nCommitted and tagged {tag}.")
        print(f"Push with: git push origin main --tags")
    else:
        print("\nFiles updated (--no-git: skipped commit/tag).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
