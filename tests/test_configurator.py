"""Tests for deployment configuration helpers."""

from pathlib import Path

from codex_adapter.deploy import configurator


def test_inject_shell_profiles_updates_bash_and_zsh(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    env_file = tmp_path / ".codex-adapter.env"
    env_file.write_text("OPENAI_BASE_URL=http://localhost:4000/v1\n")

    bashrc = tmp_path / ".bashrc"
    zshrc = tmp_path / ".zshrc"
    bashrc.write_text("# bash\n")
    zshrc.write_text("# zsh\n")

    profiles = configurator.inject_shell_profiles(env_file)

    assert profiles == [bashrc, zshrc]
    for profile in (bashrc, zshrc):
        content = profile.read_text()
        assert configurator.SHELL_MARKER_BEGIN in content
        assert f'. "{env_file}"' in content

    assert configurator.inject_shell_profiles(env_file) == [bashrc, zshrc]
    assert bashrc.read_text().count(configurator.SHELL_MARKER_BEGIN) == 1
    assert zshrc.read_text().count(configurator.SHELL_MARKER_BEGIN) == 1


def test_inject_shell_profiles_creates_default_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    env_file = tmp_path / ".codex-adapter.env"
    env_file.write_text("OPENAI_BASE_URL=http://localhost:4000/v1\n")

    profiles = configurator.inject_shell_profiles(env_file)

    assert profiles == [tmp_path / ".bashrc", tmp_path / ".zshrc"]
    assert (tmp_path / ".bashrc").exists()
    assert (tmp_path / ".zshrc").exists()
