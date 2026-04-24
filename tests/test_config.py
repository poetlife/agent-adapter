"""Tests for config module."""

from pathlib import Path

import pytest

from codex_adapter.config import ModelEntry, Preset, list_presets, load_preset


SAMPLE_PRESET_DATA = {
    "provider": "test-provider",
    "description": "Test provider for unit tests",
    "env_key": "TEST_API_KEY",
    "models": [
        {
            "name": "test-model-1",
            "litellm_model": "test/model-1",
            "api_base": "https://api.test.com",
            "max_tokens": 4096,
        },
        {
            "name": "test-model-2",
            "litellm_model": "test/model-2",
            "api_base": "https://api.test.com",
        },
    ],
}


class TestPreset:
    def test_from_dict(self):
        preset = Preset.from_dict(SAMPLE_PRESET_DATA)
        assert preset.provider == "test-provider"
        assert preset.env_key == "TEST_API_KEY"
        assert len(preset.models) == 2
        assert preset.models[0].name == "test-model-1"
        assert preset.models[0].litellm_model == "test/model-1"
        assert preset.models[0].max_tokens == 4096
        assert preset.models[1].max_tokens == 8192  # default

    def test_from_yaml(self, tmp_path: Path):
        import yaml

        yaml_path = tmp_path / "test.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(SAMPLE_PRESET_DATA, f)

        preset = Preset.from_yaml(yaml_path)
        assert preset.provider == "test-provider"
        assert len(preset.models) == 2

    def test_description_optional(self):
        data = {**SAMPLE_PRESET_DATA}
        del data["description"]
        preset = Preset.from_dict(data)
        assert preset.description == ""

    def test_thinking_fields(self):
        data = {
            "provider": "deepseek",
            "env_key": "DEEPSEEK_API_KEY",
            "models": [{
                "name": "deepseek-v4-flash",
                "litellm_model": "deepseek-v4-flash",
                "api_base": "https://api.deepseek.com",
                "supports_thinking": True,
                "default_thinking": "enabled",
                "reasoning_effort": "high",
            }],
        }
        preset = Preset.from_dict(data)
        m = preset.models[0]
        assert m.supports_thinking is True
        assert m.default_thinking == "enabled"
        assert m.reasoning_effort == "high"

    def test_thinking_fields_defaults(self):
        preset = Preset.from_dict(SAMPLE_PRESET_DATA)
        m = preset.models[0]
        assert m.supports_thinking is False
        assert m.default_thinking == "disabled"
        assert m.reasoning_effort == "high"

    def test_api_base_inherited_from_preset(self):
        """Model without api_base should inherit from top-level preset api_base."""
        data = {
            "provider": "deepseek",
            "env_key": "DEEPSEEK_API_KEY",
            "api_base": "https://api.deepseek.com",
            "models": [{
                "name": "deepseek-v4-flash",
                "litellm_model": "deepseek-v4-flash",
            }],
        }
        preset = Preset.from_dict(data)
        assert preset.models[0].api_base == "https://api.deepseek.com"


class TestLoadPreset:
    def test_load_builtin_deepseek(self):
        preset = load_preset("deepseek")
        assert preset.provider == "deepseek"
        assert preset.env_key == "DEEPSEEK_API_KEY"
        assert len(preset.models) == 2
        names = [m.name for m in preset.models]
        assert "deepseek-v4-flash" in names
        assert "deepseek-v4-pro" in names
        # Both should support thinking
        for m in preset.models:
            assert m.supports_thinking is True

    def test_load_custom_preset(self, tmp_path: Path):
        import yaml

        custom_path = tmp_path / "custom.yaml"
        with open(custom_path, "w") as f:
            yaml.dump(SAMPLE_PRESET_DATA, f)

        preset = load_preset("custom", custom_dir=tmp_path)
        assert preset.provider == "test-provider"

    def test_custom_overrides_builtin(self, tmp_path: Path):
        """Custom preset with same name as built-in should take priority."""
        import yaml

        custom_data = {**SAMPLE_PRESET_DATA, "provider": "custom-deepseek"}
        custom_path = tmp_path / "deepseek.yaml"
        with open(custom_path, "w") as f:
            yaml.dump(custom_data, f)

        preset = load_preset("deepseek", custom_dir=tmp_path)
        assert preset.provider == "custom-deepseek"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError, match="not-a-real-preset"):
            load_preset("not-a-real-preset")


class TestListPresets:
    def test_includes_builtin(self):
        presets = list_presets()
        assert "deepseek" in presets

    def test_includes_custom(self, tmp_path: Path):
        import yaml

        custom_path = tmp_path / "my-custom.yaml"
        with open(custom_path, "w") as f:
            yaml.dump(SAMPLE_PRESET_DATA, f)

        presets = list_presets(custom_dir=tmp_path)
        assert "my-custom" in presets
        assert "deepseek" in presets
