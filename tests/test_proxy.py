"""Tests for the proxy server."""

import pytest
from starlette.testclient import TestClient

from codex_adapter.config import Preset
from codex_adapter.proxy import create_app


SAMPLE_PRESET = Preset.from_dict({
    "provider": "deepseek",
    "description": "Test",
    "env_key": "DEEPSEEK_API_KEY",
    "models": [
        {
            "name": "deepseek-chat",
            "litellm_model": "deepseek/deepseek-chat",
            "api_base": "https://api.deepseek.com",
            "max_tokens": 8192,
        },
        {
            "name": "deepseek-coder",
            "litellm_model": "deepseek/deepseek-coder",
            "api_base": "https://api.deepseek.com",
            "max_tokens": 8192,
        },
    ],
})


@pytest.fixture
def app():
    return create_app(SAMPLE_PRESET)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) == 2
        slugs = [m["slug"] for m in data["models"]]
        assert "deepseek-chat" in slugs
        assert "deepseek-coder" in slugs

    def test_model_metadata(self, client):
        resp = client.get("/v1/models")
        data = resp.json()
        for model in data["models"]:
            # Required fields for Codex CLI ModelInfo
            assert "slug" in model
            assert "display_name" in model
            assert "shell_type" in model
            assert "visibility" in model
            assert "supported_reasoning_levels" in model
            assert "context_window" in model


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["provider"] == "deepseek"
