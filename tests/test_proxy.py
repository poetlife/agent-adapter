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
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        names = [m["id"] for m in data["data"]]
        assert "deepseek-chat" in names
        assert "deepseek-coder" in names

    def test_model_owned_by(self, client):
        resp = client.get("/v1/models")
        data = resp.json()
        for model in data["data"]:
            assert model["owned_by"] == "deepseek"


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["provider"] == "deepseek"
