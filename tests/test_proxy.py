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


class FakeSSEStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        self._iter = iter(self.chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


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


class TestLiteLLMIntegration:
    def test_responses_endpoint_uses_litellm_request_path(self, client, monkeypatch):
        captured = {}

        async def fake_request_chat_completion(preset, body, model_name=None):
            captured["preset"] = preset.provider
            captured["body"] = body
            captured["model_name"] = model_name
            return {
                "id": "chatcmpl-1",
                "model": "deepseek/deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from LiteLLM"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            }

        monkeypatch.setattr("codex_adapter.proxy.request_chat_completion", fake_request_chat_completion)
        monkeypatch.setattr("codex_adapter.proxy.serialize_completion_response", lambda response: response)

        resp = client.post("/v1/responses", json={"model": "deepseek-chat", "input": "Hi"})

        assert resp.status_code == 200
        assert captured["preset"] == "deepseek"
        assert captured["model_name"] == "deepseek-chat"
        assert captured["body"]["model"] == "deepseek/deepseek-chat"
        assert resp.json()["output_text"] == "Hello from LiteLLM"

    def test_streaming_responses_endpoint_translates_litellm_stream(self, client, monkeypatch):
        async def fake_request_chat_completion(preset, body, model_name=None):
            return object()

        async def fake_serialize_completion_stream(_response):
            chunks = [
                b'data: {"id":"chatcmpl-1","model":"deepseek/deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-1","model":"deepseek/deepseek-chat","choices":[{"index":0,"delta":{"content":"Hello!"},"finish_reason":null}]}\n\n',
                b'data: {"id":"chatcmpl-1","model":"deepseek/deepseek-chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n',
                b"data: [DONE]\n\n",
            ]
            async for chunk in FakeSSEStream(chunks):
                yield chunk

        monkeypatch.setattr("codex_adapter.proxy.request_chat_completion", fake_request_chat_completion)
        monkeypatch.setattr("codex_adapter.proxy.serialize_completion_stream", fake_serialize_completion_stream)

        with client.stream("POST", "/v1/responses", json={"model": "deepseek-chat", "input": "Hi", "stream": True}) as resp:
            body = "".join(resp.iter_text())

        assert resp.status_code == 200
        assert "response.created" in body
        assert "response.output_text.delta" in body
        assert "response.completed" in body
