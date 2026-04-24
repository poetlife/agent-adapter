"""Tests for the shared LiteLLM client helpers."""

import pytest

from providers.catalog import Preset
from providers.litellm_client import (
    REQUEST_TIMEOUT_SECONDS,
    build_completion_kwargs,
    litellm_error_message,
    litellm_error_status_code,
    request_chat_completion,
    serialize_completion_response,
    serialize_completion_stream,
)


SAMPLE_PRESET = Preset.from_dict({
    "provider": "deepseek",
    "env_key": "DEEPSEEK_API_KEY",
    "models": [
        {
            "name": "deepseek-v4-flash",
            "litellm_model": "deepseek/deepseek-v4-flash",
            "api_base": "https://api.deepseek.com",
        }
    ],
})


class FakeModelDump:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, exclude_none=True):
        return self.payload


class FakeStream:
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


def test_build_completion_kwargs(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    kwargs = build_completion_kwargs(
        SAMPLE_PRESET,
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
        model_name="deepseek-v4-flash",
    )

    assert kwargs["model"] == "deepseek/deepseek-v4-flash"
    assert kwargs["api_base"] == "https://api.deepseek.com"
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["timeout"] == REQUEST_TIMEOUT_SECONDS
    assert kwargs["stream"] is True


@pytest.mark.asyncio
async def test_request_chat_completion_calls_litellm(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {"id": "chatcmpl-1"}

    monkeypatch.setattr("providers.litellm_client.acompletion", fake_acompletion)

    response = await request_chat_completion(
        SAMPLE_PRESET,
        {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Hi"}]},
        model_name="deepseek-v4-flash",
    )

    assert response == {"id": "chatcmpl-1"}
    assert captured["model"] == "deepseek/deepseek-v4-flash"
    assert captured["api_key"] == "sk-test"


def test_serialize_completion_response_accepts_model_dump():
    response = FakeModelDump({"id": "chatcmpl-1", "choices": []})
    assert serialize_completion_response(response)["id"] == "chatcmpl-1"


@pytest.mark.asyncio
async def test_serialize_completion_stream_formats_sse():
    stream = FakeStream([
        FakeModelDump({"id": "chunk-1", "choices": [{"delta": {"content": "Hi"}}]}),
        FakeModelDump({"id": "chunk-2", "choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ])

    chunks = []
    async for chunk in serialize_completion_stream(stream):
        chunks.append(chunk.decode("utf-8"))

    assert chunks[0] == 'data: {"id": "chunk-1", "choices": [{"delta": {"content": "Hi"}}]}\n\n'
    assert chunks[-1] == "data: [DONE]\n\n"


def test_error_helpers():
    class FakeError(Exception):
        status_code = 429
        message = "rate limited"

    exc = FakeError()
    assert litellm_error_status_code(exc) == 429
    assert litellm_error_message(exc) == "rate limited"
