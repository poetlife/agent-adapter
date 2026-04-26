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
    stream_chat_as_responses_sse,
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


@pytest.mark.asyncio
async def test_stream_chat_as_responses_preserves_reasoning_before_tool_calls():
    stream = FakeStream([
        FakeModelDump({
            "id": "chunk-1",
            "model": "deepseek-v4-flash",
            "choices": [{"delta": {"reasoning_content": "Need commands. "}}],
        }),
        FakeModelDump({
            "id": "chunk-2",
            "model": "deepseek-v4-flash",
            "choices": [{"delta": {"reasoning_content": "Run both."}}],
        }),
        FakeModelDump({
            "id": "chunk-3",
            "model": "deepseek-v4-flash",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_001",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'},
                    }]
                }
            }],
        }),
        FakeModelDump({
            "id": "chunk-4",
            "model": "deepseek-v4-flash",
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 1,
                        "id": "call_002",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": '{"cmd":"pwd"}'},
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }),
    ])

    events = []
    async for chunk in stream_chat_as_responses_sse(stream, "deepseek-v4-flash"):
        text = chunk.decode("utf-8")
        if text.startswith("event: response.completed"):
            import json
            events.append(json.loads(text.split("data: ", 1)[1]))

    output = events[0]["response"]["output"]
    assert [item["type"] for item in output] == ["reasoning", "function_call", "function_call"]
    assert output[0]["summary"][0]["text"] == "Need commands. Run both."
    assert [item["call_id"] for item in output[1:]] == ["call_001", "call_002"]


def test_error_helpers():
    class FakeError(Exception):
        status_code = 429
        message = "rate limited"

    exc = FakeError()
    assert litellm_error_status_code(exc) == 429
    assert litellm_error_message(exc) == "rate limited"


# ---------------------------------------------------------------------------
# Helpers for parsing SSE event streams
# ---------------------------------------------------------------------------

def _parse_sse_events(raw_chunks: list[bytes]) -> list[dict]:
    """Parse SSE bytes into a list of {event_type, data} dicts."""
    import json as _json
    events: list[dict] = []
    for chunk in raw_chunks:
        text = chunk.decode("utf-8")
        # Each chunk is "event: <type>\ndata: <json>\n\n"
        lines = text.strip().split("\n")
        event_type = ""
        data_str = ""
        for line in lines:
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                data_str = line[len("data: "):]
        if event_type and data_str:
            events.append({"event": event_type, "data": _json.loads(data_str)})
    return events


# ---------------------------------------------------------------------------
# Tests for incremental SSE events
# ---------------------------------------------------------------------------

class TestStreamIncrementalEvents:
    """Verify the stream translator emits the complete set of Responses API
    incremental SSE events (added, delta, done) for each item type."""

    @pytest.mark.asyncio
    async def test_text_only_stream_emits_full_lifecycle(self):
        """A pure text stream should emit: created, output_item.added,
        content_part.added, output_text.delta(s), output_text.done,
        content_part.done, output_item.done, completed."""
        stream = FakeStream([
            FakeModelDump({
                "id": "c1", "model": "test-model",
                "choices": [{"delta": {"content": "Hello"}}],
            }),
            FakeModelDump({
                "id": "c2", "model": "test-model",
                "choices": [{"delta": {"content": " world"}}],
            }),
        ])

        chunks = []
        async for chunk in stream_chat_as_responses_sse(stream, "test-model"):
            chunks.append(chunk)

        events = _parse_sse_events(chunks)
        event_types = [e["event"] for e in events]

        assert event_types == [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]

        # Verify done payloads carry full accumulated text
        text_done = next(e for e in events if e["event"] == "response.output_text.done")
        assert text_done["data"]["text"] == "Hello world"

        part_done = next(e for e in events if e["event"] == "response.content_part.done")
        assert part_done["data"]["part"]["text"] == "Hello world"

        item_done = next(
            e for e in events
            if e["event"] == "response.output_item.done" and e["data"]["item"]["type"] == "message"
        )
        assert item_done["data"]["item"]["content"][0]["text"] == "Hello world"

        # Verify completed output
        completed = next(e for e in events if e["event"] == "response.completed")
        assert completed["data"]["response"]["output_text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_reasoning_stream_emits_full_lifecycle(self):
        """Reasoning-only stream emits: created, output_item.added,
        reasoning_summary_text.delta(s), reasoning_summary_text.done,
        output_item.done, completed."""
        stream = FakeStream([
            FakeModelDump({
                "id": "c1", "model": "m",
                "choices": [{"delta": {"reasoning_content": "Think "}}],
            }),
            FakeModelDump({
                "id": "c2", "model": "m",
                "choices": [{"delta": {"reasoning_content": "hard."}}],
            }),
        ])

        chunks = []
        async for chunk in stream_chat_as_responses_sse(stream, "m"):
            chunks.append(chunk)

        events = _parse_sse_events(chunks)
        event_types = [e["event"] for e in events]

        assert event_types == [
            "response.created",
            "response.output_item.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.output_item.done",
            "response.completed",
        ]

        reasoning_done = next(e for e in events if e["event"] == "response.reasoning_summary_text.done")
        assert reasoning_done["data"]["text"] == "Think hard."

    @pytest.mark.asyncio
    async def test_tool_call_stream_emits_incremental_events(self):
        """Tool call chunks should emit: output_item.added per call,
        function_call_arguments.delta per argument chunk,
        function_call_arguments.done + output_item.done per call."""
        stream = FakeStream([
            FakeModelDump({
                "id": "c1", "model": "m",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "ls", "arguments": '{"p'},
                        }]
                    }
                }],
            }),
            FakeModelDump({
                "id": "c2", "model": "m",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": 'ath":"."}'},
                        }]
                    }
                }],
            }),
        ])

        chunks = []
        async for chunk in stream_chat_as_responses_sse(stream, "m"):
            chunks.append(chunk)

        events = _parse_sse_events(chunks)
        event_types = [e["event"] for e in events]

        assert "response.output_item.added" in event_types
        assert "response.function_call_arguments.delta" in event_types
        assert "response.function_call_arguments.done" in event_types
        assert "response.output_item.done" in event_types

        # The added event should carry the function_call item
        fc_added = [
            e for e in events
            if e["event"] == "response.output_item.added"
            and e["data"].get("item", {}).get("type") == "function_call"
        ]
        assert len(fc_added) == 1
        assert fc_added[0]["data"]["item"]["call_id"] == "call_abc"
        assert fc_added[0]["data"]["item"]["name"] == "ls"

        # Two argument delta events
        arg_deltas = [e for e in events if e["event"] == "response.function_call_arguments.delta"]
        assert len(arg_deltas) == 2
        assert arg_deltas[0]["data"]["delta"] == '{"p'
        assert arg_deltas[1]["data"]["delta"] == 'ath":"."}'

        # Done event has full arguments
        arg_done = next(e for e in events if e["event"] == "response.function_call_arguments.done")
        assert arg_done["data"]["arguments"] == '{"path":"."}'

    @pytest.mark.asyncio
    async def test_reasoning_plus_tool_calls_full_event_sequence(self):
        """Full scenario: reasoning deltas → tool call deltas.
        Verify the complete event sequence and ordering."""
        stream = FakeStream([
            FakeModelDump({
                "id": "c1", "model": "deepseek-v4-flash",
                "choices": [{"delta": {"reasoning_content": "Let me think."}}],
            }),
            FakeModelDump({
                "id": "c2", "model": "deepseek-v4-flash",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run", "arguments": '{"x":1}'},
                        }]
                    }
                }],
            }),
            FakeModelDump({
                "id": "c3", "model": "deepseek-v4-flash",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 1,
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "run", "arguments": '{"x":2}'},
                        }]
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }),
        ])

        chunks = []
        async for chunk in stream_chat_as_responses_sse(stream, "deepseek-v4-flash"):
            chunks.append(chunk)

        events = _parse_sse_events(chunks)
        event_types = [e["event"] for e in events]

        # Verify ordering: reasoning deltas precede tool call added events
        # (reasoning done is emitted after the stream ends, so it comes after
        # the tool call added/delta events from the stream loop).
        last_reasoning_delta_idx = max(
            i for i, e in enumerate(events)
            if e["event"] == "response.reasoning_summary_text.delta"
        )
        first_fc_added_idx = next(
            i for i, e in enumerate(events)
            if e["event"] == "response.output_item.added"
            and e["data"].get("item", {}).get("type") == "function_call"
        )
        assert last_reasoning_delta_idx < first_fc_added_idx, \
            "Last reasoning delta should precede first function_call added"

        # Two function_call output_item.added events
        fc_added = [
            e for e in events
            if e["event"] == "response.output_item.added"
            and e["data"].get("item", {}).get("type") == "function_call"
        ]
        assert len(fc_added) == 2

        # Two function_call_arguments.done events
        fc_done = [e for e in events if e["event"] == "response.function_call_arguments.done"]
        assert len(fc_done) == 2

        # Completed output order: reasoning, function_call, function_call
        completed = next(e for e in events if e["event"] == "response.completed")
        output = completed["data"]["response"]["output"]
        assert [item["type"] for item in output] == [
            "reasoning", "function_call", "function_call",
        ]
        assert output[0]["summary"][0]["text"] == "Let me think."
        assert completed["data"]["response"]["usage"]["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_empty_stream_emits_created_and_completed(self):
        """An empty stream should still emit response.created + response.completed."""
        stream = FakeStream([])

        chunks = []
        async for chunk in stream_chat_as_responses_sse(stream, "m"):
            chunks.append(chunk)

        events = _parse_sse_events(chunks)
        event_types = [e["event"] for e in events]

        assert event_types == ["response.created", "response.completed"]
        completed = events[-1]["data"]["response"]
        assert completed["output"] == []
        assert completed["status"] == "completed"
