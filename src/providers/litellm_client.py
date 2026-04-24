"""Shared LiteLLM request helpers.

All upstream model traffic should flow through this module so request building,
serialization, and error handling stay consistent across proxy endpoints.

Response translation (Chat Completions → Responses API) is delegated to
LiteLLM's built-in ``LiteLLMCompletionResponsesConfig`` transformer and
``LiteLLMCompletionStreamingIterator``, with thin post-processing for
Codex CLI compatibility (``output_text`` injection, reasoning format fix).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, AsyncIterator

from litellm import acompletion
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.utils import ModelResponse

from providers.catalog import Preset

REQUEST_TIMEOUT_SECONDS = 300


def build_completion_kwargs(
    preset: Preset,
    body: dict[str, Any],
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build the canonical LiteLLM kwargs for one chat completion request."""
    model_entry = preset.resolve_model(model_name or str(body.get("model", "")))
    if model_entry is None:
        raise ValueError("No models defined in preset")

    api_key = os.environ.get(preset.env_key, "")
    if not api_key:
        raise ValueError(f"Environment variable {preset.env_key} is not set")

    kwargs = dict(body)
    kwargs["model"] = model_entry.litellm_model
    kwargs["api_base"] = model_entry.api_base
    kwargs["api_key"] = api_key
    kwargs["timeout"] = REQUEST_TIMEOUT_SECONDS
    return kwargs


async def request_chat_completion(
    preset: Preset,
    body: dict[str, Any],
    model_name: str | None = None,
) -> Any:
    """Run a chat completion through the LiteLLM SDK."""
    return await acompletion(**build_completion_kwargs(preset, body, model_name=model_name))


def serialize_completion_response(response: Any) -> dict[str, Any]:
    """Convert a LiteLLM response object into a plain JSON-serializable dict."""
    data = _to_serializable_object(response)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict-like LiteLLM response, got {type(data).__name__}")
    return data


async def serialize_completion_stream(response: Any) -> AsyncIterator[bytes]:
    """Convert a LiteLLM async stream into raw OpenAI-style SSE data lines."""
    async for chunk in response:
        yield _sse_data_line(_to_serializable_object(chunk))
    yield b"data: [DONE]\n\n"


def litellm_error_status_code(exc: Exception) -> int:
    """Best-effort extraction of an HTTP status code from a LiteLLM error."""
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, (int, str)):
        return 500
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return 500


def litellm_error_message(exc: Exception) -> str:
    """Return a readable upstream error message."""
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message
    text = str(exc)
    return text if text else exc.__class__.__name__


def _to_serializable_object(obj: Any) -> Any:
    """Convert LiteLLM / Pydantic objects into plain Python data structures."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if hasattr(obj, "dict"):
        try:
            return obj.dict(exclude_none=True)
        except TypeError:
            return obj.dict()
    if hasattr(obj, "json"):
        return json.loads(obj.json())
    raise TypeError(f"Unsupported LiteLLM object type: {type(obj).__name__}")


def _sse_data_line(data: Any) -> bytes:
    """Format a LiteLLM chunk as one SSE data frame."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Response translation: Chat Completions → Responses API  (via LiteLLM)
# ---------------------------------------------------------------------------

def transform_chat_to_responses(
    chat_response: Any,
    original_model: str = "",
) -> dict[str, Any]:
    """Convert a Chat Completions response to Responses API format.

    Delegates to LiteLLM's built-in transformer, then applies Codex CLI
    compatibility fixes:

    * ``output_text`` — LiteLLM exposes it as a ``@property`` which
      ``model_dump()`` excludes; we inject it explicitly.
    * Reasoning format — LiteLLM uses ``content[output_text]``, Codex
      expects ``summary[summary_text]``.
    * ``object`` field — LiteLLM keeps ``"chat.completion"``; Codex
      expects ``"response"``.
    * Excess fields — strip keys that Codex CLI does not expect.
    """
    if not isinstance(chat_response, ModelResponse):
        if isinstance(chat_response, dict):
            model_resp = ModelResponse(**chat_response)
        else:
            model_resp = _to_model_response(chat_response)
    else:
        model_resp = chat_response

    responses_obj = (
        LiteLLMCompletionResponsesConfig
        .transform_chat_completion_response_to_responses_api_response(
            request_input="",
            responses_api_request={},
            chat_completion_response=model_resp,
        )
    )

    result = responses_obj.model_dump(exclude_none=True, mode="json")

    # --- Codex CLI compatibility post-processing ---
    result["object"] = "response"
    result["output_text"] = _compute_output_text(result.get("output", []))
    result["output"] = _fix_reasoning_format(result.get("output", []))

    if original_model:
        result["model"] = original_model

    # Strip fields that Codex CLI does not expect in a Responses API response.
    for key in ("parallel_tool_calls", "tool_choice", "tools", "text",
                "temperature", "top_p", "max_output_tokens", "truncation"):
        result.pop(key, None)

    return result


async def stream_chat_as_responses_sse(
    chat_stream: Any,
    original_model: str = "",
) -> AsyncIterator[bytes]:
    """Translate a LiteLLM chat completion stream into Responses API SSE bytes.

    Wraps the ``CustomStreamWrapper`` from ``acompletion(stream=True)`` in
    LiteLLM's ``LiteLLMCompletionStreamingIterator``, which produces typed
    Responses API events.  Each event is serialized to SSE format for
    ``Starlette.StreamingResponse``.
    """
    if not isinstance(chat_stream, CustomStreamWrapper):
        # Fallback: raw async iterator — wrap manually.
        async for chunk in _fallback_stream_translation(chat_stream, original_model):
            yield chunk
        return

    iterator = LiteLLMCompletionStreamingIterator(
        model=original_model,
        litellm_custom_stream_wrapper=chat_stream,
        request_input="",
        responses_api_request={},
    )

    async for event in iterator:
        event_dict = (
            event.model_dump(exclude_none=True, mode="json")
            if hasattr(event, "model_dump")
            else dict(event)
        )

        event_type = event_dict.get("type", "unknown")

        # Fix reasoning format throughout the stream.
        if event_type in (
            "response.output_item.added",
            "response.output_item.done",
        ):
            item = event_dict.get("item", {})
            if item.get("type") == "reasoning":
                event_dict["item"] = _fix_single_reasoning_item(item)

        # Patch the completed event.
        if event_type == "response.completed" and "response" in event_dict:
            resp = event_dict["response"]
            resp["object"] = "response"
            resp["output"] = _fix_reasoning_format(resp.get("output", []))
            resp["output_text"] = _compute_output_text(resp.get("output", []))
            for key in ("parallel_tool_calls", "tool_choice", "tools", "text",
                        "temperature", "top_p", "max_output_tokens", "truncation"):
                resp.pop(key, None)

        yield _sse_event(event_type, event_dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_model_response(obj: Any) -> ModelResponse:
    """Best-effort conversion to ``ModelResponse``."""
    if hasattr(obj, "model_dump"):
        return ModelResponse(**obj.model_dump(exclude_none=True))
    if hasattr(obj, "dict"):
        return ModelResponse(**obj.dict(exclude_none=True))
    raise TypeError(f"Cannot convert {type(obj).__name__} to ModelResponse")


def _fix_reasoning_format(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert LiteLLM reasoning items to Codex CLI expected format.

    LiteLLM:  ``type="reasoning", content=[{type: "output_text", text: "…"}]``
    Codex:    ``type="reasoning", summary=[{type: "summary_text", text: "…"}]``
    """
    return [
        _fix_single_reasoning_item(item) if item.get("type") == "reasoning" else item
        for item in output
    ]


def _fix_single_reasoning_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert one reasoning output item to Codex ``summary`` format."""
    content = item.get("content", [])
    if not content and item.get("summary"):
        return item  # Already in the right format.

    summary: list[dict[str, Any]] = []
    for c in content:
        if isinstance(c, dict):
            text = c.get("text", "")
            if text:
                summary.append({"type": "summary_text", "text": text})

    new_item = dict(item)
    new_item.pop("content", None)
    # Drop fields Codex doesn't expect on reasoning items.
    new_item.pop("status", None)
    new_item.pop("role", None)
    new_item["summary"] = summary
    return new_item


def _compute_output_text(output: list[dict[str, Any]]) -> str:
    """Compute the ``output_text`` shortcut from output items."""
    parts: list[str] = []
    for item in output:
        if item.get("type") == "message":
            for c in item.get("content", []):
                if isinstance(c, dict) and c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "".join(parts)


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    """Format one Server-Sent Event."""
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    ).encode("utf-8")


async def _fallback_stream_translation(
    stream: Any,
    original_model: str,
) -> AsyncIterator[bytes]:
    """Minimal fallback when the stream is not a ``CustomStreamWrapper``.

    Re-implements just enough of the old ``translate_stream()`` logic to
    keep the proxy functional if LiteLLM returns an unexpected type.
    """
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    accumulated_text = ""

    yield _sse_event("response.created", {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "status": "in_progress",
            "model": original_model,
            "output": [],
        },
    })

    text_started = False

    async for raw_chunk in stream:
        chunk_data = _to_serializable_object(raw_chunk)
        if not isinstance(chunk_data, dict):
            continue

        for choice in chunk_data.get("choices", []):
            delta = choice.get("delta", {})
            text_delta = delta.get("content")
            if text_delta:
                if not text_started:
                    text_started = True
                    yield _sse_event("response.output_item.added", {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "message", "id": msg_id,
                            "role": "assistant", "content": [],
                        },
                    })
                    yield _sse_event("response.content_part.added", {
                        "type": "response.content_part.added",
                        "item_id": msg_id, "content_index": 0,
                        "part": {"type": "output_text", "text": ""},
                    })
                accumulated_text += text_delta
                yield _sse_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": msg_id, "content_index": 0,
                    "delta": text_delta,
                })

    import time as _time
    final_output: list[dict[str, Any]] = []
    if accumulated_text:
        final_output.append({
            "type": "message", "id": msg_id, "role": "assistant",
            "content": [{"type": "output_text", "text": accumulated_text}],
        })
    yield _sse_event("response.completed", {
        "type": "response.completed",
        "response": {
            "id": resp_id, "object": "response",
            "created_at": int(_time.time()), "model": original_model,
            "output": final_output, "output_text": accumulated_text,
            "status": "completed", "usage": {},
        },
    })
