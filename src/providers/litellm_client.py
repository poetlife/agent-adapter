"""Shared LiteLLM request helpers.

All upstream model traffic should flow through this module so request building,
serialization, and error handling stay consistent across proxy endpoints.

Response translation (Chat Completions → Responses API) is delegated to
LiteLLM's built-in ``LiteLLMCompletionResponsesConfig`` transformer for
non-streaming responses, with a custom streaming translator that preserves
DeepSeek ``reasoning_content`` across tool-call turns.  Thin post-processing
ensures Codex CLI compatibility (``output_text`` injection, reasoning format fix).
"""

from __future__ import annotations

import json
import os
import time as _time
import uuid
from typing import Any, AsyncIterator

from litellm import acompletion
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

    The local translator preserves DeepSeek ``reasoning_content`` in final
    Responses history when a thinking-mode turn calls tools.  DeepSeek requires
    that reasoning to be passed back with the historical assistant tool call.
    """
    async for chunk in _translate_stream_preserving_reasoning(chat_stream, original_model):
        yield chunk


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


async def _translate_stream_preserving_reasoning(
    stream: Any,
    original_model: str,
) -> AsyncIterator[bytes]:
    """Translate chat stream chunks while preserving reasoning before tool calls.

    Emits the full set of Responses API incremental SSE events so Codex CLI
    can render progress during streaming:

    - ``response.created``
    - ``response.output_item.added`` / ``.done`` for reasoning, message, and
      function_call items
    - ``response.reasoning_summary_text.delta`` / ``.done``
    - ``response.content_part.added`` / ``.done``
    - ``response.output_text.delta`` / ``.done``
    - ``response.function_call_arguments.delta`` / ``.done``
    - ``response.completed``
    """
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
    model = original_model
    usage: dict[str, Any] = {}
    accumulated_text = ""
    accumulated_reasoning = ""
    # Tracks each tool call by its stream index.
    # Values: {"id": str, "name": str, "arguments": str, "added": bool}
    accumulated_tool_calls: dict[int, dict[str, Any]] = {}
    created_emitted = False
    reasoning_added = False
    text_added = False

    async for raw_chunk in stream:
        chunk_data = _to_serializable_object(raw_chunk)
        if not isinstance(chunk_data, dict):
            continue

        if "model" in chunk_data:
            model = chunk_data["model"]
        if chunk_data.get("usage"):
            raw_usage = chunk_data["usage"]
            usage = {
                "input_tokens": raw_usage.get("prompt_tokens", 0),
                "output_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            }

        if not created_emitted:
            created_emitted = True
            yield _sse_event("response.created", {
                "type": "response.created",
                "response": {
                    "id": resp_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": model,
                    "output": [],
                },
            })

        for choice in chunk_data.get("choices", []):
            delta = choice.get("delta", {})
            reasoning_delta = delta.get("reasoning_content")
            if reasoning_delta:
                if not reasoning_added:
                    reasoning_added = True
                    yield _sse_event("response.output_item.added", {
                        "type": "response.output_item.added",
                        "item": {"type": "reasoning", "id": reasoning_id, "summary": []},
                    })
                accumulated_reasoning += reasoning_delta
                yield _sse_event("response.reasoning_summary_text.delta", {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": reasoning_id,
                    "summary_index": 0,
                    "delta": reasoning_delta,
                })

            text_delta = delta.get("content")
            if text_delta:
                if not text_added:
                    text_added = True
                    yield _sse_event("response.output_item.added", {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "message",
                            "id": msg_id,
                            "role": "assistant",
                            "content": [],
                        },
                    })
                    yield _sse_event("response.content_part.added", {
                        "type": "response.content_part.added",
                        "item_id": msg_id,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": ""},
                    })
                accumulated_text += text_delta
                yield _sse_event("response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "content_index": 0,
                    "delta": text_delta,
                })

            for tool_delta in delta.get("tool_calls", []) or []:
                index = tool_delta.get("index", 0)
                tool_call = accumulated_tool_calls.setdefault(
                    index,
                    {"id": "", "name": "", "arguments": "", "added": False},
                )
                if tool_delta.get("id"):
                    tool_call["id"] = tool_delta["id"]
                function_delta = tool_delta.get("function", {})
                if function_delta.get("name"):
                    tool_call["name"] += function_delta["name"]

                # Emit output_item.added the first time we see enough info
                # (call id available and not yet emitted).
                if tool_call["id"] and not tool_call["added"]:
                    tool_call["added"] = True
                    yield _sse_event("response.output_item.added", {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": tool_call["id"],
                            "call_id": tool_call["id"],
                            "name": tool_call.get("name", ""),
                            "arguments": "",
                        },
                    })

                arg_delta = function_delta.get("arguments")
                if arg_delta:
                    tool_call["arguments"] += arg_delta
                    yield _sse_event("response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": tool_call["id"],
                        "delta": arg_delta,
                    })

    # --- Emit "done" events for each item that was streamed. ---

    # Reasoning done
    if reasoning_added:
        yield _sse_event("response.reasoning_summary_text.done", {
            "type": "response.reasoning_summary_text.done",
            "item_id": reasoning_id,
            "summary_index": 0,
            "text": accumulated_reasoning,
        })
        yield _sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": reasoning_id,
                "summary": [{"type": "summary_text", "text": accumulated_reasoning}],
            },
        })

    # Text message done
    if text_added:
        yield _sse_event("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": msg_id,
            "content_index": 0,
            "text": accumulated_text,
        })
        yield _sse_event("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": msg_id,
            "content_index": 0,
            "part": {"type": "output_text", "text": accumulated_text},
        })
        yield _sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": accumulated_text}],
            },
        })

    # Function call done events
    for index in sorted(accumulated_tool_calls):
        tool_call = accumulated_tool_calls[index]
        if not tool_call.get("id"):
            continue
        fc_item = {
            "type": "function_call",
            "id": tool_call["id"],
            "call_id": tool_call["id"],
            "name": tool_call.get("name", ""),
            "arguments": tool_call.get("arguments", ""),
        }
        yield _sse_event("response.function_call_arguments.done", {
            "type": "response.function_call_arguments.done",
            "item_id": tool_call["id"],
            "arguments": tool_call.get("arguments", ""),
        })
        yield _sse_event("response.output_item.done", {
            "type": "response.output_item.done",
            "item": fc_item,
        })

    # --- Build final output for response.completed ---
    final_output: list[dict[str, Any]] = []
    if accumulated_reasoning:
        final_output.append({
            "type": "reasoning",
            "id": reasoning_id,
            "summary": [{"type": "summary_text", "text": accumulated_reasoning}],
        })
    if accumulated_text:
        final_output.append({
            "type": "message",
            "id": msg_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": accumulated_text}],
        })
    for index in sorted(accumulated_tool_calls):
        tool_call = accumulated_tool_calls[index]
        if tool_call.get("id"):
            final_output.append({
                "type": "function_call",
                "id": tool_call["id"],
                "call_id": tool_call["id"],
                "name": tool_call.get("name", ""),
                "arguments": tool_call.get("arguments", ""),
            })

    if not created_emitted:
        yield _sse_event("response.created", {
            "type": "response.created",
            "response": {
                "id": resp_id,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": [],
            },
        })
    yield _sse_event("response.completed", {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": int(_time.time()),
            "model": model,
            "output": final_output,
            "output_text": accumulated_text,
            "status": "completed",
            "usage": usage,
        },
    })
