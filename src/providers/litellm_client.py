"""Shared LiteLLM request helpers.

All upstream model traffic should flow through this module so request building,
serialization, and error handling stay consistent across proxy endpoints.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from litellm import acompletion

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
