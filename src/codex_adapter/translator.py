"""Translate between OpenAI Responses API and Chat Completions API.

Codex CLI (latest) sends requests to POST /v1/responses (Responses API).
DeepSeek and most non-OpenAI providers only support POST /v1/chat/completions.

This module converts:
  Responses API request  →  Chat Completions request
  Chat Completions response  →  Responses API response
  Chat Completions SSE stream  →  Responses API SSE stream
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator


# ---------------------------------------------------------------------------
# Request translation: Responses API → Chat Completions
# ---------------------------------------------------------------------------

def responses_request_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses API request body into a Chat Completions request body.

    Responses API fields → Chat Completions mapping:
        instructions      → system message
        input (str)       → single user message
        input (list)      → messages array (with role mapping)
        max_output_tokens → max_tokens
        model             → model
        temperature       → temperature
        tools             → tools (function format)
        stream            → stream
    """
    messages: list[dict[str, Any]] = []

    # System message from 'instructions'
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # Convert 'input' to messages
    raw_input = body.get("input", "")
    if isinstance(raw_input, str):
        if raw_input:
            messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        messages.extend(_convert_input_items(raw_input))

    chat_body: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": messages,
    }

    # max_output_tokens → max_tokens
    if "max_output_tokens" in body:
        chat_body["max_tokens"] = body["max_output_tokens"]

    # Temperature
    if "temperature" in body:
        chat_body["temperature"] = body["temperature"]

    # Top-p
    if "top_p" in body:
        chat_body["top_p"] = body["top_p"]

    # Stream
    if body.get("stream", False):
        chat_body["stream"] = True
        chat_body["stream_options"] = {"include_usage": True}

    # Tools: convert Responses API tools to Chat Completions format
    tools = body.get("tools")
    if tools:
        chat_tools = _convert_tools(tools)
        if chat_tools:
            chat_body["tools"] = chat_tools

    # Tool choice
    if "tool_choice" in body:
        chat_body["tool_choice"] = body["tool_choice"]

    return chat_body


def _convert_input_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API input items to Chat Completions messages."""
    messages: list[dict[str, Any]] = []

    for item in items:
        item_type = item.get("type", "")
        role = item.get("role", "user")

        if item_type == "message" or "content" in item:
            # Standard message item
            content = item.get("content", "")
            if isinstance(content, list):
                content = _convert_content_parts(content)
            messages.append({"role": role, "content": content})

        elif item_type == "function_call":
            # Tool call from previous turn (assistant message with tool_calls)
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }],
            })

        elif item_type == "function_call_output":
            # Tool result
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })

        else:
            # Fallback: treat as a user message if it has text content
            text = item.get("text", item.get("content", ""))
            if text:
                messages.append({"role": role, "content": text})

    return messages


def _convert_content_parts(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Convert Responses API content parts to Chat Completions format."""
    converted = []
    for part in parts:
        part_type = part.get("type", "")
        if part_type in ("input_text", "text", "output_text"):
            converted.append({
                "type": "text",
                "text": part.get("text", ""),
            })
        elif part_type == "input_image":
            converted.append({
                "type": "image_url",
                "image_url": {"url": part.get("image_url", part.get("url", ""))},
            })
        else:
            # Pass through unknown types as text if possible
            text = part.get("text", "")
            if text:
                converted.append({"type": "text", "text": text})

    # If all parts are text, simplify to a single string
    if all(p["type"] == "text" for p in converted):
        return " ".join(p["text"] for p in converted)

    return converted


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API tools to Chat Completions tool format."""
    chat_tools = []
    for tool in tools:
        tool_type = tool.get("type", "")
        if tool_type == "function":
            # Already in function format, pass through
            chat_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        # Skip built-in tools (web_search, code_interpreter, etc.) as
        # they are OpenAI-specific and not supported by third-party models
    return chat_tools


# ---------------------------------------------------------------------------
# Response translation: Chat Completions → Responses API
# ---------------------------------------------------------------------------

def chat_response_to_responses(chat_resp: dict[str, Any], original_model: str = "") -> dict[str, Any]:
    """Convert a Chat Completions response to a Responses API response."""
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    output: list[dict[str, Any]] = []

    choices = chat_resp.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        role = message.get("role", "assistant")
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        # Text output
        if content is not None:
            output.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": role,
                "content": [
                    {"type": "output_text", "text": content}
                ],
            })

        # Tool calls
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                output.append({
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                })

    # Build output_text shortcut
    output_text = ""
    for item in output:
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    output_text += c.get("text", "")

    # Usage mapping
    chat_usage = chat_resp.get("usage", {})
    usage = {
        "input_tokens": chat_usage.get("prompt_tokens", 0),
        "output_tokens": chat_usage.get("completion_tokens", 0),
        "total_tokens": chat_usage.get("total_tokens", 0),
    }

    # Determine status from finish_reason
    finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
    status = "completed" if finish_reason == "stop" else "incomplete"

    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": chat_resp.get("model", original_model),
        "output": output,
        "output_text": output_text,
        "status": status,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Streaming translation: Chat Completions SSE → Responses API SSE
# ---------------------------------------------------------------------------

async def translate_stream(
    chat_stream: AsyncIterator[bytes],
    original_model: str = "",
) -> AsyncIterator[bytes]:
    """Translate a Chat Completions SSE stream to Responses API SSE events.

    Chat Completions stream format:
        data: {"choices":[{"delta":{"content":"Hello"}}]}
        data: [DONE]

    Responses API stream format:
        event: response.created
        data: {"id":"resp_xxx","object":"response","status":"in_progress"}

        event: response.output_item.added
        data: {"type":"message",...}

        event: response.content_part.added
        data: {"type":"output_text","text":""}

        event: response.output_text.delta
        data: {"type":"output_text","delta":"Hello"}

        event: response.completed
        data: {"id":"resp_xxx","output":[...],"status":"completed"}

        event: response.done
        data: [DONE]
    """
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    accumulated_text = ""
    accumulated_tool_calls: dict[int, dict[str, Any]] = {}  # index → partial tool call
    model = original_model
    usage: dict[str, Any] = {}
    created_emitted = False

    async for raw_chunk in chat_stream:
        line = raw_chunk.decode("utf-8", errors="replace").strip() if isinstance(raw_chunk, bytes) else raw_chunk.strip()

        # SSE streams may have multiple lines per chunk
        for sub_line in line.split("\n"):
            sub_line = sub_line.strip()
            if not sub_line:
                continue

            if sub_line.startswith("data: "):
                data_str = sub_line[6:]
            elif sub_line.startswith("data:"):
                data_str = sub_line[5:]
            else:
                continue

            if data_str.strip() == "[DONE]":
                # Emit final completed event
                final_output: list[dict[str, Any]] = []
                if accumulated_text:
                    final_output.append({
                        "type": "message",
                        "id": msg_id,
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": accumulated_text}],
                    })
                for _idx, tc in sorted(accumulated_tool_calls.items()):
                    final_output.append({
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex[:24]}",
                        "call_id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", ""),
                    })

                completed = {
                    "id": resp_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": model,
                    "output": final_output,
                    "output_text": accumulated_text,
                    "status": "completed",
                    "usage": usage,
                }
                yield _sse_event("response.completed", completed)
                yield b"event: done\ndata: [DONE]\n\n"
                return

            # Parse JSON
            try:
                chunk_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Update model
            if "model" in chunk_data:
                model = chunk_data["model"]

            # Emit response.created once
            if not created_emitted:
                created_emitted = True
                created_event = {
                    "id": resp_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": model,
                    "output": [],
                }
                yield _sse_event("response.created", created_event)

                # Emit output_item.added for the message
                yield _sse_event("response.output_item.added", {
                    "type": "message",
                    "id": msg_id,
                    "role": "assistant",
                    "content": [],
                })
                yield _sse_event("response.content_part.added", {
                    "type": "output_text",
                    "text": "",
                })

            # Extract choices
            choices = chunk_data.get("choices", [])

            # Extract usage if present (from stream_options include_usage)
            if "usage" in chunk_data and chunk_data["usage"]:
                u = chunk_data["usage"]
                usage = {
                    "input_tokens": u.get("prompt_tokens", 0),
                    "output_tokens": u.get("completion_tokens", 0),
                    "total_tokens": u.get("total_tokens", 0),
                }

            for choice in choices:
                delta = choice.get("delta", {})

                # Text content delta
                text_delta = delta.get("content")
                if text_delta:
                    accumulated_text += text_delta
                    yield _sse_event("response.output_text.delta", {
                        "type": "output_text",
                        "delta": text_delta,
                    })

                # Tool call deltas
                tc_deltas = delta.get("tool_calls", [])
                for tc_delta in tc_deltas:
                    idx = tc_delta.get("index", 0)
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc_delta.get("id", ""),
                            "name": "",
                            "arguments": "",
                        }
                    tc = accumulated_tool_calls[idx]
                    if tc_delta.get("id"):
                        tc["id"] = tc_delta["id"]
                    func = tc_delta.get("function", {})
                    if func.get("name"):
                        tc["name"] += func["name"]
                    if func.get("arguments"):
                        tc["arguments"] += func["arguments"]


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    """Format a single SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
