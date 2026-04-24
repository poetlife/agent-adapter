"""Translate between OpenAI Responses API and Chat Completions API.

Codex CLI (latest) sends requests to POST /v1/responses (Responses API).
DeepSeek and most non-OpenAI providers only support POST /v1/chat/completions.

This module converts:
  Responses API request  →  Chat Completions request  (with DeepSeek thinking support)
  Chat Completions response  →  Responses API response  (with reasoning_content mapping)
  Chat Completions SSE stream  →  Responses API SSE stream

DeepSeek V4 thinking mode mapping:
  Codex reasoning.effort  →  DeepSeek thinking.type + reasoning_effort
    "low" / "medium"      →  thinking.type="enabled", reasoning_effort="high"
    "high"                →  thinking.type="enabled", reasoning_effort="high"
    "max"                 →  thinking.type="enabled", reasoning_effort="max"
    (absent)              →  use model preset defaults

  DeepSeek reasoning_content  →  Codex reasoning output item
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

# Best-effort import of proxy debug logger; translator can also be used standalone.
_logger = logging.getLogger("codex_adapter.debug")


@dataclass
class ModelConfig:
    """Per-model config passed to translation functions for thinking support."""

    supports_thinking: bool = False
    default_thinking: str = "disabled"   # "enabled" or "disabled"
    reasoning_effort: str = "high"       # "high" or "max"


# ---------------------------------------------------------------------------
# Request translation: Responses API → Chat Completions
# ---------------------------------------------------------------------------

def responses_request_to_chat(
    body: dict[str, Any],
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Convert a Responses API request body into a Chat Completions request body.

    Responses API fields → Chat Completions mapping:
        instructions      → system message
        input (str)       → single user message
        input (list)      → messages array (with role mapping)
        max_output_tokens → max_tokens
        model             → model
        temperature       → temperature (dropped when thinking is enabled)
        tools             → tools (function format)
        stream            → stream
        reasoning.effort  → thinking + reasoning_effort (DeepSeek V4)
    """
    if model_config is None:
        model_config = ModelConfig()

    messages: list[dict[str, Any]] = []

    # System message from 'instructions'
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # --- Thinking mode translation ---
    thinking_enabled = False
    ds_reasoning_effort = model_config.reasoning_effort

    # Check Codex's reasoning.effort parameter
    reasoning = body.get("reasoning")
    if reasoning and isinstance(reasoning, dict):
        effort = reasoning.get("effort", "")
        if effort:
            thinking_enabled = True
            # Map Codex effort levels to DeepSeek reasoning_effort
            # DeepSeek only supports "high" and "max"; low/medium map to high
            if effort in ("low", "medium", "high"):
                ds_reasoning_effort = "high"
            elif effort in ("max", "xhigh"):
                ds_reasoning_effort = "max"
            else:
                ds_reasoning_effort = "high"
    elif model_config.supports_thinking and model_config.default_thinking == "enabled":
        # Use model preset defaults if no explicit reasoning param
        thinking_enabled = True

    # Convert 'input' to messages
    raw_input = body.get("input", "")
    if isinstance(raw_input, str):
        if raw_input:
            messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        messages.extend(_convert_input_items(raw_input, thinking_enabled=thinking_enabled))

    chat_body: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": messages,
    }

    # max_output_tokens → max_tokens
    if "max_output_tokens" in body:
        chat_body["max_tokens"] = body["max_output_tokens"]

    if model_config.supports_thinking:
        if thinking_enabled:
            chat_body["thinking"] = {"type": "enabled"}
            chat_body["reasoning_effort"] = ds_reasoning_effort
            # DeepSeek 不支持 temperature/top_p 在思考模式下
            # 所以不传这些参数
        else:
            chat_body["thinking"] = {"type": "disabled"}
            # Temperature (only when thinking is disabled)
            if "temperature" in body:
                chat_body["temperature"] = body["temperature"]
            # Top-p
            if "top_p" in body:
                chat_body["top_p"] = body["top_p"]
    else:
        # Non-thinking model: pass through temperature/top_p normally
        if "temperature" in body:
            chat_body["temperature"] = body["temperature"]
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


def _convert_input_items(
    items: list[dict[str, Any]],
    *,
    thinking_enabled: bool = False,
) -> list[dict[str, Any]]:
    """Convert Responses API input items to Chat Completions messages.

    Also handles reasoning_content from previous assistant turns for DeepSeek
    multi-turn tool-call conversations.

    Role mapping: 'developer' → 'system' (DeepSeek doesn't support 'developer').

    DeepSeek thinking mode requires reasoning_content to be passed back in the
    assistant message. When a 'reasoning' item precedes a 'function_call' item,
    we merge them into a single assistant message with both reasoning_content
    and tool_calls.

    Some clients do not emit a fresh 'reasoning' item for every assistant
    continuation after tool outputs. In thinking mode we carry forward the most
    recent assistant reasoning_content across an immediate tool-result
    continuation so DeepSeek still receives the required field.
    """
    messages: list[dict[str, Any]] = []
    # Buffer reasoning_content from 'reasoning' items to attach to the next
    # assistant message (function_call or regular message).
    pending_reasoning: str | None = None
    last_assistant_reasoning: str | None = None
    last_item_was_tool_output = False

    for item in items:
        item_type = item.get("type", "")
        role = item.get("role", "user")
        # DeepSeek (and most non-OpenAI providers) only accept:
        # system, user, assistant, tool
        if role == "developer":
            role = "system"

        if item_type == "reasoning":
            # Buffer reasoning text — it will be attached to the next
            # assistant message (function_call or text message).
            text = ""
            # reasoning items may have summary list or direct text
            summary = item.get("summary", [])
            if isinstance(summary, list):
                for s in summary:
                    if isinstance(s, dict):
                        text += s.get("text", "")
                    elif isinstance(s, str):
                        text += s
            if not text:
                text = item.get("text", item.get("content", ""))
            if pending_reasoning:
                pending_reasoning += text
            else:
                pending_reasoning = text
            last_item_was_tool_output = False

        elif item_type == "function_call":
            # Tool call from previous turn (assistant message with tool_calls)
            msg: dict[str, Any] = {
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
            }
            # Attach buffered reasoning_content (DeepSeek requirement).
            # If the client omitted a fresh reasoning item for an immediate
            # post-tool continuation, reuse the previous assistant reasoning.
            if pending_reasoning:
                msg["reasoning_content"] = pending_reasoning
                pending_reasoning = None
            elif "reasoning_content" in item:
                msg["reasoning_content"] = item["reasoning_content"]
            elif thinking_enabled and last_item_was_tool_output and last_assistant_reasoning:
                msg["reasoning_content"] = last_assistant_reasoning
            if msg.get("reasoning_content"):
                last_assistant_reasoning = msg["reasoning_content"]
            messages.append(msg)
            last_item_was_tool_output = False

        elif item_type == "function_call_output":
            # Tool result
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })
            last_item_was_tool_output = True

        elif item_type == "message" or "content" in item:
            # Standard message item
            content = item.get("content", "")
            if isinstance(content, list):
                content = _convert_content_parts(content)
            msg = {"role": role, "content": content}
            # Attach buffered reasoning to assistant messages.
            if role == "assistant" and pending_reasoning:
                msg["reasoning_content"] = pending_reasoning
                pending_reasoning = None
            elif "reasoning_content" in item:
                msg["reasoning_content"] = item["reasoning_content"]
            elif role == "assistant" and thinking_enabled and last_item_was_tool_output and last_assistant_reasoning:
                msg["reasoning_content"] = last_assistant_reasoning
            if role == "assistant" and msg.get("reasoning_content"):
                last_assistant_reasoning = msg["reasoning_content"]
            messages.append(msg)
            last_item_was_tool_output = False

        else:
            # Fallback: treat as a user message if it has text content
            text = item.get("text", item.get("content", ""))
            if text:
                messages.append({"role": role, "content": text})
            last_item_was_tool_output = False

    # If there's leftover buffered reasoning with no following assistant message,
    # emit it as a standalone assistant message
    if pending_reasoning:
        trailing_msg = {
            "role": "assistant",
            "content": "",
            "reasoning_content": pending_reasoning,
        }
        messages.append(trailing_msg)
        last_assistant_reasoning = trailing_msg["reasoning_content"]

    # Merge consecutive assistant messages.
    # Codex CLI emits each function_call as a separate item, but the Chat
    # Completions API forbids consecutive assistant messages.  We fold them
    # into a single assistant message with a combined tool_calls array.
    messages = _merge_consecutive_assistant(messages)

    return messages


def _merge_consecutive_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive assistant messages into one.

    DeepSeek (and most Chat Completions providers) reject consecutive
    assistant messages.  Codex CLI's Responses API format emits one item
    per function_call, which each become a separate assistant message
    after ``_convert_input_items``.

    This function merges them:
      - ``tool_calls`` arrays are concatenated.
      - ``reasoning_content`` strings are concatenated (newline-separated).
      - ``content`` is kept from the first non-None value; later text is
        appended (unlikely but defensive).
    """
    if not messages:
        return messages

    merged: list[dict[str, Any]] = []
    for msg in messages:
        if (
            merged
            and msg["role"] == "assistant"
            and merged[-1]["role"] == "assistant"
        ):
            prev = merged[-1]
            # Merge tool_calls
            if msg.get("tool_calls"):
                prev.setdefault("tool_calls", [])
                prev["tool_calls"].extend(msg["tool_calls"])
            # Merge reasoning_content
            if msg.get("reasoning_content"):
                if prev.get("reasoning_content"):
                    prev["reasoning_content"] += "\n" + msg["reasoning_content"]
                else:
                    prev["reasoning_content"] = msg["reasoning_content"]
            # Merge content (prefer non-None)
            new_content = msg.get("content")
            if new_content is not None:
                old_content = prev.get("content")
                if old_content is None or old_content == "":
                    prev["content"] = new_content
                elif isinstance(old_content, str) and isinstance(new_content, str):
                    prev["content"] = old_content + new_content
        else:
            merged.append(msg)

    return merged


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
    """Convert a Chat Completions response to a Responses API response.

    Handles DeepSeek's reasoning_content field:
      - reasoning_content from the assistant message is mapped to a
        "reasoning" output item in the Responses API format, placed
        before the text message output.
    """
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    output: list[dict[str, Any]] = []

    choices = chat_resp.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        role = message.get("role", "assistant")
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        tool_calls = message.get("tool_calls")

        # Reasoning/thinking output (DeepSeek V4)
        if reasoning_content:
            output.append({
                "type": "reasoning",
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "summary": [
                    {"type": "summary_text", "text": reasoning_content}
                ],
            })

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

    Event data format follows the OpenAI Responses API spec as consumed by
    Codex CLI (Rust):
      - Every data payload MUST have a "type" field matching the event name.
      - response.created / response.completed wrap the response in a "response" key.
      - response.output_item.added / .done wrap the item in an "item" key.
      - Delta events use top-level "delta", "item_id", etc.
    """
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    reasoning_id = f"rs_{uuid.uuid4().hex[:24]}"
    accumulated_text = ""
    accumulated_reasoning = ""
    accumulated_tool_calls: dict[int, dict[str, Any]] = {}  # index → partial tool call
    model = original_model
    usage: dict[str, Any] = {}
    created_emitted = False
    reasoning_part_emitted = False
    text_part_emitted = False

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
                # Build final output items for response.completed
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
                for _idx, tc in sorted(accumulated_tool_calls.items()):
                    final_output.append({
                        "type": "function_call",
                        "id": f"fc_{uuid.uuid4().hex[:24]}",
                        "call_id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", ""),
                    })

                # Emit response.output_item.done for each accumulated item
                for item in final_output:
                    yield _sse_event("response.output_item.done", {
                        "type": "response.output_item.done",
                        "item": item,
                    })

                # Emit response.completed (Codex CLI reads "response" sub-field)
                completed_resp = {
                    "id": resp_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": model,
                    "output": final_output,
                    "output_text": accumulated_text,
                    "status": "completed",
                    "usage": usage,
                }
                yield _sse_event("response.completed", {
                    "type": "response.completed",
                    "response": completed_resp,
                })
                return

            # Parse JSON
            try:
                chunk_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Detect in-stream error objects from backend.
            # Some providers (e.g. DeepSeek) may return HTTP 200 but send an
            # error JSON object in the SSE data instead of a normal chunk.
            if "error" in chunk_data and "choices" not in chunk_data:
                error_obj = chunk_data["error"]
                error_msg = error_obj.get("message", "Unknown upstream error")
                error_code = error_obj.get("code", "upstream_error")
                _logger.warning(
                    "[STREAM-ERR] In-stream error from backend:\n%s",
                    json.dumps(chunk_data, ensure_ascii=False, indent=2),
                )
                # Emit response.created if not yet emitted so Codex CLI sees
                # the response lifecycle.
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
                # Emit structured error and response.failed for Codex CLI
                yield _sse_event("error", {
                    "error": {
                        "message": error_msg,
                        "type": str(error_code),
                        "code": str(error_code),
                    }
                })
                failed_resp = {
                    "id": resp_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": model,
                    "output": [],
                    "output_text": "",
                    "status": "failed",
                    "error": {
                        "code": str(error_code),
                        "message": error_msg,
                    },
                    "usage": usage,
                }
                yield _sse_event("response.failed", {
                    "type": "response.failed",
                    "response": failed_resp,
                })
                return

            # Update model
            if "model" in chunk_data:
                model = chunk_data["model"]

            # Emit response.created once
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

                # Reasoning content delta (DeepSeek V4 thinking)
                reasoning_delta = delta.get("reasoning_content")
                if reasoning_delta:
                    if not reasoning_part_emitted:
                        reasoning_part_emitted = True
                        yield _sse_event("response.output_item.added", {
                            "type": "response.output_item.added",
                            "item": {
                                "type": "reasoning",
                                "id": reasoning_id,
                                "summary": [],
                            },
                        })
                    accumulated_reasoning += reasoning_delta
                    yield _sse_event("response.reasoning_summary_text.delta", {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": reasoning_id,
                        "summary_index": 0,
                        "delta": reasoning_delta,
                    })

                # Text content delta
                text_delta = delta.get("content")
                if text_delta:
                    if not text_part_emitted:
                        text_part_emitted = True
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
