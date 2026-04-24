"""Proxy server that translates Responses API → Chat Completions API.

Codex CLI sends:  POST /v1/responses  (Responses API format)
This proxy:       POST /v1/chat/completions  (via LiteLLM to DeepSeek / etc.)

The proxy handles both streaming and non-streaming requests, translating
the protocol in both directions transparently.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import uvicorn
from rich.console import Console
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse, Response
from starlette.routing import Route

from codex_adapter.config import Preset
from codex_adapter.litellm_client import (
    litellm_error_message,
    litellm_error_status_code,
    request_chat_completion,
    serialize_completion_response,
    serialize_completion_stream,
)
from codex_adapter.logging_utils import (
    init_logging,
    log_debug,
    request_log_context,
    resolve_trace_id,
)
from codex_adapter.translator import (
    ModelConfig,
    chat_response_to_responses,
    responses_request_to_chat,
    translate_stream,
)

console = Console()


def _sanitize_messages_for_log(messages: list[dict]) -> list[dict]:
    """Summarize messages for debug log — truncate long content, keep structure."""
    result = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.get("role", "?")}
        # Content — truncate to 200 chars
        content = msg.get("content")
        if content is None:
            entry["content"] = None
        elif isinstance(content, str):
            entry["content"] = content[:200] + ("..." if len(content) > 200 else "")
        else:
            entry["content"] = "[list]"
        # reasoning_content
        if "reasoning_content" in msg:
            rc = msg["reasoning_content"]
            entry["reasoning_content"] = rc[:100] + ("..." if len(rc) > 100 else "") if rc else rc
        # tool_calls — just count + names
        if "tool_calls" in msg:
            tcs = msg["tool_calls"]
            entry["tool_calls"] = [
                {"id": tc.get("id", "?"), "name": tc.get("function", {}).get("name", "?")}
                for tc in tcs
            ]
        # tool_call_id
        if "tool_call_id" in msg:
            entry["tool_call_id"] = msg["tool_call_id"]
        result.append(entry)
    return result


def _strip_reasoning_for_retry(chat_body: dict[str, Any]) -> dict[str, Any]:
    """Disable thinking and strip reasoning_content for one retry attempt."""
    fallback_body = dict(chat_body)
    fallback_body.pop("thinking", None)
    fallback_body.pop("reasoning_effort", None)
    fallback_body["messages"] = [dict(msg) for msg in chat_body.get("messages", [])]
    for msg in fallback_body["messages"]:
        msg.pop("reasoning_content", None)
    return fallback_body


def _should_retry_without_reasoning(exc: Exception, chat_body: dict[str, Any]) -> bool:
    """Retry once when DeepSeek rejects reasoning_content in multi-turn context."""
    return (
        litellm_error_status_code(exc) == 400
        and "reasoning_content" in litellm_error_message(exc)
        and chat_body.get("thinking", {}).get("type") == "enabled"
    )


def create_app(preset: Preset) -> Starlette:
    """Create the ASGI application with Responses API → Chat Completions translation."""

    async def handle_responses(request: Request) -> Response:
        """Handle POST /v1/responses — the core translation endpoint."""
        body = await request.json()
        trace_id, trace_id_source = resolve_trace_id(body, request.headers)
        model_name = body.get("model", "")
        is_stream = body.get("stream", False)

        with request_log_context(trace_id):
            # Find matching model entry for thinking config
            model_entry = preset.resolve_model(model_name)

            # Build model config for translator
            model_config = ModelConfig(
                supports_thinking=model_entry.supports_thinking if model_entry else False,
                default_thinking=model_entry.default_thinking if model_entry else "disabled",
                reasoning_effort=model_entry.reasoning_effort if model_entry else "high",
            )

            # Translate Responses API request → Chat Completions request
            chat_body = responses_request_to_chat(body, model_config=model_config)

            # Map model name to LiteLLM identifier
            if model_entry:
                chat_body["model"] = model_entry.litellm_model

            log_debug("[REQ] Responses API input", {
                "trace_id_source": trace_id_source,
                "model": model_name,
                "stream": is_stream,
                "input_item_count": len(body.get("input", [])) if isinstance(body.get("input"), list) else "string",
                "reasoning": body.get("reasoning"),
                "tools_count": len(body.get("tools", [])),
            })
            raw_input = body.get("input", "")
            if isinstance(raw_input, list):
                log_debug("[REQ] Raw input items", [
                    {k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v)
                     for k, v in item.items()}
                    for item in raw_input
                ])
            log_debug("[REQ] Translated Chat Completions body", {
                "model": chat_body.get("model"),
                "thinking": chat_body.get("thinking"),
                "reasoning_effort": chat_body.get("reasoning_effort"),
                "message_count": len(chat_body.get("messages", [])),
                "messages": _sanitize_messages_for_log(chat_body.get("messages", [])),
            })

        if is_stream:
            return await _handle_streaming(chat_body, model_name, trace_id)
        return await _handle_non_streaming(chat_body, model_name, trace_id)

    async def _handle_non_streaming(chat_body: dict, original_model: str, trace_id: str) -> JSONResponse:
        """Forward non-streaming request and translate response."""
        with request_log_context(trace_id):
            try:
                chat_resp = await request_chat_completion(preset, chat_body, model_name=original_model)
            except Exception as exc:
                status_code = litellm_error_status_code(exc)
                error_msg = litellm_error_message(exc)
                log_debug("[ERR] Non-streaming backend error", {
                    "status": status_code,
                    "error": error_msg[:2000],
                })
                return JSONResponse(
                    {"error": {"message": error_msg, "type": "upstream_error", "code": status_code}},
                    status_code=status_code,
                )

            chat_resp = serialize_completion_response(chat_resp)
            responses_resp = chat_response_to_responses(chat_resp, original_model)
            return JSONResponse(responses_resp)

    async def _handle_streaming(chat_body: dict, original_model: str, trace_id: str) -> StreamingResponse:
        """Forward streaming request and translate SSE events."""

        async def event_generator():
            nonlocal chat_body
            with request_log_context(trace_id):
                try:
                    stream = await request_chat_completion(preset, chat_body, model_name=original_model)
                    async for chunk in translate_stream(serialize_completion_stream(stream), original_model):
                        yield chunk
                    return
                except Exception as exc:
                    if _should_retry_without_reasoning(exc, chat_body):
                        chat_body = _strip_reasoning_for_retry(chat_body)
                        try:
                            stream = await request_chat_completion(preset, chat_body, model_name=original_model)
                            async for chunk in translate_stream(serialize_completion_stream(stream), original_model):
                                yield chunk
                            return
                        except Exception as retry_exc:
                            exc = retry_exc

                    error_msg = litellm_error_message(exc)
                    error_code = litellm_error_status_code(exc)
                    log_debug("[ERR] Backend returned non-200", {
                        "status": error_code,
                        "error": error_msg,
                        "chat_body_messages": _sanitize_messages_for_log(chat_body.get("messages", [])),
                        "thinking": chat_body.get("thinking"),
                    })
                    error_event = {
                        "error": {
                            "message": error_msg,
                            "type": "upstream_error",
                            "code": error_code,
                        }
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
                    # Codex CLI expects response.failed with proper wrapping
                    import time as _time
                    import uuid as _uuid
                    failed_resp = {
                        "id": f"resp_{_uuid.uuid4().hex[:24]}",
                        "object": "response",
                        "created_at": int(_time.time()),
                        "model": original_model,
                        "output": [],
                        "output_text": "",
                        "status": "failed",
                        "error": {
                            "code": str(error_code),
                            "message": error_msg,
                        },
                        "usage": None,
                    }
                    yield f"event: response.failed\ndata: {json.dumps({'type': 'response.failed', 'response': failed_resp})}\n\n".encode()
                    return

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def handle_models(_request: Request) -> JSONResponse:
        """Handle GET /v1/models — list available models.

        Returns the Codex CLI model catalog format (ModelsResponse):
        {"models": [ModelInfo, ...]}

        Each ModelInfo must include metadata fields that Codex CLI expects
        (slug, display_name, shell_type, supported_reasoning_levels, etc.).
        """
        models = []
        for m in preset.models:
            # Build reasoning levels based on model thinking support
            if m.supports_thinking:
                reasoning_levels = [
                    {"effort": "low", "description": "Fast responses with lighter reasoning"},
                    {"effort": "medium", "description": "Balanced speed and reasoning depth"},
                    {"effort": "high", "description": "Greater reasoning depth"},
                ]
                default_reasoning = "medium"
            else:
                reasoning_levels = []
                default_reasoning = None

            models.append({
                "slug": m.name,
                "display_name": m.name,
                "description": m.description or f"{preset.provider} model",
                "default_reasoning_level": default_reasoning,
                "supported_reasoning_levels": reasoning_levels,
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 1,
                "additional_speed_tiers": [],
                "availability_nux": None,
                "upgrade": None,
                "base_instructions": "",
                "supports_reasoning_summaries": m.supports_thinking,
                "default_reasoning_summary": "none",
                "support_verbosity": False,
                "default_verbosity": None,
                "apply_patch_tool_type": "freeform",
                "web_search_tool_type": "text",
                "truncation_policy": {"mode": "tokens", "limit": 10000},
                "supports_parallel_tool_calls": True,
                "supports_image_detail_original": False,
                "context_window": m.context_length,
                "max_context_window": m.context_length,
                "effective_context_window_percent": 90,
                "experimental_supported_tools": [],
                "input_modalities": ["text"],
            })
        return JSONResponse({"models": models})

    async def handle_chat_completions(request: Request) -> Response:
        """Handle POST /v1/chat/completions — pass-through for direct Chat Completions calls."""
        body = await request.json()
        trace_id, trace_id_source = resolve_trace_id(body, request.headers)
        model_name = body.get("model", "")
        is_stream = body.get("stream", False)

        with request_log_context(trace_id):
            log_debug("[REQ] Direct Chat Completions input", {
                "trace_id_source": trace_id_source,
                "model": model_name,
                "stream": is_stream,
            })

            # Map model name
            model_entry = preset.resolve_model(model_name)
            if model_entry:
                body["model"] = model_entry.litellm_model

            if is_stream:
                try:
                    stream = await request_chat_completion(preset, body, model_name=model_name)
                except Exception as exc:
                    error_code = litellm_error_status_code(exc)
                    error_msg = litellm_error_message(exc)
                    log_debug("[ERR] Direct chat streaming backend error", {
                        "status": error_code,
                        "error": error_msg[:2000],
                    })
                    return JSONResponse(
                        {"error": {"message": error_msg, "type": "upstream_error", "code": error_code}},
                        status_code=error_code,
                    )

                async def stream_passthrough():
                    with request_log_context(trace_id):
                        try:
                            async for chunk in serialize_completion_stream(stream):
                                yield chunk
                        except Exception as exc:
                            error_payload = {
                                "error": {
                                    "message": litellm_error_message(exc),
                                    "type": "upstream_error",
                                    "code": litellm_error_status_code(exc),
                                }
                            }
                            log_debug("[ERR] Direct chat streaming passthrough error", error_payload)
                            yield f"data: {json.dumps(error_payload)}\n\n".encode("utf-8")

                return StreamingResponse(
                    stream_passthrough(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )

            try:
                resp = await request_chat_completion(preset, body, model_name=model_name)
            except Exception as exc:
                error_code = litellm_error_status_code(exc)
                error_msg = litellm_error_message(exc)
                log_debug("[ERR] Direct chat backend error", {
                    "status": error_code,
                    "error": error_msg[:2000],
                })
                return JSONResponse(
                    {"error": {"message": error_msg, "type": "upstream_error", "code": error_code}},
                    status_code=error_code,
                )
            return JSONResponse(serialize_completion_response(resp))

    async def handle_health(_request: Request) -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse({"status": "ok", "provider": preset.provider})

    routes = [
        Route("/v1/responses", handle_responses, methods=["POST"]),
        Route("/v1/models", handle_models, methods=["GET"]),
        Route("/v1/chat/completions", handle_chat_completions, methods=["POST"]),
        Route("/health", handle_health, methods=["GET"]),
    ]

    return Starlette(routes=routes)


def start_proxy(
    preset: Preset,
    port: int = 4000,
    host: str = "0.0.0.0",
    debug: bool = False,
) -> None:
    """Start the proxy server (blocking)."""
    log_path = init_logging(debug=debug)

    # Validate API key
    api_key = os.environ.get(preset.env_key)
    if not api_key:
        console.print(
            f"[bold red]Error:[/] Environment variable [bold]{preset.env_key}[/] is not set.\n"
            f"Please set it first:\n\n"
            f"  export {preset.env_key}=your-api-key-here\n",
        )
        sys.exit(1)

    console.print()
    console.print("[bold green]Codex Adapter Proxy[/]")
    console.print(f"  Provider   : [cyan]{preset.provider}[/]")
    console.print(f"  Models     : [cyan]{', '.join(m.name for m in preset.models)}[/]")
    console.print(f"  Listen     : [cyan]http://{host}:{port}[/]")
    console.print("  Translate  : [yellow]Responses API → Chat Completions[/]")
    console.print(f"  Debug log  : [cyan]{log_path}[/]")
    console.print()
    console.print("[bold yellow]Codex CLI usage:[/]")
    console.print(f"  export OPENAI_BASE_URL=http://localhost:{port}/v1")
    console.print("  export OPENAI_API_KEY=sk-placeholder")
    console.print(f"  codex --model {preset.models[0].name} \"your prompt\"")
    console.print()
    console.print("[dim]Press Ctrl+C to stop the proxy.[/]")
    console.print()

    app = create_app(preset)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
    )
