"""Proxy server that translates Responses API → Chat Completions API.

Codex CLI sends:  POST /v1/responses  (Responses API format)
This proxy:       POST /v1/chat/completions  (to LiteLLM / DeepSeek / etc.)

The proxy handles both streaming and non-streaming requests, translating
the protocol in both directions transparently.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import uvicorn
import yaml
from rich.console import Console
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse, Response
from starlette.routing import Route

from codex_adapter.config import Preset, get_user_config_dir
from codex_adapter.translator import (
    ModelConfig,
    chat_response_to_responses,
    responses_request_to_chat,
    translate_stream,
)

console = Console()


def _build_backend_url(preset: Preset, model_name: str | None = None) -> str:
    """Determine the backend Chat Completions URL for a given model."""
    # Find the matching model in the preset
    target = None
    if model_name:
        for m in preset.models:
            if m.name == model_name:
                target = m
                break
    if target is None and preset.models:
        target = preset.models[0]

    if target is None:
        raise ValueError("No models defined in preset")

    api_base = target.api_base.rstrip("/")
    return f"{api_base}/chat/completions"


def _get_api_key(preset: Preset) -> str:
    """Get the API key from environment."""
    key = os.environ.get(preset.env_key, "")
    if not key:
        raise ValueError(f"Environment variable {preset.env_key} is not set")
    return key


def create_app(preset: Preset) -> Starlette:
    """Create the ASGI application with Responses API → Chat Completions translation."""

    async def handle_responses(request: Request) -> Response:
        """Handle POST /v1/responses — the core translation endpoint."""
        body = await request.json()
        model_name = body.get("model", "")
        is_stream = body.get("stream", False)

        # Find matching model entry for thinking config
        model_entry = None
        for m in preset.models:
            if m.name == model_name:
                model_entry = m
                break
        if model_entry is None and preset.models:
            model_entry = preset.models[0]

        # Build model config for translator
        model_config = ModelConfig(
            supports_thinking=model_entry.supports_thinking if model_entry else False,
            default_thinking=model_entry.default_thinking if model_entry else "disabled",
            reasoning_effort=model_entry.reasoning_effort if model_entry else "high",
        )

        # Translate Responses API request → Chat Completions request
        chat_body = responses_request_to_chat(body, model_config=model_config)

        # Map model name to litellm_model identifier
        if model_entry:
            chat_body["model"] = model_entry.litellm_model

        # Build backend request
        backend_url = _build_backend_url(preset, model_name)
        api_key = _get_api_key(preset)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if is_stream:
            return await _handle_streaming(backend_url, headers, chat_body, model_name)
        else:
            return await _handle_non_streaming(backend_url, headers, chat_body, model_name)

    async def _handle_non_streaming(
        url: str, headers: dict, chat_body: dict, original_model: str
    ) -> JSONResponse:
        """Forward non-streaming request and translate response."""
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(url, json=chat_body, headers=headers)

        if resp.status_code != 200:
            return JSONResponse(
                {"error": {"message": resp.text, "type": "upstream_error", "code": resp.status_code}},
                status_code=resp.status_code,
            )

        chat_resp = resp.json()
        responses_resp = chat_response_to_responses(chat_resp, original_model)
        return JSONResponse(responses_resp)

    async def _handle_streaming(
        url: str, headers: dict, chat_body: dict, original_model: str
    ) -> StreamingResponse:
        """Forward streaming request and translate SSE events."""

        async def event_generator():
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", url, json=chat_body, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        error_msg = error_body.decode("utf-8", errors="replace")
                        error_event = {
                            "error": {
                                "message": error_msg,
                                "type": "upstream_error",
                                "code": resp.status_code,
                            }
                        }
                        yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
                        # Codex CLI expects every stream to end with response.completed + done
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
                            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                            "error": error_event["error"],
                        }
                        yield f"event: response.completed\ndata: {json.dumps(failed_resp)}\n\n".encode()
                        yield b"event: done\ndata: [DONE]\n\n"
                        return

                    async for chunk in translate_stream(resp.aiter_lines(), original_model):
                        yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def handle_models(request: Request) -> JSONResponse:
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
        model_name = body.get("model", "")
        is_stream = body.get("stream", False)

        # Map model name
        for m in preset.models:
            if m.name == model_name:
                body["model"] = m.litellm_model
                break

        backend_url = _build_backend_url(preset, model_name)
        api_key = _get_api_key(preset)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if is_stream:
            async def stream_passthrough():
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream("POST", backend_url, json=body, headers=headers) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk

            return StreamingResponse(
                stream_passthrough(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(backend_url, json=body, headers=headers)
            return JSONResponse(resp.json(), status_code=resp.status_code)

    async def handle_health(request: Request) -> JSONResponse:
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
    console.print(f"  Translate  : [yellow]Responses API → Chat Completions[/]")
    console.print()
    console.print("[bold yellow]Codex CLI usage:[/]")
    console.print(f"  export OPENAI_BASE_URL=http://localhost:{port}/v1")
    console.print(f"  export OPENAI_API_KEY=sk-placeholder")
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
