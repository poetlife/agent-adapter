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

        # Translate Responses API request → Chat Completions request
        chat_body = responses_request_to_chat(body)

        # Map model name to litellm_model identifier
        for m in preset.models:
            if m.name == model_name:
                chat_body["model"] = m.litellm_model
                break

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
                        error_event = {
                            "error": {
                                "message": error_body.decode("utf-8", errors="replace"),
                                "type": "upstream_error",
                                "code": resp.status_code,
                            }
                        }
                        yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
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
        """Handle GET /v1/models — list available models."""
        models = []
        for m in preset.models:
            models.append({
                "id": m.name,
                "object": "model",
                "owned_by": preset.provider,
            })
        return JSONResponse({"object": "list", "data": models})

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
