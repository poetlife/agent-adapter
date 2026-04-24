# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Codex Adapter is a protocol translation proxy. Codex CLI speaks **Responses API** (`POST /v1/responses`), but DeepSeek and other providers only support **Chat Completions API** (`POST /v1/chat/completions`). This proxy sits between them and translates both directions transparently, including streaming SSE events and thinking mode parameters.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest -v

# Run a single test file
uv run pytest tests/test_translator.py -v

# Run a single test
uv run pytest tests/test_translator.py::TestResponsesRequestToChat::test_simple_string_input -v

# Start proxy in foreground (development)
export DEEPSEEK_API_KEY=sk-xxx
uv run codex-adapter start --preset deepseek --debug

# One-shot configure (writes env file, codex config, shell profile)
uv run codex-adapter setup -p deepseek -k sk-xxx

# Background service management
uv run codex-adapter service start
uv run codex-adapter service status
uv run codex-adapter service stop
```

## Architecture

```
Codex CLI → POST /v1/responses → proxy.py → translator.py → POST /v1/chat/completions → DeepSeek
                                                ← translates response back ←
```

**Core data flow** (3 files to understand):

- **`proxy.py`** — Starlette ASGI app with 4 routes. Receives requests, calls translator, forwards to backend via httpx async client. Handles both streaming and non-streaming.
- **`translator.py`** — The heart of the project (~530 lines). `responses_request_to_chat()` converts inbound requests, `chat_response_to_responses()` converts responses, `translate_stream()` handles SSE event-by-event translation. Also maps Codex `reasoning.effort` → DeepSeek `thinking` params.
- **`config.py`** — Preset system. `Preset` and `ModelEntry` dataclasses loaded from YAML files. Built-in presets live in `presets/deepseek.yaml`, user custom presets in `~/.config/codex-adapter/presets/`.

**CLI layer** (`cli.py`): Click-based. Commands: `start`, `list`, `setup`, `deploy`, `service` (group with start/stop/restart/status/logs/install-systemd).

**Deploy module** (`deploy/`): Python-native deployment for future platform integration.
- `installer.py` — dependency detection/installation (Python 3.12+, uv, Node.js 20+, Codex CLI)
- `configurator.py` — generates `~/.codex-adapter.env`, `~/.codex/config.toml`, injects shell profile
- `service_manager.py` — background process lifecycle via PID file + subprocess
- `systemd.py` — generates and installs systemd unit files

## Key Design Decisions

- **All I/O is async** (httpx, Starlette, uvicorn). Don't introduce sync blocking calls in the request path.
- **Thinking mode mapping is non-trivial**: DeepSeek's minimum is `high`, so Codex's `low`/`medium` both map to `high`. When thinking is enabled, `temperature`/`top_p` are dropped (DeepSeek restriction).
- **Deploy module uses pure functions** where possible (e.g., `generate_env_content()`, `generate_unit()`) for testability and future API/platform reuse.
- **Preset YAML is the configuration boundary** — all provider-specific details (api_base, model names, thinking support) are encapsulated in presets, not hardcoded.

## Testing

Tests are in `tests/` using pytest + pytest-asyncio. The translator tests (`test_translator.py`, 437 lines) are the most important — they document the exact translation behavior for requests, responses, streaming, thinking mode, and tool calls. Use them as the spec when modifying translation logic.
