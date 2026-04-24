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
Codex CLI → POST /v1/responses → proxy.py → translator.py → litellm_client.py → LiteLLM SDK → DeepSeek
                                                            ← translates response back ←
```

**Core data flow** (3 files to understand):

- **`proxy.py`** — Starlette ASGI app with 4 routes. Receives requests, calls translator, then dispatches via the shared LiteLLM client. Handles both streaming and non-streaming.
- **`translator.py`** — The heart of the project (~530 lines). `responses_request_to_chat()` converts inbound requests, `chat_response_to_responses()` converts responses, `translate_stream()` handles SSE event-by-event translation. Also maps Codex `reasoning.effort` → DeepSeek `thinking` params.
- **`litellm_client.py`** — The single upstream call path. Builds LiteLLM kwargs from preset data, executes async chat completions, and normalizes regular/streaming responses plus error metadata.
- **`config.py`** — Preset system. `Preset` and `ModelEntry` dataclasses loaded from YAML files. Built-in presets live in `presets/deepseek.yaml`, user custom presets in `~/.config/codex-adapter/presets/`.

**CLI layer** (`cli.py`): Click-based. Commands: `start`, `list`, `setup`, `deploy`, `service` (group with start/stop/restart/status/logs/install-systemd).

**Deploy module** (`deploy/`): Python-native deployment for future platform integration.
- `installer.py` — dependency detection/installation (Python 3.12+, uv, Node.js 20+, Codex CLI)
- `configurator.py` — generates `~/.codex-adapter.env`, `~/.codex/config.toml`, injects shell profile
- `service_manager.py` — background process lifecycle via PID file + subprocess
- `systemd.py` — generates and installs systemd unit files

## Key Design Decisions

- **All I/O is async** (httpx, Starlette, uvicorn). Don't introduce sync blocking calls in the request path.
- **All upstream model access must go through LiteLLM** via `codex_adapter.litellm_client`. Do not call provider chat-completions endpoints directly from `proxy.py` or other request-path code with raw `httpx`.
- **Thinking mode mapping is non-trivial**: DeepSeek's minimum is `high`, so Codex's `low`/`medium` both map to `high`. When thinking is enabled, `temperature`/`top_p` are dropped (DeepSeek restriction).
- **Deploy module uses pure functions** where possible (e.g., `generate_env_content()`, `generate_unit()`) for testability and future API/platform reuse.
- **Preset YAML is the configuration boundary** — all provider-specific details (api_base, model names, thinking support) are encapsulated in presets, not hardcoded.
- **Dependencies must be pinned exactly** in `pyproject.toml` (`==` only for runtime, dev, and build-system dependencies), and every dependency change must update `uv.lock` in the same change.

## Testing

Tests are in `tests/` using pytest + pytest-asyncio. The translator tests (`test_translator.py`, 437 lines) are the most important — they document the exact translation behavior for requests, responses, streaming, thinking mode, and tool calls. Use them as the spec when modifying translation logic.

## Principles

### Single Source of Truth

同一件事在整个仓库里必须只有一个事实标准来源。这是跨所有脚本、workflow 和文档的强约束，优先级高于"就近实现"和"临时便捷"。

**判定标准（满足任意一条即视为"同一件事"）：**

- 同一个判断（例如：Python/Node 版本是否满足要求、某工具是否可用）
- 同一类副作用（例如：写一个 structured log、通过 subprocess 启动守护进程、kubectl apply 本地 patch）
- 同一个数据源（例如：服务端口映射、模型配置清单、预设文件路径）

**禁止行为：**

- 脚本 A 和方法 1 做了判断，脚本 B 做同样的判断时另写方法 2（即使"两行就搞定"）。
- 把公共函数复制一份做"轻微定制"而不是汇总到公共模块。
- 在文档中给出与代码实现不一致的描述，制造第二个事实来源。

**改动前必做：**

1. 先查 `docs/ssot-registry.md`（如不存在则无需操作），确认要做的"判断/操作/数据读取"是否已有登记。
2. 若已有登记：直接复用该入口；不满足需求时改进那一个入口，不要在调用侧绕开。
3. 若没有登记但这是一个跨脚本反复出现的模式：抽到公共模块（如 `deploy/*.py` 或 `config.py`），并在注册表中登记一行。

> 注册表维护在 `docs/ssot-registry.md`，AGENTS.md 不收录具体条目以避免膨胀。注册表中每行记录一个入口（函数/变量/模块路径）及其解决的问题。当前项目规模小，注册表可选；当代码量增长后应补建。
