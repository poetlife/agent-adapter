# SSOT Registry

本文件登记所有跨脚本/跨模块的共享入口（函数、变量、模块路径），确保同一件事只有一个事实标准来源。新增共享逻辑前请先查此表。

---

## 🗂 Provider 与预设

| 入口 | 所属模块 | 解决的问题 | 被谁使用 |
|---|---|---|---|
| `Preset` / `ModelEntry` dataclass | `providers.catalog` | provider 预设结构定义与 YAML 解析 | cli, codex_setup, entrypoints, deploy |
| `Preset.resolve_model(name)` | `providers.catalog` | 公开模型名到具体 provider 模型的统一解析与默认回退 | entrypoints, providers.litellm_client |
| `load_preset(name, custom_dir)` | `providers.catalog` | 按名称加载 provider 预设（自定义优先，回退内置） | cli, deploy |
| `list_presets(custom_dir)` | `providers.catalog` | 列出全部可用预设 | cli, deploy |
| `get_user_config_dir()` / `get_user_presets_dir()` / `get_builtin_presets_dir()` | `providers.catalog` | provider 配置目录与内置预设目录定位 | cli, deploy, logging |
| `providers/presets/*.yaml` | `src/providers/presets/` | provider API 地址、模型清单、thinking 能力等唯一数据源 | providers.catalog |

## 🔄 协议与入口

| 入口 | 所属模块 | 解决的问题 | 被谁使用 |
|---|---|---|---|
| `ModelConfig` | `protocols.responses_chat` | Responses -> Chat 转换时的 thinking 配置结构 | entrypoints.responses_proxy |
| `responses_request_to_chat()` | `protocols.responses_chat` | Responses API → Chat Completions 请求翻译（含 DeepSeek thinking 定制、reasoning carry-forward） | entrypoints.responses_proxy |
| `transform_chat_to_responses()` | `providers.litellm_client` | Chat Completions → Responses API 响应翻译（委托 LiteLLM 内置 `LiteLLMCompletionResponsesConfig`，加 Codex 格式修正） | entrypoints.responses_proxy |
| `stream_chat_as_responses_sse()` | `providers.litellm_client` | Chat Completions 流 → Responses API SSE 事件流（委托 LiteLLM `LiteLLMCompletionStreamingIterator`） | entrypoints.responses_proxy |
| `generate_codex_model_catalog()` | `protocols.codex_model_catalog` | 生成 Codex CLI 兼容的模型目录输出，避免 setup/proxy 各写一份 | codex_setup, entrypoints.responses_proxy |
| `create_app()` / `start_proxy()` | `entrypoints.responses_proxy` | HTTP 入口、请求生命周期、错误包装、SSE 输出 | cli |

## 🌐 上游访问

| 入口 | 所属模块 | 解决的问题 | 被谁使用 |
|---|---|---|---|
| `build_completion_kwargs()` / `request_chat_completion()` / `serialize_completion_*()` | `providers.litellm_client` | 统一通过 LiteLLM 构造和发起上游请求，并序列化普通/流式响应 | entrypoints.responses_proxy |
| `litellm_error_status_code()` / `litellm_error_message()` | `providers.litellm_client` | 统一上游错误提取与包装 | entrypoints.responses_proxy |

## 🪵 公共基础设施

| 入口 | 所属模块 | 解决的问题 | 被谁使用 |
|---|---|---|---|
| `init_logging()` / `request_log_context()` / `resolve_trace_id()` / `log_*()` | `common.logging` | 统一日志初始化、trace id 绑定和结构化日志输出 | entrypoints, protocols |
| `get_app_config_dir(app_name)` | `common.runtime_paths` | 公共运行时配置目录定位 | providers.catalog, common.logging |

## 🚀 Codex Adapter

| 入口 | 所属模块 | 解决的问题 | 被谁使用 |
|---|---|---|---|
| `generate_codex_config_toml()` / `write_model_catalog()` / `print_setup_instructions()` | `codex_adapter.codex_setup` | Codex CLI 快速配置与本地文件生成 | cli, deploy |
| `configure_all()` / `write_env_file()` / `write_codex_config_file()` | `codex_adapter.deploy.configurator` | 一键部署配置 | cli |
| `install_all()` / `service_manager.*` / `systemd.*` | `codex_adapter.deploy.*` | 安装依赖与后台服务管理 | cli |

