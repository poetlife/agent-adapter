# SSOT Registry

本文件登记所有跨脚本/跨模块的共享入口（函数、变量、模块路径），确保同一件事只有一个事实标准来源。新增共享逻辑前请先查此表。

---

## 🗂 配置与预设

| 入口 | 所属模块 | 解决的问题 | 被谁使用 |
|---|---|---|---|
| `Preset` / `ModelEntry` dataclass | `config.py` | 预设数据的结构定义及 YAML 解析 | cli.py, proxy.py, codex_setup.py, configurator.py |
| `load_preset(name, custom_dir)` | `config.py` | 按名称加载预设（自定义目录优先，回退内置） | cli.py, proxy.py, configurator.py |
| `list_presets(custom_dir)` | `config.py` | 列出所有可用预设名称 | cli.py, configurator.py |
| `get_user_config_dir()` | `config.py` | 返回用户配置目录 `~/.config/codex-adapter/` | proxy.py, config.py 内部 |
| `get_user_presets_dir()` | `config.py` | 返回用户自定义预设目录 | cli.py, configurator.py |
| `get_builtin_presets_dir()` | `config.py` | 返回内置预设目录 | cli.py（仅用于 list） |
| `presets/deepseek.yaml` | `src/codex_adapter/presets/` | 厂商 API 基地址、模型清单、thinking 支持标记（YAML 为唯一数据源） | config.py |

## 🔄 协议翻译

| 入口 | 所属模块 | 解决的问题 | 被谁使用 |
|---|---|---|---|
| `ModelConfig` dataclass | `translator.py` | 单模型 thinking 配置：是否支持、默认开关、reasoning_effort | proxy.py |
| `responses_request_to_chat(body, model_config)` | `translator.py` | 将 Responses API 请求体转为 Chat Completions 请求体 | proxy.py |
| `chat_response_to_responses(chat_resp, original_model)` | `translator.py` | 将 Chat Completions 响应体转为 Responses API 响应体 | proxy.py |
| `translate_stream(...)` | `translator.py` | SSE 流式翻译：逐 event 转换 Chat Completions → Responses 格式 | proxy.py |

## 🚀 部署与配置

| 入口 | 所属模块 | 解决的问题 | 被谁使用 |
|---|---|---|---|
| `CheckResult` / `OSInfo` dataclass | `deploy/installer.py` | 依赖检查结果与 OS 信息的结构化定义 | installer.py 内部 |
| `detect_os()` | `deploy/installer.py` | 检测操作系统类型和架构 | installer.py 内部 |
| `check_python()` / `check_uv()` / `check_node()` / `check_codex_cli()` | `deploy/installer.py` | 逐一检查依赖项版本和安装状态 | cli.py (deploy), installer.py 内部 |
| `check_all()` / `install_all(project_dir)` | `deploy/installer.py` | 批量检查和安装全量依赖 | cli.py (deploy) |
| `generate_env_content(api_key, env_key, preset_name, port, project_dir)` | `deploy/configurator.py` | 生成 `.codex-adapter.env` 文件内容 | configurator.py 内部 |
| `write_env_file(...)` | `deploy/configurator.py` | 写入 `.codex-adapter.env` 并设置权限 | cli.py (setup), configurator.py 内部 |
| `generate_codex_config_toml(port, model, all_models)` | `codex_setup.py` | 生成 `~/.codex/config.toml` 片段 | codex_setup.py, configurator.py |
| `generate_model_catalog(models)` | `codex_setup.py` | 为 Codex CLI 生成 model-catalog.json 数据结构 | codex_setup.py 内部 |
| `write_model_catalog(models)` | `codex_setup.py` | 将 model-catalog.json 写入磁盘 | codex_setup.py, configurator.py |
| `detect_codex_cli()` | `codex_setup.py` | 检查 codex 命令是否在 PATH 中并返回路径 | codex_setup.py |

## ⚙️ 服务生命周期

| 入口 | 所属模块 | 解决的问题 | 被谁使用 |
|---|---|---|---|
| `ServiceStatus` dataclass | `deploy/service_manager.py` | 后台服务状态的结构化定义 | service_manager.py 内部 |
| `is_running()` / `health_check(port, ...)` | `deploy/service_manager.py` | 检测服务是否在运行、健康检查 | service_manager.py 内部, cli.py |
| `start(preset, port, ...)` | `deploy/service_manager.py` | 以 subprocess 启动后台代理 | cli.py (service start) |
| `stop()` | `deploy/service_manager.py` | 停止后台代理进程 | cli.py (service stop) |
| `status()` / `print_status()` | `deploy/service_manager.py` | 查询并展示服务状态 | cli.py (service status) |
| `logs(follow, lines)` | `deploy/service_manager.py` | 查看后台日志 | cli.py (service logs) |
| `is_systemd_available()` / `is_unit_installed()` | `deploy/systemd.py` | 判断 systemd 环境及单元安装状态 | systemd.py 内部 |
| `generate_unit(preset, port, ...)` | `deploy/systemd.py` | 生成 systemd unit 文件内容 | systemd.py 内部 |
| `install_unit(preset, port, ...)` | `deploy/systemd.py` | 安装/卸载 systemd unit 文件 | cli.py (service install-systemd) |
| `start_via_systemd()` / `stop_via_systemd()` | `deploy/systemd.py` | 通过 systemctl 启动/停止服务 | cli.py (service install-systemd) |

## 🚧 已知潜在重复（需要关注）

| 关注点 | 涉及位置 | 风险说明 |
|---|---|---|
| `installer.py` 中的 `_version_gte()` | `deploy/installer.py:110` | 版本比较工具函数，当前只在 installer 内部使用。若其他模块也需要版本比较，应抽到公共 utils |
| `installer.py` 中的 `check_uv()` vs `service_manager.py` 中的 `_find_uv()` | 两个模块各自独立查找 uv 路径 | 前者用于依赖检查，后者用于启动 subprocess。若逻辑不一致会导致"安装后仍找不到"。跨模块使用时需统一 |
| `codex_setup.py` 中的 `detect_codex_cli()` vs `installer.py` 中的 `check_codex_cli()` | 两个不同签名但检测同一件事 | 当前各自服务于不同上下文（PATH 查询 vs 版本检查），可接受；若未来扩展应合入公共接口 |
| `codex_setup.py` 中的 `generate_shell_exports()` vs `configurator.py` 中的 `generate_env_content()` | 两者生成环境变量配置但目标不同（shell profile vs .env 文件） | 若后续字段增多应拆出公共的"环境变量字典"生成函数 |
