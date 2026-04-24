# Codex Adapter

> Adapt OpenAI Codex CLI to work with DeepSeek and other models — zero config on the Codex side.

最新版 Codex CLI 使用 OpenAI 的 **Responses API** (`/v1/responses`)，而 DeepSeek 等第三方模型只支持 **Chat Completions API** (`/v1/chat/completions`)。

本项目在本地启动一个轻量代理服务器，**自动完成两种协议之间的双向翻译**，对 Codex CLI 完全透明——无需配置 `wire_api`，无需改动 Codex CLI 本身。

## 架构

```
Codex CLI                    Codex Adapter Proxy              DeepSeek API
──────────                   ───────────────────              ────────────
POST /v1/responses    →      接收 Responses API 请求
(Responses API 格式)          ↓ 翻译请求格式
                             POST /v1/chat/completions  →    处理请求
                             ↓ 翻译响应格式               ←  返回结果
                      ←      返回 Responses API 响应
```

**翻译内容包括：**
- `instructions` ↔ system message
- `input` ↔ messages 数组
- `max_output_tokens` ↔ `max_tokens`
- function tool calls 双向映射
- SSE 流式事件格式完整转换

## 快速开始

```bash
# 1. 克隆并安装
git clone <repo-url> && cd agent-adapter
uv sync

# 2. 设置 API Key
export DEEPSEEK_API_KEY=sk-your-key-here

# 3. 启动代理
uv run codex-adapter start --preset deepseek

# 4. 在另一个终端中使用 Codex CLI（原生 Responses API 模式）
export OPENAI_BASE_URL=http://localhost:4000/v1
export OPENAI_API_KEY=sk-placeholder
codex --model deepseek-chat "help me fix this bug"
```

## 命令

### `codex-adapter start`

启动协议翻译代理服务器。

```bash
uv run codex-adapter start --preset deepseek              # 默认端口 4000
uv run codex-adapter start --preset deepseek --port 8080  # 自定义端口
uv run codex-adapter start --preset deepseek --debug      # 开启调试日志
```

### `codex-adapter list`

列出所有可用的模型预设。

```bash
uv run codex-adapter list
```

### `codex-adapter setup`

显示 Codex CLI 的配置指引。

```bash
uv run codex-adapter setup --preset deepseek
uv run codex-adapter setup --preset deepseek --write-config  # 自动写入 config.toml
```

## 支持的模型

### DeepSeek

| 模型名称 | 说明 |
|---------|------|
| `deepseek-chat` | DeepSeek 通用对话模型 |
| `deepseek-coder` | DeepSeek 代码生成模型 |
| `deepseek-reasoner` | DeepSeek 推理模型 |

## 添加自定义模型

在 `~/.config/codex-adapter/presets/` 目录下创建 YAML 文件：

```yaml
# ~/.config/codex-adapter/presets/my-provider.yaml
provider: my-provider
description: "My custom provider"
env_key: MY_PROVIDER_API_KEY

models:
  - name: my-model
    litellm_model: openai/my-model
    api_base: https://api.my-provider.com/v1
    max_tokens: 4096
```

然后使用：

```bash
uv run codex-adapter start --preset my-provider
```

## API 端点

代理服务器提供以下端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/responses` | POST | 核心端点：接收 Responses API 请求，翻译后转发 |
| `/v1/chat/completions` | POST | 直通端点：直接转发 Chat Completions 请求 |
| `/v1/models` | GET | 列出可用模型 |
| `/health` | GET | 健康检查 |

## 开发

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest -v

# 运行单个测试文件
uv run pytest tests/test_translator.py -v
```

## 工作原理

1. Codex CLI 发送 Responses API 请求到 `POST /v1/responses`
2. 代理将 `instructions` 转为 system message，将 `input` 转为 messages 数组
3. 代理向 DeepSeek 等后端发送标准 Chat Completions 请求
4. 后端返回 Chat Completions 响应
5. 代理将响应翻译回 Responses API 格式返回给 Codex CLI
6. 流式响应 (SSE) 同样完整翻译事件格式

## License

MIT
