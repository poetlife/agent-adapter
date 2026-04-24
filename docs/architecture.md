# Architecture

## Goals

当前仓库不再把所有能力都压在 `codex_adapter` 包下面，而是按“共享能力 / provider 能力 / 协议转换 / 入口 / adapter 自身”分层。这样后续新增其他 adapter 时，可以直接复用上游访问、日志、协议转换和预设目录，而不是再复制一套。

核心链路：

`entrypoints` -> `protocols`（请求翻译）-> `providers.litellm_client`（请求 + 响应翻译）-> `DeepSeek / Hunyuan / other providers`

`codex_adapter` 只负责 Codex 场景下的 CLI、setup 和 deploy。

## Package Layout

```text
src/
  common/
    logging.py            # 统一日志与 trace id
    runtime_paths.py      # 公共运行时路径
  providers/
    catalog.py            # Preset / ModelEntry / 预设加载
    litellm_client.py     # 统一 LiteLLM 请求入口 + 响应翻译（Chat→Responses，委托 LiteLLM 内置转换器）
    presets/              # 内置 provider 预设 YAML
  protocols/
    responses_chat.py     # Responses API → Chat Completions 请求翻译（含 DeepSeek thinking 定制）
    codex_model_catalog.py# Codex 模型目录输出
  entrypoints/
    responses_proxy.py    # HTTP 入口和请求生命周期
  codex_adapter/
    cli.py                # Codex adapter CLI
    codex_setup.py        # Codex 快速配置
    deploy/               # 安装/部署/后台服务
```

## Dependency Rules

- `common` 只能依赖 Python 标准库或通用第三方库，不能反向依赖 adapter / provider。
- `providers` 可以依赖 `common`，负责“如何找到模型、如何通过 LiteLLM 调上游”。
- `protocols` 可以依赖 `common` 和 `providers` 的类型定义，负责纯协议 / wire format 转换。
- `entrypoints` 负责编排请求生命周期、日志上下文、错误包装和 HTTP 输出，不直接硬编码 provider 细节。
- `codex_adapter` 只放 Codex 特有的 CLI、setup、deploy 逻辑，不做共享基础设施的唯一实现。

## Single Source Of Truth

- provider 预设结构与加载入口：`providers.catalog`
- 统一日志与 trace id：`common.logging`
- 上游 LiteLLM 调用：`providers.litellm_client`
- Responses→Chat 请求翻译（含 DeepSeek thinking 定制）：`protocols.responses_chat`
- Chat→Responses 响应翻译（委托 LiteLLM 内置转换器）：`providers.litellm_client`
- Codex 模型目录：`protocols.codex_model_catalog`

## Migration Notes

- 新代码应优先引用顶层包：`common`、`providers`、`protocols`、`entrypoints`。
- 如果未来新增其他 adapter，应放在与 `codex_adapter` 并列的位置，并复用上面的顶层能力，而不是继续向 `codex_adapter` 内部塞公共模块。
