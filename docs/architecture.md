# Architecture

## Goals

当前仓库不再把所有能力都压在 `codex_adapter` 包下面，而是按“共享能力 / provider 能力 / 协议转换 / 入口 / adapter 自身”分层。这样后续新增其他 adapter 时，可以直接复用上游访问、日志、协议转换和预设目录，而不是再复制一套。

核心链路：

`entrypoints` -> `protocols` -> `providers.litellm_client` -> `DeepSeek / Hunyuan / other providers`

`codex_adapter` 只负责 Codex 场景下的 CLI、setup、deploy 和兼容导出。

## Package Layout

```text
src/
  common/
    logging.py            # 统一日志与 trace id
    runtime_paths.py      # 公共运行时路径
  providers/
    catalog.py            # Preset / ModelEntry / 预设加载
    litellm_client.py     # 统一 LiteLLM 请求入口
    presets/              # 内置 provider 预设 YAML
  protocols/
    responses_chat.py     # Responses API <-> Chat Completions 转换
    codex_model_catalog.py# Codex 模型目录输出
  entrypoints/
    responses_proxy.py    # HTTP 入口和请求生命周期
  codex_adapter/
    cli.py                # Codex adapter CLI
    codex_setup.py        # Codex 快速配置
    deploy/               # 安装/部署/后台服务
    config.py             # 兼容 re-export
    proxy.py              # 兼容 re-export
    translator.py         # 兼容 re-export
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
- Responses/Chat 协议转换：`protocols.responses_chat`
- Codex 模型目录：`protocols.codex_model_catalog`

## Migration Notes

- 旧路径 `codex_adapter.config` / `logging_utils` / `litellm_client` / `translator` / `proxy` 目前保留为兼容 re-export。
- 新代码应优先引用顶层包：`common`、`providers`、`protocols`、`entrypoints`。
- 如果未来新增其他 adapter，应放在与 `codex_adapter` 并列的位置，并复用上面的顶层能力，而不是继续向 `codex_adapter` 内部塞公共模块。
