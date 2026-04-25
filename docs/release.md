# Release Guide

## 目标

版本发布时只做一件事：推送符合 `vX.Y.Z` 格式的 tag。其余事情由 GitHub Actions 自动完成，确保用户可以直接从 Release 安装，不需要再从源码构建。

## 自动化流水线

触发条件：推送 `v*.*.*` tag，例如 `v0.1.2`。

`Release` workflow 会按顺序执行：

1. 安装锁定依赖（`uv sync --locked`）
2. 校验 `GITHUB_REF_NAME` 与 `pyproject.toml` 的 `project.version` 完全一致
3. 运行完整测试（`uv run pytest -v`）
4. 构建发布产物（`uv build`）
5. 校验 `dist/` 中恰好生成一个 `wheel` 和一个 `sdist`
6. 使用刚构建出的 `wheel` 执行一次安装冒烟验证：
   `uv tool run --from <wheel> codex-adapter --version`
7. 创建 GitHub Release，并上传 `dist/*`

## 发布步骤

```bash
# 1. 修改版本号
$EDITOR pyproject.toml

# 2. 提交主分支
git push origin main

# 3. 创建并推送 tag
git tag vX.Y.Z
git push origin vX.Y.Z
```

如果 workflow 成功，Release 页面会出现两个可下载产物：

- `codex_adapter-X.Y.Z-py3-none-any.whl`
- `codex_adapter-X.Y.Z.tar.gz`

## 用户安装方式

推荐直接安装 Release 里的 `wheel`：

```bash
uv tool install --from "https://github.com/poetlife/agent-adapter/releases/download/vX.Y.Z/codex_adapter-X.Y.Z-py3-none-any.whl" codex-adapter
```

或者：

```bash
pipx install "https://github.com/poetlife/agent-adapter/releases/download/vX.Y.Z/codex_adapter-X.Y.Z-py3-none-any.whl"
```

已有 Python 环境时，也可以：

```bash
pip install "https://github.com/poetlife/agent-adapter/releases/download/vX.Y.Z/codex_adapter-X.Y.Z-py3-none-any.whl"
```

## 常见失败原因

- tag 不是 `vX.Y.Z` 形式，workflow 不会触发
- tag 版本和 `pyproject.toml` 里的版本不一致，workflow 会直接失败
- 测试失败或构建失败，GitHub Release 不会创建
