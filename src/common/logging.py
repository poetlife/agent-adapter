"""Shared Loguru-based application logging and request trace helpers."""

from __future__ import annotations

import json
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Mapping

from loguru import logger as _base_logger

from common.runtime_paths import get_app_config_dir

_TRACE_ID_VAR: ContextVar[str] = ContextVar("codex_adapter_trace_id", default="-")

_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
    "trace_id={extra[trace_id]} | {message}{extra[payload_text]}\n"
)

_BODY_TRACE_ID_PATHS = (
    ("trace_id",),
    ("traceId",),
    ("request_id",),
    ("requestId",),
    ("metadata", "trace_id"),
    ("metadata", "traceId"),
    ("metadata", "request_id"),
    ("metadata", "requestId"),
)

_HEADER_TRACE_ID_KEYS = (
    "x-trace-id",
    "x-request-id",
    "trace-id",
    "request-id",
)


def _inject_defaults(record: dict[str, Any]) -> None:
    extra = record["extra"]
    extra.setdefault("trace_id", _TRACE_ID_VAR.get())
    extra.setdefault("payload_text", "")


logger = _base_logger.patch(_inject_defaults)


def init_logging(debug: bool = False) -> Path:
    """Configure Loguru sinks and return the file-backed debug log path."""
    log_dir = get_app_config_dir("codex-adapter")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "debug.log"

    _base_logger.remove()
    _base_logger.add(
        log_path,
        level="DEBUG",
        encoding="utf-8",
        rotation="10 MB",
        retention=2,
        format=_LOG_FORMAT,
        backtrace=debug,
        diagnose=debug,
    )
    if debug:
        _base_logger.add(
            sys.stderr,
            level="DEBUG",
            colorize=True,
            format=_LOG_FORMAT,
            backtrace=True,
            diagnose=True,
        )

    logger.info("=== codex-adapter log started ===")
    return log_path


def get_current_trace_id() -> str:
    """Return the request trace id currently bound to this execution context."""
    return _TRACE_ID_VAR.get()


@contextmanager
def request_log_context(trace_id: str | None = None) -> Iterator[str]:
    """Bind a request trace id for all logs emitted in this context."""
    resolved_trace_id = trace_id or f"req_{uuid.uuid4().hex[:12]}"
    token = _TRACE_ID_VAR.set(resolved_trace_id)
    try:
        yield resolved_trace_id
    finally:
        _TRACE_ID_VAR.reset(token)


def resolve_trace_id(
    body: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Extract a trace id from request body or headers, else generate one."""
    if body:
        for path in _BODY_TRACE_ID_PATHS:
            value = _lookup_nested(body, path)
            if isinstance(value, str) and value.strip():
                return value.strip(), "body:" + ".".join(path)

    if headers:
        lowered_headers = {key.lower(): value for key, value in headers.items()}
        for key in _HEADER_TRACE_ID_KEYS:
            value = lowered_headers.get(key)
            if value and value.strip():
                return value.strip(), f"header:{key}"

        traceparent = lowered_headers.get("traceparent")
        trace_id = _trace_id_from_traceparent(traceparent)
        if trace_id:
            return trace_id, "header:traceparent"

    return f"req_{uuid.uuid4().hex[:12]}", "generated"


def log_debug(message: str, data: Any = None) -> None:
    """Emit a debug log with optional pretty-printed payload."""
    _log("DEBUG", message, data)


def log_info(message: str, data: Any = None) -> None:
    """Emit an info log with optional pretty-printed payload."""
    _log("INFO", message, data)


def log_warning(message: str, data: Any = None) -> None:
    """Emit a warning log with optional pretty-printed payload."""
    _log("WARNING", message, data)


def log_error(message: str, data: Any = None) -> None:
    """Emit an error log with optional pretty-printed payload."""
    _log("ERROR", message, data)


def log_exception(message: str, data: Any = None) -> None:
    """Emit an exception log with traceback and optional pretty payload."""
    logger.bind(
        trace_id=get_current_trace_id(),
        payload_text=_serialize_payload(data),
    ).exception(message)


def _log(level: str, message: str, data: Any = None) -> None:
    logger.bind(
        trace_id=get_current_trace_id(),
        payload_text=_serialize_payload(data),
    ).log(level, message)


def _serialize_payload(data: Any) -> str:
    if data is None:
        return ""
    try:
        serialized = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        serialized = repr(data)
    return "\n" + serialized


def _lookup_nested(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _trace_id_from_traceparent(traceparent: str | None) -> str | None:
    if not traceparent:
        return None
    parts = traceparent.strip().split("-")
    if len(parts) != 4:
        return None
    trace_id = parts[1]
    if len(trace_id) == 32 and all(ch in "0123456789abcdefABCDEF" for ch in trace_id):
        return trace_id.lower()
    return None
