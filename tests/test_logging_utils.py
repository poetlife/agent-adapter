"""Tests for shared logging helpers."""

from common.logging import log_debug, logger, request_log_context, resolve_trace_id


def test_resolve_trace_id_from_body_metadata():
    trace_id, source = resolve_trace_id({"metadata": {"traceId": "trace-body-123"}})
    assert trace_id == "trace-body-123"
    assert source == "body:metadata.traceId"


def test_resolve_trace_id_from_traceparent_header():
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    trace_id, source = resolve_trace_id(headers={"traceparent": traceparent})
    assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert source == "header:traceparent"


def test_request_log_context_includes_trace_id_in_log_output():
    lines: list[str] = []
    sink_id = logger.add(
        lines.append,
        format="{extra[trace_id]}|{message}{extra[payload_text]}",
    )

    try:
        with request_log_context("trace-log-123"):
            log_debug("hello", {"ok": True})
    finally:
        logger.remove(sink_id)

    joined = "\n".join(lines)
    assert "trace-log-123|hello" in joined
    assert '"ok": true' in joined
