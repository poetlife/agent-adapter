"""Shared cross-adapter utilities."""

from common.logging import (
    get_current_trace_id,
    init_logging,
    log_debug,
    log_error,
    log_exception,
    log_info,
    log_warning,
    logger,
    request_log_context,
    resolve_trace_id,
)
from common.runtime_paths import get_app_config_dir
