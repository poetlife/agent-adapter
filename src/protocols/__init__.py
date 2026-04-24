"""Protocol translation and wire-shape helpers."""

from protocols.codex_model_catalog import generate_codex_model_catalog
from protocols.responses_chat import (
    ModelConfig,
    chat_response_to_responses,
    responses_request_to_chat,
    translate_stream,
)
