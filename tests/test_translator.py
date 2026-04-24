"""Tests for the Responses API → Chat Completions translator and LiteLLM response translation."""

import json

import pytest

from protocols.responses_chat import (
    ModelConfig,
    responses_request_to_chat,
)
from providers.litellm_client import (
    transform_chat_to_responses,
    _fix_reasoning_format,
    _compute_output_text,
)


class TestResponsesRequestToChat:
    """Test converting Responses API requests to Chat Completions format."""

    def test_simple_string_input(self):
        body = {
            "model": "deepseek-chat",
            "input": "Hello, how are you?",
        }
        result = responses_request_to_chat(body)
        assert result["model"] == "deepseek-chat"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, how are you?"

    def test_instructions_become_system_message(self):
        body = {
            "model": "deepseek-chat",
            "instructions": "You are a helpful assistant.",
            "input": "Hi",
        }
        result = responses_request_to_chat(body)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."
        assert result["messages"][1]["role"] == "user"

    def test_max_output_tokens_mapped(self):
        body = {
            "model": "deepseek-chat",
            "input": "Hi",
            "max_output_tokens": 2048,
        }
        result = responses_request_to_chat(body)
        assert result["max_tokens"] == 2048
        assert "max_output_tokens" not in result

    def test_stream_flag(self):
        body = {
            "model": "deepseek-chat",
            "input": "Hi",
            "stream": True,
        }
        result = responses_request_to_chat(body)
        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}

    def test_temperature_passthrough(self):
        body = {
            "model": "deepseek-chat",
            "input": "Hi",
            "temperature": 0.7,
        }
        result = responses_request_to_chat(body)
        assert result["temperature"] == 0.7

    def test_list_input_with_messages(self):
        body = {
            "model": "deepseek-chat",
            "input": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        result = responses_request_to_chat(body)
        assert len(result["messages"]) == 3
        assert result["messages"][0]["content"] == "Hello"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][2]["content"] == "How are you?"

    def test_function_call_in_input(self):
        body = {
            "model": "deepseek-chat",
            "input": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_abc123",
                    "name": "get_weather",
                    "arguments": '{"city": "Beijing"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc123",
                    "output": "Sunny, 25°C",
                },
            ],
        }
        result = responses_request_to_chat(body)
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "user"
        # Function call → assistant message with tool_calls
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"
        # Function output → tool message
        assert result["messages"][2]["role"] == "tool"
        assert result["messages"][2]["content"] == "Sunny, 25°C"

    def test_function_tools_converted(self):
        body = {
            "model": "deepseek-chat",
            "input": "Hi",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        }
        result = responses_request_to_chat(body)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_builtin_tools_skipped(self):
        """Built-in OpenAI tools (web_search, etc.) should be dropped."""
        body = {
            "model": "deepseek-chat",
            "input": "Hi",
            "tools": [
                {"type": "web_search"},
                {"type": "code_interpreter"},
            ],
        }
        result = responses_request_to_chat(body)
        # No tools should be included since they're all built-in
        assert "tools" not in result or result["tools"] == []

    def test_content_parts_text(self):
        body = {
            "model": "deepseek-chat",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hello "},
                        {"type": "input_text", "text": "world"},
                    ],
                }
            ],
        }
        result = responses_request_to_chat(body)
        # All text parts should be joined
        assert result["messages"][0]["content"] == "Hello  world"

    def test_empty_input(self):
        body = {"model": "deepseek-chat", "input": ""}
        result = responses_request_to_chat(body)
        assert result["messages"] == []


class TestChatResponseToResponses:
    """Test converting Chat Completions responses to Responses API format (via LiteLLM)."""

    def test_simple_text_response(self):
        chat_resp = {
            "id": "chatcmpl-123",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = transform_chat_to_responses(chat_resp)
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["output_text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        msg_items = [o for o in result["output"] if o["type"] == "message"]
        assert len(msg_items) >= 1

    def test_tool_call_response(self):
        chat_resp = {
            "id": "chatcmpl-456",
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Beijing"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        result = transform_chat_to_responses(chat_resp)
        # LiteLLM maps tool_calls finish_reason to "completed" (standard
        # Responses API behaviour: tool calls are output items in a completed response)
        assert result["status"] == "completed"
        # Should have a function_call output item
        fc_items = [o for o in result["output"] if o["type"] == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "get_weather"
        assert fc_items[0]["call_id"] == "call_abc"

    def test_preserves_original_model(self):
        chat_resp = {
            "id": "chatcmpl-789",
            "model": "deepseek/deepseek-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = transform_chat_to_responses(chat_resp, original_model="deepseek-chat")
        # When original_model is specified, it should be used
        assert result["model"] == "deepseek-chat"

    def test_empty_choices(self):
        chat_resp = {"id": "chatcmpl-0", "model": "x", "choices": [], "usage": {}}
        result = transform_chat_to_responses(chat_resp)
        assert result["output_text"] == ""


# ---------------------------------------------------------------------------
# Thinking mode tests (DeepSeek V4)
# ---------------------------------------------------------------------------

THINKING_MODEL = ModelConfig(
    supports_thinking=True,
    default_thinking="enabled",
    reasoning_effort="high",
)

NON_THINKING_MODEL = ModelConfig(
    supports_thinking=False,
    default_thinking="disabled",
    reasoning_effort="high",
)


class TestThinkingModeRequest:
    """Test thinking mode translation in requests."""

    def test_default_thinking_enabled(self):
        """Model with default_thinking=enabled should add thinking params."""
        body = {"model": "deepseek-v4-flash", "input": "Hi"}
        result = responses_request_to_chat(body, model_config=THINKING_MODEL)
        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "high"
        # temperature should NOT be present when thinking is enabled
        assert "temperature" not in result

    def test_reasoning_effort_high(self):
        """Codex reasoning.effort=high → DeepSeek reasoning_effort=high."""
        body = {
            "model": "deepseek-v4-flash",
            "input": "Hi",
            "reasoning": {"effort": "high"},
        }
        result = responses_request_to_chat(body, model_config=THINKING_MODEL)
        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "high"

    def test_reasoning_effort_max(self):
        """Codex reasoning.effort=max → DeepSeek reasoning_effort=max."""
        body = {
            "model": "deepseek-v4-pro",
            "input": "Complex problem",
            "reasoning": {"effort": "max"},
        }
        result = responses_request_to_chat(body, model_config=THINKING_MODEL)
        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "max"

    def test_reasoning_effort_low_maps_to_high(self):
        """Codex reasoning.effort=low → DeepSeek reasoning_effort=high (lowest available)."""
        body = {
            "model": "deepseek-v4-flash",
            "input": "Hi",
            "reasoning": {"effort": "low"},
        }
        result = responses_request_to_chat(body, model_config=THINKING_MODEL)
        assert result["thinking"] == {"type": "enabled"}
        assert result["reasoning_effort"] == "high"

    def test_reasoning_effort_medium_maps_to_high(self):
        """Codex reasoning.effort=medium → DeepSeek reasoning_effort=high."""
        body = {
            "model": "deepseek-v4-flash",
            "input": "Hi",
            "reasoning": {"effort": "medium"},
        }
        result = responses_request_to_chat(body, model_config=THINKING_MODEL)
        assert result["reasoning_effort"] == "high"

    def test_thinking_drops_temperature(self):
        """When thinking is enabled, temperature/top_p should NOT be sent."""
        body = {
            "model": "deepseek-v4-flash",
            "input": "Hi",
            "temperature": 0.7,
            "top_p": 0.9,
            "reasoning": {"effort": "high"},
        }
        result = responses_request_to_chat(body, model_config=THINKING_MODEL)
        assert "temperature" not in result
        assert "top_p" not in result

    def test_non_thinking_model_ignores_reasoning(self):
        """Non-thinking model should pass temperature through normally."""
        body = {
            "model": "some-model",
            "input": "Hi",
            "temperature": 0.7,
        }
        result = responses_request_to_chat(body, model_config=NON_THINKING_MODEL)
        assert result["temperature"] == 0.7
        assert "thinking" not in result
        assert "reasoning_effort" not in result

    def test_thinking_disabled_allows_temperature(self):
        """Thinking model without reasoning param and default disabled → temperature ok."""
        model_cfg = ModelConfig(
            supports_thinking=True,
            default_thinking="disabled",
            reasoning_effort="high",
        )
        body = {
            "model": "deepseek-v4-flash",
            "input": "Hi",
            "temperature": 0.5,
        }
        result = responses_request_to_chat(body, model_config=model_cfg)
        assert result["thinking"] == {"type": "disabled"}
        assert result["temperature"] == 0.5


class TestThinkingModeResponse:
    """Test reasoning_content mapping in responses (via LiteLLM)."""

    def test_reasoning_content_mapped(self):
        """DeepSeek reasoning_content → Responses API reasoning output item."""
        chat_resp = {
            "id": "chatcmpl-123",
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "让我想想这个问题...\n首先需要分析...",
                    "content": "答案是42。",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        }
        result = transform_chat_to_responses(chat_resp)

        # Should have reasoning item BEFORE message item
        types = [o["type"] for o in result["output"]]
        assert "reasoning" in types
        assert "message" in types
        reasoning_idx = types.index("reasoning")
        message_idx = types.index("message")
        assert reasoning_idx < message_idx

        reasoning_item = result["output"][reasoning_idx]
        assert reasoning_item["summary"][0]["text"] == "让我想想这个问题...\n首先需要分析..."
        assert result["output_text"] == "答案是42。"

    def test_no_reasoning_content(self):
        """Without reasoning_content, no reasoning output item."""
        chat_resp = {
            "id": "chatcmpl-456",
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        result = transform_chat_to_responses(chat_resp)
        reasoning_items = [o for o in result["output"] if o["type"] == "reasoning"]
        assert len(reasoning_items) == 0
        msg_items = [o for o in result["output"] if o["type"] == "message"]
        assert len(msg_items) >= 1

    def test_reasoning_with_tool_calls(self):
        """Reasoning + tool call should produce reasoning + function_call items."""
        chat_resp = {
            "id": "chatcmpl-789",
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "需要查询天气信息...",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Beijing"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
        }
        result = transform_chat_to_responses(chat_resp)
        types = [o["type"] for o in result["output"]]
        assert "reasoning" in types
        assert "function_call" in types
        assert result["status"] == "completed"


class TestMultiTurnReasoningContent:
    """Test multi-turn conversations with reasoning_content (the root cause scenario)."""

    def test_reasoning_content_attached_to_function_call(self):
        """Reasoning item before function_call should merge reasoning_content into assistant msg."""
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "message", "role": "user", "content": "What's the weather?"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I need to check weather"}]},
                {"type": "function_call", "call_id": "call_001", "name": "get_weather", "arguments": '{"city":"Beijing"}'},
                {"type": "function_call_output", "call_id": "call_001", "output": '{"temp": 25}'},
            ],
        }
        config = ModelConfig(supports_thinking=True, default_thinking="enabled")
        result = responses_request_to_chat(body, model_config=config)
        msgs = result["messages"]

        # The assistant message (function_call) should have reasoning_content
        assistant_msg = [m for m in msgs if m["role"] == "assistant"][0]
        assert "reasoning_content" in assistant_msg
        assert assistant_msg["reasoning_content"] == "I need to check weather"
        assert "tool_calls" in assistant_msg

    def test_reasoning_content_present_when_thinking_enabled(self):
        """When thinking is enabled and history has reasoning_content, both should be in request."""
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
                {"type": "message", "role": "assistant", "content": "Hi there!"},
                {"type": "message", "role": "user", "content": "Follow up"},
            ],
        }
        config = ModelConfig(supports_thinking=True, default_thinking="enabled")
        result = responses_request_to_chat(body, model_config=config)

        # Thinking should be enabled
        assert result["thinking"]["type"] == "enabled"
        # Assistant message should carry reasoning_content
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["reasoning_content"] == "thinking..."

    def test_tool_continuation_reuses_previous_reasoning_when_missing(self):
        """Assistant continuation after tool output should inherit prior reasoning in thinking mode."""
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "message", "role": "user", "content": "Wait for the command"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Need to wait longer"}]},
                {"type": "function_call", "call_id": "call_001", "name": "write_stdin", "arguments": '{"session_id":1,"yield_time_ms":30000}'},
                {"type": "function_call_output", "call_id": "call_001", "output": "still running"},
                {"type": "function_call", "call_id": "call_002", "name": "write_stdin", "arguments": '{"session_id":1,"yield_time_ms":60000}'},
            ],
        }
        config = ModelConfig(supports_thinking=True, default_thinking="enabled")
        result = responses_request_to_chat(body, model_config=config)

        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0]["reasoning_content"] == "Need to wait longer"
        assert assistant_msgs[1]["reasoning_content"] == "Need to wait longer"

    def test_reasoning_not_reused_across_user_turn(self):
        """Previous assistant reasoning should not leak into a fresh user-triggered turn."""
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "message", "role": "user", "content": "First task"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "old reasoning"}]},
                {"type": "message", "role": "assistant", "content": "Done"},
                {"type": "message", "role": "user", "content": "New task"},
                {"type": "message", "role": "assistant", "content": "Fresh answer"},
            ],
        }
        config = ModelConfig(supports_thinking=True, default_thinking="enabled")
        result = responses_request_to_chat(body, model_config=config)

        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0]["reasoning_content"] == "old reasoning"
        assert "reasoning_content" not in assistant_msgs[1]

    def test_tool_continuation_does_not_reuse_reasoning_when_thinking_disabled(self):
        """Reasoning carry-forward should only happen when this request actually uses thinking."""
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "message", "role": "user", "content": "Wait"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "old reasoning"}]},
                {"type": "function_call", "call_id": "call_001", "name": "write_stdin", "arguments": '{"session_id":1,"yield_time_ms":30000}'},
                {"type": "function_call_output", "call_id": "call_001", "output": "still running"},
                {"type": "function_call", "call_id": "call_002", "name": "write_stdin", "arguments": '{"session_id":1,"yield_time_ms":60000}'},
            ],
        }
        config = ModelConfig(supports_thinking=True, default_thinking="disabled")
        result = responses_request_to_chat(body, model_config=config)

        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert result["thinking"]["type"] == "disabled"
        assert assistant_msgs[0]["reasoning_content"] == "old reasoning"
        assert "reasoning_content" not in assistant_msgs[1]


class TestConsecutiveAssistantMerge:
    """Test merging of consecutive assistant messages (parallel tool calls)."""

    def test_parallel_tool_calls_merged(self):
        """Multiple function_call items should merge into one assistant message."""
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "message", "role": "user", "content": "Read both files"},
                {"type": "function_call", "call_id": "call_001", "name": "read_file", "arguments": '{"path":"a.py"}'},
                {"type": "function_call", "call_id": "call_002", "name": "read_file", "arguments": '{"path":"b.py"}'},
                {"type": "function_call_output", "call_id": "call_001", "output": "content a"},
                {"type": "function_call_output", "call_id": "call_002", "output": "content b"},
            ],
        }
        config = ModelConfig(supports_thinking=True, default_thinking="enabled")
        result = responses_request_to_chat(body, model_config=config)

        # Should have exactly one assistant message
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        # With both tool_calls combined
        assert len(assistant_msgs[0]["tool_calls"]) == 2
        names = [tc["function"]["name"] for tc in assistant_msgs[0]["tool_calls"]]
        assert names == ["read_file", "read_file"]

    def test_three_parallel_tool_calls_merged(self):
        """Three consecutive function_calls merge into one assistant message with 3 tool_calls."""
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "message", "role": "user", "content": "Do three things"},
                {"type": "function_call", "call_id": "c1", "name": "tool1", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "tool2", "arguments": "{}"},
                {"type": "function_call", "call_id": "c3", "name": "tool3", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "r1"},
                {"type": "function_call_output", "call_id": "c2", "output": "r2"},
                {"type": "function_call_output", "call_id": "c3", "output": "r3"},
            ],
        }
        config = ModelConfig(supports_thinking=True, default_thinking="enabled")
        result = responses_request_to_chat(body, model_config=config)

        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert len(assistant_msgs[0]["tool_calls"]) == 3

    def test_assistant_text_plus_tool_call_merged(self):
        """Assistant text message followed by function_call should merge."""
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "message", "role": "user", "content": "Help me"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check"}]},
                {"type": "function_call", "call_id": "call_001", "name": "search", "arguments": '{"q":"test"}'},
                {"type": "function_call_output", "call_id": "call_001", "output": "found it"},
            ],
        }
        config = ModelConfig(supports_thinking=True, default_thinking="enabled")
        result = responses_request_to_chat(body, model_config=config)

        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        # Content preserved
        assert assistant_msgs[0]["content"] == "Let me check"
        # Tool call merged in
        assert len(assistant_msgs[0]["tool_calls"]) == 1
        # Reasoning preserved
        assert assistant_msgs[0].get("reasoning_content") == "thinking..."

    def test_no_consecutive_assistant_in_output(self):
        """No matter the input pattern, output must never have consecutive assistant messages."""
        body = {
            "model": "deepseek-v4-flash",
            "input": [
                {"type": "message", "role": "user", "content": "Complex task"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "step 1"}]},
                {"type": "function_call", "call_id": "c1", "name": "t1", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "t2", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "r1"},
                {"type": "function_call_output", "call_id": "c2", "output": "r2"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "step 2"}]},
                {"type": "function_call", "call_id": "c3", "name": "t3", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c3", "output": "r3"},
                {"type": "message", "role": "user", "content": "Continue"},
            ],
        }
        config = ModelConfig(supports_thinking=True, default_thinking="enabled")
        result = responses_request_to_chat(body, model_config=config)

        # Verify no consecutive assistant messages
        msgs = result["messages"]
        for i in range(1, len(msgs)):
            if msgs[i]["role"] == "assistant":
                assert msgs[i - 1]["role"] != "assistant", (
                    f"Consecutive assistant messages at [{i-1}] and [{i}]"
                )


class TestLiteLLMResponseHelpers:
    """Test the LiteLLM response post-processing helpers."""

    def test_fix_reasoning_format_converts_content_to_summary(self):
        """LiteLLM reasoning content[output_text] → Codex summary[summary_text]."""
        output = [{
            "type": "reasoning",
            "id": "rs_123",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "thinking...", "annotations": []}],
        }]
        fixed = _fix_reasoning_format(output)
        assert "content" not in fixed[0]
        assert "status" not in fixed[0]
        assert "role" not in fixed[0]
        assert fixed[0]["summary"] == [{"type": "summary_text", "text": "thinking..."}]

    def test_fix_reasoning_format_preserves_non_reasoning(self):
        output = [
            {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
            {"type": "function_call", "name": "foo", "arguments": "{}"},
        ]
        fixed = _fix_reasoning_format(output)
        assert fixed == output

    def test_fix_reasoning_format_already_summary(self):
        """Items already in summary format should pass through."""
        output = [{
            "type": "reasoning",
            "id": "rs_456",
            "summary": [{"type": "summary_text", "text": "already correct"}],
        }]
        fixed = _fix_reasoning_format(output)
        assert fixed[0]["summary"][0]["text"] == "already correct"

    def test_compute_output_text(self):
        output = [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "Hello "}]},
            {"type": "message", "content": [{"type": "output_text", "text": "world"}]},
        ]
        assert _compute_output_text(output) == "Hello world"

    def test_compute_output_text_empty(self):
        assert _compute_output_text([]) == ""
        assert _compute_output_text([{"type": "function_call"}]) == ""

    def test_transform_output_text_in_result(self):
        """output_text must be a real key in the dict (not just a property)."""
        chat_resp = {
            "id": "chatcmpl-ot",
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = transform_chat_to_responses(chat_resp)
        assert "output_text" in result
        assert result["output_text"] == "Hi"

    def test_transform_object_field_is_response(self):
        """object field must be 'response', not 'chat.completion'."""
        chat_resp = {
            "id": "chatcmpl-obj",
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = transform_chat_to_responses(chat_resp)
        assert result["object"] == "response"

    def test_transform_strips_excess_fields(self):
        """Codex CLI doesn't expect certain fields in the response."""
        chat_resp = {
            "id": "chatcmpl-strip",
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = transform_chat_to_responses(chat_resp)
        for key in ("parallel_tool_calls", "tool_choice", "tools", "text", "temperature"):
            assert key not in result, f"Unexpected key {key!r} in result"
