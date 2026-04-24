"""Tests for the Responses API → Chat Completions translator."""

import json

import pytest

from codex_adapter.translator import (
    chat_response_to_responses,
    responses_request_to_chat,
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
    """Test converting Chat Completions responses to Responses API format."""

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
        result = chat_response_to_responses(chat_resp)
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["output_text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"

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
        result = chat_response_to_responses(chat_resp)
        assert result["status"] == "incomplete"  # tool_calls means not done yet
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
        result = chat_response_to_responses(chat_resp, original_model="deepseek-chat")
        # Should use the chat response's model if available
        assert result["model"] == "deepseek/deepseek-chat"

    def test_empty_choices(self):
        chat_resp = {"id": "chatcmpl-0", "model": "x", "choices": [], "usage": {}}
        result = chat_response_to_responses(chat_resp)
        assert result["output"] == []
        assert result["output_text"] == ""
