"""Tests for the Kimi K2.5 tool call fixer."""

import json

from main import (
    ToolCallAccumulator,
    _contains_tool_tokens,
    _parse_tool_calls_from_text,
    fix_non_streaming_response,
)


# ---------------------------------------------------------------------------
# ToolCallAccumulator tests
# ---------------------------------------------------------------------------


class TestToolCallAccumulator:
    def test_no_tool_tokens(self):
        acc = ToolCallAccumulator()
        result = acc.feed("Hello, this is normal text")
        assert result == "Hello, this is normal text"
        assert len(acc.tool_calls) == 0
        assert not acc.in_section

    def test_single_tool_call(self):
        text = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>'
            '{"path": "/tmp/test.txt"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        acc = ToolCallAccumulator()
        clean = acc.feed(text)
        assert clean == ""
        assert len(acc.tool_calls) == 1
        tc = acc.tool_calls[0]
        assert tc.function_name == "read_file"
        assert json.loads(tc.arguments) == {"path": "/tmp/test.txt"}
        assert tc.call_id.startswith("call_")
        assert acc.finished

    def test_multiple_tool_calls(self):
        text = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.read:1<|tool_call_argument_begin|>'
            '{"file": "a.txt"}'
            '<|tool_call_end|>'
            '<|tool_call_begin|>functions.write:2<|tool_call_argument_begin|>'
            '{"file": "b.txt", "content": "hello"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        acc = ToolCallAccumulator()
        clean = acc.feed(text)
        assert clean == ""
        assert len(acc.tool_calls) == 2
        assert acc.tool_calls[0].function_name == "read"
        assert acc.tool_calls[1].function_name == "write"
        assert json.loads(acc.tool_calls[1].arguments) == {
            "file": "b.txt",
            "content": "hello",
        }

    def test_text_before_tool_section(self):
        text = (
            'Let me think about this...'
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>'
            '{"query": "test"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        acc = ToolCallAccumulator()
        clean = acc.feed(text)
        assert clean == "Let me think about this..."
        assert len(acc.tool_calls) == 1
        assert acc.tool_calls[0].function_name == "search"

    def test_text_with_spaces_around_tokens(self):
        text = (
            '<|tool_calls_section_begin|> '
            '<|tool_call_begin|> functions.test:0 <|tool_call_argument_begin|> '
            '{"key": "value"} '
            '<|tool_call_end|> '
            '<|tool_calls_section_end|>'
        )
        acc = ToolCallAccumulator()
        clean = acc.feed(text)
        assert clean == ""
        assert len(acc.tool_calls) == 1
        assert acc.tool_calls[0].function_name == "test"
        assert json.loads(acc.tool_calls[0].arguments) == {"key": "value"}

    def test_function_name_without_prefix(self):
        """Tool call ID without 'functions.' prefix."""
        text = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>my_tool:5<|tool_call_argument_begin|>'
            '{"x": 1}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        acc = ToolCallAccumulator()
        clean = acc.feed(text)
        assert acc.tool_calls[0].function_name == "my_tool"

    def test_function_name_without_colon_suffix(self):
        """Tool call ID without ':N' suffix."""
        text = (
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.bash<|tool_call_argument_begin|>'
            '{"cmd": "ls"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        acc = ToolCallAccumulator()
        clean = acc.feed(text)
        assert acc.tool_calls[0].function_name == "bash"

    def test_chunked_feeding(self):
        """Simulate streaming by feeding one character at a time."""
        full_text = (
            'Reasoning here. '
            '<|tool_calls_section_begin|>'
            '<|tool_call_begin|>functions.process:28<|tool_call_argument_begin|>'
            '{"action": "log", "sessionId": "fresh-crest"}'
            '<|tool_call_end|>'
            '<|tool_calls_section_end|>'
        )
        acc = ToolCallAccumulator()
        all_clean = ""
        for ch in full_text:
            all_clean += acc.feed(ch)
        # Flush remaining buffer
        all_clean += acc.buffer
        assert all_clean.strip() == "Reasoning here."
        assert len(acc.tool_calls) == 1
        assert acc.tool_calls[0].function_name == "process"
        assert json.loads(acc.tool_calls[0].arguments) == {
            "action": "log",
            "sessionId": "fresh-crest",
        }

    def test_chunked_feeding_splits_across_token(self):
        """Token boundary falls in the middle of a token marker."""
        acc = ToolCallAccumulator()
        # Feed the text in arbitrary chunks that split tokens
        chunks = [
            "Hello <|tool_calls",  # partial token
            "_section_begin|><|tool_call_be",
            "gin|>functions.foo:1<|tool_call_ar",
            'gument_begin|>{"a": 1}<|tool_call_end',
            "|><|tool_calls_section_end|>",
        ]
        all_clean = ""
        for chunk in chunks:
            all_clean += acc.feed(chunk)
        all_clean += acc.buffer
        assert all_clean.strip() == "Hello"
        assert len(acc.tool_calls) == 1
        assert acc.tool_calls[0].function_name == "foo"

    def test_real_world_example(self):
        """The exact example from the user's bug report."""
        text = (
            " L'output è stato troncato, ma vedo che è tornato qualcosa. "
            "Vediamo il risultato completo e verifichiamo se contiene le informazioni necessarie. "
            "<|tool_calls_section_begin|> "
            "<|tool_call_begin|> functions.process:28 "
            '<|tool_call_argument_begin|> {"action": "log", "sessionId": "fresh-crest"} '
            "<|tool_call_end|> "
            "<|tool_calls_section_end|>"
        )
        acc = ToolCallAccumulator()
        clean = acc.feed(text)
        clean += acc.buffer
        assert "tool_calls" not in clean
        assert "<|" not in clean
        assert len(acc.tool_calls) == 1
        tc = acc.tool_calls[0]
        assert tc.function_name == "process"
        assert json.loads(tc.arguments) == {
            "action": "log",
            "sessionId": "fresh-crest",
        }


# ---------------------------------------------------------------------------
# _parse_tool_calls_from_text tests
# ---------------------------------------------------------------------------


class TestParseToolCallsFromText:
    def test_basic(self):
        text = (
            "Some reasoning "
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>"
            '{"x": 1}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        clean, calls = _parse_tool_calls_from_text(text)
        assert "tool_calls" not in clean
        assert clean.strip() == "Some reasoning"
        assert len(calls) == 1

    def test_no_tool_tokens(self):
        text = "Just regular text"
        clean, calls = _parse_tool_calls_from_text(text)
        assert clean == "Just regular text"
        assert len(calls) == 0


# ---------------------------------------------------------------------------
# fix_non_streaming_response tests
# ---------------------------------------------------------------------------


class TestFixNonStreamingResponse:
    def test_fixes_reasoning_content(self):
        body = {
            "id": "chatcmpl-test",
            "model": "moonshotai/Kimi-K2.5",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": None,
                        "reasoning_content": (
                            "Let me check. "
                            "<|tool_calls_section_begin|> "
                            "<|tool_call_begin|> functions.process:28 "
                            '<|tool_call_argument_begin|> {"action": "log", "sessionId": "fresh-crest"} '
                            "<|tool_call_end|> "
                            "<|tool_calls_section_end|>"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        result = fix_non_streaming_response(body)
        msg = result["choices"][0]["message"]

        assert "<|" not in msg["reasoning_content"]
        assert msg["reasoning_content"].strip() == "Let me check."
        assert msg["tool_calls"] is not None
        assert len(msg["tool_calls"]) == 1

        tc = msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "process"
        assert json.loads(tc["function"]["arguments"]) == {
            "action": "log",
            "sessionId": "fresh-crest",
        }
        assert tc["id"].startswith("call_")

        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_fixes_content_field(self):
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<|tool_calls_section_begin|>"
                            "<|tool_call_begin|>functions.bash:0<|tool_call_argument_begin|>"
                            '{"cmd": "ls"}'
                            "<|tool_call_end|>"
                            "<|tool_calls_section_end|>"
                        ),
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        result = fix_non_streaming_response(body)
        msg = result["choices"][0]["message"]

        # Content was only tool tokens, so it should be None
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "bash"

    def test_preserves_existing_tool_calls(self):
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<|tool_calls_section_begin|>"
                            "<|tool_call_begin|>functions.new_tool:0<|tool_call_argument_begin|>"
                            '{"a": 1}'
                            "<|tool_call_end|>"
                            "<|tool_calls_section_end|>"
                        ),
                        "tool_calls": [
                            {
                                "id": "call_existing",
                                "type": "function",
                                "function": {
                                    "name": "existing_tool",
                                    "arguments": '{"b": 2}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        result = fix_non_streaming_response(body)
        msg = result["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["function"]["name"] == "existing_tool"
        assert msg["tool_calls"][1]["function"]["name"] == "new_tool"

    def test_no_fix_needed(self):
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        result = fix_non_streaming_response(body)
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["message"]["tool_calls"] is None
        assert result["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# _contains_tool_tokens tests
# ---------------------------------------------------------------------------


class TestContainsToolTokens:
    def test_positive(self):
        assert _contains_tool_tokens("<|tool_calls_section_begin|>")
        assert _contains_tool_tokens("text <|tool_call_begin|> more")

    def test_negative(self):
        assert not _contains_tool_tokens("normal text")
        assert not _contains_tool_tokens("")
        assert not _contains_tool_tokens("<|other_token|>")
