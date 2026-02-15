"""
Kimi K2.5 Tool Call Fixer - OpenAI-compatible proxy middleware.

Intercepts raw tool call tokens (e.g. <|tool_calls_section_begin|>) that Kimi K2.5
sometimes emits inside reasoning_content or content, and converts them into proper
OpenAI-format tool_calls in the response.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
TIMEOUT = float(os.environ.get("TIMEOUT", "600"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kimi-tool-call-fixer")

# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------

TOK_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TOK_SECTION_END = "<|tool_calls_section_end|>"
TOK_CALL_BEGIN = "<|tool_call_begin|>"
TOK_CALL_END = "<|tool_call_end|>"
TOK_ARG_BEGIN = "<|tool_call_argument_begin|>"

ALL_TOKENS = [
    TOK_SECTION_BEGIN,
    TOK_SECTION_END,
    TOK_CALL_BEGIN,
    TOK_CALL_END,
    TOK_ARG_BEGIN,
]

# Regex that matches the full tool_calls section (greedy across newlines)
TOOL_SECTION_RE = re.compile(
    re.escape(TOK_SECTION_BEGIN) + r"(.*?)" + re.escape(TOK_SECTION_END),
    re.DOTALL,
)

# Regex for individual tool calls within a section
TOOL_CALL_RE = re.compile(
    re.escape(TOK_CALL_BEGIN)
    + r"\s*(.*?)\s*"
    + re.escape(TOK_ARG_BEGIN)
    + r"\s*(.*?)\s*"
    + re.escape(TOK_CALL_END),
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@dataclass
class ParsedToolCall:
    """A single parsed tool call extracted from raw tokens."""

    call_id: str
    function_name: str
    arguments: str  # JSON string


@dataclass
class ToolCallAccumulator:
    """
    State machine that accumulates streamed text and detects tool-call token
    sequences.  Works character-by-character (well, chunk-by-chunk) so it can
    handle tokens split across SSE chunks.
    """

    buffer: str = ""
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    in_section: bool = False
    finished: bool = False

    # ---- internal per-call state ----
    _in_call: bool = False
    _in_args: bool = False
    _current_id_buf: str = ""
    _current_args_buf: str = ""

    def feed(self, text: str) -> str:
        """
        Feed a chunk of text.  Returns the portion of text that is *not* part
        of a tool-call token sequence (i.e. the "clean" text that should still
        be forwarded as reasoning/content).
        """
        self.buffer += text
        return self._consume()

    def _consume(self) -> str:
        """Process the buffer and return clean (non-tool-call) text."""
        clean = ""

        while self.buffer:
            # --- Check for any token prefix at current position ---
            # If the remaining buffer is a prefix of any token we need to wait
            # for more data before deciding.
            if self._is_partial_token_prefix(self.buffer):
                break

            # --- Try to match full tokens at the start of buffer ---
            matched = False

            if self.buffer.startswith(TOK_SECTION_BEGIN):
                self.in_section = True
                self.buffer = self.buffer[len(TOK_SECTION_BEGIN):]
                logger.info("Tool call section STARTED — intercepting raw tool tokens")
                matched = True

            elif self.buffer.startswith(TOK_SECTION_END):
                self.in_section = False
                self.finished = True
                self.buffer = self.buffer[len(TOK_SECTION_END):]
                logger.info(
                    "Tool call section ENDED — %d tool call(s) parsed",
                    len(self.tool_calls),
                )
                matched = True

            elif self.buffer.startswith(TOK_CALL_BEGIN):
                self._in_call = True
                self._in_args = False
                self._current_id_buf = ""
                self._current_args_buf = ""
                self.buffer = self.buffer[len(TOK_CALL_BEGIN):]
                logger.debug("Parsing new tool call …")
                matched = True

            elif self.buffer.startswith(TOK_ARG_BEGIN):
                self._in_args = True
                self.buffer = self.buffer[len(TOK_ARG_BEGIN):]
                matched = True

            elif self.buffer.startswith(TOK_CALL_END):
                # Finalize this tool call
                self._finalize_call()
                self.buffer = self.buffer[len(TOK_CALL_END):]
                matched = True

            if not matched:
                ch = self.buffer[0]
                self.buffer = self.buffer[1:]
                if self._in_call:
                    if self._in_args:
                        self._current_args_buf += ch
                    else:
                        self._current_id_buf += ch
                elif self.in_section:
                    # Whitespace / noise between calls inside the section — skip
                    pass
                else:
                    clean += ch

        return clean

    def _finalize_call(self) -> None:
        raw_id = self._current_id_buf.strip()
        raw_args = self._current_args_buf.strip()

        # The ID from the model looks like "functions.process:28" where
        # "functions." is a namespace prefix. Strip it to get the function
        # name, and build a proper call_id.
        func_name = raw_id
        if func_name.startswith("functions."):
            func_name = func_name[len("functions."):]

        # Some IDs contain a numeric suffix after colon (e.g. "read:3");
        # the number is just an index, not part of the name, but some models
        # may also use it as the call id. We'll preserve the original raw_id
        # as the basis for the call_id.

        # Strip the colon-suffix from the function name if present
        name_without_suffix = func_name.split(":")[0] if ":" in func_name else func_name

        if not name_without_suffix:
            logger.error(
                "Parsed tool call with EMPTY function name (raw_id=%r). "
                "This tool call will likely fail downstream.",
                raw_id,
            )

        call_id = f"call_{uuid.uuid4().hex[:24]}"

        # Validate / normalise the arguments JSON
        try:
            args_obj = json.loads(raw_args)
            args_str = json.dumps(args_obj)
        except json.JSONDecodeError:
            logger.error(
                "Could not parse tool call arguments as valid JSON for '%s' "
                "(call_id=%s): %r",
                name_without_suffix,
                call_id,
                raw_args[:200] + ("…" if len(raw_args) > 200 else ""),
            )
            args_str = raw_args

        self.tool_calls.append(
            ParsedToolCall(
                call_id=call_id,
                function_name=name_without_suffix,
                arguments=args_str,
            )
        )

        args_preview = args_str[:120] + ("…" if len(args_str) > 120 else "")
        logger.info(
            "Parsed tool call: %s (call_id=%s) args=%s",
            name_without_suffix,
            call_id,
            args_preview,
        )

        self._in_call = False
        self._in_args = False
        self._current_id_buf = ""
        self._current_args_buf = ""

    @staticmethod
    def _is_partial_token_prefix(buf: str) -> bool:
        """
        Return True if `buf` is a non-empty prefix of any known token but is
        NOT yet a complete token.  This tells the accumulator to wait for more
        data.
        """
        for tok in ALL_TOKENS:
            if tok.startswith(buf) and buf != tok:
                return True
        return False


def _contains_tool_tokens(text: str) -> bool:
    """Quick check whether a string contains any raw tool-call tokens."""
    return TOK_SECTION_BEGIN in text or TOK_CALL_BEGIN in text


# ---------------------------------------------------------------------------
# Non-streaming response fixer
# ---------------------------------------------------------------------------


def _parse_tool_calls_from_text(text: str) -> tuple[str, list[ParsedToolCall]]:
    """
    Parse and remove tool-call token sequences from *text*, returning the
    cleaned text and a list of ParsedToolCalls.
    """
    acc = ToolCallAccumulator()
    clean = acc.feed(text)
    # Flush anything remaining in the buffer that isn't a partial token
    clean += acc.buffer
    acc.buffer = ""
    return clean, acc.tool_calls


def fix_non_streaming_response(body: dict[str, Any]) -> dict[str, Any]:
    """
    Inspect a non-streaming chat completion response and, if any choice
    contains raw tool-call tokens in content or reasoning_content, parse
    them out and attach proper tool_calls.
    """
    choices = body.get("choices", [])
    for choice in choices:
        msg = choice.get("message", {})

        all_tool_calls: list[ParsedToolCall] = []

        # Check reasoning_content
        rc = msg.get("reasoning_content")
        if rc and _contains_tool_tokens(rc):
            clean_rc, tc = _parse_tool_calls_from_text(rc)
            cleaned = clean_rc.rstrip()
            msg["reasoning_content"] = cleaned if cleaned else None
            all_tool_calls.extend(tc)

        # Check content
        content = msg.get("content")
        if isinstance(content, str) and _contains_tool_tokens(content):
            clean_c, tc = _parse_tool_calls_from_text(content)
            msg["content"] = clean_c.rstrip() if clean_c.strip() else None
            all_tool_calls.extend(tc)

        if all_tool_calls:
            existing = msg.get("tool_calls") or []
            for pc in all_tool_calls:
                existing.append(
                    {
                        "id": pc.call_id,
                        "type": "function",
                        "function": {
                            "name": pc.function_name,
                            "arguments": pc.arguments,
                        },
                    }
                )
            msg["tool_calls"] = existing
            choice["finish_reason"] = "tool_calls"
            logger.info(
                "Non-streaming fix: converted %d raw tool call token(s) → "
                "native tool_calls [%s]",
                len(all_tool_calls),
                ", ".join(tc.function_name for tc in all_tool_calls),
            )

    return body


# ---------------------------------------------------------------------------
# Streaming response fixer
# ---------------------------------------------------------------------------


def _make_tool_call_chunks(
    tool_calls: list[ParsedToolCall],
    chunk_template: dict[str, Any],
    choice_index: int = 0,
) -> list[str]:
    """
    Build SSE `data: ...` lines that represent tool_call deltas for the
    parsed tool calls, following the OpenAI streaming format.
    """
    lines: list[str] = []

    for idx, tc in enumerate(tool_calls):
        # First chunk for this tool_call: includes function name
        delta_name: dict[str, Any] = {
            "tool_calls": [
                {
                    "index": idx,
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.function_name,
                        "arguments": "",
                    },
                }
            ]
        }
        chunk_name = _build_chunk(chunk_template, choice_index, delta_name, None)
        lines.append(f"data: {json.dumps(chunk_name)}\n\n")

        # Stream arguments in a single piece (could be chunked further but
        # this keeps it simple and correct).
        delta_args: dict[str, Any] = {
            "tool_calls": [
                {
                    "index": idx,
                    "function": {
                        "arguments": tc.arguments,
                    },
                }
            ]
        }
        chunk_args = _build_chunk(chunk_template, choice_index, delta_args, None)
        lines.append(f"data: {json.dumps(chunk_args)}\n\n")

    return lines


def _build_chunk(
    template: dict[str, Any],
    choice_index: int,
    delta: dict[str, Any],
    finish_reason: str | None,
) -> dict[str, Any]:
    """Create a streaming chunk dict from a template."""
    return {
        "id": template.get("id", ""),
        "object": "chat.completion.chunk",
        "created": template.get("created", int(time.time())),
        "model": template.get("model", ""),
        "system_fingerprint": template.get("system_fingerprint"),
        "choices": [
            {
                "index": choice_index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


async def _stream_with_tool_fix(
    upstream_response: httpx.Response,
) -> AsyncGenerator[str]:
    """
    Async generator that reads the upstream SSE stream, intercepts tool-call
    tokens, and yields fixed SSE lines.

    Returns an async generator of bytes.
    """
    accumulator = ToolCallAccumulator()
    chunk_template: dict[str, Any] = {}
    last_choice_index = 0
    saw_tool_tokens = False
    emitted_finish = False  # True once we've sent a finish_reason=tool_calls chunk

    try:
        async for raw_line in upstream_response.aiter_lines():
            # SSE lines
            line = raw_line.strip()

            if not line:
                yield "\n"
                continue

            if line.startswith(":"):
                # SSE comment — pass through
                yield line + "\n"
                continue

            if line == "data: [DONE]":
                # Before emitting [DONE], flush any accumulated tool calls
                if accumulator.tool_calls:
                    saw_tool_tokens = True
                    logger.info(
                        "Stream [DONE]: flushing %d remaining tool call(s)",
                        len(accumulator.tool_calls),
                    )
                    for tc_line in _make_tool_call_chunks(
                        accumulator.tool_calls, chunk_template, last_choice_index
                    ):
                        yield tc_line

                    # Emit a finish_reason=tool_calls chunk
                    finish_chunk = _build_chunk(
                        chunk_template, last_choice_index, {}, "tool_calls"
                    )
                    yield f"data: {json.dumps(finish_chunk)}\n\n"
                    emitted_finish = True

                if saw_tool_tokens:
                    logger.info("Stream complete — tool call tokens were intercepted and converted")

                yield "data: [DONE]\n\n"
                continue

            if not line.startswith("data: "):
                yield line + "\n"
                continue

            json_str = line[len("data: "):]
            try:
                chunk = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Failed to parse SSE chunk as JSON: %s", json_str[:200])
                yield line + "\n"
                continue

            # Capture template info for building our own chunks
            if not chunk_template:
                chunk_template = {
                    "id": chunk.get("id", ""),
                    "model": chunk.get("model", ""),
                    "created": chunk.get("created", int(time.time())),
                    "system_fingerprint": chunk.get("system_fingerprint"),
                }

            choices = chunk.get("choices", [])
            if not choices:
                yield f"data: {json.dumps(chunk)}\n\n"
                continue

            choice = choices[0]
            last_choice_index = choice.get("index", 0)
            delta = choice.get("delta", {})

            modified = False

            # --- Process reasoning_content ---
            rc = delta.get("reasoning_content")
            if rc and isinstance(rc, str):
                if _contains_tool_tokens(rc) or accumulator.in_section or accumulator._in_call:
                    clean = accumulator.feed(rc)
                    if clean:
                        delta["reasoning_content"] = clean
                    else:
                        del delta["reasoning_content"]
                    modified = True

            # --- Process content ---
            content = delta.get("content")
            if content and isinstance(content, str):
                if _contains_tool_tokens(content) or accumulator.in_section or accumulator._in_call:
                    clean = accumulator.feed(content)
                    if clean:
                        delta["content"] = clean
                    else:
                        del delta["content"]
                    modified = True

            # If tool calls just finished in this chunk, emit them now
            if accumulator.finished:
                saw_tool_tokens = True
                logger.info(
                    "Emitting %d parsed tool call(s) into stream",
                    len(accumulator.tool_calls),
                )
                # Emit any remaining clean text first
                if modified:
                    # Only emit if there's meaningful delta content left
                    has_content = (
                        delta.get("reasoning_content")
                        or delta.get("content")
                        or delta.get("role")
                    )
                    if has_content:
                        chunk["choices"][0]["delta"] = delta
                        chunk["choices"][0]["finish_reason"] = None
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Emit tool call chunks
                for tc_line in _make_tool_call_chunks(
                    accumulator.tool_calls, chunk_template, last_choice_index
                ):
                    yield tc_line

                # Emit finish_reason=tool_calls since the upstream will
                # only send finish_reason=stop (which is wrong for tool calls)
                finish_chunk = _build_chunk(
                    chunk_template, last_choice_index, {}, "tool_calls"
                )
                yield f"data: {json.dumps(finish_chunk)}\n\n"
                emitted_finish = True

                # Reset accumulator for potential additional sections
                accumulator.tool_calls.clear()
                accumulator.finished = False
                continue

            if modified:
                has_content = (
                    delta.get("reasoning_content")
                    or delta.get("content")
                    or delta.get("role")
                )
                if has_content:
                    chunk["choices"][0]["delta"] = delta
                    yield f"data: {json.dumps(chunk)}\n\n"
                # If delta is now empty (all content was tool tokens), skip this chunk
                continue

            # No modification needed — pass through
            # But if we detected tool calls earlier, fix the finish_reason
            if saw_tool_tokens and choice.get("finish_reason") == "stop":
                if emitted_finish:
                    # We already sent a finish_reason=tool_calls chunk;
                    # suppress the upstream's finish_reason=stop entirely
                    continue
                choice["finish_reason"] = "tool_calls"

            yield f"data: {json.dumps(chunk)}\n\n"

    except httpx.ReadError as exc:
        logger.error("Upstream connection error while streaming: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error in stream processing: %s", exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

# Long-lived HTTP client — created/destroyed with the app lifespan so that
# streaming responses can still read from the upstream connection after the
# request handler has returned.
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT))
    logger.info("HTTP client created (timeout=%ss)", TIMEOUT)
    yield
    await _http_client.aclose()
    _http_client = None
    logger.info("HTTP client closed")


app = FastAPI(
    title="Kimi K2.5 Tool Call Fixer",
    description="OpenAI-compatible proxy that fixes raw tool-call tokens from Kimi K2.5",
    lifespan=_lifespan,
)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str) -> Response:
    """
    Forward every request to the upstream server unchanged.
    For chat completion responses (streaming or not), intercept and fix
    any raw tool-call tokens.
    """
    assert _http_client is not None, "HTTP client not initialised"
    client = _http_client

    target_url = f"{SERVER_URL.rstrip('/')}/{path}"

    # Read the raw body
    body = await request.body()

    # Forward all headers except host
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    # Determine if this is a streaming chat completion request
    is_streaming = False
    is_chat_completion = "chat/completions" in path
    if is_chat_completion and body:
        try:
            req_body = json.loads(body)
            is_streaming = req_body.get("stream", False)
        except json.JSONDecodeError:
            pass

    if is_streaming and is_chat_completion:
        # Streaming: we need to read the response as a stream
        upstream_req = client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )
        upstream_resp = await client.send(upstream_req, stream=True)

        # Build response headers
        resp_headers = dict(upstream_resp.headers)
        resp_headers.pop("content-length", None)
        resp_headers.pop("transfer-encoding", None)
        resp_headers.pop("content-encoding", None)

        generator = _stream_with_tool_fix(upstream_resp)

        async def stream_and_close():
            try:
                async for chunk in generator:
                    yield chunk.encode() if isinstance(chunk, str) else chunk
            except httpx.ReadError as exc:
                logger.error(
                    "Upstream read error during stream consumption: %s", exc
                )
            except Exception as exc:
                logger.error(
                    "Error during stream consumption: %s", exc, exc_info=True
                )
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(
            stream_and_close(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type="text/event-stream",
        )
    else:
        # Non-streaming: forward and optionally fix
        upstream_resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
        )

        resp_body = upstream_resp.content
        resp_headers = dict(upstream_resp.headers)

        if is_chat_completion and upstream_resp.status_code == 200:
            try:
                resp_json = json.loads(resp_body)
                # Check if any fixing is needed
                needs_fix = False
                for choice in resp_json.get("choices", []):
                    msg = choice.get("message", {})
                    rc = msg.get("reasoning_content", "")
                    ct = msg.get("content", "") or ""
                    if _contains_tool_tokens(rc) or _contains_tool_tokens(ct):
                        needs_fix = True
                        break

                if needs_fix:
                    logger.info(
                        "Detected raw tool tokens in non-streaming response — fixing"
                    )
                    resp_json = fix_non_streaming_response(resp_json)
                    resp_body = json.dumps(resp_json).encode()
                    resp_headers["content-length"] = str(len(resp_body))
            except json.JSONDecodeError:
                pass

        # Remove hop-by-hop headers
        for h in ("transfer-encoding", "content-encoding"):
            resp_headers.pop(h, None)

        return Response(
            content=resp_body,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info("Starting Kimi K2.5 Tool Call Fixer on %s:%d", host, port)
    logger.info("Proxying to: %s", SERVER_URL)
    uvicorn.run(app, host=host, port=port)
