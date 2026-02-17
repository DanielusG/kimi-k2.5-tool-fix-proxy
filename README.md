# kimi-tool-call-fixer

An OpenAI-compatible proxy that fixes Kimi K2.5's raw tool call tokens in streaming and non-streaming responses.

## The Problem

Kimi K2.5 sometimes emits tool call special tokens (`<|tool_calls_section_begin|>`, `<|tool_call_begin|>`, etc.) inside `reasoning_content` or `content` instead of producing native `tool_calls` in the API response. This breaks any client expecting standard OpenAI-format tool calls.

## What This Does

This proxy sits between your client and the Kimi-compatible API server. It forwards all requests unmodified and intercepts responses, parsing raw tool call tokens into proper OpenAI-format `tool_calls` with correct `finish_reason: "tool_calls"`. Responses without raw tokens pass through untouched.

## Usage

### Docker (recommended)

```bash
git clone https://github.com/DanielusG/kimi-k2.5-tool-fix-proxy.git
cd kimi-tool-call-fixer
SERVER_URL=https://your-ip:your-port/v1 docker compose up -d
```

Then point your client at `http://localhost:8080` instead of the upstream API.

### Manual

```bash
pip install -r requirements.txt
SERVER_URL=https://your-ip:your-port/v1 python main.py
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `SERVER_URL` | `http://localhost:8000` | Upstream API base URL |
| `PORT` | `8080` | Proxy listen port |
| `HOST` | `0.0.0.0` | Bind address |
| `TIMEOUT` | `600` | Upstream request timeout (seconds) |
| `LOG_LEVEL` | `INFO` | Logging level |

## Tests

```bash
pip install pytest
pytest test_main.py -v
```

## License

MIT
