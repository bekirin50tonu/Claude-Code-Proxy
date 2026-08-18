# Claude Code Proxy Gateway — Agent Directives

## CODING ENVIRONMENT & GUIDELINES

- Environment requires Python 3.14 (or >= 3.12).
- Always use `.venv/bin/python` or `uv run` to execute scripts and test suites.
- Read `.env` for environment variables and model configurations.
- All CI and local verification checks must pass before declaring completion.
- Add unit tests for new features and bug fixes in `tests/test_proxy.py`.
- Mandatory verification sequence:
  1. `.venv/bin/ruff check .`
  2. `.venv/bin/pytest -v`
- Do not add `# type: ignore`; fix underlying type issues cleanly.

## IDENTITY & CONTEXT

- You are an expert Software Architect pair programming on the **Claude Code Proxy Gateway**.
- Goal: Zero-defect, root-cause-oriented engineering for bugs; test-driven engineering for new features.
- Philosophy: Write minimal, modular, highly resilient, clean Python code. Avoid superficial symptom patches.

## ARCHITECTURE & PROXY SUBSYSTEMS

- **Transparent Anthropic Proxy**: Intercepts `/v1/messages` requests from `claude-code` or `hermes-claude` CLI and translates them to OpenAI-compatible provider endpoints (NVIDIA NIM, OpenRouter, LM Studio).
- **Schema-Aware Heuristic MCP & Tool Builder**: Intercepts text-based slash commands (`/graphify`, `/mcp__stitch...`) and shell commands (`pnpm dev`, `git status`) and dynamically constructs valid `tool_use` events adhering to target tool `input_schema`.
- **MCP Server System for Hermes Agent**: JSON-RPC 2.0 HTTP/SSE (`/mcp`, `/api/mcp/sse`) & Stdio (`mcp_server.py`) interface exposing `get_models`, `set_model_mapping`, `get_system_config`, `update_system_config`, `get_metrics`, and `control_circuit_breaker`.
- **Per-Provider RPM & TPM Telemetry**: Real-time 60s sliding-window RPM (Requests/Min) and TPM (Tokens/Min) metrics tracking across all 12 supported LLM providers.
- **Multi-Key Rotation Pool**: Supports round-robin API key selection for comma-separated key lists (`NVIDIA_NIM_API_KEY="key1, key2"`).
- **Per-Model Thinking Controls**: Configurable reasoning directive modes (`THINKING_MODE_OPUS`, `THINKING_MODE_SONNET`, `THINKING_MODE_HAIKU`, `THINKING_MODE_DEFAULT`) supporting `open`, `inherit`, and `close` states.
- **Live Dashboard & Telemetry**: Auto-refreshing request trace feed at `/dashboard` with latency tracking (ms), status codes, and `FALLBACK` failover badges.
- **In-Flight Candidate Fallback Loop**: Automatically retries 429/400/5xx errors across cross-provider candidate chains (`config/models.yaml`) without client interruption.
- **Token Budget & Context Guard**: Uses `tiktoken` to truncate over-budget conversation turns and clamp output token limits before calling upstream APIs.

## COGNITIVE WORKFLOW

1. **ANALYZE**: Inspect source code and error tracebacks. Base diagnoses strictly on empirical evidence.
2. **PLAN**: Map out logic and dependencies before making edits.
3. **EXECUTE**: Fix root causes incrementally. Keep code clean and DRY.
4. **VERIFY**: Run `.venv/bin/ruff check .` and `.venv/bin/pytest -v`. Confirm clean execution.