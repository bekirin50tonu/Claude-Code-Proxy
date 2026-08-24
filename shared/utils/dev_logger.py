"""DevLogger — write raw request, response, and result payloads to logs/ directory."""

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


class DevLogger:
    """Developer logger persisting raw transaction payloads to logs/ directory."""

    def __init__(self, logs_dir: Path | None = None) -> None:
        self.logs_dir = logs_dir or LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.raw_jsonl_path = self.logs_dir / "raw_requests.jsonl"
        self.dev_log_path = self.logs_dir / "dev_proxy.log"
        self.errors_log_path = self.logs_dir / "errors.log"

    def record_transaction(
        self,
        request_id: str,
        method: str,
        path: str,
        client_model: str,
        mapped_model: str,
        status_code: int,
        duration_ms: float,
        request_body: dict[str, Any] | None = None,
        response_body: dict[str, Any] | str | None = None,
        upstream_response: dict[str, Any] | str | None = None,
        headers: dict[str, str] | None = None,
        error_details: dict[str, Any] | None = None,
        fallbacks_used: list[str] | None = None,
        stage_info: dict[str, Any] | None = None,
    ) -> None:
        """Write raw request (Stage 1 & 2), response (Stage 3), and result (Stage 4)."""
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        timestamp_human = time.strftime("%Y-%m-%d %H:%M:%S")

        req_b = request_body or {}
        messages = req_b.get("messages", [])
        system = req_b.get("system", "")
        tools = req_b.get("tools", [])

        last_user_msg = ""
        if isinstance(messages, list):
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str):
                        last_user_msg = c[:200]
                    elif isinstance(c, list):
                        last_user_msg = str(c)[:200]
                    break

        s_info = stage_info or {}
        sanitizations = s_info.get("sanitizations", ["AutoCloseTagParser: OK", "AngleBracketEscape: OK"])

        record = {
            "timestamp": timestamp_iso,
            "request_id": request_id,
            "stage_1_incoming": {
                "source": "Claude Code CLI",
                "method": method,
                "path": path,
                "client_model": client_model,
                "headers": headers or {},
                "messages_count": len(messages) if isinstance(messages, list) else 0,
                "tools_count": len(tools) if isinstance(tools, list) else 0,
                "last_user_message": last_user_msg,
                "body": request_body,
            },
            "stage_2_translated_upstream": {
                "target": "Upstream LLM Provider",
                "mapped_model": mapped_model,
                "fallbacks_used": fallbacks_used or [],
                "system_prompt": str(system)[:300] if system else "None",
            },
            "stage_3_upstream_response": {
                "source": "Upstream LLM Provider",
                "mapped_model": mapped_model,
                "status_code": status_code,
                "body": upstream_response or response_body,
            },
            "stage_4_outgoing_anthropic": {
                "target": "Claude Code CLI",
                "status_code": status_code,
                "duration_ms": duration_ms,
                "sanitizations_applied": sanitizations,
                "body": response_body,
            },
            "error_details": error_details,
        }

        # 1. Append JSON record to logs/raw_requests.jsonl
        try:
            with open(self.raw_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("DevLogger: Failed to write to %s: %s", self.raw_jsonl_path, exc)

        # 2. Append 4-stage formatted text record to logs/dev_proxy.log
        try:
            error_str = f"  ⚠️ ERROR DETAILS: {json.dumps(error_details)}\n" if error_details else ""
            fb_str = f" (Fallbacks used: {', '.join(fallbacks_used)})" if fallbacks_used else ""
            dev_entry = (
                f"{'=' * 85}\n"
                f"[{timestamp_human}] ID: {request_id} | {method} {path} | Status: {status_code} ({duration_ms:.1f}ms)\n"
                f"{'-' * 85}\n"
                f"STAGE 1 [INCOMING REQUEST]: Claude Code CLI -> Proxy Gateway\n"
                f"  Model: {client_model} | Msgs: {len(messages) if isinstance(messages, list) else 0} | Tools: {len(tools) if isinstance(tools, list) else 0}\n"
                f"  Last User Msg: {repr(last_user_msg)}\n"
                f"STAGE 2 [TRANSLATED UPSTREAM]: Proxy Gateway -> Upstream LLM\n"
                f"  Mapped Model: {mapped_model}{fb_str}\n"
                f"STAGE 3 [RAW UPSTREAM RESPONSE]: Upstream LLM -> Proxy Gateway\n"
                f"  Status: {status_code} | Payload: {json.dumps(upstream_response or response_body, ensure_ascii=False)[:300] if (upstream_response or response_body) else 'None'}\n"
                f"STAGE 4 [OUTGOING RESPONSE]: Proxy Gateway -> Claude Code CLI\n"
                f"  Status: {status_code} | Sanitizations: {', '.join(sanitizations)}\n"
                f"{error_str}"
                f"{'=' * 85}\n\n"
            )
            with open(self.dev_log_path, "a", encoding="utf-8") as f:
                f.write(dev_entry)
        except Exception as exc:
            logger.warning("DevLogger: Failed to write to %s: %s", self.dev_log_path, exc)

        # 3. If error occurred, write dedicated entry to logs/errors.log
        if status_code >= 400 or error_details:
            try:
                err_entry = (
                    f"[{timestamp_human}] [ERROR {status_code}] ID: {request_id} | {method} {path}\n"
                    f"  STAGE 1 (Client Model): {client_model}\n"
                    f"  STAGE 2 (Upstream Model): {mapped_model} (Tried: {fallbacks_used or []})\n"
                    f"  STAGE 3 (Upstream Output): {json.dumps(upstream_response or response_body, ensure_ascii=False)[:400]}\n"
                    f"  DETAILS: {json.dumps(error_details or response_body, ensure_ascii=False)}\n"
                    f"{'=' * 85}\n"
                )
                with open(self.errors_log_path, "a", encoding="utf-8") as f:
                    f.write(err_entry)
            except Exception as exc:
                logger.warning("DevLogger: Failed to write to %s: %s", self.errors_log_path, exc)


dev_logger = DevLogger()
