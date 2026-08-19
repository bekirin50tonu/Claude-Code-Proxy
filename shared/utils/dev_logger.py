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
    ) -> None:
        """Write raw request (from Claude Code), response (from LLM Model), and result (to Claude Code)."""
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        timestamp_human = time.strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "timestamp": timestamp_iso,
            "request_id": request_id,
            "request": {
                "source": "Claude Code CLI",
                "method": method,
                "path": path,
                "client_model": client_model,
                "headers": headers or {},
                "body": request_body,
            },
            "response": {
                "source": "Upstream LLM Model",
                "mapped_model": mapped_model,
                "fallbacks_used": fallbacks_used or [],
                "body": upstream_response or response_body,
            },
            "result": {
                "target": "Claude Code CLI",
                "status_code": status_code,
                "duration_ms": duration_ms,
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

        # 2. Append formatted text record to logs/dev_proxy.log
        try:
            error_str = f"  ERROR DETAILS: {json.dumps(error_details)}\n" if error_details else ""
            dev_entry = (
                f"[{timestamp_human}] ID: {request_id} | {method} {path} | Status: {status_code} ({duration_ms}ms)\n"
                f"  [1. REQUEST (Claude Code -> Proxy)] Model: {client_model}\n"
                f"      Payload: {json.dumps(request_body, ensure_ascii=False) if request_body else 'None'}\n"
                f"  [2. RESPONSE (LLM Model -> Proxy)] Model: {mapped_model}\n"
                f"      Payload: {json.dumps(upstream_response or response_body, ensure_ascii=False) if (upstream_response or response_body) else 'None'}\n"
                f"  [3. RESULT (Proxy -> Claude Code)] Status: {status_code}\n"
                f"      Payload: {json.dumps(response_body, ensure_ascii=False) if isinstance(response_body, dict) else (response_body or 'None')}\n"
                f"{error_str}"
                f"{'-' * 80}\n"
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
                    f"  REQUEST: Model {client_model}\n"
                    f"  RESPONSE: Upstream Model {mapped_model}\n"
                    f"  DETAILS: {json.dumps(error_details or response_body, ensure_ascii=False)}\n"
                    f"{'=' * 80}\n"
                )
                with open(self.errors_log_path, "a", encoding="utf-8") as f:
                    f.write(err_entry)
            except Exception as exc:
                logger.warning("DevLogger: Failed to write to %s: %s", self.errors_log_path, exc)


dev_logger = DevLogger()
