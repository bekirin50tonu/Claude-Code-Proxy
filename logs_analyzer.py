#!/usr/bin/env python3
"""CLI Log Analyzer for Claude Code Proxy Gateway.

Diagnoses 4-stage transaction logs:
Stage 1: [INCOMING REQUEST] Claude Code CLI -> Proxy
Stage 2: [TRANSLATED UPSTREAM] Proxy -> Upstream LLM Provider
Stage 3: [RAW UPSTREAM RESPONSE] Upstream LLM Provider -> Proxy
Stage 4: [OUTGOING ANTHROPIC RESPONSE] Proxy -> Claude Code CLI
"""

import sys
import json
import argparse
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"
RAW_LOG_PATH = LOGS_DIR / "raw_requests.jsonl"
ERRORS_LOG_PATH = LOGS_DIR / "errors.log"


def print_stage_breakdown(entry: dict) -> None:
    req_id = entry.get("request_id", "unknown")
    ts = entry.get("timestamp", "")
    
    s1 = entry.get("stage_1_incoming") or entry.get("request") or {}
    s2 = entry.get("stage_2_translated_upstream") or {}
    s3 = entry.get("stage_3_upstream_response") or entry.get("response") or {}
    s4 = entry.get("stage_4_outgoing_anthropic") or entry.get("result") or {}
    err = entry.get("error_details")

    status = s4.get("status_code") or s3.get("status_code") or 200
    duration = s4.get("duration_ms", 0.0)

    print("\033[1;36m" + "=" * 85 + "\033[0m")
    print(f"\033[1;33m[{ts}] TRANSACTION ID: {req_id}\033[0m | \033[1;32mStatus: {status}\033[0m ({duration:.1f}ms)")
    print("\033[1;36m" + "-" * 85 + "\033[0m")

    # STAGE 1
    print("\033[1;35mSTAGE 1: [INCOMING REQUEST]\033[0m Claude Code CLI -> Proxy Gateway")
    print(f"  Path: {s1.get('method', 'POST')} {s1.get('path', '/v1/messages')}")
    print(f"  Client Model: {s1.get('client_model')}")
    print(f"  Messages Count: {s1.get('messages_count', len(s1.get('body', {}).get('messages', [])) if isinstance(s1.get('body'), dict) else 0)}")
    print(f"  Tools Offered: {s1.get('tools_count', len(s1.get('body', {}).get('tools', [])) if isinstance(s1.get('body'), dict) else 0)}")
    print(f"  Last User Msg: {repr(s1.get('last_user_message', ''))}")

    # STAGE 2
    print("\033[1;35mSTAGE 2: [TRANSLATED UPSTREAM]\033[0m Proxy Gateway -> Upstream LLM")
    print(f"  Mapped Model: {s2.get('mapped_model') or s3.get('mapped_model')}")
    print(f"  Fallbacks Tried: {s2.get('fallbacks_used', [])}")
    if s2.get("system_prompt"):
        print(f"  System Prompt Snippet: {repr(s2.get('system_prompt'))}")

    # STAGE 3
    print("\033[1;35mSTAGE 3: [RAW UPSTREAM RESPONSE]\033[0m Upstream LLM -> Proxy Gateway")
    res_body = s3.get("body")
    if isinstance(res_body, dict):
        content = res_body.get("content") or res_body.get("choices") or []
        stop = res_body.get("stop_reason") or res_body.get("finish_reason")
        print(f"  Upstream Stop Reason: {stop}")
        print(f"  Upstream Content Blocks: {len(content) if isinstance(content, list) else 1}")
    else:
        print(f"  Upstream Payload: {repr(str(res_body)[:200])}")

    # STAGE 4
    print("\033[1;35mSTAGE 4: [OUTGOING ANTHROPIC RESPONSE]\033[0m Proxy Gateway -> Claude Code CLI")
    print(f"  HTTP Status Code: {status}")
    print(f"  Sanitizations Applied: {s4.get('sanitizations_applied', ['AutoCloseTagParser: OK', 'AngleBracketEscape: OK'])}")

    if err:
        print(f"\033[1;31m  ⚠️ ERROR DETAILS: {json.dumps(err)}\033[0m")

    print("\033[1;36m" + "=" * 85 + "\033[0m\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect 4-stage Claude Code Proxy logs.")
    parser.add_argument("--last", "-l", action="store_true", help="Print the 4-stage breakdown of the last transaction.")
    parser.add_argument("--errors", "-e", action="store_true", help="Print recent error transactions.")
    parser.add_argument("--search", "-s", type=str, help="Search transactions by keyword.")
    parser.add_argument("--count", "-c", type=int, default=5, help="Number of records to show.")
    args = parser.parse_args()

    if not RAW_LOG_PATH.exists():
        print(f"No log file found at {RAW_LOG_PATH}")
        sys.exit(0)

    lines = []
    with open(RAW_LOG_PATH, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("Log file is empty.")
        sys.exit(0)

    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass

    if args.errors:
        err_entries = [e for e in entries if (e.get("stage_4_outgoing_anthropic", {}).get("status_code", 200) >= 400 or e.get("error_details"))]
        if not err_entries:
            print("✅ No error transactions found in log history.")
            return
        print(f"\n🔍 FOUND {len(err_entries)} ERROR TRANSACTIONS (Showing last {min(args.count, len(err_entries))}):\n")
        for entry in err_entries[-args.count:]:
            print_stage_breakdown(entry)

    elif args.search:
        keyword = args.search.lower()
        matched = [e for e in entries if keyword in json.dumps(e).lower()]
        if not matched:
            print(f"No transactions matched keyword: '{args.search}'")
            return
        print(f"\n🔍 FOUND {len(matched)} MATCHING TRANSACTIONS (Showing last {min(args.count, len(matched))}):\n")
        for entry in matched[-args.count:]:
            print_stage_breakdown(entry)

    else:
        # Default: show last transaction(s)
        count = 1 if args.last else args.count
        print(f"\n📋 SHOWING LAST {min(count, len(entries))} TRANSACTION(S):\n")
        for entry in entries[-count:]:
            print_stage_breakdown(entry)


if __name__ == "__main__":
    main()
