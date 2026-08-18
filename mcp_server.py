#!/usr/bin/env python3
"""Standalone Stdio MCP Server for Hermes Agent.

Reads JSON-RPC 2.0 requests from stdin, executes proxy management tools,
and outputs JSON-RPC 2.0 responses to stdout.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from api.mcp import MCP_TOOLS_DEFINITIONS, execute_mcp_tool

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(line_buffering=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def run_stdio_mcp_server() -> None:
    """Read JSON-RPC messages line-by-line from stdin and respond on stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception:
            err_resp = {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {}) or {}

        if method == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "claude-code-proxy-mcp-stdio",
                        "version": "1.0.0",
                        "description": "Claude Code Proxy Stdio MCP Server for Hermes Agent",
                    },
                },
            }
        elif method in ("notifications/initialized", "initialized", "ping"):
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        elif method == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": MCP_TOOLS_DEFINITIONS},
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {}) or {}
            tool_res = asyncio.run(execute_mcp_tool(tool_name, arguments))
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": tool_res,
            }
        else:
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: '{method}'"},
            }

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        run_stdio_mcp_server()
    except (KeyboardInterrupt, SystemExit):
        pass
