"""Subagent Policy Engine — YAML-driven execution & security policy for subagents."""

import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

POLICY_FILE = Path(__file__).parent.parent.parent / "config" / "subagent_policy.yaml"


class SubagentPolicyEngine:
    def __init__(self) -> None:
        self.decisions_audit_log: list[dict[str, Any]] = []
        self._policy: dict[str, Any] = {}
        self.load_policy()

    def load_policy(self) -> None:
        if POLICY_FILE.exists():
            try:
                self._policy = yaml.safe_load(POLICY_FILE.read_text(encoding="utf-8")) or {}
            except Exception as e:
                logger.warning(f"Failed to load subagent_policy.yaml: {e}")
                self._policy = self._default_policy()
        else:
            self._policy = self._default_policy()

    def _default_policy(self) -> dict[str, Any]:
        return {
            "subagents": {
                "enabled": True,
                "max_recursion_depth": 3,
                "allowed_tools": ["*"],
                "blocked_tools": ["rm -rf", "git push --force"],
                "background_execution": {
                    "allowed": True,
                    "max_duration_seconds": 300,
                },
            }
        }

    def save_policy(self, new_policy: dict[str, Any]) -> None:
        self._policy = new_policy
        try:
            POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
            POLICY_FILE.write_text(yaml.safe_dump(new_policy, default_flow_style=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save subagent policy: {e}")

    def evaluate_action(self, tool_name: str, arguments: dict[str, Any] | None = None) -> tuple[bool, str]:
        sub_cfg = self._policy.get("subagents", {})
        if not sub_cfg.get("enabled", True):
            decision = (False, "Subagents execution globally disabled by policy.")
            self._record_decision(tool_name, decision)
            return decision

        blocked = sub_cfg.get("blocked_tools", [])
        arg_str = str(arguments or "")
        for b in blocked:
            if b in tool_name or b in arg_str:
                decision = (False, f"Tool or argument contains blocked pattern '{b}'.")
                self._record_decision(tool_name, decision)
                return decision

        decision = (True, "Allowed by subagent policy.")
        self._record_decision(tool_name, decision)
        return decision

    def _record_decision(self, action: str, result: tuple[bool, str]) -> None:
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "action": action,
            "allowed": result[0],
            "reason": result[1],
        }
        self.decisions_audit_log.insert(0, entry)
        if len(self.decisions_audit_log) > 100:
            self.decisions_audit_log = self.decisions_audit_log[:100]

    def get_policy(self) -> dict[str, Any]:
        return self._policy

    def get_decisions(self) -> list[dict[str, Any]]:
        return self.decisions_audit_log


subagent_policy_engine = SubagentPolicyEngine()
