"""Claude Code Settings Bridge — Unified configuration bridge between Claude Code CLI and local proxy."""

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

USER_SETTINGS_PATH = Path.home() / ".claude.json"
PROJECT_SETTINGS_PATH = Path.cwd() / ".claude.json"


class ClaudeSettingsManager:
    """Manages merged configuration hierarchy (User -> Project -> Local)."""

    def _read_json(self, path: Path) -> dict[str, Any]:
        if path.exists() and path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8")) or {}
            except Exception as e:
                logger.warning(f"Failed to read Claude settings from '{path}': {e}")
        return {}

    def _write_json(self, path: Path, data: dict[str, Any]) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"Failed to write Claude settings to '{path}': {e}")
            return False

    def get_merged_settings(self) -> dict[str, Any]:
        user_cfg = self._read_json(USER_SETTINGS_PATH)
        project_cfg = self._read_json(PROJECT_SETTINGS_PATH)

        merged = {}
        merged.update(user_cfg)
        merged.update(project_cfg)
        return merged

    def set_setting(self, key: str, value: Any, scope: str = "project") -> bool:
        target_path = USER_SETTINGS_PATH if scope.lower() == "user" else PROJECT_SETTINGS_PATH
        cfg = self._read_json(target_path)
        cfg[key] = value
        return self._write_json(target_path, cfg)

    def sync_proxy_to_claude(self) -> dict[str, Any]:
        """Sync local proxy settings into Claude Code project config."""
        from config import settings

        proxy_env = {
            "ANTHROPIC_BASE_URL": "http://localhost:8090",
            "ANTHROPIC_AUTH_TOKEN": settings.GATEWAY_AUTH_TOKEN or "local-proxy-token",
        }
        self.set_setting("env", proxy_env, scope="project")
        logger.info("Synced Claude Code CLI configuration to point to local proxy gateway (http://localhost:8090).")
        return {"status": "success", "synced_env": proxy_env}


claude_settings_manager = ClaudeSettingsManager()
