"""The repo-committed SessionStart hook must keep pointing at a real script.

`.claude/settings.json` names a script by path. Nothing at runtime validates
that coupling: rename or delete the script and the hook fails silently in every
cloud session, which is precisely where nobody is watching. This binds them.

Skipped in midas-core, which ships no root `.claude/settings.json` — same
convention as the `site/src/lib/rails.ts` guard in test_reason_codes.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = REPO_ROOT / ".claude" / "settings.json"

pytestmark = pytest.mark.skipif(
    not SETTINGS.exists(), reason=".claude/settings.json is live-only"
)


def _hook_commands() -> list[str]:
    config = json.loads(SETTINGS.read_text(encoding="utf-8"))
    return [
        hook["command"]
        for group in config.get("hooks", {}).get("SessionStart", [])
        for hook in group.get("hooks", [])
        if hook.get("type") == "command"
    ]


def test_settings_is_valid_json_with_a_sessionstart_hook():
    commands = _hook_commands()
    assert commands, "expected at least one SessionStart command hook"


def test_every_hook_command_points_at_a_script_that_exists():
    """$CLAUDE_PROJECT_DIR resolves to the repo root in a session."""
    missing = []
    for command in _hook_commands():
        for match in re.findall(r'"?\$CLAUDE_PROJECT_DIR"?(/[\w./-]+)', command):
            if not (REPO_ROOT / match.lstrip("/")).is_file():
                missing.append(f"{command!r} -> {match} does not exist")
    assert not missing, "hook points at a missing script:\n  " + "\n  ".join(missing)


def test_ensure_venv_is_cloud_only():
    """It must no-op on a dev machine — it would otherwise rebuild a local venv.

    The guard is the CLAUDE_CODE_REMOTE check; without it, every local session
    start would run a full dependency install against the developer's checkout.
    """
    source = (REPO_ROOT / "scripts" / "ensure_venv.sh").read_text(encoding="utf-8")
    assert "CLAUDE_CODE_REMOTE" in source
    assert "exit 0" in source
