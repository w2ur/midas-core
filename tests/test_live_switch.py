"""Unit tests for engine/live_switch.py — global kill switch.

Fail-safe principle: any unexpected condition → False (default OFF).
Uses tmp_path + monkeypatch fixtures; is_live_enabled() accepts an optional
path parameter so tests can point at temp files without touching the real config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine import live_switch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


# ---------------------------------------------------------------------------
# File-based behaviour (no env override)
# ---------------------------------------------------------------------------


class TestFileBehaviour:
    def test_absent_file_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIDAS_LIVE", raising=False)
        missing = tmp_path / "live_switch.json"
        assert live_switch.is_live_enabled(path=missing) is False

    def test_live_enabled_true_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIDAS_LIVE", raising=False)
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": True})
        assert live_switch.is_live_enabled(path=p) is True

    def test_live_enabled_false_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIDAS_LIVE", raising=False)
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": False})
        assert live_switch.is_live_enabled(path=p) is False

    def test_live_enabled_missing_key_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIDAS_LIVE", raising=False)
        p = tmp_path / "live_switch.json"
        _write_config(p, {"other_key": "value"})
        assert live_switch.is_live_enabled(path=p) is False

    def test_malformed_json_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIDAS_LIVE", raising=False)
        p = tmp_path / "live_switch.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not valid json}")
        assert live_switch.is_live_enabled(path=p) is False


# ---------------------------------------------------------------------------
# Env override — MIDAS_LIVE wins over file
# ---------------------------------------------------------------------------


class TestEnvOverride:
    def test_env_zero_overrides_true_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIDAS_LIVE", "0")
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": True})
        assert live_switch.is_live_enabled(path=p) is False

    def test_env_false_lowercase_overrides_true_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIDAS_LIVE", "false")
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": True})
        assert live_switch.is_live_enabled(path=p) is False

    def test_env_false_uppercase_overrides_true_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIDAS_LIVE", "FALSE")
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": True})
        assert live_switch.is_live_enabled(path=p) is False

    def test_env_one_overrides_false_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIDAS_LIVE", "1")
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": False})
        assert live_switch.is_live_enabled(path=p) is True

    def test_env_true_lowercase_overrides_false_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIDAS_LIVE", "true")
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": False})
        assert live_switch.is_live_enabled(path=p) is True

    def test_env_true_uppercase_overrides_false_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIDAS_LIVE", "TRUE")
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": False})
        assert live_switch.is_live_enabled(path=p) is True

    def test_unknown_env_value_falls_back_to_file_true(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIDAS_LIVE", "maybe")
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": True})
        assert live_switch.is_live_enabled(path=p) is True

    def test_unknown_env_value_falls_back_to_file_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIDAS_LIVE", "maybe")
        p = tmp_path / "live_switch.json"
        _write_config(p, {"live_enabled": False})
        assert live_switch.is_live_enabled(path=p) is False

    def test_env_one_overrides_absent_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIDAS_LIVE", "1")
        missing = tmp_path / "live_switch.json"
        assert live_switch.is_live_enabled(path=missing) is True


# ---------------------------------------------------------------------------
# Fail-safe: never raises
# ---------------------------------------------------------------------------


class TestFailSafe:
    def test_never_raises_on_any_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MIDAS_LIVE", raising=False)
        # Directory instead of a file — causes a read error.
        bad_path = tmp_path / "a_directory"
        bad_path.mkdir()
        assert live_switch.is_live_enabled(path=bad_path) is False

    def test_default_path_resolves_without_error(self):
        # Calling with no arguments must not raise, regardless of file state.
        result = live_switch.is_live_enabled()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Guard: committed default must always be OFF
# ---------------------------------------------------------------------------


class TestCommittedDefault:
    @pytest.mark.live_cast
    def test_committed_default_is_off(self, monkeypatch):
        """Guard CI against anyone committing live_enabled=true.

        Uses the real data/agent_config/live_switch.json without any path
        override, with MIDAS_LIVE absent so the file value is authoritative.
        """
        from engine.config import get_config

        monkeypatch.delenv("MIDAS_LIVE", raising=False)
        config_path = get_config().agent_config_dir / "live_switch.json"
        data = __import__("json").loads(config_path.read_text(encoding="utf-8"))
        assert data.get("live_enabled") is False, (
            "data/agent_config/live_switch.json must ship with live_enabled=false"
        )
        assert live_switch.is_live_enabled() is False, (
            "is_live_enabled() must return False with no env override and the committed config"
        )
