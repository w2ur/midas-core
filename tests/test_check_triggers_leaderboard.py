import json
import logging
from datetime import date

import pytest

from engine.config import get_config


def test_watcher_refreshes_current_json_after_fire(midas_data_root, monkeypatch):
    from scripts import check_triggers

    get_config().leaderboard_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        check_triggers,
        "_build_portfolio_summaries",
        lambda: {"satoshi": {"agent_id": "satoshi"}},
    )
    monkeypatch.setattr(
        check_triggers,
        "_build_leaderboard_rows",
        lambda summaries, on: [{"rank": 1, "agent": "satoshi", "return_pct": 5.0}],
    )

    check_triggers.refresh_leaderboard_artifact(
        trigger="trigger-fire", on=date(2026, 5, 23)
    )

    payload = json.loads((get_config().leaderboard_dir / "current.json").read_text())
    assert payload["trigger"] == "trigger-fire"
    assert payload["rows"][0]["agent"] == "satoshi"
    assert payload["updated_at"].endswith("Z")


def test_watcher_leaderboard_refresh_swallows_errors(
    midas_data_root, monkeypatch, caplog
):
    """Critical contract: a leaderboard refresh failure must never raise."""
    from scripts import check_triggers

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(check_triggers, "_build_portfolio_summaries", _boom)

    with caplog.at_level(logging.WARNING):
        check_triggers.refresh_leaderboard_artifact(
            trigger="trigger-fire", on=date(2026, 5, 23)
        )

    assert any(
        "leaderboard refresh failed" in r.message.lower() for r in caplog.records
    )
