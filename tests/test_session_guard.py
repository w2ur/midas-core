"""Tests for the session freshness guard.

Regression cover for the 2026-07-31 stall: a sandbox that fires on time,
suspends minutes in, and resumes ~63 hours later with a repo view that has
gone stale underneath it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timedelta, timezone

import pytest

from scripts import session_guard
from scripts.session_guard import (
    SessionAnchor,
    StaleSessionError,
    anchor_session,
    assert_session_fresh,
    clear_anchor,
    load_anchor,
)


@pytest.fixture
def anchored(tmp_path, monkeypatch):
    """Anchor written to a temp dir, with git calls stubbed out."""
    monkeypatch.setattr(session_guard, "_anchor_path", lambda: tmp_path / "anchor.json")

    state = {"origin_main": "base000", "subjects": [], "changed": []}

    def fake_git(*args: str) -> str:
        if args[:2] == ("rev-parse", "origin/main"):
            return state["origin_main"]
        if args[0] == "log":
            return "\n".join(state["subjects"])
        if args[0] == "diff":
            return "\n".join(state["changed"])
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(session_guard, "_git", fake_git)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    return state, tmp_path


def _write_anchor(tmp_path, *, started_at, session_date=date(2026, 7, 31)):
    (tmp_path / "anchor.json").write_text(
        json.dumps(
            SessionAnchor(
                session_date=session_date,
                base_sha="base000",
                started_at=started_at,
            ).to_dict()
        ),
        encoding="utf-8",
    )


def test_anchor_roundtrip(anchored):
    _, tmp_path = anchored
    a = anchor_session(date(2026, 7, 31))
    assert a.base_sha == "base000"
    loaded = load_anchor()
    assert loaded is not None
    assert loaded.session_date == date(2026, 7, 31)
    assert loaded.base_sha == "base000"

    clear_anchor()
    assert load_anchor() is None


def test_missing_anchor_is_fatal(anchored):
    with pytest.raises(StaleSessionError, match="no session anchor"):
        assert_session_fresh("author")


def test_fresh_session_passes(anchored):
    _, tmp_path = anchored
    _write_anchor(
        tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(minutes=20),
        session_date=datetime.now(timezone.utc).date(),
    )
    assert_session_fresh("author")  # no raise


def test_stalled_sandbox_is_caught(anchored):
    """The 2026-07-31 case: 63 hours of wall clock between anchor and check."""
    _, tmp_path = anchored
    _write_anchor(
        tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(hours=63),
        session_date=datetime.now(timezone.utc).date(),
    )
    with pytest.raises(StaleSessionError, match="stalled and resumed|running"):
        assert_session_fresh("author")


def test_date_rollover_is_caught(anchored):
    _, tmp_path = anchored
    _write_anchor(
        tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        session_date=datetime.now(timezone.utc).date() - timedelta(days=2),
    )
    with pytest.raises(StaleSessionError, match="already closed|dated"):
        assert_session_fresh("author")


def test_superseding_session_on_main_is_caught(anchored):
    state, tmp_path = anchored
    _write_anchor(
        tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(minutes=20),
        session_date=datetime.now(timezone.utc).date(),
    )
    state["origin_main"] = "newer99"
    state["subjects"] = ["chore: weekend refresh 2026-08-01"]
    with pytest.raises(StaleSessionError, match="superseded"):
        assert_session_fresh("push")


def test_ledger_movement_on_main_is_caught(anchored):
    state, tmp_path = anchored
    _write_anchor(
        tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(minutes=20),
        session_date=datetime.now(timezone.utc).date(),
    )
    state["origin_main"] = "newer99"
    state["subjects"] = ["chore(triggers): execute ord_x 2026-08-01"]
    state["changed"] = ["data/portfolios/satoshi/portfolio.json"]
    with pytest.raises(StaleSessionError, match="ledger moved"):
        assert_session_fresh("push")


def test_harmless_main_movement_is_allowed(anchored):
    """Prices/sentiment/site commits land mid-session routinely and are fine."""
    state, tmp_path = anchored
    _write_anchor(
        tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(minutes=20),
        session_date=datetime.now(timezone.utc).date(),
    )
    state["origin_main"] = "newer99"
    state["subjects"] = ["[data] 2026-07-31 sentiment digests"]
    state["changed"] = ["data/sentiment/x.json", "site/src/pages/index.astro"]
    assert_session_fresh("author")  # no raise
