"""Tests for engine.manager_report — pure read/compute helpers for the private
Streamlit Manager page (app/pages/06_manager.py).

TDD: these tests were written before the implementation.

The Manager report is a *private* local view (Streamlit is not deployed). It reads:
  - data/portfolios/the-manager/{portfolio,snapshots}.json   (PAPER book)
  - data/portfolios/baseline-manager/{portfolio,snapshots}.json (Gate C twin)
  - data/orders/manager-review/{date}.json                   (daily decisions)
  - data/orders/manager-review/resolved.json                 (matured outcomes)

These helpers stay pandas/plotly-free so they can be unit-tested without a UI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.config import get_config, reset_config_cache
from engine.manager_report import (
    book_paths,
    build_manager_summary,
    load_decisions,
    load_resolved,
    read_portfolio,
    read_snapshots,
    return_pct,
)


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv("MIDAS_DATA_DIR", raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


# ---------------------------------------------------------------------------
# read_snapshots
# ---------------------------------------------------------------------------


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_snapshots_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_snapshots(tmp_path / "nope.json") == []


def test_read_snapshots_malformed_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "snapshots.json"
    p.write_text("{not json", encoding="utf-8")
    assert read_snapshots(p) == []


def test_read_snapshots_sorts_by_date(tmp_path: Path) -> None:
    p = tmp_path / "snapshots.json"
    _write(
        p,
        [
            {"date": "2026-06-10", "portfolio_value": 2010.0},
            {"date": "2026-06-08", "portfolio_value": 2000.0},
            {"date": "2026-06-09", "portfolio_value": 1990.0},
        ],
    )
    out = read_snapshots(p)
    assert [r["date"] for r in out] == ["2026-06-08", "2026-06-09", "2026-06-10"]
    assert out[0]["portfolio_value"] == 2000.0


def test_read_snapshots_skips_rows_without_value(tmp_path: Path) -> None:
    p = tmp_path / "snapshots.json"
    _write(
        p,
        [
            {"date": "2026-06-08", "portfolio_value": 2000.0},
            {"date": "2026-06-09"},  # no portfolio_value -> dropped
            {"portfolio_value": 2010.0},  # no date -> dropped
        ],
    )
    out = read_snapshots(p)
    assert out == [{"date": "2026-06-08", "portfolio_value": 2000.0}]


# ---------------------------------------------------------------------------
# return_pct
# ---------------------------------------------------------------------------


def test_return_pct_empty_is_none() -> None:
    assert return_pct([]) is None


def test_return_pct_single_point_is_zero() -> None:
    assert return_pct([{"date": "2026-06-08", "portfolio_value": 2000.0}]) == 0.0


def test_return_pct_basic() -> None:
    snaps = [
        {"date": "2026-06-08", "portfolio_value": 2000.0},
        {"date": "2026-06-09", "portfolio_value": 2100.0},
    ]
    assert return_pct(snaps) == pytest.approx(5.0)


def test_return_pct_zero_base_is_none() -> None:
    snaps = [
        {"date": "2026-06-08", "portfolio_value": 0.0},
        {"date": "2026-06-09", "portfolio_value": 100.0},
    ]
    assert return_pct(snaps) is None


# ---------------------------------------------------------------------------
# build_manager_summary
# ---------------------------------------------------------------------------


def test_summary_no_data_has_not_run() -> None:
    s = build_manager_summary([], [], initial=2000.0)
    assert s["has_run"] is False
    assert s["manager_nav"] is None
    assert s["gap_pct"] is None


def test_summary_gap_is_manager_minus_baseline() -> None:
    manager = [
        {"date": "2026-06-08", "portfolio_value": 2000.0},
        {"date": "2026-06-20", "portfolio_value": 2200.0},  # +10%
    ]
    baseline = [
        {"date": "2026-06-08", "portfolio_value": 2000.0},
        {"date": "2026-06-20", "portfolio_value": 2080.0},  # +4%
    ]
    s = build_manager_summary(manager, baseline, initial=2000.0)
    assert s["has_run"] is True
    assert s["manager_nav"] == pytest.approx(2200.0)
    assert s["baseline_nav"] == pytest.approx(2080.0)
    assert s["manager_return_pct"] == pytest.approx(10.0)
    assert s["baseline_return_pct"] == pytest.approx(4.0)
    assert s["gap_pct"] == pytest.approx(6.0)


def test_summary_baseline_absent_gap_none_but_manager_present() -> None:
    manager = [
        {"date": "2026-06-08", "portfolio_value": 2000.0},
        {"date": "2026-06-20", "portfolio_value": 2200.0},
    ]
    s = build_manager_summary(manager, [], initial=2000.0)
    assert s["has_run"] is True
    assert s["manager_nav"] == pytest.approx(2200.0)
    assert s["baseline_nav"] is None
    assert s["gap_pct"] is None


# ---------------------------------------------------------------------------
# read_portfolio
# ---------------------------------------------------------------------------


def test_read_portfolio_missing_is_none(tmp_path: Path) -> None:
    assert read_portfolio(tmp_path / "portfolio.json") is None


def test_read_portfolio_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "portfolio.json"
    payload = {"cash": 1234.5, "positions": [{"ticker": "ASML.AS", "shares": 2}]}
    _write(p, payload)
    out = read_portfolio(p)
    assert out is not None
    assert out["cash"] == 1234.5
    assert out["positions"][0]["ticker"] == "ASML.AS"


# ---------------------------------------------------------------------------
# load_decisions
# ---------------------------------------------------------------------------


def test_load_decisions_missing_dir_is_empty(tmp_path: Path) -> None:
    assert load_decisions(tmp_path / "nope") == []


def test_load_decisions_excludes_resolved_and_sorts_desc(tmp_path: Path) -> None:
    review = tmp_path / "manager-review"
    review.mkdir()
    _write(review / "2026-06-10.json", {"date": "2026-06-10", "conviction": 8})
    _write(review / "2026-06-08.json", {"date": "2026-06-08", "conviction": None})
    _write(review / "resolved.json", [{"date": "2026-06-08", "ticker": "X"}])
    out = load_decisions(review)
    assert [d["date"] for d in out] == ["2026-06-10", "2026-06-08"]
    assert all("ticker" not in d for d in out)  # resolved.json excluded


# ---------------------------------------------------------------------------
# load_resolved
# ---------------------------------------------------------------------------


def test_load_resolved_missing_is_empty(tmp_path: Path) -> None:
    assert load_resolved(tmp_path / "resolved.json") == []


def test_load_resolved_returns_list(tmp_path: Path) -> None:
    p = tmp_path / "resolved.json"
    _write(
        p,
        [
            {
                "date": "2026-06-08",
                "ticker": "ASML.AS",
                "action": "buy",
                "realized_return_pct": 3.21,
                "alpha_vs_msci_pct": 1.04,
            }
        ],
    )
    out = load_resolved(p)
    assert len(out) == 1
    assert out[0]["ticker"] == "ASML.AS"
    assert out[0]["alpha_vs_msci_pct"] == 1.04


# ---------------------------------------------------------------------------
# book_paths
# ---------------------------------------------------------------------------


@pytest.mark.live_cast
def test_book_paths_review_dir_matches_legacy() -> None:
    """review_dir must be sourced from channels_prefix config, not id string-strip.

    The config-sourced prefix for 'the-manager' is 'manager', so the review dir
    must be byte-identical to the pre-SP2 legacy path data/orders/manager-review.
    """
    paths = book_paths("the-manager")
    assert paths["review_dir"] == get_config().orders_dir / "manager-review"
    assert (
        paths["resolved"]
        == get_config().orders_dir / "manager-review" / "resolved.json"
    )
    assert (
        paths["portfolio"]
        == get_config().portfolios_dir / "the-manager" / "portfolio.json"
    )
    assert (
        paths["snapshots"]
        == get_config().portfolios_dir / "the-manager" / "snapshots.json"
    )
