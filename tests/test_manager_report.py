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
    authored_status,
    book_paths,
    build_manager_summary,
    index_manager_inbox,
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


# ---------------------------------------------------------------------------
# index_manager_inbox + authored_status
#
# Regression cover for the 2026-07-18 confusion: the decision log rendered an
# armed-but-unfired conditional order identically to an executed market buy, so
# "BUY QQQ3.L €250" read as a fill when it had actually expired unfired. These
# tests pin the terminal-status join (inbox = filled/expired/cancelled, pending
# = armed) that the log now uses to disambiguate authored intent from outcome.
# ---------------------------------------------------------------------------


def _jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_index_manager_inbox_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert index_manager_inbox(tmp_path / "nope") == {}


def test_index_manager_inbox_scans_all_files_last_wins(tmp_path: Path) -> None:
    inbox = tmp_path / "manager-inbox"
    # A conditional authored 06-30 fills on 07-07 — its terminal record lives in
    # the file dated by the FILL day, not the authoring day. The index must scan
    # every *.jsonl (glob order is sorted, so a later file wins on a dup id).
    _jsonl(inbox / "2026-07-07.jsonl", [{"order_id": "a", "status": "filled"}])
    _jsonl(
        inbox / "2026-07-16.jsonl",
        [
            {"order_id": "b", "status": "rejected", "reason": "TRIGGER_EXPIRED"},
            {"order_id": "a", "status": "rejected", "reason": "later"},
        ],
    )
    idx = index_manager_inbox(inbox)
    assert idx["a"]["reason"] == "later"  # later file wins
    assert idx["b"]["status"] == "rejected"


def test_index_manager_inbox_tolerates_blank_and_bad_lines(tmp_path: Path) -> None:
    inbox = tmp_path / "manager-inbox"
    inbox.mkdir(parents=True)
    (inbox / "2026-07-07.jsonl").write_text(
        '\n{"order_id": "a", "status": "filled"}\n{bad json\n\n',
        encoding="utf-8",
    )
    idx = index_manager_inbox(inbox)
    assert set(idx) == {"a"}


def _decision(date_str: str, positions: list[dict]) -> dict:
    return {"date": date_str, "conviction": 6, "positions": positions}


def _outbox(outbox_dir: Path, date_str: str, orders: list[dict]) -> None:
    """Write a day's manager outbox — the broker's emitted-order record."""
    _jsonl(outbox_dir / f"{date_str}.jsonl", orders)


def test_authored_status_market_fill_is_filled(tmp_path: Path) -> None:
    outbox = tmp_path / "manager-outbox"
    _outbox(outbox, "2026-07-13", [{"order_id": "ord_x", "ticker": "NVDA"}])
    inbox_index = {"ord_x": {"status": "filled"}}
    d = _decision("2026-07-13", [{"ticker": "NVDA", "action": "BUY", "size_eur": 300}])
    out = authored_status(
        d,
        inbox_index=inbox_index,
        pending_dir=tmp_path / "manager-pending",
        outbox_dir=outbox,
    )
    assert out == ["filled"]


def test_authored_status_armed_pending_order(tmp_path: Path) -> None:
    outbox = tmp_path / "manager-outbox"
    _outbox(outbox, "2026-07-14", [{"order_id": "ord_btc", "ticker": "BTC-EUR"}])
    pending = tmp_path / "manager-pending"
    pending.mkdir(parents=True)
    (pending / "ord_btc.json").write_text("{}", encoding="utf-8")
    d = _decision(
        "2026-07-14",
        [
            {
                "ticker": "BTC-EUR",
                "action": "BUY",
                "size_eur": 300,
                "trigger": {"op": ">=", "level": 58000.0},
            }
        ],
    )
    out = authored_status(d, inbox_index={}, pending_dir=pending, outbox_dir=outbox)
    assert out == ["armed"]


def test_authored_status_expired_carries_date(tmp_path: Path) -> None:
    outbox = tmp_path / "manager-outbox"
    _outbox(outbox, "2026-07-02", [{"order_id": "ord_qqq", "ticker": "QQQ3.L"}])
    inbox_index = {
        "ord_qqq": {
            "status": "rejected",
            "reason": "TRIGGER_EXPIRED",
            "ts_filled": "2026-07-16T01:06:12.508888Z",
        }
    }
    d = _decision(
        "2026-07-02", [{"ticker": "QQQ3.L", "action": "BUY", "size_eur": 250}]
    )
    out = authored_status(
        d,
        inbox_index=inbox_index,
        pending_dir=tmp_path / "manager-pending",
        outbox_dir=outbox,
    )
    assert out == ["expired 07-16"]


def test_authored_status_cancelled(tmp_path: Path) -> None:
    outbox = tmp_path / "manager-outbox"
    _outbox(outbox, "2026-07-05", [{"order_id": "ord_foo", "ticker": "FOO"}])
    inbox_index = {
        "ord_foo": {
            "status": "rejected",
            "reason": "CANCELLED_BY_AGENT",
            "ts_filled": "2026-07-06T20:00:00Z",
        }
    }
    d = _decision("2026-07-05", [{"ticker": "FOO", "action": "BUY", "size_eur": 100}])
    out = authored_status(
        d,
        inbox_index=inbox_index,
        pending_dir=tmp_path / "manager-pending",
        outbox_dir=outbox,
    )
    assert out == ["cancelled 07-06"]


def test_authored_status_other_rejection_names_reason(tmp_path: Path) -> None:
    outbox = tmp_path / "manager-outbox"
    _outbox(outbox, "2026-07-05", [{"order_id": "ord_foo", "ticker": "FOO"}])
    inbox_index = {"ord_foo": {"status": "rejected", "reason": "NOTIONAL_CAP_EXCEEDED"}}
    d = _decision("2026-07-05", [{"ticker": "FOO", "action": "BUY", "size_eur": 100}])
    out = authored_status(
        d,
        inbox_index=inbox_index,
        pending_dir=tmp_path / "manager-pending",
        outbox_dir=outbox,
    )
    assert out == ["rejected: NOTIONAL_CAP_EXCEEDED"]


def test_authored_status_unknown_when_in_flight(tmp_path: Path) -> None:
    # An emitted order with no inbox record and no pending file yet — in flight.
    outbox = tmp_path / "manager-outbox"
    _outbox(outbox, "2026-07-05", [{"order_id": "ord_foo", "ticker": "FOO"}])
    d = _decision("2026-07-05", [{"ticker": "FOO", "action": "BUY", "size_eur": 100}])
    out = authored_status(
        d,
        inbox_index={},
        pending_dir=tmp_path / "manager-pending",
        outbox_dir=outbox,
    )
    assert out == [""]


def test_authored_status_bad_date_degrades_gracefully(tmp_path: Path) -> None:
    d = _decision("not-a-date", [{"ticker": "FOO", "action": "BUY", "size_eur": 100}])
    out = authored_status(
        d,
        inbox_index={},
        pending_dir=tmp_path / "manager-pending",
        outbox_dir=tmp_path / "manager-outbox",
    )
    assert out == [""]


def test_authored_status_skipped_position_does_not_shift_labels(
    tmp_path: Path,
) -> None:
    """The core regression: a position the broker SKIPS (no store price, HOLD,
    zero size) must not shift the status of later positions.

    The Manager authors [XYZ, QQQ3.L]. XYZ has no store price, so
    manager_decision_to_orders skips it WITHOUT bumping seq — only QQQ3.L is
    emitted, as ord_...001. Joining by position index would attribute _001's
    fill to XYZ and leave QQQ3.L unresolved (the exact mislabeling this feature
    exists to prevent). Joining via the outbox's ticker->order_id map keeps each
    label on its own ticker.
    """
    outbox = tmp_path / "manager-outbox"
    # Only QQQ3.L reached the broker; XYZ was filtered before an id was assigned.
    _outbox(
        outbox,
        "2026-07-20",
        [{"order_id": "ord_2026-07-20_the-manager_001", "ticker": "QQQ3.L"}],
    )
    inbox_index = {"ord_2026-07-20_the-manager_001": {"status": "filled"}}
    d = _decision(
        "2026-07-20",
        [
            {"ticker": "XYZ", "action": "BUY", "size_eur": 200},
            {"ticker": "QQQ3.L", "action": "BUY", "size_eur": 250},
        ],
    )
    out = authored_status(
        d,
        inbox_index=inbox_index,
        pending_dir=tmp_path / "manager-pending",
        outbox_dir=outbox,
    )
    assert out == ["", "filled"]  # NOT ["filled", ""]


def test_authored_status_repeated_ticker_consumes_in_emit_order(
    tmp_path: Path,
) -> None:
    outbox = tmp_path / "manager-outbox"
    _outbox(
        outbox,
        "2026-07-20",
        [
            {"order_id": "ord_a", "ticker": "FOO"},
            {"order_id": "ord_b", "ticker": "FOO"},
        ],
    )
    inbox_index = {"ord_a": {"status": "filled"}}
    pending = tmp_path / "manager-pending"
    pending.mkdir(parents=True)
    (pending / "ord_b.json").write_text("{}", encoding="utf-8")
    d = _decision(
        "2026-07-20",
        [
            {"ticker": "FOO", "action": "BUY", "size_eur": 100},
            {"ticker": "FOO", "action": "BUY", "size_eur": 100},
        ],
    )
    out = authored_status(
        d,
        inbox_index=inbox_index,
        pending_dir=pending,
        outbox_dir=outbox,
    )
    assert out == ["filled", "armed"]


def test_authored_status_missing_outbox_is_unknown(tmp_path: Path) -> None:
    inbox_index = {"ord_x": {"status": "filled"}}
    d = _decision("2026-07-13", [{"ticker": "NVDA", "action": "BUY", "size_eur": 300}])
    out = authored_status(
        d,
        inbox_index=inbox_index,
        pending_dir=tmp_path / "manager-pending",
        outbox_dir=tmp_path / "manager-outbox",  # dir absent
    )
    assert out == [""]
