"""Branch coverage for scripts.fetch_ohlcv.main() — the nightly, unattended cron
entry point.

Every symbol now runs with a 1-day revision window — the trailing stored bar is
re-requested and replaced when its value changed. The two halves of that
behaviour are pinned separately:

- **No churn.** A cash equity or ETF bar IS final at the 22:30 UTC fetch
  (measured: SPY and AAPL, 0 of 23 trailing bars drifted), so the re-fetch is
  identical, `merge_rows` finds nothing to replace, and every stored byte
  survives. This is why widening the window to ~1,100 committed files is safe.
- **Revision.** Crypto (24/7) and commodity futures (`=F`, whose next Globex
  session has already opened at 22:30 UTC — GC=F drifted on 13 of the last 22
  bars, worst +2.865%) are still forming when written, and must be corrected.

All tests here drive `main()` end-to-end with `_fetch_symbol` and
`_fetch_ticker_info` monkeypatched to synthetic, network-free responses —
no yfinance call is ever made. `_fetch_symbol`'s replacement filters a fixed
per-symbol time series down to the requested `[start, end]` window, mirroring
real yfinance's window semantics, so a wrongly-widened or wrongly-narrowed
window is observable as a byte-level diff, not just a silently-skipped branch.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

import scripts.fetch_ohlcv as fo
from engine.config import get_config

_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _yf_frame(rows: dict[str, list]) -> pd.DataFrame:
    """Build a synthetic yfinance-shaped OHLCV frame (see tests/test_ohlcv_ingest.py)."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in rows], name="Date")
    data = [rows[d] for d in rows]
    return pd.DataFrame(data, index=idx, columns=pd.Index(_FIELDS))


def _make_fake_fetch_symbol(frames: dict[str, dict[str, list]]):
    """Build a `_fetch_symbol` replacement over a fixed per-symbol series.

    Filters each symbol's series down to the `[start, end]` window main()
    actually requests, mirroring yfinance's window semantics. A window that
    is wrongly widened (or wrongly narrowed) surfaces as different rows
    coming back, not as an untested no-op.
    """

    def fake(symbol: str, start: date, end: date) -> pd.DataFrame | None:
        series = frames.get(symbol)
        if not series:
            return None
        windowed = {
            d: v for d, v in series.items() if start <= date.fromisoformat(d) <= end
        }
        if not windowed:
            return None
        return _yf_frame(windowed)

    return fake


def _tight_line(d: str, close: float) -> str:
    """A store line with non-default JSON spacing (no space after ':'/',').

    A re-serializing implementation would normalize this back to spaced
    form via json.dumps' default separators — same control as
    tests/test_ohlcv_ingest.py's mixed-rewrite test. Proves an "untouched"
    row was left alone at the byte level, not merely re-derived to an
    equal-looking value.
    """
    return (
        f'{{"date":"{d}","open":{close},"high":{close},"low":{close},'
        f'"close":{close},"adj_close":{close},"volume":100}}'
    )


def _canonical_line(d: str, ohlcv: list) -> str:
    """The exact line the store writes for a frame row.

    `row_to_record` coerces every cell through `safe_float`/`safe_int`, so an
    integer 1 in the frame lands as `1.0` on disk. Building the expectation
    through the same coercion keeps "unchanged" meaning byte-identical.
    """
    o, h, low, c, adj, v = ohlcv
    return json.dumps(
        {
            "date": d,
            "open": float(o),
            "high": float(h),
            "low": float(low),
            "close": float(c),
            "adj_close": float(adj),
            "volume": int(v),
        }
    )


def _write_raw(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["fetch_ohlcv.py", *argv])
    return fo.main()


# --- equity no-churn (why the universal window is safe) --------------------


def test_equity_unchanged_trailing_bar_stays_byte_identical(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An equity now gets the revision window like everything else, and it
    still costs nothing: a cash-equity bar is final at fetch time, so the
    re-fetched trailing row equals the stored one and `merge_rows` finds
    nothing to replace. Every pre-existing byte survives — including the
    non-canonically-spaced row before the window, which proves an untouched
    row is carried across the rewrite verbatim rather than re-serialized."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)

    yesterday_bar = [1, 2, 0.5, 185.0, 185.0, 1]
    today_bar = [1, 2, 0.5, 200.12, 200.12, 1]

    path = get_config().ohlcv_dir / "AAPL.jsonl"
    day_before_line = _tight_line(day_before.isoformat(), 180.0)
    # Stored in the same form the fetch would produce → the re-fetch is a no-op.
    yesterday_line = _canonical_line(yesterday.isoformat(), yesterday_bar)
    _write_raw(path, [day_before_line, yesterday_line])
    before = path.read_text(encoding="utf-8")

    frames = {
        "AAPL": {
            # An equity close does not move after 22:30 UTC — measured over 23
            # trading days, SPY and AAPL drifted on 0 of them.
            yesterday.isoformat(): yesterday_bar,
            today.isoformat(): today_bar,
        }
    }
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "AAPL"])
    assert rc == 0

    text = path.read_text(encoding="utf-8")
    assert text.startswith(before)  # nothing about the pre-existing bytes moved
    assert text.splitlines() == [
        day_before_line,
        yesterday_line,  # unchanged — the window found nothing to revise
        _canonical_line(today.isoformat(), today_bar),
    ]


# --- futures + crypto revision (the bars that are NOT final at fetch) ------


def test_futures_stale_by_one_day_revises_trailing_row(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: 7777b4cc3 — the revision window was gated to crypto, so a
    commodity-futures bar was frozen at whatever partial value the 22:30 UTC
    cron happened to catch.

    `GC=F` at 22:30 UTC is mid-Globex: the next session has already opened, so
    the bar is still forming. Measured over the store's 23 most recent bars it
    differed from its final value on 13 of the 22 days the two records share
    (worst +2.865% on 2026-07-29); `PL=F` (+3.365%) and `CL=F` (+2.516%) behave
    the same way. Worse, the corrected skip condition means every run now writes
    a same-day partial — with the crypto gate in place futures went from ~61%
    wrong to ~100% wrong. The trailing row must be replaced by its final
    value."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)

    path = get_config().ohlcv_dir / "GC=F.jsonl"
    day_before_line = _tight_line(day_before.isoformat(), 3310.0)
    partial_yesterday_line = _canonical_line(
        yesterday.isoformat(), [3320.0, 3355.0, 3315.0, 3341.6, 3341.6, 1200]
    )
    _write_raw(path, [day_before_line, partial_yesterday_line])

    final_yesterday_bar = [3320.0, 3372.4, 3315.0, 3437.3, 3437.3, 1850]
    today_bar = [3437.3, 3460.0, 3430.0, 3451.9, 3451.9, 900]
    frames = {
        "GC=F": {
            yesterday.isoformat(): final_yesterday_bar,
            today.isoformat(): today_bar,
        }
    }
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "GC=F"])
    assert rc == 0

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        day_before_line,  # before the window — byte-identical
        _canonical_line(yesterday.isoformat(), final_yesterday_bar),  # revised
        _canonical_line(today.isoformat(), today_bar),  # appended
    ]
    assert json.loads(lines[1])["close"] == 3437.3  # not the frozen 3341.6


def test_crypto_stale_by_one_day_revises_trailing_row(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = date.today()
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)

    path = get_config().ohlcv_dir / "BTC-EUR.jsonl"
    day_before_line = _tight_line(day_before.isoformat(), 54000.0)
    stale_yesterday_line = json.dumps(
        {
            "date": yesterday.isoformat(),
            "open": 55151.82,
            "high": 55893.35,
            "low": 55010.56,
            "close": 55649.74,
            "adj_close": 55649.74,
            "volume": 20443908096,
        }
    )
    _write_raw(path, [day_before_line, stale_yesterday_line])

    frames = {
        "BTC-EUR": {
            yesterday.isoformat(): [1, 2, 0.5, 55545.09, 55545.09, 1],
            today.isoformat(): [1, 2, 0.5, 56061.41, 56061.41, 1],
        }
    }
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "BTC-EUR"])
    assert rc == 0

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0] == day_before_line  # untouched row stays byte-identical
    assert json.loads(lines[1])["close"] == 55545.09  # trailing row revised
    assert json.loads(lines[2])["close"] == 56061.41  # new row appended


# --- skip path ---------------------------------------------------------


def test_skip_path_no_fetch_no_mutation_when_store_covers_today(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = date.today()
    path = get_config().ohlcv_dir / "AAPL.jsonl"
    today_line = _tight_line(today.isoformat(), 200.0)
    _write_raw(path, [today_line])
    before = path.read_text(encoding="utf-8")

    def fake(symbol: str, start: date, end: date) -> pd.DataFrame | None:
        raise AssertionError("must not fetch when the store already covers today")

    monkeypatch.setattr(fo, "_fetch_symbol", fake)
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "AAPL"])
    assert rc == 0
    assert path.read_text(encoding="utf-8") == before


# --- backfill (cheap, lower risk — operator-invoked, not on the cron) ------


def test_backfill_refetches_full_window_and_keeps_pre_existing_row(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    today = date.today()
    old = today - timedelta(days=4)
    path = get_config().ohlcv_dir / "AAPL.jsonl"
    _write_raw(path, [_tight_line(old.isoformat(), 100.0)])

    frames = {"AAPL": {today.isoformat(): [1, 2, 0.5, 210.0, 210.0, 1]}}
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(
        monkeypatch, ["--symbols", "AAPL", "--backfill", "--history-days", "5"]
    )
    assert rc == 0

    dates = [
        json.loads(line)["date"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert old.isoformat() in dates  # append-only: pre-existing row survives
    assert today.isoformat() in dates
