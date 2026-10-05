"""engine.market_calendar: buckets and bucket lag, on synthetic stores."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from engine.market_calendar import (
    MAX_BUCKET_LAG_DAYS,
    MIN_BUCKET_POPULATION,
    bucket_lag,
    bucket_of,
    is_stale,
    store_bucket,
)


def _write(store: Path, symbol: str, dates: list[date]) -> Path:
    path = store / f"{symbol}.jsonl"
    path.write_text(
        "".join(json.dumps({"date": d.isoformat(), "close": 1.0}) + "\n" for d in dates),
        encoding="utf-8",
    )
    return path


def _weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


ON = date(2026, 10, 2)  # a Friday
SESSIONS = _weekdays(date(2026, 9, 7), ON)


def test_bucket_of_matches_the_fetch_scripts_rules():
    assert bucket_of("AIR.PA") == ".PA"
    assert bucket_of("BT.A.L") == ".L"
    assert bucket_of("BTC-EUR") == "crypto"
    assert bucket_of("EURUSD=X") == "fx"
    assert bucket_of("AAPL") == ""
    assert bucket_of("HBAR-USD") == ""
    assert bucket_of("HBAR-USD", crypto=frozenset({"HBAR-USD"})) == "crypto"


def test_store_bucket_reads_crypto_from_the_pair_shape():
    assert store_bucket("HBAR-USD") == "crypto"  # not in the fee allowlist
    assert store_bucket("BRK-B") == ""  # a share class
    assert store_bucket("4GLD.DE") == ".DE"


def test_fetch_ohlcv_rates_holes_with_the_same_function():
    from scripts import fetch_ohlcv

    assert fetch_ohlcv.hole_bucket is bucket_of


@pytest.fixture
def us_store(tmp_path: Path) -> Path:
    for sym in ("A", "B", "C", "D", "E", "F"):
        _write(tmp_path, sym, SESSIONS)
    return tmp_path


@pytest.mark.parametrize(
    "as_of, lag",
    [
        (ON, 0),
        (date(2026, 10, 1), 1),
        (date(2026, 9, 30), 2),
        (date(2026, 9, 25), 5),  # Fri -> Fri: five sessions, the weekend counts none
        (date(2026, 10, 3), 0),  # a price newer than the bucket is not behind it
    ],
)
def test_lag_counts_the_buckets_trading_days(us_store, as_of, lag):
    result = bucket_lag("AAPL", as_of, ON, store=us_store)
    assert (result.bucket, result.reference, result.lag) == ("", ON, lag)


def test_is_stale_is_strictly_more_than_the_tolerance(us_store):
    assert MAX_BUCKET_LAG_DAYS == 1
    assert not is_stale("AAPL", date(2026, 10, 1), ON, store=us_store)
    assert is_stale("AAPL", date(2026, 9, 30), ON, store=us_store)


def test_a_date_only_a_minority_holds_is_not_a_bucket_day(us_store):
    """Two early writers of 10-05 out of six do not move the reference."""
    later = ON + timedelta(days=3)
    for sym in ("A", "B"):
        _write(us_store, sym, SESSIONS + [later])
    result = bucket_lag("AAPL", ON, later, store=us_store)
    assert (result.reference, result.lag) == (ON, 0)


def test_a_date_exactly_half_the_bucket_holds_is_a_bucket_day(us_store):
    """Regression (review of feat/stage1-asof-reads, 2026-10-03): the spec
    (plan 1.3) takes the newest date held by AT LEAST half the bucket. With a
    strict majority, three of six holding 10-02 left the reference at 10-01,
    so a price frozen at 09-30 read lag 1 and filled at a two-session-old
    close instead of refusing."""
    for sym in ("D", "E", "F"):
        _write(us_store, sym, SESSIONS[:-1])
    result = bucket_lag("AAPL", date(2026, 9, 30), ON, store=us_store)
    assert (result.reference, result.lag) == (ON, 2)
    assert is_stale("AAPL", date(2026, 9, 30), ON, store=us_store)


def test_rows_after_the_trade_date_are_ignored(us_store):
    """A replay of a past date reads the bucket as of that date."""
    result = bucket_lag("AAPL", date(2026, 9, 29), date(2026, 9, 30), store=us_store)
    assert (result.reference, result.lag) == (date(2026, 9, 30), 1)


def test_a_dead_file_does_not_dilute_the_majority(us_store):
    """MATIC-shaped: a file whose last row predates the window is not live.
    Seven dead files outnumber the six live ones, so counting them would leave
    no date with a majority and the rail would abstain."""
    for n in range(7):
        _write(us_store, f"DEAD{n}", _weekdays(date(2025, 3, 3), date(2025, 3, 24)))
    result = bucket_lag("AAPL", date(2026, 9, 30), ON, store=us_store)
    assert (result.population, result.lag) == (6, 2)


def test_a_small_bucket_abstains(tmp_path):
    for sym in ("A.F", "B.F"):
        _write(tmp_path, sym, SESSIONS)
    assert MIN_BUCKET_POPULATION > 2
    result = bucket_lag("A.F", date(2026, 9, 1), ON, store=tmp_path)
    assert result.lag is None
    assert not is_stale("A.F", date(2026, 9, 1), ON, store=tmp_path)


def test_a_rewritten_file_is_reread(us_store):
    """The per-file date cache keys on (mtime, size): a store write between
    two reads in one process must be seen."""
    assert bucket_lag("AAPL", ON, ON, store=us_store).reference == ON
    later = ON + timedelta(days=3)
    for sym in ("A", "B", "C", "D"):
        path = _write(us_store, sym, SESSIONS + [later])
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert bucket_lag("AAPL", ON, later, store=us_store).reference == later


def test_a_row_not_in_the_ingest_layout_is_still_read(tmp_path):
    for sym in ("A", "B", "C", "D", "E"):
        _write(tmp_path, sym, SESSIONS)
    (tmp_path / "F.jsonl").write_text(
        "".join(json.dumps({"close": 1.0, "date": d.isoformat()}) + "\n" for d in SESSIONS),
        encoding="utf-8",
    )
    assert bucket_lag("F", ON, ON, store=tmp_path).population == 6
