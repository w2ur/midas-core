"""Branch coverage for scripts.fetch_ohlcv.main() — the nightly, unattended cron
entry point.

Task 3 review flagged that only the crypto/revise branch had ever been run (a
one-off manual probe against BTC-EUR). Nothing exercised the equity/FX
pure-append path or the skip path through main()'s actual control flow.
That gap matters most for the invariant that protects the store: an equity
or FX symbol must NEVER get `revise_from` set, or ~1,100 committed files get
silently rewritten every night.

All tests here drive `main()` end-to-end with `_fetch_symbol` and
`_fetch_ticker_info` monkeypatched to synthetic, network-free responses —
no yfinance call is ever made. `_fetch_symbol`'s replacement filters a fixed
per-symbol time series down to the requested `[start, end]` window, mirroring
real yfinance's window semantics, so a wrongly-widened window (e.g. the
crypto revision gate leaking onto a non-crypto symbol) is observable as a
byte-level diff, not just a silently-skipped branch.
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


def _write_raw(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["fetch_ohlcv.py", *argv])
    return fo.main()


# --- equity/FX pure-append (the load-bearing invariant) --------------------


def test_equity_stale_by_one_day_is_pure_append_byte_identical(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-crypto symbol must never get revise_from set, so every
    previously-stored row survives byte-for-byte. The synthetic response for
    "yesterday" deliberately differs from the stored value — if the equity
    path ever leaked revise_from, this test would catch a real byte change,
    not just an unexercised branch."""
    today = date.today()
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)

    path = get_config().ohlcv_dir / "AAPL.jsonl"
    day_before_line = _tight_line(day_before.isoformat(), 180.0)
    yesterday_line = _tight_line(yesterday.isoformat(), 185.0)
    _write_raw(path, [day_before_line, yesterday_line])
    before = path.read_text(encoding="utf-8")

    frames = {
        "AAPL": {
            # Differs from the stored 185.0 — would show up as a revision
            # if the equity path ever set revise_from.
            yesterday.isoformat(): [1, 2, 0.5, 999.99, 999.99, 1],
            today.isoformat(): [1, 2, 0.5, 200.12, 200.12, 1],
        }
    }
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "AAPL"])
    assert rc == 0

    text = path.read_text(encoding="utf-8")
    assert text.startswith(before)  # nothing about the pre-existing bytes moved
    lines = text.splitlines()
    assert lines == [
        day_before_line,
        yesterday_line,  # unchanged — no revision happened
        json.dumps(
            {
                "date": today.isoformat(),
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 200.12,
                "adj_close": 200.12,
                "volume": 1,
            }
        ),
    ]


# --- crypto revision (the mirror case — proves the gate discriminates) -----


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
