"""Branch coverage for scripts.fetch_ohlcv.main() — the nightly, unattended cron
entry point.

Since 2026-08-12 `main()` fetches through YESTERDAY, never today, so no forming
bar is ever stored (`_fetch_end()` below is the fixtures' anchor for that).
Every symbol still runs with a 1-day revision window — the trailing stored bar
is re-requested and replaced when its value changed — but what it catches is
the vendor revising an already-complete bar, not a partial one this run wrote
itself. The two halves of that behaviour are pinned separately:

- **No churn.** A cash equity or ETF close does not move once its session has
  ended (measured: SPY and AAPL, 0 of 23 trailing bars drifted), so the
  re-fetch is identical, `merge_rows` finds nothing to replace, and every
  stored byte survives. This is why widening the window to ~1,100 committed
  files is safe.
- **Revision.** Commodity futures (`=F`) and crypto do get revised by the
  vendor after the fact — GC=F drifted on 13 of the last 22 bars, worst
  +2.865% — and must be corrected. (Those figures were measured under the
  22:30 UTC schedule, when the same bars were also still forming at fetch
  time; that second cause is gone, the vendor revisions are not.)

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
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import scripts.fetch_ohlcv as fo
from engine.corporate_actions import CorporateAction
from engine.ohlcv_ingest import QuarantinedRow
from engine.config import get_config
from engine.portfolio import PortfolioManager
from engine.types import Trade

_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _yf_frame(rows: dict[str, list]) -> pd.DataFrame:
    """Build a synthetic yfinance-shaped OHLCV frame (see tests/test_ohlcv_ingest.py)."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in rows], name="Date")
    data = [rows[d] for d in rows]
    return pd.DataFrame(data, index=idx, columns=pd.Index(_FIELDS))


def _fetch_end() -> date:
    """The newest day a run asks the vendor for: yesterday, never today.

    `scripts.fetch_ohlcv.main` sets `end = date.today() - 1` (2026-08-12) so
    that every bar it stores is a COMPLETE daily bar — asking for today serves
    a partial one on any 24/7 instrument. Fixtures anchor here rather than on
    `date.today()` because `_make_fake_fetch_symbol` honours the requested
    window: a fixture row dated today is simply never requested, which turns a
    revision test into a silent no-op instead of a failure.
    """
    return date.today() - timedelta(days=1)


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
    end = _fetch_end()
    prev = end - timedelta(days=1)
    prev2 = end - timedelta(days=2)

    prev_bar = [1, 2, 0.5, 185.0, 185.0, 1]
    end_bar = [1, 2, 0.5, 200.12, 200.12, 1]

    path = get_config().ohlcv_dir / "AAPL.jsonl"
    prev2_line = _tight_line(prev2.isoformat(), 180.0)
    # Stored in the same form the fetch would produce → the re-fetch is a no-op.
    prev_line = _canonical_line(prev.isoformat(), prev_bar)
    _write_raw(path, [prev2_line, prev_line])
    before = path.read_text(encoding="utf-8")

    frames = {
        "AAPL": {
            # A cash-equity close does not move once the session has ended —
            # measured over 23 trading days, SPY and AAPL drifted on 0 of them.
            prev.isoformat(): prev_bar,
            end.isoformat(): end_bar,
        }
    }
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "AAPL"])
    assert rc == 0

    text = path.read_text(encoding="utf-8")
    assert text.startswith(before)  # nothing about the pre-existing bytes moved
    assert text.splitlines() == [
        prev2_line,
        prev_line,  # unchanged — the window found nothing to revise
        _canonical_line(end.isoformat(), end_bar),
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
    the same way. Since 2026-08-12 `end` is yesterday, so the stored bar is no
    longer a partial one this run wrote itself — but the vendor still revises a
    complete futures bar, which is exactly what the window now exists to catch.
    The trailing row must be replaced by its final value."""
    end = _fetch_end()
    prev = end - timedelta(days=1)
    prev2 = end - timedelta(days=2)

    path = get_config().ohlcv_dir / "GC=F.jsonl"
    prev2_line = _tight_line(prev2.isoformat(), 3310.0)
    superseded_prev_line = _canonical_line(
        prev.isoformat(), [3320.0, 3355.0, 3315.0, 3341.6, 3341.6, 1200]
    )
    _write_raw(path, [prev2_line, superseded_prev_line])

    revised_prev_bar = [3320.0, 3372.4, 3315.0, 3437.3, 3437.3, 1850]
    end_bar = [3437.3, 3460.0, 3430.0, 3451.9, 3451.9, 900]
    frames = {
        "GC=F": {
            prev.isoformat(): revised_prev_bar,
            end.isoformat(): end_bar,
        }
    }
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "GC=F"])
    assert rc == 0

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        prev2_line,  # before the window — byte-identical
        _canonical_line(prev.isoformat(), revised_prev_bar),  # revised
        _canonical_line(end.isoformat(), end_bar),  # appended
    ]
    assert json.loads(lines[1])["close"] == 3437.3  # not the superseded 3341.6


def test_crypto_stale_by_one_day_revises_trailing_row(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    end = _fetch_end()
    prev = end - timedelta(days=1)
    prev2 = end - timedelta(days=2)

    path = get_config().ohlcv_dir / "BTC-EUR.jsonl"
    prev2_line = _tight_line(prev2.isoformat(), 54000.0)
    stale_prev_line = json.dumps(
        {
            "date": prev.isoformat(),
            "open": 55151.82,
            "high": 55893.35,
            "low": 55010.56,
            "close": 55649.74,
            "adj_close": 55649.74,
            "volume": 20443908096,
        }
    )
    _write_raw(path, [prev2_line, stale_prev_line])

    frames = {
        "BTC-EUR": {
            prev.isoformat(): [1, 2, 0.5, 55545.09, 55545.09, 1],
            end.isoformat(): [1, 2, 0.5, 56061.41, 56061.41, 1],
        }
    }
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "BTC-EUR"])
    assert rc == 0

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0] == prev2_line  # untouched row stays byte-identical
    assert json.loads(lines[1])["close"] == 55545.09  # trailing row revised
    assert json.loads(lines[2])["close"] == 56061.41  # new row appended


# --- skip path ---------------------------------------------------------


def test_the_run_never_asks_the_vendor_for_todays_bar(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`end` is yesterday, so no forming bar can enter the store (2026-08-12).

    Pinned directly rather than left to the fixtures that depend on it. Asking
    for today costs nothing on a cash market — the day has not opened — but a
    24/7 instrument is served a partial bar the moment the UTC day opens
    (measured: Yahoo returns a same-day BTC-USD close at 07:49 UTC). The 20:00
    session publishes whatever the store holds and `add_snapshot` freezes it,
    so a partial captured at 06:00 becomes a permanent published mark.
    """
    requested: list[date] = []

    def fake(symbol: str, start: date, end: date) -> pd.DataFrame | None:
        requested.append(end)
        return None

    monkeypatch.setattr(fo, "_fetch_symbol", fake)
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    _run_main(monkeypatch, ["--symbols", "BTC-EUR"])

    assert requested, "the run asked the vendor for nothing at all"
    assert requested[0] == date.today() - timedelta(days=1)
    assert requested[0] != date.today()


def test_skip_path_no_fetch_no_mutation_when_store_covers_end(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The skip is keyed on `end`, not on the calendar day.

    Since 2026-08-12 `end` is yesterday, so this is the ordinary
    second-run-in-one-day case: the store already holds the newest day this
    run would ask for, and nothing is requested.
    """
    end = _fetch_end()
    path = get_config().ohlcv_dir / "AAPL.jsonl"
    end_line = _tight_line(end.isoformat(), 200.0)
    _write_raw(path, [end_line])
    before = path.read_text(encoding="utf-8")

    def fake(symbol: str, start: date, end: date) -> pd.DataFrame | None:
        raise AssertionError("must not fetch when the store already covers end")

    monkeypatch.setattr(fo, "_fetch_symbol", fake)
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "AAPL"])
    assert rc == 0
    assert path.read_text(encoding="utf-8") == before


# --- backfill (cheap, lower risk — operator-invoked, not on the cron) ------


def test_backfill_refetches_full_window_and_keeps_pre_existing_row(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    end = _fetch_end()
    old = end - timedelta(days=4)
    path = get_config().ohlcv_dir / "AAPL.jsonl"
    _write_raw(path, [_tight_line(old.isoformat(), 100.0)])

    frames = {"AAPL": {end.isoformat(): [1, 2, 0.5, 210.0, 210.0, 1]}}
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
    assert end.isoformat() in dates


# --- resweep (rewrites committed history — full-window revision) -----------


def test_resweep_revises_a_wrong_row_deep_in_history(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: e98ea8cf5 — partial bars written on the day they formed were
    frozen throughout the store's history. --backfill re-fetched the window but
    wrote append-only, so it could not correct them; only the trailing day was
    revisable."""
    path = get_config().ohlcv_dir / "GC=F.jsonl"

    def bar(c: float) -> list:
        return [c, c, c, c, c, 1]

    _write_raw(
        path,
        [
            _canonical_line("2026-06-10", bar(3300.0)),
            _canonical_line(
                "2026-06-11", bar(9999.9)
            ),  # frozen partial, deep in history
            _canonical_line("2026-06-12", bar(3350.0)),
        ],
    )

    frames = {
        "GC=F": {
            "2026-06-10": bar(3300.0),
            "2026-06-11": bar(3311.1),
            "2026-06-12": bar(3350.0),
        }
    }
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--symbols", "GC=F", "--resweep"])
    assert rc == 0

    closes = [
        json.loads(line)["close"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert closes == [3300.0, 3311.1, 3350.0]


def test_resweep_requires_an_explicit_symbol_list(
    monkeypatch: pytest.MonkeyPatch,
    midas_data_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--resweep rewrites committed history; it must never run on the whole
    universe by accident.

    Pins the guard's own message (not just "some SystemExit fired") so this
    test cannot pass if the two --resweep guards' messages were swapped."""
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, ["--resweep"])
    assert excinfo.value.code == 2
    assert "--resweep requires an explicit --symbols list" in capsys.readouterr().err


def test_resweep_and_backfill_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    midas_data_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--resweep and --backfill both force a full-window fetch; combining them
    is nonsensical and must be rejected rather than silently picking one.

    Pins the guard's own message so this test cannot pass if the two
    --resweep guards' messages were swapped."""
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, ["--resweep", "--backfill", "--symbols", "GC=F"])
    assert excinfo.value.code == 2
    assert "--resweep and --backfill are mutually exclusive" in capsys.readouterr().err


def test_neither_resweep_nor_symbols_proceeds_normally(
    monkeypatch: pytest.MonkeyPatch, midas_data_root: Path
) -> None:
    """Companion to the two guard tests above: with neither --resweep nor
    --symbols set, main() must proceed rather than exit. Nothing else in the
    suite calls main() without --symbols, so without this test a guard
    mis-scoped from `if args.resweep and not args.symbols:` down to
    `if not args.symbols:` (firing unconditionally) would pass both guard
    tests above and go undetected.

    `_all_symbols` is monkeypatched to a fixed list and `--dry-run` used so
    this stays hermetic — no universe-resolver network fallback, no fetch."""
    monkeypatch.setattr(fo, "_all_symbols", lambda: ["AAPL"])
    rc = _run_main(monkeypatch, ["--dry-run"])
    assert rc == 0


# --- corporate actions (split detection wired into --resweep) --------------


def test_resweep_detects_a_split_and_adjusts_the_holding_agents_position(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: nothing in the engine handled corporate actions — a
    position held through a split kept its pre-split share count against a
    post-split price, silently mis-valuing the book by the split ratio.

    --resweep is the one fetch path that re-requests a full historical
    window against already-committed rows, which is exactly the comparison
    detect_split needs: every pre-split row divided by the SAME ratio,
    unlike ordinary drift's scattered-ratio signature (see
    engine.corporate_actions). A holder's position must come out adjusted;
    an unrelated agent's book must be untouched."""
    start = date(2026, 6, 1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(20)]

    def bar(c: float) -> list:
        return [c, c, c, c, c, 1]

    path = get_config().ohlcv_dir / "CRWD.jsonl"
    # Pre-split store: raw close is 4x what a fresh fetch now returns — Yahoo
    # retroactively rewrites raw close (not just adj_close) after a split.
    # The last 5 rows are post-split: the nightly cron appended them at
    # correct prices after the split, so the store already agrees there.
    # detect_split requires that transition to exist (a wholly-drifted
    # overlap fails closed — it is also what a units mismatch looks like).
    fresh_closes = [100.0 + i for i in range(20)]
    stale_closes = [c * 4.0 for c in fresh_closes[:15]] + fresh_closes[15:]
    _write_raw(path, [_canonical_line(d, bar(c)) for d, c in zip(dates, stale_closes)])

    frames = {"CRWD": {d: bar(c) for d, c in zip(dates, fresh_closes)}}
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    portfolios = PortfolioManager(base_dir=get_config().portfolios_dir)
    portfolios.initialize("holder", initial_capital=5000.0)
    portfolios.apply_trade(
        "holder",
        Trade(
            id="t1",
            timestamp=datetime(2026, 5, 20, 20, 0),
            action="BUY",
            ticker="CRWD",
            shares=3.0,
            price=400.0,
            total=1200.0,
            fees=0.0,
            reasoning="Pre-split buy",
        ),
    )
    portfolios.initialize("bystander", initial_capital=5000.0)
    portfolios.apply_trade(
        "bystander",
        Trade(
            id="t2",
            timestamp=datetime(2026, 5, 20, 20, 0),
            action="BUY",
            ticker="AAPL",
            shares=5.0,
            price=100.0,
            total=500.0,
            fees=0.0,
            reasoning="Unrelated position",
        ),
    )

    rc = _run_main(monkeypatch, ["--symbols", "CRWD", "--resweep"])
    assert rc == 0

    holder = portfolios.load("holder")
    position = next(p for p in holder.positions if p.ticker == "CRWD")
    assert position.shares == pytest.approx(12.0)
    assert position.avg_cost == pytest.approx(100.0)
    assert position.shares * position.avg_cost == pytest.approx(1200.0)

    bystander = portfolios.load("bystander")
    aapl = next(p for p in bystander.positions if p.ticker == "AAPL")
    assert aapl.shares == 5.0
    assert aapl.avg_cost == 100.0

    # The store itself is still correctly revised regardless of the split
    # adjustment running alongside it.
    closes = [
        json.loads(line)["close"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert closes == fresh_closes


def test_resweep_does_not_adjust_ordinary_scattered_drift(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Class-D drift (ALV.DE/BMW.DE's real signature: many rows, each at its
    own distinct ratio) must never trigger a position adjustment — a false
    positive here would silently multiply a real position by a bogus ratio.

    The drift is followed by rows the store already agrees on, so the run
    reaches the detector's real gates rather than being turned away early by
    the "no transition" fail-closed rule."""
    start = date(2026, 6, 1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(20)]

    def bar(c: float) -> list:
        return [c, c, c, c, c, 1]

    path = get_config().ohlcv_dir / "ALV.DE.jsonl"
    fresh_closes = [100.0] * 20
    ratios = [0.960 + 0.002 * i for i in range(15)] + [1.0] * 5
    stale_closes = [c * r for c, r in zip(fresh_closes, ratios)]
    _write_raw(path, [_canonical_line(d, bar(c)) for d, c in zip(dates, stale_closes)])

    frames = {"ALV.DE": {d: bar(c) for d, c in zip(dates, fresh_closes)}}
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    portfolios = PortfolioManager(base_dir=get_config().portfolios_dir)
    portfolios.initialize("holder", initial_capital=5000.0)
    portfolios.apply_trade(
        "holder",
        Trade(
            id="t1",
            timestamp=datetime(2026, 5, 20, 20, 0),
            action="BUY",
            ticker="ALV.DE",
            shares=3.0,
            price=95.0,
            total=285.0,
            fees=0.0,
            reasoning="Drift-affected buy",
        ),
    )

    rc = _run_main(monkeypatch, ["--symbols", "ALV.DE", "--resweep"])
    assert rc == 0

    holder = portfolios.load("holder")
    position = next(p for p in holder.positions if p.ticker == "ALV.DE")
    assert position.shares == 3.0
    assert position.avg_cost == 95.0


# --- --resweep-held (the scheduled trigger for split detection) ------------


def test_resweep_held_and_symbols_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    midas_data_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--resweep-held resolves its own symbol list from held positions; combining
    it with an explicit --symbols would make one of the two pointless."""
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, ["--resweep-held", "--symbols", "AAPL"])
    assert excinfo.value.code == 2
    assert (
        "--resweep-held resolves its own --symbols list — do not pass both"
        in capsys.readouterr().err
    )


def test_resweep_held_and_backfill_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    midas_data_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, ["--resweep-held", "--backfill"])
    assert excinfo.value.code == 2
    assert (
        "--resweep-held and --backfill are mutually exclusive"
        in capsys.readouterr().err
    )


def test_resweep_held_does_not_require_an_explicit_symbol_list(
    monkeypatch: pytest.MonkeyPatch, midas_data_root: Path
) -> None:
    """Companion to the plain --resweep guard: --resweep-held is the one way
    to run a full-window resweep WITHOUT an explicit --symbols list, since it
    resolves its own from _collect_holdings()."""
    monkeypatch.setattr(fo, "_collect_holdings", lambda: {"AAPL"})
    # The symbol must actually resolve: since W2.5 a run where every symbol
    # comes back empty exits 1, so stubbing `_fetch_symbol` to None would test
    # the failure-rate guard instead of the symbol-list resolution this is about.
    monkeypatch.setattr(
        fo,
        "_fetch_symbol",
        _make_fake_fetch_symbol({"AAPL": {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]}}),
    )
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)
    rc = _run_main(monkeypatch, ["--resweep-held"])
    assert rc == 0


def test_resweep_held_resolves_symbols_from_collect_holdings(
    monkeypatch: pytest.MonkeyPatch, midas_data_root: Path, capsys
) -> None:
    """Reuses _collect_holdings() rather than reimplementing it — pinned by
    monkeypatching that exact function and asserting its output drives the
    resolved symbol list (via --dry-run, so nothing is fetched)."""
    monkeypatch.setattr(fo, "_collect_holdings", lambda: {"MSFT", "CRWD", "AI.PA"})
    rc = _run_main(monkeypatch, ["--resweep-held", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Resolved 3 symbols to fetch." in out
    for sym in ("MSFT", "CRWD", "AI.PA"):
        assert sym in out


def test_resweep_held_with_no_open_positions_is_a_clean_no_op(
    monkeypatch: pytest.MonkeyPatch, midas_data_root: Path, capsys
) -> None:
    """An empty holdings set (e.g. all 12 books flat) must exit cleanly
    without attempting any fetch — proven by a _fetch_symbol that raises if
    called at all."""
    monkeypatch.setattr(fo, "_collect_holdings", lambda: set())

    def fail_if_called(symbol: str, start: date, end: date):
        raise AssertionError("must not fetch when no positions are held")

    monkeypatch.setattr(fo, "_fetch_symbol", fail_if_called)
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

    rc = _run_main(monkeypatch, ["--resweep-held"])
    assert rc == 0
    assert "nothing to resweep" in capsys.readouterr().out


def test_resweep_held_detects_a_split_within_a_90_day_window(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through the actual trigger the scheduled workflow uses:
    --resweep-held --history-days 90 (the window the workflow passes).
    Proves both halves the coordinator asked for — the held-ticker set
    resolves via _collect_holdings(), and a 90-day window comfortably clears
    detect_split's 10-row minimum (15 daily rows land well inside it here) —
    end to end, including the agent's position actually being adjusted."""
    today = date.today()
    start_date = today - timedelta(days=30)
    dates = [(start_date + timedelta(days=i)).isoformat() for i in range(20)]

    def bar(c: float) -> list:
        return [c, c, c, c, c, 1]

    path = get_config().ohlcv_dir / "CRWD.jsonl"
    # 15 stale pre-split rows, then 5 the nightly cron already appended at
    # post-split prices — the transition detect_split requires.
    fresh_closes = [100.0 + i for i in range(20)]
    stale_closes = [c * 4.0 for c in fresh_closes[:15]] + fresh_closes[15:]
    _write_raw(path, [_canonical_line(d, bar(c)) for d, c in zip(dates, stale_closes)])

    frames = {"CRWD": {d: bar(c) for d, c in zip(dates, fresh_closes)}}
    monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol(frames))
    monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)
    monkeypatch.setattr(fo, "_collect_holdings", lambda: {"CRWD"})

    portfolios = PortfolioManager(base_dir=get_config().portfolios_dir)
    portfolios.initialize("holder", initial_capital=5000.0)
    portfolios.apply_trade(
        "holder",
        Trade(
            id="t1",
            timestamp=datetime(2026, 5, 20, 20, 0),
            action="BUY",
            ticker="CRWD",
            shares=3.0,
            price=400.0,
            total=1200.0,
            fees=0.0,
            reasoning="Pre-split buy",
        ),
    )

    rc = _run_main(monkeypatch, ["--resweep-held", "--history-days", "90"])
    assert rc == 0

    holder = portfolios.load("holder")
    position = next(p for p in holder.positions if p.ticker == "CRWD")
    assert position.shares == pytest.approx(12.0)
    assert position.avg_cost == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Mass-failure exit code (2026-08-07 review, W2.5)
# ---------------------------------------------------------------------------


def _all_symbols_fail(symbol: str, start: date, end: date):
    return None


def _cover(symbols: list[str], close: float = 1.5) -> None:
    """Give each symbol a store file, i.e. a history of having served rows.

    The gate's denominator is store coverage, so a test that does not seed the
    store is testing the "cannot measure" branch whatever else it sets up.
    """
    for symbol in symbols:
        _write_raw(
            get_config().ohlcv_dir / f"{symbol}.jsonl",
            [_tight_line("2026-08-05", close)],
        )


class TestFailureRateGate:
    """`fetch_ohlcv.py` exited 0 regardless of the failure count, so a total
    vendor outage produced a green run, "No OHLCV changes to commit", and a
    session pricing a stale store the next evening — with nothing anywhere
    saying the data had not arrived.

    The denominator is the symbols the store ALREADY COVERS. Against the full
    symbol list the gate was mis-calibrated from the day it shipped and fired
    on every full-universe run — see `TestUnresolvedSymbolsAreNotAnOutage`.
    """

    def test_total_outage_exits_nonzero(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        _cover(["AAPL", "MSFT", "SAP.DE", "BP.L"])
        monkeypatch.setattr(fo, "_fetch_symbol", _all_symbols_fail)
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

        rc = _run_main(monkeypatch, ["--symbols", "AAPL,MSFT,SAP.DE,BP.L"])

        assert rc == fo.EXIT_VENDOR_OUTAGE
        assert "vendor-side or network failure" in capsys.readouterr().err

    def test_a_few_dead_symbols_are_not_a_failure(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control, and the reason the threshold is a rate rather than a
        count: `MATIC-USD` and `UNI-USD` have served nothing since March and
        April 2025. Any absolute floor would fire every night or never."""
        _cover(_MANY)
        good = _make_fake_fetch_symbol(
            {s: {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]} for s in _MANY[:-1]}
        )
        monkeypatch.setattr(fo, "_fetch_symbol", good)
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

        rc = _run_main(monkeypatch, ["--symbols", ",".join(_MANY)])
        assert rc == 0  # 1 of 20 dead = 5%, under the 10% limit

    def test_the_threshold_binds_just_past_the_limit(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The falsifying pair for the rate itself."""
        _cover(_MANY)
        # 3 of 20 = 15%, over the limit.
        good = _make_fake_fetch_symbol(
            {s: {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]} for s in _MANY[:-3]}
        )
        monkeypatch.setattr(fo, "_fetch_symbol", good)
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

        assert (
            _run_main(monkeypatch, ["--symbols", ",".join(_MANY)])
            == fo.EXIT_VENDOR_OUTAGE
        )

    def test_already_current_symbols_stay_in_the_denominator(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a second run in one UTC day must not read as an outage.

        A covered symbol whose store already holds `end` is skipped before the
        fetch, but it is *evidence the store is healthy* and belongs in the
        denominator. Excluding it collapses the population onto the handful of
        permanently-dead names that are still attempted (their store is stale,
        so they never take the skip branch) — `MATIC-USD` and `UNI-USD` alone
        read 2/2 = 100% and fail a fully current store. Reachable by a re-run,
        by the documented `workflow_dispatch`, or by the Tue-Sat full run
        (`0 6 * * 2-6`) delayed past midnight into Sunday — its `end` is then
        Saturday, so it writes the Saturday rows for the 24/7 names before the
        Sun-Mon crypto-only run (`0 6 * * 0,1`), whose `end` is also Saturday,
        asks for them.
        """
        end = _fetch_end().isoformat()
        current = [f"CUR{i}" for i in range(20)]
        for symbol in current:
            _write_raw(
                get_config().ohlcv_dir / f"{symbol}.jsonl", [_tight_line(end, 1.5)]
            )
        # Two covered-but-dead names, still attempted because their store is old.
        _cover(["MATIC-USD", "UNI-USD"])

        monkeypatch.setattr(fo, "_fetch_symbol", _make_fake_fetch_symbol({}))
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

        rc = _run_main(
            monkeypatch, ["--symbols", ",".join(current + ["MATIC-USD", "UNI-USD"])]
        )

        assert rc == 0, "2 dead names against 22 covered symbols is not an outage"

    def test_a_bootstrap_where_nothing_answers_is_still_an_outage(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """The empty store has no coverage, but "nothing served" still decides.

        Returning 0 here would reopen W2.5's exact hole on the one run with
        nothing to fall back on — a fork's first fetch, or a run after the
        store directory moved.
        """
        monkeypatch.setattr(fo, "_fetch_symbol", _all_symbols_fail)
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

        rc = _run_main(monkeypatch, ["--symbols", "AAPL,MSFT,SAP.DE,BP.L"])

        assert rc == fo.EXIT_VENDOR_OUTAGE
        assert "no prior coverage" in capsys.readouterr().err

    def test_a_healthy_bootstrap_with_a_bad_universe_still_passes(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The falsifying pair for the bootstrap branch.

        Without this, the test above passes just as happily against a rule that
        fails every empty-store run — which is the fork-hostile version.
        """
        good = _make_fake_fetch_symbol(
            {s: {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]} for s in _MANY}
        )
        monkeypatch.setattr(fo, "_fetch_symbol", good)
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

        rc = _run_main(
            monkeypatch,
            ["--symbols", ",".join(_MANY + [f"BAD{i}.PA" for i in range(200)])],
        )

        assert rc == 0

    def test_empty_vendor_frame_is_reported_on_stderr(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """An empty frame used to return None without a word."""
        monkeypatch.setattr(fo.yf, "download", lambda *a, **k: pd.DataFrame())
        assert fo._fetch_symbol("AAPL", date(2026, 8, 1), date(2026, 8, 6)) is None
        assert "AAPL: vendor returned no rows" in capsys.readouterr().err


class TestUnresolvedSymbolsAreNotAnOutage:
    """Regression: 5599e64f6 — the gate fired on its own steady state.

    A symbol that has never served a row and has no store file is a ticker
    that does not resolve at the vendor, not a vendor that stopped answering.
    Counting the two together put the 2026-08-07 baseline (121 failures of
    1,150 = 10.5%) permanently over the 10% limit, so every full-universe run
    exited 1, the commit step was skipped, and the price store stopped
    advancing for four days — the outage the gate exists to detect, produced
    by the gate. 118 of those 121 are Refinitiv-style codes in
    `data/universes/stoxx600.json` that Yahoo has no route for.
    """

    def test_the_production_shape_does_not_fail_the_run(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact 2026-08-07 proportions: 10.5% of the list, 0.2% of coverage."""
        covered = [f"COV{i}" for i in range(1030)]
        never_served = [f"BADTICKER{i}.PA" for i in range(121)]
        _cover(covered)

        # Every covered symbol answers except two — NKLA and ROG.SW on the night.
        good = _make_fake_fetch_symbol(
            {s: {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]} for s in covered[:-2]}
        )
        monkeypatch.setattr(fo, "_fetch_symbol", good)
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

        rc = _run_main(monkeypatch, ["--symbols", ",".join(covered + never_served)])

        assert rc == 0, "the steady-state baseline must not fail the run"

    def test_the_old_denominator_would_have_failed_this(
        self, midas_data_root: Path
    ) -> None:
        """The control: prove the shape above really is over the old limit.

        Without this, `test_the_production_shape_does_not_fail_the_run` passes
        just as happily against a gate that never fires at all.
        """
        total, unresolved_count, covered_failures = 1151, 121, 2
        assert unresolved_count / total > fo.MAX_FAILURE_RATE
        assert covered_failures / (total - unresolved_count) < fo.MAX_FAILURE_RATE

    def test_a_covered_symbol_going_dark_still_fails(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """The gate must still bind on what it was built for.

        Unresolvable tickers alongside a real outage must not dilute it — under
        the old denominator they would have, by inflating the population the
        rate is taken over.
        """
        covered = [f"COV{i}" for i in range(20)]
        _cover(covered)
        # 5 of 20 covered symbols go dark = 25%, over the limit.
        good = _make_fake_fetch_symbol(
            {s: {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]} for s in covered[:-5]}
        )
        monkeypatch.setattr(fo, "_fetch_symbol", good)
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

        rc = _run_main(
            monkeypatch,
            ["--symbols", ",".join(covered + [f"BAD{i}.PA" for i in range(200)])],
        )

        assert rc == fo.EXIT_VENDOR_OUTAGE
        assert "the store already covers" in capsys.readouterr().err

    def test_unresolved_symbols_are_reported_not_silently_tolerated(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """Not failing the run is not the same as saying nothing.

        ~120 unfetchable tickers in a committed universe is a real defect; it
        is just not one a red nightly run can fix, and holding the price store
        hostage to it is what broke the desk.
        """
        _cover(["AAPL"])
        good = _make_fake_fetch_symbol(
            {"AAPL": {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]}}
        )
        monkeypatch.setattr(fo, "_fetch_symbol", good)
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)

        rc = _run_main(monkeypatch, ["--symbols", "AAPL,AIRP.PA,BNPP.PA"])

        err = capsys.readouterr().err
        assert rc == 0
        assert "2 symbol(s) have never served a row" in err
        assert "AIRP.PA" in err

    def test_the_report_survives_a_quarantine_night(
        self, midas_data_root: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ) -> None:
        """`if total_quarantined: return 1` used to sit above this report.

        So on any night the tripwire refused a row, the ~120-unroutable-ticker
        warning — the entire point of the report — was never printed. The
        quarantine check is now last, after every report.
        """
        _cover(["AAPL"])
        good = _make_fake_fetch_symbol(
            {"AAPL": {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]}}
        )
        monkeypatch.setattr(fo, "_fetch_symbol", good)
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)
        monkeypatch.setattr(fo, "_write_rows", lambda *a, **k: fo.MergeResult(0, 0, 1))

        rc = _run_main(monkeypatch, ["--symbols", "AAPL,AIRP.PA"])

        assert rc == fo.EXIT_QUARANTINED
        assert "have never served a row" in capsys.readouterr().err


class TestExitCodes:
    """The workflow must tell a deliberate refusal from a crash.

    A bare `!cancelled()` push gate commits whatever is on disk, including a
    store half-written by an exception at symbol 500 of 1,150 — and the session
    that prices it writes one immutable snapshot row per book, so some mark at
    today and others at yesterday with no correction possible.
    """

    def test_the_deliberate_exits_are_distinct_from_one(self) -> None:
        assert fo.EXIT_QUARANTINED != 1
        assert fo.EXIT_VENDOR_OUTAGE != 1
        assert fo.EXIT_QUARANTINED != fo.EXIT_VENDOR_OUTAGE

    def test_committable_exits_name_exactly_the_deliberate_ones(self) -> None:
        assert set(fo.COMMITTABLE_EXITS) == {
            0,
            fo.EXIT_QUARANTINED,
            fo.EXIT_VENDOR_OUTAGE,
        }
        assert 1 not in fo.COMMITTABLE_EXITS, "an unhandled traceback exits 1"

    # The workflow half of this contract is asserted in tests/test_ci_guards.py
    # (TestPushGateMatchesExitCodes). That module is live-only, and
    # .github/workflows/ does not exist in midas-core — so a workflow-reading
    # test HERE is an unconditional failure in the public repo's suite. Caught
    # by running core's suite after the sync, which `sync_core check` cannot
    # see: the file is byte-identical in both repos, and that is the problem.


_MANY = [f"SYM{i}" for i in range(20)]


class TestAdjudicationClosesTheQuarantineLoop:
    """The tripwire could refuse a row but nothing could ever accept one.

    A real corporate action has the same shape as the units flip the tripwire
    hunts, and `resweep-held-tickers` sweeps HELD tickers only — so a universe
    ticker nobody holds froze indefinitely (MNST: store stuck on 2026-08-10 at
    91.43 through a 2:1 split, still buyable by four books at twice its price,
    with `fetch-ohlcv` red on every full-universe run since 08-13).
    """

    ACTION = CorporateAction("MNST", "2026-08-11", shares_ratio=2.0)
    REFUSED = (
        QuarantinedRow("MNST", "2026-08-11", "new-row", 91.43, 45.53, 0.497976),
    )

    def _arrange(self, monkeypatch, refused, actions):
        _cover(["MNST"])
        monkeypatch.setattr(
            fo,
            "_fetch_symbol",
            _make_fake_fetch_symbol({"MNST": {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]}}),
        )
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)
        # Stateful: the main loop's merge refuses rows; the adjudication
        # re-merge that follows actually lands them. A stub that returned
        # (0, 0) for BOTH would model a store still frozen, which is now its
        # own test below.
        calls: list[int] = []

        def write_rows(*a, **k):
            calls.append(1)
            if len(calls) == 1:
                return fo.MergeResult(0, 0, len(refused), refused)
            return fo.MergeResult(len(refused) or 1, 0)

        monkeypatch.setattr(fo, "_write_rows", write_rows)
        monkeypatch.setattr(fo, "_fetch_actions", lambda symbol: list(actions))

    def test_a_quarantined_row_the_calendar_explains_goes_green(self, midas_data_root, monkeypatch):
        self._arrange(monkeypatch, self.REFUSED, [self.ACTION])
        assert _run_main(monkeypatch, ["--symbols", "MNST"]) == 0

    def test_a_quarantined_row_the_calendar_cannot_explain_stays_red(self, midas_data_root, monkeypatch):
        """The tripwire's whole purpose survives: no calendar entry, no ingest."""
        self._arrange(monkeypatch, self.REFUSED, [])
        assert _run_main(monkeypatch, ["--symbols", "MNST"]) == fo.EXIT_QUARANTINED

    def test_a_count_without_row_detail_still_goes_red(self, midas_data_root, monkeypatch):
        """Fail closed when the refusal is invisible rather than explained.

        Found while building this: keying the exit on the adjudication result
        alone let a `MergeResult` that reports `quarantined=1` with no rows
        attached exit 0 — a refused row turning the run GREEN by being
        undescribed. The exit now subtracts only positively-explained rows
        from the tripwire's own count, so anything unaccounted for stays red.
        Same family as every other "check that cannot fail" in this repo.
        """
        self._arrange(monkeypatch, (), [self.ACTION])
        monkeypatch.setattr(fo, "_write_rows", lambda *a, **k: fo.MergeResult(0, 0, 1))
        assert _run_main(monkeypatch, ["--symbols", "MNST"]) == fo.EXIT_QUARANTINED

    def test_an_explained_row_whose_remerge_lands_nothing_stays_red(
        self, midas_data_root, monkeypatch
    ):
        """The calendar explaining a refusal is not proof the store moved.

        If the wider re-fetch omits the refused dates — Yahoo serves a date
        over one window and not another, in both directions — the store stays
        frozen on the pre-action close. Counting those rows as explained would
        exit 0, and every later night would refuse them, hit the `already`
        branch and exit 0 again: a permanent freeze, invisible.
        """
        self._arrange(monkeypatch, self.REFUSED, [self.ACTION])
        monkeypatch.setattr(
            fo,
            "_write_rows",
            lambda *a, **k: fo.MergeResult(0, 0, len(self.REFUSED), self.REFUSED),
        )
        assert _run_main(monkeypatch, ["--symbols", "MNST"]) == fo.EXIT_QUARANTINED

    def test_the_ledger_records_the_adjudication(self, midas_data_root, monkeypatch):
        self._arrange(monkeypatch, self.REFUSED, [self.ACTION])
        _run_main(monkeypatch, ["--symbols", "MNST"])
        rows = [
            json.loads(line)
            for line in fo._ledger_path().read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["symbol"] == "MNST"
        assert rows[0]["effective"] == "2026-08-11"
        assert rows[0]["shares_ratio"] == 2.0
        assert rows[0]["source"] == "vendor-calendar"

    def test_the_same_action_is_never_applied_to_holdings_twice(
        self, midas_data_root, monkeypatch
    ):
        """The money path. `apply_split` mutates real share counts.

        The nightly fetch and the weekly resweep can both see one action; two
        applications halve a book twice, and the cost-basis invariant hides it
        (shares x avg_cost stays put on every application).
        """
        applied: list[float] = []
        self._arrange(monkeypatch, self.REFUSED, [self.ACTION])
        monkeypatch.setattr(
            fo,
            "_apply_split_to_holders",
            lambda symbol, ratio: applied.append(ratio) or ["world"],
        )
        _run_main(monkeypatch, ["--symbols", "MNST"])
        _run_main(monkeypatch, ["--symbols", "MNST"])
        assert applied == [2.0], f"apply_split ran {len(applied)} times, expected 1"

    def test_apply_split_receives_the_shares_ratio_not_the_price_ratio(
        self, midas_data_root, monkeypatch
    ):
        """Feeding it 0.5 on a 2:1 would HALVE a real holding, silently."""
        seen: list[float] = []
        self._arrange(monkeypatch, self.REFUSED, [self.ACTION])
        monkeypatch.setattr(
            fo, "_apply_split_to_holders", lambda symbol, ratio: seen.append(ratio) or []
        )
        _run_main(monkeypatch, ["--symbols", "MNST"])
        assert seen == [2.0]
        assert seen != [self.ACTION.price_ratio]

    def test_an_unreadable_ledger_refuses_to_adjudicate(self, midas_data_root, monkeypatch):
        """Fail closed: cannot prove an action is unapplied, so do not apply it."""
        self._arrange(monkeypatch, self.REFUSED, [self.ACTION])
        path = fo._ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json\n")
        assert _run_main(monkeypatch, ["--symbols", "MNST"]) == fo.EXIT_QUARANTINED


class TestBothSplitPathsShareOneLedger:
    """`detect_split`'s resweep path bypassed the ledger entirely.

    The nightly path adjudicates from the calendar and records the action;
    the weekly resweep inferred a ratio from drift and applied it with no
    ledger check and no ledger write. Both call `_apply_split_to_holders`, so
    one real action could be applied twice — and the sequence is not
    hypothetical: an unrestated split (MNST's shape) is adjudicated from the
    calendar tonight, the vendor restates history days later (JMAT.L proves
    that happens late), and Monday's resweep then finds exactly the drift
    `detect_split` was built to find. Shares x4, with the cost-basis
    invariant holding on every application so nothing looks wrong.
    """

    ACTION = CorporateAction("MNST", "2026-08-11", shares_ratio=2.0)

    def _arrange(self, monkeypatch, actions):
        _cover(["MNST"])
        monkeypatch.setattr(
            fo,
            "_fetch_symbol",
            _make_fake_fetch_symbol({"MNST": {"2026-08-06": [1, 2, 0.5, 1.5, 1.5, 100]}}),
        )
        monkeypatch.setattr(fo, "_fetch_ticker_info", lambda symbol: None)
        monkeypatch.setattr(fo, "_write_rows", lambda *a, **k: fo.MergeResult(1, 0))
        monkeypatch.setattr(fo, "detect_split", lambda stored, df: 2.0)
        monkeypatch.setattr(fo, "_fetch_actions", lambda symbol: list(actions))

    def _run_resweep(self, monkeypatch, applied):
        monkeypatch.setattr(
            fo,
            "_apply_split_to_holders",
            lambda symbol, ratio: applied.append(ratio) or ["world"],
        )
        _run_main(monkeypatch, ["--resweep", "--symbols", "MNST"])

    def test_the_resweep_records_what_it_applied(self, midas_data_root, monkeypatch):
        self._arrange(monkeypatch, [self.ACTION])
        self._run_resweep(monkeypatch, [])
        rows = [
            json.loads(line)
            for line in fo._ledger_path().read_text().splitlines()
            if line.strip()
        ]
        assert [(r["symbol"], r["effective"]) for r in rows] == [("MNST", "2026-08-11")]

    def test_an_action_adjudicated_nightly_is_not_reapplied_by_the_resweep(
        self, midas_data_root, monkeypatch
    ):
        """The money path, across BOTH entry points."""
        self._arrange(monkeypatch, [self.ACTION])
        applied: list[float] = []
        # Stand in for last night's calendar adjudication.
        fo._record_adjudication(self.ACTION, ["world"], 1)
        self._run_resweep(monkeypatch, applied)
        assert applied == [], "the resweep re-applied an action already in the ledger"

    def test_a_drift_only_detection_still_applies(self, midas_data_root, monkeypatch):
        """No calendar entry means no shared identity — keep the old behaviour.

        Refusing here would silently drop the detection `detect_split` exists
        for. It is applied unledgered and says so in the log.
        """
        self._arrange(monkeypatch, [])
        applied: list[float] = []
        self._run_resweep(monkeypatch, applied)
        assert applied == [2.0]

    def test_an_unreadable_ledger_leaves_holdings_alone(self, midas_data_root, monkeypatch):
        self._arrange(monkeypatch, [self.ACTION])
        path = fo._ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json\n")
        applied: list[float] = []
        self._run_resweep(monkeypatch, applied)
        assert applied == []
