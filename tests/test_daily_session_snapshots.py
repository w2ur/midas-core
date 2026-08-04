"""Regression tests for the daily session snapshot pricing helper.

Background: Until 2026-05-10, `step_update_snapshots` priced positions by
`prices_df.iloc[-1].to_dict()` from a left-joined DataFrame across all held
tickers, then `dict.get(ticker, avg_cost)`. When a ticker had no row for the
DataFrame's last date (e.g., European tickers lagging US closes by one day
in the OHLCV store), the dict held NaN for that key — `dict.get` returned
the NaN, not the default — and `portfolio_value` came out NaN. World and
yolo-sapiens-eur recorded NaN snapshots throughout the experiment.

Fix: per-ticker `latest_close_on_or_before`, with avg_cost fallback.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from engine.portfolio import PortfolioManager
from engine.types import Portfolio, Position
from scripts.daily_session import _compute_positions_value, step_update_snapshots


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_portfolio(positions: list[Position], cash: float = 0.0) -> Portfolio:
    return Portfolio(
        cash=cash,
        positions=positions,
        last_updated=date(2026, 5, 9),
        currency="EUR",
    )


def test_uses_per_ticker_latest_close_when_dates_differ(tmp_path: Path) -> None:
    """ASML.AS lagging by a day must not poison the AAPL valuation."""
    store = tmp_path / "ohlcv"
    store.mkdir()
    _write_jsonl(
        store / "AAPL.jsonl",
        [
            {"date": "2026-05-07", "close": 287.44, "adj_close": 287.44},
            {"date": "2026-05-08", "close": 293.32, "adj_close": 293.32},
        ],
    )
    _write_jsonl(
        store / "ASML.AS.jsonl",
        [{"date": "2026-05-07", "close": 1300.0, "adj_close": 1300.0}],
    )
    portfolio = _make_portfolio(
        [
            Position(
                ticker="AAPL",
                shares=5.0,
                avg_cost=263.4,
                date_opened=date(2026, 4, 17),
                grid_level=0,
            ),
            Position(
                ticker="ASML.AS",
                shares=2.0,
                avg_cost=1232.0,
                date_opened=date(2026, 4, 20),
                grid_level=0,
            ),
        ]
    )

    pv = _compute_positions_value(portfolio, date(2026, 5, 8), store=store)

    # AAPL @ 293.32 (Friday close) + ASML.AS @ 1300.0 (Thursday carry-forward)
    assert pv == pytest.approx(5 * 293.32 + 2 * 1300.0)


def test_falls_back_to_avg_cost_when_no_price_in_store(tmp_path: Path) -> None:
    store = tmp_path / "ohlcv"
    store.mkdir()
    portfolio = _make_portfolio(
        [
            Position(
                ticker="UNKNOWN",
                shares=10.0,
                avg_cost=50.0,
                date_opened=date(2026, 4, 17),
                grid_level=0,
            )
        ]
    )

    pv = _compute_positions_value(portfolio, date(2026, 5, 8), store=store)

    assert pv == pytest.approx(500.0)


def test_empty_portfolio_returns_zero(tmp_path: Path) -> None:
    store = tmp_path / "ohlcv"
    store.mkdir()
    portfolio = _make_portfolio([])

    assert _compute_positions_value(portfolio, date(2026, 5, 8), store=store) == 0.0


def test_uses_close_on_target_date_not_later_rows(tmp_path: Path) -> None:
    """If the snapshot date is mid-history, future rows must not be used."""
    store = tmp_path / "ohlcv"
    store.mkdir()
    _write_jsonl(
        store / "AAPL.jsonl",
        [
            {"date": "2026-05-05", "close": 100.0, "adj_close": 100.0},
            {"date": "2026-05-06", "close": 200.0, "adj_close": 200.0},
            {"date": "2026-05-07", "close": 300.0, "adj_close": 300.0},
        ],
    )
    portfolio = _make_portfolio(
        [
            Position(
                ticker="AAPL",
                shares=1.0,
                avg_cost=50.0,
                date_opened=date(2026, 4, 17),
                grid_level=0,
            )
        ]
    )

    pv = _compute_positions_value(portfolio, date(2026, 5, 6), store=store)

    assert pv == pytest.approx(200.0)


def test_stalled_market_date_does_not_rewrite_a_previous_sessions_row(
    midas_data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: 0da774525 — on 2026-08-03 the OHLCV store had not advanced
    past the weekend, so `step_update_snapshots` re-used market date 2026-08-02
    and overwrote the weekend refresh's already-published row with a portfolio
    that included Monday's fills. `/archive/2026-08-02` changed after the fact.
    """
    from engine.config import get_config

    portfolios_dir = get_config().portfolios_dir
    manager = PortfolioManager(base_dir=portfolios_dir)
    manager.initialize("test-agent", initial_capital=10_000.0, currency="EUR")

    # Sunday's refresh publishes a row for market date 2026-08-02.
    manager.add_snapshot(
        strategy_id="test-agent",
        snapshot_date=date(2026, 8, 2),
        portfolio_value=10_000.0,
        cash=10_000.0,
        positions_value=0.0,
        benchmarks={"sp500": 7470.3},
        session_date=date(2026, 8, 2),
    )

    # Monday's session runs against the same stalled market date.
    monkeypatch.setattr(
        "scripts.daily_session.date",
        type("D", (date,), {"today": staticmethod(lambda: date(2026, 8, 3))}),
    )
    step_update_snapshots(
        {"date": "2026-08-02", "benchmarks": {"sp500": 7470.3}},
    )

    snapshots = manager.load_snapshots("test-agent")
    rows = [s for s in snapshots if s["date"] == "2026-08-02"]
    assert len(rows) == 1
    assert rows[0]["portfolio_value"] == pytest.approx(10_000.0)
    assert rows[0]["session_date"] == "2026-08-02"
