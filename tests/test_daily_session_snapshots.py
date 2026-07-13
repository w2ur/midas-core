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

from engine.types import Portfolio, Position
from scripts.daily_session import _compute_positions_value


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
