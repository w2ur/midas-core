"""Regression tests for the daily session snapshot pricing helper.

Background: Until 2026-05-10, `step_update_snapshots` priced positions by
`prices_df.iloc[-1].to_dict()` from a left-joined DataFrame across all held
tickers, then `dict.get(ticker, avg_cost)`. When a ticker had no row for the
DataFrame's last date (e.g., European tickers lagging US closes by one day
in the OHLCV store), the dict held NaN for that key — `dict.get` returned
the NaN, not the default — and `portfolio_value` came out NaN. World and
yolo-sapiens-eur recorded NaN snapshots throughout the experiment.

Fix: per-ticker `latest_close_on_or_before`, with avg_cost fallback.

FX background: until this fix, `_compute_positions_value` summed a
foreign-currency price straight into the book's base currency with no
conversion — the fill path (`engine.paper_broker`) has always converted via
`_ticker_currency` + `engine.fx.convert`, the valuation path never did.
Five live books were wrong by -13.39% to +4.42% of positions value. The
tests below that hold a USD-default ticker (`AAPL`, `UNKNOWN` — no suffix,
no override) in a EUR book were *already* cross-currency; they just never
noticed because the old code never converted. They now seed an EURUSD=X
rate and assert the converted total.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from engine.config import get_config
from engine.portfolio import PortfolioManager
from engine.restatement import MissingPriceError
from engine.types import Portfolio, Position, Trade
from scripts.daily_session import _compute_positions_value, step_update_snapshots


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    import json

    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_portfolio(
    positions: list[Position], cash: float = 0.0, currency: str = "EUR"
) -> Portfolio:
    return Portfolio(
        cash=cash,
        positions=positions,
        last_updated=date(2026, 5, 9),
        currency=currency,
    )


def test_uses_per_ticker_latest_close_when_dates_differ(midas_data_root: Path) -> None:
    """ASML.AS lagging by a day must not poison the AAPL valuation."""
    store = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
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
    # AAPL resolves to USD (no suffix, no override); the book is EUR.
    _write_jsonl(
        store / "EURUSD=X.jsonl",
        [{"date": "2026-05-07", "close": 1.25, "adj_close": 1.25}],
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

    # AAPL @ 293.32 USD (Friday close), converted USD->EUR at 1/1.25, plus
    # ASML.AS @ 1300.0 EUR (Thursday carry-forward, no conversion needed).
    assert pv == pytest.approx(5 * 293.32 * (1 / 1.25) + 2 * 1300.0)


def test_refuses_rather_than_falling_back_to_avg_cost(midas_data_root: Path) -> None:
    """Changed 2026-08-07 (review W4.5): this used to value at `avg_cost`.

    Cost is not a valuation — it is the last price at which somebody was
    willing to transact, which may be months stale. And snapshots are
    immutable, so a snapshot written from cost is permanent. The session now
    skips that book's row for the day instead; `step_update_snapshots`
    catches this per portfolio, so one unpriceable book does not abort the
    session.
    """
    store = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    # UNKNOWN resolves to USD (no suffix, no override); the book is EUR.
    _write_jsonl(
        store / "EURUSD=X.jsonl",
        [{"date": "2026-05-07", "close": 1.25, "adj_close": 1.25}],
    )
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

    with pytest.raises(MissingPriceError):
        _compute_positions_value(portfolio, date(2026, 5, 8), store=store)


def test_empty_portfolio_returns_zero(tmp_path: Path) -> None:
    store = tmp_path / "ohlcv"
    store.mkdir()
    portfolio = _make_portfolio([])

    assert _compute_positions_value(portfolio, date(2026, 5, 8), store=store) == 0.0


def test_uses_close_on_target_date_not_later_rows(midas_data_root: Path) -> None:
    """If the snapshot date is mid-history, future rows must not be used."""
    store = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        store / "AAPL.jsonl",
        [
            {"date": "2026-05-05", "close": 100.0, "adj_close": 100.0},
            {"date": "2026-05-06", "close": 200.0, "adj_close": 200.0},
            {"date": "2026-05-07", "close": 300.0, "adj_close": 300.0},
        ],
    )
    _write_jsonl(
        store / "EURUSD=X.jsonl",
        [{"date": "2026-05-05", "close": 1.25, "adj_close": 1.25}],
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

    assert pv == pytest.approx(200.0 * (1 / 1.25))


def test_converts_foreign_currency_position_to_book_currency(
    midas_data_root: Path,
) -> None:
    """A EUR book holding a USD ticker must be converted, not summed raw.

    This pins the exact defect from the brief: `total += p.shares * price`
    with no FX conversion.
    """
    store = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        store / "MSFT.AS.jsonl",  # EUR-suffix ticker, matches the book currency
        [{"date": "2026-05-08", "close": 100.0, "adj_close": 100.0}],
    )
    _write_jsonl(
        store / "RTX.jsonl",  # no suffix -> resolves to USD
        [{"date": "2026-05-08", "close": 120.0, "adj_close": 120.0}],
    )
    _write_jsonl(
        store / "EURUSD=X.jsonl",  # stored rate is EUR->USD
        [{"date": "2026-05-08", "close": 1.25, "adj_close": 1.25}],
    )
    portfolio = _make_portfolio(
        [
            Position(
                ticker="MSFT.AS",
                shares=3.0,
                avg_cost=90.0,
                date_opened=date(2026, 4, 17),
                grid_level=0,
            ),
            Position(
                ticker="RTX",
                shares=10.0,
                avg_cost=110.0,
                date_opened=date(2026, 4, 17),
                grid_level=0,
            ),
        ],
        currency="EUR",
    )

    pv = _compute_positions_value(portfolio, date(2026, 5, 8), store=store)

    eur_leg = 3.0 * 100.0
    usd_leg_converted = 10.0 * 120.0 * (1 / 1.25)  # USD->EUR
    assert pv == pytest.approx(eur_leg + usd_leg_converted)
    # Not the unconverted raw sum, which the old defect would have returned.
    assert pv != pytest.approx(eur_leg + 10.0 * 120.0)


def test_single_currency_book_unchanged_by_fx_fix(midas_data_root: Path) -> None:
    """A book whose every holding matches its own currency must be byte-identical.

    Seven of twelve live books never needed FX conversion; this fix must not
    move them. A USD book holding a plain-suffix (USD-resolving) ticker
    should value exactly `shares * price`, same as before this change.
    """
    store = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        store / "AAPL.jsonl",
        [{"date": "2026-05-08", "close": 293.32, "adj_close": 293.32}],
    )
    portfolio = _make_portfolio(
        [
            Position(
                ticker="AAPL",
                shares=5.0,
                avg_cost=263.4,
                date_opened=date(2026, 4, 17),
                grid_level=0,
            )
        ],
        currency="USD",
    )

    pv = _compute_positions_value(portfolio, date(2026, 5, 8), store=store)

    assert pv == pytest.approx(5 * 293.32)


def test_missing_fx_rate_raises_named_exception(midas_data_root: Path) -> None:
    """A missing FX rate must raise, never silently sum the unconverted price.

    Reuses `engine.restatement.MissingPriceError` (`what="FX rate"`) — the
    same exception `revalue_snapshot` raises for the identical gap, so the
    valuation and restatement paths fail the same way on the same condition.
    """
    store = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        store / "RTX.jsonl",
        [{"date": "2026-05-08", "close": 120.0, "adj_close": 120.0}],
    )
    # Deliberately no EURUSD=X fixture.
    portfolio = _make_portfolio(
        [
            Position(
                ticker="RTX",
                shares=10.0,
                avg_cost=110.0,
                date_opened=date(2026, 4, 17),
                grid_level=0,
            )
        ],
        currency="EUR",
    )

    with pytest.raises(MissingPriceError, match="FX rate"):
        _compute_positions_value(portfolio, date(2026, 5, 8), store=store)


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


def test_step_update_snapshots_skips_only_the_book_missing_fx_rate(
    midas_data_root: Path,
) -> None:
    """A missing FX rate must not abort the whole session.

    `_compute_positions_value` raises loud (no wrong number gets published
    for the affected book), but `step_update_snapshots` catches it per
    portfolio so the other book still gets a correct snapshot the same run.
    """
    store = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        store / "RTX.jsonl",
        [{"date": "2026-05-08", "close": 120.0, "adj_close": 120.0}],
    )
    _write_jsonl(
        store / "AAPL.jsonl",
        [{"date": "2026-05-08", "close": 293.32, "adj_close": 293.32}],
    )
    # No EURUSD=X fixture, so "eur-book" (holding RTX, USD) cannot be priced.

    portfolios_dir = get_config().portfolios_dir
    manager = PortfolioManager(base_dir=portfolios_dir)
    manager.initialize("eur-book", initial_capital=1_000.0, currency="EUR")
    manager.apply_trade(
        "eur-book",
        Trade(
            id="t1",
            timestamp=datetime(2026, 5, 8, 20, 0, 0),
            action="BUY",
            ticker="RTX",
            shares=5.0,
            price=120.0,
            total=600.0,
            fees=0.0,
            reasoning="test fixture",
        ),
    )
    manager.initialize("usd-book", initial_capital=1_000.0, currency="USD")
    manager.apply_trade(
        "usd-book",
        Trade(
            id="t2",
            timestamp=datetime(2026, 5, 8, 20, 0, 0),
            action="BUY",
            ticker="AAPL",
            shares=2.0,
            price=293.32,
            total=586.64,
            fees=0.0,
            reasoning="test fixture",
        ),
    )

    snapshotted = step_update_snapshots(
        {"date": "2026-05-08", "benchmarks": {}},
    )

    assert "eur-book" not in snapshotted
    assert "usd-book" in snapshotted

    usd_snapshots = manager.load_snapshots("usd-book")
    rows = [s for s in usd_snapshots if s["date"] == "2026-05-08"]
    assert len(rows) == 1
    eur_snapshots = manager.load_snapshots("eur-book")
    assert not [s for s in eur_snapshots if s["date"] == "2026-05-08"]
