"""Tests for engine.quotes — quote-unit resolution and normalisation.

Regression: `engine.paper_broker._ticker_currency` resolved a ticker's
currency from a suffix heuristic alone. Two independent defects followed:

1. Every `.L` ticker was mapped to GBP. The LSE quotes most of its order
   book in **pence** (`GBp` at the vendor), not pounds, so a `.L` position
   was valued 100x high. `world` bought 8 `LLOY.L` at 116.60 and the ledger
   booked `total: 1090.19` for what is ~EUR 11 of Lloyds. And `.L` is not
   even uniformly sterling — `PHAG.L` quotes in USD.
2. Every European suffix the heuristic did not enumerate fell through to
   USD — 126 tickers in `world`'s own universe (`.ST` SEK, `.MC` EUR,
   `.BR` EUR, `.HE` EUR, `.OL` NOK, `.CO` DKK, `.VI` EUR, `.IR` EUR,
   `.LS` EUR, `.LU` EUR, `.WA` PLN, `.AT` EUR).

The fix resolves the quote unit from the vendor (persisted into the
`data/tickers.json` registry by `scripts/fetch_ohlcv.py`, which already
fetches `.info` per symbol) and falls back to the heuristic only for
tickers the vendor cannot answer for. `GBp` is a unit, not a currency: it
is normalised to GBP/100 in exactly one place, `engine.quotes.normalise_quote`.

Fixtures only — no network. `.github/workflows/tests.yml` checks out with
`fetch-depth: 1` and CI has no network guarantee.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.config import get_config
from engine.quotes import (
    Quote,
    latest_price,
    normalise_quote,
    quote_currency,
    ticker_currency,
)


# ---------------------------------------------------------------------------
# Fixtures — the committed registry / override map, seeded in a tmp data root
# ---------------------------------------------------------------------------


def _seed_registry(entries: dict[str, str | None]) -> None:
    """Write data/tickers.json with {symbol: currency} (name/type filled in)."""
    path = get_config().tickers_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                sym: {"name": sym, "type": "equity", "currency": ccy}
                for sym, ccy in entries.items()
            }
        ),
        encoding="utf-8",
    )


def _seed_overrides(entries: dict[str, str]) -> None:
    path = get_config().ticker_currencies_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")


def _seed_ohlcv(ticker: str, close: float, on: str = "2026-06-01") -> None:
    path = get_config().ohlcv_dir / f"{ticker}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": on, "close": close}) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. The real cases, pinned against what the vendor actually reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", ["LLOY.L", "FCIT.L", "SGLN.L", "III.L"])
def test_lse_pence_tickers_resolve_to_gbp_at_one_hundredth(
    midas_data_root: Path, ticker: str
) -> None:
    """Yahoo reports `GBp` for these four. GBp is pence: ISO GBP, scale 1/100."""
    _seed_registry({ticker: "GBp"})
    assert quote_currency(ticker) == "GBp"
    assert ticker_currency(ticker) == "GBP"
    assert normalise_quote(ticker, 116.60) == Quote(pytest.approx(1.166), "GBP")


def test_phag_l_is_usd_not_gbp(midas_data_root: Path) -> None:
    """`.L` is not uniformly sterling — PHAG.L quotes in USD at the vendor.

    This is the case the suffix heuristic cannot express at all: no rule on
    the string `PHAG.L` distinguishes it from `LLOY.L`.
    """
    _seed_registry({"PHAG.L": "USD"})
    assert ticker_currency("PHAG.L") == "USD"
    assert normalise_quote("PHAG.L", 54.87) == Quote(pytest.approx(54.87), "USD")


@pytest.mark.parametrize(
    ("ticker", "vendor_currency"),
    [
        ("SAN.MC", "EUR"),  # Madrid — heuristic said USD
        ("NDA-FI.HE", "EUR"),  # Helsinki — heuristic said USD
        ("KBC.BR", "EUR"),  # Brussels — heuristic said USD
        ("EQNR.OL", "NOK"),  # Oslo — NOT eurozone, and heuristic said USD
        ("NOVO-B.CO", "DKK"),  # Copenhagen — NOT eurozone
        ("VOLV-B.ST", "SEK"),  # Stockholm — NOT eurozone
    ],
)
def test_european_suffixes_the_heuristic_missed(
    midas_data_root: Path, ticker: str, vendor_currency: str
) -> None:
    _seed_registry({ticker: vendor_currency})
    assert ticker_currency(ticker) == vendor_currency


# ---------------------------------------------------------------------------
# 2. Resolution order: override > registry > heuristic
# ---------------------------------------------------------------------------


def test_override_map_beats_the_registry(midas_data_root: Path) -> None:
    """The hand-maintained override file stays the top layer.

    It is the only way to correct a ticker the vendor answers wrongly, or
    one it has stopped answering for at all (WDFC.SW is delisted at Yahoo
    yet still carried in the committed override map).
    """
    _seed_registry({"MSFT": "USD"})
    _seed_overrides({"MSFT": "EUR"})
    assert ticker_currency("MSFT") == "EUR"


def test_registry_beats_the_suffix_heuristic(midas_data_root: Path) -> None:
    _seed_registry({"SGLN.MI": "EUR", "PHAG.L": "USD"})
    assert ticker_currency("PHAG.L") == "USD"  # heuristic would say GBP


def test_heuristic_is_the_fallback_when_the_vendor_is_silent(
    midas_data_root: Path,
) -> None:
    """Unknown ticker, empty registry → suffix heuristic, unchanged behaviour."""
    _seed_registry({})
    assert ticker_currency("AAPL") == "USD"
    assert ticker_currency("AIR.PA") == "EUR"
    assert ticker_currency("SIKA.SW") == "CHF"
    assert ticker_currency("7203.T") == "JPY"
    assert ticker_currency("BTC-EUR") == "EUR"
    # The LSE default quoting convention is pence, so an unregistered `.L`
    # falls back to GBp rather than GBP.
    assert quote_currency("UNKNOWN.L") == "GBp"
    assert ticker_currency("UNKNOWN.L") == "GBP"


def test_the_heuristic_keys_on_the_exact_suffix_not_a_string_ending(
    midas_data_root: Path,
) -> None:
    """`EQNR.OL` must not be swallowed by the `.L` (pence) rule.

    The old heuristic was an `endswith` chain; it survived only because it
    never enumerated a colliding suffix. Adding Oslo, Athens and Toronto
    made the collisions real (`OL`/`L`, `AT`/`T`, `TO`/`O`).
    """
    _seed_registry({})
    assert quote_currency("EQNR.OL") == "NOK"
    assert quote_currency("LLOY.L") == "GBp"
    assert quote_currency("OPAP.AT") == "EUR"
    assert quote_currency("7203.T") == "JPY"
    assert quote_currency("RY.TO") == "CAD"
    # An unenumerated suffix keeps the pre-existing USD default.
    assert quote_currency("FOO.ZZ") == "USD"


def test_registry_entry_without_a_currency_falls_back(midas_data_root: Path) -> None:
    """A pre-currency registry entry (name only) must not resolve to None."""
    path = get_config().tickers_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"AIR.PA": {"name": "Airbus", "type": "equity"}}))
    assert ticker_currency("AIR.PA") == "EUR"


# ---------------------------------------------------------------------------
# 3. The /100 happens exactly once — latest_price is the single store reader
# ---------------------------------------------------------------------------


def test_latest_price_normalises_the_store_quote(midas_data_root: Path) -> None:
    _seed_registry({"LLOY.L": "GBp"})
    _seed_ohlcv("LLOY.L", 116.60)
    assert latest_price("LLOY.L", date(2026, 6, 1)) == Quote(
        pytest.approx(1.166), "GBP"
    )


def test_latest_price_is_none_when_the_store_has_no_row(midas_data_root: Path) -> None:
    _seed_registry({"LLOY.L": "GBp"})
    assert latest_price("LLOY.L", date(2026, 6, 1)) is None


def test_normalise_quote_is_idempotent_in_currency_but_not_in_price(
    midas_data_root: Path,
) -> None:
    """Guards against a second caller re-applying the scale.

    `normalise_quote` takes a RAW store quote and returns an ISO-denominated
    one. Feeding its own output back in divides by 100 again — which is
    exactly the double-application this test exists to make visible if a
    future caller pipes one into the other.
    """
    _seed_registry({"LLOY.L": "GBp"})
    once = normalise_quote("LLOY.L", 116.60)
    twice = normalise_quote("LLOY.L", once.price)
    assert once.price == pytest.approx(1.166)
    assert twice.price == pytest.approx(0.01166)
    assert once.currency == twice.currency == "GBP"


def test_paper_broker_reexport_still_resolves(midas_data_root: Path) -> None:
    """`engine.paper_broker._ticker_currency` is imported by three modules."""
    from engine.paper_broker import _ticker_currency

    _seed_registry({"LLOY.L": "GBp"})
    assert _ticker_currency("LLOY.L") == "GBP"


# ---------------------------------------------------------------------------
# 4. All four pricing paths agree on the same GBp position
#
# CLAUDE.md: "Three pricing paths and the fill path all consume this; they
# must all agree." 8 LLOY.L at 116.60p = GBP 9.328 = EUR 10.97 at 0.85
# EUR/GBP — not the EUR 1090.19 the ledger booked.
# ---------------------------------------------------------------------------

_LLOY_SHARES = 8.0
_LLOY_PENCE = 116.60
_EURGBP = 0.85
# 8 * 1.1660 GBP = 9.328 GBP; GBP->EUR = 1/0.85.
_LLOY_EUR = _LLOY_SHARES * (_LLOY_PENCE / 100.0) / _EURGBP


@pytest.fixture
def _gbp_book(midas_data_root: Path) -> Path:
    _seed_registry({"LLOY.L": "GBp"})
    _seed_ohlcv("LLOY.L", _LLOY_PENCE)
    _seed_ohlcv("EURGBP=X", _EURGBP)
    return midas_data_root


def test_valuation_path_prices_a_pence_position_in_pounds(_gbp_book: Path) -> None:
    from engine.valuation import portfolio_mtm

    summary = {
        "cash": 100.0,
        "currency": "EUR",
        "positions": [{"ticker": "LLOY.L", "shares": _LLOY_SHARES}],
    }
    assert portfolio_mtm(summary, date(2026, 6, 1)) == pytest.approx(100.0 + _LLOY_EUR)


def test_restatement_path_prices_a_pence_position_in_pounds(_gbp_book: Path) -> None:
    from engine.restatement import revalue_snapshot

    _, positions_value = revalue_snapshot(
        {"LLOY.L": _LLOY_SHARES},
        cash=0.0,
        market_date=date(2026, 6, 1),
        currency="EUR",
    )
    assert positions_value == pytest.approx(_LLOY_EUR)


def test_snapshot_path_prices_a_pence_position_in_pounds(_gbp_book: Path) -> None:
    from engine.types import Portfolio, Position
    from scripts.daily_session import _compute_positions_value

    portfolio = Portfolio(
        cash=0.0,
        positions=[
            Position(
                ticker="LLOY.L",
                shares=_LLOY_SHARES,
                avg_cost=_LLOY_PENCE / 100.0,
                date_opened=date(2026, 6, 1),
                grid_level=0,
            )
        ],
        last_updated=date(2026, 6, 1),
        currency="EUR",
    )
    assert _compute_positions_value(portfolio, date(2026, 6, 1)) == pytest.approx(
        _LLOY_EUR
    )


def test_fill_path_books_a_pence_order_in_pounds(_gbp_book: Path) -> None:
    """The fill notional and the recorded fill_price are both GBP, not pence."""
    from engine.orders import Order, append_order
    from engine.paper_broker import fill_day
    from engine.portfolio import PortfolioManager

    trade_date = date(2026, 6, 1)
    pm = PortfolioManager(base_dir=get_config().portfolios_dir)
    pm.initialize("world", initial_capital=10_000.0, currency="EUR")
    append_order(
        trade_date,
        Order(
            order_id="ord_lloy",
            agent_id="world",
            ts=datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc),
            action="BUY",
            ticker="LLOY.L",
            shares=_LLOY_SHARES,
            reasoning="pence regression",
            currency="EUR",
        ),
    )
    fills = fill_day(trade_date, pm)
    assert [f.status for f in fills] == ["filled"]
    assert fills[0].fill_currency == "GBP"
    assert fills[0].fill_price == pytest.approx(_LLOY_PENCE / 100.0)
    assert fills[0].notional_base == pytest.approx(_LLOY_EUR)


def test_triggered_fire_books_a_pence_order_in_pounds(_gbp_book: Path) -> None:
    """The watcher hands `_execute_triggered_order` a RAW quote too."""
    from engine.orders import Order
    from engine.paper_broker import _execute_triggered_order
    from engine.portfolio import PortfolioManager

    pm = PortfolioManager(base_dir=get_config().portfolios_dir)
    pm.initialize("world", initial_capital=10_000.0, currency="EUR")
    order = Order(
        order_id="ord_lloy_trig",
        agent_id="world",
        ts=datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc),
        action="BUY",
        ticker="LLOY.L",
        shares=_LLOY_SHARES,
        reasoning="pence regression",
        currency="EUR",
    )
    fill = _execute_triggered_order(order, date(2026, 6, 1), pm, _LLOY_PENCE)
    assert fill is not None and fill.status == "filled"
    assert fill.fill_currency == "GBP"
    assert fill.fill_price == pytest.approx(_LLOY_PENCE / 100.0)
    assert fill.notional_base == pytest.approx(_LLOY_EUR)


def test_a_malformed_registry_currency_falls_back_to_the_heuristic(
    midas_data_root: Path,
) -> None:
    """The vendor really does answer with a number sometimes.

    A full-universe sweep on 2026-08-07 returned `"3.3"` for ENX.AS. If that
    reached `engine.fx.convert` it would be unconvertible and take the whole
    holding book's valuation to None — a worse failure than the heuristic.
    `engine.tickers` refuses to write it; this is the read-side guard on the
    committed, hand-editable file.
    """
    _seed_registry({"ENX.AS": "3.3", "HMB.ST": "9.2"})
    assert ticker_currency("ENX.AS") == "EUR"  # .AS heuristic
    assert ticker_currency("HMB.ST") == "SEK"  # .ST heuristic
