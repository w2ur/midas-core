"""Foreign exchange conversion helpers.

Reads daily FX rates from the committed OHLCV store at data/market/ohlcv/
(pairs fetched via the forex-majors universe). Primary use case: converting
non-EUR portfolio values to EUR for cross-agent comparison and real-money
reporting.

Rates are daily closes — intraday precision is neither available in the
store nor needed for portfolio-level reporting.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Iterable

from engine.config import get_config

# Direct pairs available in the forex-majors universe.
# Each entry: (from, to) → yfinance ticker, inverted (True = stored rate is to/from, use 1/rate)
_DIRECT: dict[tuple[str, str], tuple[str, bool]] = {
    ("USD", "EUR"): (
        "EURUSD=X",
        True,
    ),  # Stored rate is EUR→USD, so EUR per USD = 1/rate
    ("GBP", "EUR"): ("EURGBP=X", True),
    ("JPY", "EUR"): ("EURJPY=X", True),
    ("EUR", "USD"): ("EURUSD=X", False),
    ("EUR", "GBP"): ("EURGBP=X", False),
    ("EUR", "JPY"): ("EURJPY=X", False),
}


def _load_store_series(ticker: str) -> dict[str, float]:
    """Return {date_iso: raw close} for a ticker, or empty dict if missing.

    Raw `close`, never `adj_close` — same basis as every other read path
    (`engine.ohlcv_store` module docstring). For an FX pair the two fields
    are equal anyway (a currency pair pays no dividend); reading `close`
    keeps the rule uniform rather than resting on that.
    """
    path = get_config().ohlcv_dir / f"{ticker}.jsonl"
    if not path.exists():
        return {}
    series: dict[str, float] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = row.get("date")
            close = row.get("close")
            if d and close is not None:
                series[d] = float(close)
    return series


def _latest_on_or_before(series: dict[str, float], target: date) -> float | None:
    """Return the latest value with date ≤ target, or None if none."""
    target_iso = target.isoformat()
    eligible = [d for d in series if d <= target_iso]
    if not eligible:
        return None
    return series[max(eligible)]


def get_rate(
    from_currency: str, to_currency: str, on: date | None = None
) -> float | None:
    """Return the exchange rate: how many `to_currency` per 1 `from_currency` on `on`.

    Returns None if the rate cannot be computed from the available data.
    Uses the most recent available close on or before `on` (defaults to today).
    """
    if from_currency == to_currency:
        return 1.0
    if on is None:
        on = date.today()

    key = (from_currency, to_currency)
    if key in _DIRECT:
        ticker, inverted = _DIRECT[key]
        series = _load_store_series(ticker)
        val = _latest_on_or_before(series, on)
        if val is None or val == 0:
            return None
        return 1.0 / val if inverted else val

    # Indirect via USD — e.g., CHF→EUR = (USD per CHF) / (USD per EUR) = 1/USDCHF × 1/(1/EURUSD)
    # Compose: from→USD then USD→to.
    if from_currency != "USD" and to_currency != "USD":
        to_usd = get_rate(from_currency, "USD", on)
        from_usd = get_rate("USD", to_currency, on)
        if to_usd is None or from_usd is None:
            return None
        return to_usd * from_usd

    # Fallback pairs quoted against USD.
    usd_pair_map = {
        ("CHF", "USD"): ("USDCHF=X", True),
        ("USD", "CHF"): ("USDCHF=X", False),
        ("CAD", "USD"): ("USDCAD=X", True),
        ("USD", "CAD"): ("USDCAD=X", False),
        ("AUD", "USD"): ("AUDUSD=X", False),
        ("USD", "AUD"): ("AUDUSD=X", True),
        ("NZD", "USD"): ("NZDUSD=X", False),
        ("USD", "NZD"): ("NZDUSD=X", True),
    }
    if key in usd_pair_map:
        ticker, inverted = usd_pair_map[key]
        series = _load_store_series(ticker)
        val = _latest_on_or_before(series, on)
        if val is None or val == 0:
            return None
        return 1.0 / val if inverted else val

    return None


def convert(
    amount: float, from_currency: str, to_currency: str, on: date | None = None
) -> float | None:
    """Convert `amount` from one currency to another using the rate on `on`.

    Returns None if the rate is unavailable. Useful for portfolio valuation.
    """
    rate = get_rate(from_currency, to_currency, on)
    if rate is None:
        return None
    return amount * rate


def to_eur(amount: float, from_currency: str, on: date | None = None) -> float | None:
    """Convenience: convert `amount` from `from_currency` to EUR on `on`."""
    return convert(amount, from_currency, "EUR", on)
