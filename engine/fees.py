"""Fee model for paper fills — per-asset-class fee schedule.

This module implements a realistic fee model for the Midas paper broker.
Fees apply from the deploy date of this module forward only; historical
portfolio states and inbox records authored before this change remain
untouched (no backfill, no restatement).

Fee schedule basis:
- equity/ETF: Interactive Brokers Ireland (IBIE) tiered fixed rate,
  approximately 0.05% with a EUR 1.25 minimum per order (2026-06 IBKR research).
- crypto: Kraken spot taker rate at base tier (0.40%). The maker rate of
  0.25% applies to limit orders on the live broker but is not modeled here;
  paper fills use the conservative taker assumption.
- fx: Spread proxy of approximately 0.002% (2 pips on a 10k notional unit),
  representing OANDA Europe mid-spread estimate for major pairs.

Usage:
    from engine.fees import classify_ticker, fee_for

    asset_class = classify_ticker("BTC-EUR")   # -> "crypto"
    fee = fee_for("BTC-EUR", 4000.0)           # -> 16.0
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from engine.triggers import is_crypto_ticker

# ---------------------------------------------------------------------------
# Fee schedule constants
# ---------------------------------------------------------------------------

# equity/ETF: IBIE tiered rate ~0.05% with EUR 1.25 floor per order
_EQUITY_RATE = 0.0005  # 0.05%
_EQUITY_FLOOR = 1.25  # EUR 1.25 minimum commission
_EQUITY_FLOOR_CURRENCY = "EUR"  # the currency that floor is quoted in

# crypto: Kraken spot taker at base tier (0.40%)
# Note: live-broker optimization uses maker rate (0.25%) for limit orders;
# paper fills use the conservative taker assumption.
_CRYPTO_RATE = 0.0040  # 0.40%

# fx: spread proxy (~0.002%) for OANDA Europe major pair mid-spread estimate
_FX_RATE = 0.00002  # 0.002%

AssetClass = Literal["equity", "crypto", "fx"]


# ---------------------------------------------------------------------------
# Config-driven rate resolvers (prefer jurisdiction config, fall back to
# module-constant defaults so legacy numbers are byte-identical when no
# jurisdiction block is present in roster.yaml).
# ---------------------------------------------------------------------------


def _rates() -> dict:
    from engine.config import get_config

    return get_config().jurisdiction.fees or {}


def _equity_rate() -> float:
    return float(_rates().get("equity", {}).get("rate_pct", _EQUITY_RATE * 100)) / 100


def _equity_floor() -> float:
    return float(_rates().get("equity", {}).get("floor", _EQUITY_FLOOR))


def _equity_floor_currency() -> str:
    return str(
        _rates().get("equity", {}).get("floor_currency", _EQUITY_FLOOR_CURRENCY)
    )


def _crypto_rate() -> float:
    return float(_rates().get("crypto", {}).get("taker_pct", _CRYPTO_RATE * 100)) / 100


def _fx_rate() -> float:
    return float(_rates().get("fx", {}).get("spread_pct", _FX_RATE * 100)) / 100


def classify_ticker(ticker: str) -> AssetClass:
    """Classify a ticker symbol into an asset class for fee computation.

    Classification rules (in priority order):
    1. crypto  — base symbol in the known crypto allowlist with a fiat quote
                 (e.g. BTC-EUR, ETH-USD). Uses is_crypto_ticker from engine.triggers.
    2. fx      — ticker ending with ``=X`` (yfinance FX pair convention, e.g. EURUSD=X).
    3. equity  — everything else (equities, ETFs, indices).

    Parameters
    ----------
    ticker:
        Raw ticker symbol as used in orders and the OHLCV store.

    Returns
    -------
    AssetClass
        One of "crypto", "fx", or "equity".
    """
    # Normalize once so the function is safe to reuse outside the broker path
    # (the OHLCV store is uppercase-by-convention, but a future fee-disclosure
    # UI or the Manager context may call this with raw/None input).
    ticker = (ticker or "").upper()
    if is_crypto_ticker(ticker):
        return "crypto"
    if ticker.endswith("=X"):
        return "fx"
    return "equity"


def fee_for(
    ticker: str,
    notional: float,
    currency: str | None = None,
    on: date | None = None,
) -> float:
    """Compute the brokerage fee for a single fill.

    Parameters
    ----------
    ticker:
        Ticker symbol — used to determine the asset class.
    notional:
        Trade notional in the agent's base currency (post-FX conversion).
    currency:
        The book's base currency. Only the equity **floor** depends on it:
        the floor is a fixed EUR amount (IBIE's EUR 1.25 minimum), and it was
        being charged as-is on the USD books — a dollar-denominated book paid
        "$1.25" for a minimum that is actually EUR 1.25 (2026-08-07 review,
        W7.4). The percentage rates are scale-free and need no conversion.
        Omit on a book already denominated in the floor's currency.
    on:
        Date for the FX lookup; defaults to the latest available rate.

    Returns
    -------
    float
        Fee amount in the agent's base currency. Always >= 0.

    Notes
    -----
    If the floor cannot be converted (no FX rate), the unconverted floor is
    used rather than raising. This is deliberate and bounded: the floor binds
    only on orders small enough for a rate-based fee to fall under ~EUR 1.25,
    the error is cents, and on the broker path the FX rate is necessarily
    available already — the notional itself was converted with it, and a
    missing rate is `NO_FX_RATE` long before this is called.
    """
    asset_class = classify_ticker(ticker)

    if asset_class == "equity":
        return max(_equity_floor_in(currency, on), _equity_rate() * notional)
    elif asset_class == "crypto":
        return _crypto_rate() * notional
    else:  # fx
        return _fx_rate() * notional


def _equity_floor_in(currency: str | None, on: date | None) -> float:
    """The equity commission floor expressed in *currency*."""
    floor = _equity_floor()
    floor_ccy = _equity_floor_currency()
    if not currency or currency.upper() == floor_ccy.upper():
        return floor
    from engine.fx import convert as _fx_convert

    converted = _fx_convert(floor, floor_ccy, currency, on)
    return floor if converted is None else converted
