"""Ticker registry — maps symbol → human-readable name, asset type, currency.

The registry is committed to git at data/tickers.json so the site can read it
at build time and so the sandboxed daily-session agent can see it. It is
populated and refreshed by scripts/fetch_ohlcv.py, which already calls
yfinance for every symbol once a week.

``currency`` is the vendor's quote unit **verbatim** — including sub-units
such as ``GBp`` (LSE pence), which is NOT an ISO code. Storing it raw is
deliberate: it is the only place the 100:1 distinction between a pence quote
and a sterling quote survives, and ``engine.quotes`` is the single consumer
that resolves it into an ISO currency plus a price scale. Do not "clean" a
``GBp`` here into ``GBP``; that erases the fact the store's prices are pence
and re-creates the 100x valuation defect this field was added to close.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

from engine.config import get_config


class TickerInfo(TypedDict):
    name: str | None
    type: str  # "equity" | "etf" | "crypto" | "forex" | "unknown"
    currency: str | None  # vendor quote unit verbatim, e.g. "USD", "EUR", "GBp"


Registry = dict[str, TickerInfo]


def load_registry(path: Path | None = None) -> Registry:
    """Load the registry from disk. Returns {} when the file is missing."""
    if path is None:
        path = get_config().tickers_path
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_registry(reg: Registry, path: Path | None = None) -> None:
    """Write the registry to disk, sorted by symbol for diff stability."""
    if path is None:
        path = get_config().tickers_path
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: reg[k] for k in sorted(reg)}
    with path.open("w") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)
        f.write("\n")


def merge(existing: Registry, fresh: Registry) -> Registry:
    """Merge a freshly-fetched registry into the existing one.

    Rule: when ``fresh[key].name`` is ``None``, keep the existing entry
    intact (a transient yfinance failure must not blank out a known name).
    Otherwise replace the existing entry wholesale — except that a known
    ``currency`` is carried forward when the fresh entry has none, for the
    same reason: a currency that silently reverts to ``None`` drops the
    ticker back onto the suffix heuristic, which is exactly what got ``.L``
    valued in pounds instead of pence.
    """
    out: Registry = dict(existing)
    for key, info in fresh.items():
        prior = out.get(key)
        if (
            info.get("name") is None
            and prior is not None
            and prior.get("name") is not None
        ):
            continue
        if info.get("currency") is None and prior is not None and prior.get("currency"):
            info = {**info, "currency": prior["currency"]}
        out[key] = info
    return out


_CRYPTO_PATTERN = re.compile(r"^([A-Z]{2,6})-(USD|EUR)$")
_FOREX_PATTERN = re.compile(r"^([A-Z]{3})([A-Z]{3})=X$")

_CRYPTO_NAMES: dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "XRP": "XRP",
    "ADA": "Cardano",
    "DOGE": "Dogecoin",
    "LTC": "Litecoin",
    "BCH": "Bitcoin Cash",
    "LINK": "Chainlink",
    "DOT": "Polkadot",
    "AVAX": "Avalanche",
    "MATIC": "Polygon",
    "ATOM": "Cosmos",
    "XLM": "Stellar",
    "TRX": "TRON",
    "UNI": "Uniswap",
}

_QUOTE_TYPE_MAP = {
    "EQUITY": "equity",
    "ETF": "etf",
    "MUTUALFUND": "etf",
    "CRYPTOCURRENCY": "crypto",
    "CURRENCY": "forex",
}


def _infer_type(symbol: str, info: dict | None) -> str:
    if info:
        qt = (info.get("quoteType") or "").upper()
        if qt in _QUOTE_TYPE_MAP:
            return _QUOTE_TYPE_MAP[qt]
    if _CRYPTO_PATTERN.match(symbol):
        return "crypto"
    if _FOREX_PATTERN.match(symbol):
        return "forex"
    return "unknown"


#: A quote unit is three letters — ISO 4217 (``USD``) or a vendor sub-unit
#: (``GBp``, ``ZAc``, ``ILA``). Anything else is a vendor glitch, not a
#: currency: a full-universe sweep on 2026-08-07 came back with ``"3.3"`` for
#: ENX.AS and ``"9.2"`` for HMB.ST. Those must be rejected at capture, not
#: written into the registry, where they would resolve to a currency
#: ``engine.fx`` cannot convert and take the whole book's valuation to None.
_CURRENCY_CODE = re.compile(r"^[A-Za-z]{3}$")


def _yfinance_currency(info: dict | None) -> str | None:
    """The vendor's quote unit for a symbol, verbatim (``GBp`` stays ``GBp``).

    ``financialCurrency`` is deliberately NOT used as a fallback: for an LSE
    listing it reports the reporting currency (``GBP``) while the quote is in
    pence, so falling back to it would silently mis-scale the very tickers
    this field exists to get right.

    Returns ``None`` — falling the symbol back to the suffix heuristic — when
    the vendor's answer is not shaped like a currency code.
    """
    if not info:
        return None
    value = info.get("currency")
    if isinstance(value, str) and _CURRENCY_CODE.match(value.strip()):
        return value.strip()
    return None


def _yfinance_name(info: dict | None) -> str | None:
    if not info:
        return None
    for key in ("longName", "shortName"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _shape_name(symbol: str) -> str | None:
    crypto_match = _CRYPTO_PATTERN.match(symbol)
    if crypto_match:
        base = crypto_match.group(1)
        return _CRYPTO_NAMES.get(base)  # may be None for unknown coins
    forex_match = _FOREX_PATTERN.match(symbol)
    if forex_match:
        return f"{forex_match.group(1)}/{forex_match.group(2)}"
    return None


def resolve_name(symbol: str, info: dict | None) -> TickerInfo:
    """Resolve a ticker symbol to its (name, type) entry.

    Resolution order for ``name``:
      1. yfinance ``longName`` if non-empty.
      2. yfinance ``shortName`` if non-empty.
      3. Symbol-shape heuristic (crypto static map, forex pair formatter).
      4. ``None``.

    ``type`` is taken from ``info['quoteType']`` when available, otherwise
    inferred from the symbol shape.

    ``currency`` is ``info['currency']`` verbatim, or ``None`` when the
    vendor did not answer — in which case ``engine.quotes`` falls back to its
    suffix heuristic for that symbol.
    """
    name = _yfinance_name(info) or _shape_name(symbol)
    return {
        "name": name,
        "type": _infer_type(symbol, info),
        "currency": _yfinance_currency(info),
    }
