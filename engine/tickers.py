"""Ticker name registry — maps symbol → human-readable name + asset type.

The registry is committed to git at data/tickers.json so the site can read it
at build time and so the sandboxed daily-session agent can see it. It is
populated and refreshed by scripts/fetch_ohlcv.py, which already calls
yfinance for every symbol once a week.
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
    Otherwise replace the existing entry wholesale.
    """
    out: Registry = dict(existing)
    for key, info in fresh.items():
        if info.get("name") is None and key in out and out[key].get("name") is not None:
            continue
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
    """
    name = _yfinance_name(info) or _shape_name(symbol)
    return {"name": name, "type": _infer_type(symbol, info)}
