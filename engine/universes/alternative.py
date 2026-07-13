"""Alternative data universe resolvers — congressional trades, insider buying,
high short-interest stocks.

Universe lists live in `data/universes/{name}.json`, **committed** to the
repo. File presence is authoritative. Refresh out-of-band via
`scripts/refresh_universes.py` (the weekly workflow) — the cloud sandbox
has no network and runtime fetches would crash. Curated fallback lists
seed the file on first run.
"""

from __future__ import annotations

import json

from engine.config import get_config


def _read_data(name: str) -> list[str] | None:
    path = get_config().universes_dir / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_data(name: str, tickers: list[str]) -> None:
    data_dir = get_config().universes_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{name}.json").write_text(json.dumps(tickers), encoding="utf-8")


# ---------------------------------------------------------------------------
# Congressional trades universe
# ---------------------------------------------------------------------------

# Curated fallback: ~25 stocks frequently traded by U.S. Congress members.
_CONGRESSIONAL_FALLBACK: list[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "BAC",
    "GS",
    "V",
    "MA",
    "UNH",
    "JNJ",
    "PFE",
    "XOM",
    "CVX",
    "COP",
    "LMT",
    "RTX",
    "BA",
    "NOC",
    "DIS",
    "NFLX",
    "CRM",
    "PANW",
    "PLTR",
]


def get_congressional_tickers() -> list[str]:
    """Return tickers frequently traded by U.S. Congress members."""
    cached = _read_data("congressional")
    if cached is not None:
        return cached
    tickers = sorted(_CONGRESSIONAL_FALLBACK)
    _write_data("congressional", tickers)
    return tickers


def refresh_congressional() -> list[str]:
    """Re-seed the congressional list from the curated fallback.

    Live Quiver/Finnhub integration is out of scope — this seed exists so
    the data file is reproducible from code if it's ever lost.
    """
    tickers = sorted(_CONGRESSIONAL_FALLBACK)
    _write_data("congressional", tickers)
    return tickers


# ---------------------------------------------------------------------------
# Insider buying universe
# ---------------------------------------------------------------------------

# Curated list of stocks with historically significant insider buying activity.
# PXD removed 2026-04-17 (acquired by Exxon in October 2023).
_INSIDER_FALLBACK: list[str] = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOG",
    "META",
    "BRK-B",
    "JPM",
    "BAC",
    "WFC",
    "C",
    "XOM",
    "CVX",
    "OXY",
    "COP",
    "LMT",
    "RTX",
    "NOC",
    "BA",
    "GD",
    "UNH",
    "CVS",
    "HCA",
    "CI",
    "MCK",
    "COST",
    "HD",
    "LOW",
    "TGT",
    "WMT",
]


def get_insider_tickers() -> list[str]:
    """Return stocks with historically significant insider buying activity."""
    cached = _read_data("insider")
    if cached is not None:
        return cached
    tickers = sorted(_INSIDER_FALLBACK)
    _write_data("insider", tickers)
    return tickers


def refresh_insider() -> list[str]:
    tickers = sorted(_INSIDER_FALLBACK)
    _write_data("insider", tickers)
    return tickers


# ---------------------------------------------------------------------------
# High short-interest universe
# ---------------------------------------------------------------------------

# Curated list of stocks historically carrying elevated short interest.
# Last refreshed 2026-04-17: removed BBBY, PRTY, JWN, OSTK, WISH, EXPR
# (delisted / bankrupt between 2023-2025).
_HIGH_SHORT_FALLBACK: list[str] = [
    "GME",
    "AMC",
    "SPCE",
    "PLTR",
    "RIVN",
    "LCID",
    "NKLA",
    "WKHS",
    "RIDE",
    "BYND",
    "CVNA",
    "BBWI",
    "M",
    "KSS",
    "FUBO",
    "SFIX",
    "CLOV",
    "FIZZ",
    "PUBM",
]


def get_high_short_tickers() -> list[str]:
    """Return stocks with historically high short interest."""
    cached = _read_data("high-short")
    if cached is not None:
        return cached
    tickers = sorted(_HIGH_SHORT_FALLBACK)
    _write_data("high-short", tickers)
    return tickers


def refresh_high_short() -> list[str]:
    tickers = sorted(_HIGH_SHORT_FALLBACK)
    _write_data("high-short", tickers)
    return tickers


# ---------------------------------------------------------------------------
# Bulk refresh
# ---------------------------------------------------------------------------


def refresh_all_alternatives() -> dict[str, int]:
    """Re-seed every alternative universe from its curated fallback list."""
    return {
        "congressional": len(refresh_congressional()),
        "insider": len(refresh_insider()),
        "high-short": len(refresh_high_short()),
    }
