"""Quote-unit resolution — ticker → (ISO currency, price in that currency).

Two facts the rest of the engine must never have to know individually:

1. **A ticker's quote currency is vendor data, not a property of its suffix.**
   `PHAG.L` and `LLOY.L` share a suffix and quote in different currencies
   (USD and pence respectively). The suffix heuristic that used to answer
   this question got `.L` wrong for both, and answered `USD` for every
   European suffix it did not enumerate — 126 tickers in `world`'s universe
   alone (`.ST`, `.MC`, `.BR`, `.HE`, `.OL`, `.CO`, `.VI`, `.IR`, `.LS`,
   `.LU`, `.WA`, `.AT`). Currency is now resolved from the vendor, captured
   into `data/tickers.json` by `scripts/fetch_ohlcv.py` (which already
   fetches `.info` per symbol for the name registry, so this costs no extra
   call), and the heuristic survives only as a last resort for tickers the
   vendor cannot answer for.

2. **`GBp` is a unit, not a currency.** The LSE quotes most of its order
   book in pence. `GBp 116.60` is `GBP 1.1660`. The store holds the raw
   vendor quote — pence — because that is what the vendor publishes and what
   an agent reading `data/market/ohlcv/LLOY.L.jsonl` sees. Converting it is
   the job of exactly one function here, `normalise_quote`, so that no two
   pricing paths can disagree about whether it has already happened.

Resolution order (highest first):

  1. `data/ticker_currencies.json` — the hand-maintained override map. The
     only way to correct a ticker the vendor answers wrongly, or one it has
     stopped answering for (`WDFC.SW` is delisted at Yahoo and still needs a
     currency).
  2. `data/tickers.json` — the vendor-captured registry (`currency` field).
  3. The suffix heuristic below.

Stdlib + engine only: this module is in the `midas-core` sync manifest, so
it must never import a vendor client. The vendor call lives in
`scripts/fetch_ohlcv.py`.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import NamedTuple

from engine.config import get_config, register_reset_callback
from engine.ohlcv_store import latest_close_on_or_before


class Quote(NamedTuple):
    """A price already denominated in an ISO currency.

    `price` is never a sub-unit quote: `normalise_quote` has divided a pence
    quote by 100 before this exists. `currency` is therefore always an ISO
    4217 code, never `GBp`.
    """

    price: float
    currency: str


# Vendor sub-unit codes: quote unit → (ISO currency, multiplier to reach it).
# Yahoo reports `GBp` for London pence, `ZAc` for South African cents and
# `ILA` for Israeli agorot. Only `GBp` occurs in this desk's universes today;
# the other two are listed because they are the same 100:1 trap and cost
# nothing to pre-empt (neither has an FX pair in `engine.fx`, so such a book
# would refuse to value rather than value wrongly — the right failure).
#
# **This lookup is deliberately case-SENSITIVE.** `GBp` and `GBP` differ only
# by case and mean quantities 100x apart; upper-casing before the lookup
# would divide every genuine sterling quote by 100. `GBX` (the LSE's own
# spelling of the same pence unit) is the one alias that is unambiguous in
# upper case, so it is listed explicitly rather than folded in.
_SUB_UNITS: dict[str, tuple[str, float]] = {
    "GBp": ("GBP", 0.01),
    "GBX": ("GBP", 0.01),
    "ZAc": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
}

# Exchange suffix (after the LAST dot, upper-cased) → quote unit. Last resort
# only: it fires for a ticker absent from both the override map and the
# registry, which in practice means a symbol `scripts/fetch_ohlcv.py` has
# never seen — and therefore has no price for either.
#
# `L` maps to GBp because pence is the LSE's default quoting convention. The
# exceptions (USD- and EUR-quoted ETFs like PHAG.L) are exactly what the
# registry layer above exists to carry; no rule on the string can find them.
#
# Keyed on the exact suffix rather than matched with `endswith`, so `OL`
# (Oslo) cannot be swallowed by an `.L` rule and `AT` (Athens) cannot be
# swallowed by `.T`. The old heuristic's `endswith` chain worked only because
# it never enumerated a suffix that collided.
_SUFFIX_UNITS: dict[str, str] = {
    "PA": "EUR",  # Paris
    "DE": "EUR",  # Xetra
    "F": "EUR",  # Frankfurt
    "AS": "EUR",  # Amsterdam
    "MI": "EUR",  # Milan
    "MC": "EUR",  # Madrid
    "BR": "EUR",  # Brussels
    "HE": "EUR",  # Helsinki
    "VI": "EUR",  # Vienna
    "IR": "EUR",  # Dublin
    "LS": "EUR",  # Lisbon
    "LU": "EUR",  # Luxembourg
    "AT": "EUR",  # Athens
    "ST": "SEK",  # Stockholm — NOT the eurozone
    "OL": "NOK",  # Oslo — NOT the eurozone
    "CO": "DKK",  # Copenhagen — NOT the eurozone
    "WA": "PLN",  # Warsaw — NOT the eurozone
    "L": "GBp",  # London, in pence
    "SW": "CHF",  # SIX Swiss
    "T": "JPY",  # Tokyo
    "TO": "CAD",  # Toronto
    "HK": "HKD",  # Hong Kong
    "AX": "AUD",  # ASX
}


#: A quote unit is three letters — ISO 4217 or a vendor sub-unit. Mirrors
#: `engine.tickers._CURRENCY_CODE`, which applies the same shape check at
#: capture time; this one guards the read side of the committed file.
_CURRENCY_CODE = re.compile(r"^[A-Za-z]{3}$")

_TICKER_CURRENCY_OVERRIDES: dict[str, str] | None = None
_TICKER_REGISTRY_CURRENCIES: dict[str, str] | None = None


def _load_ticker_currency_overrides() -> dict[str, str]:
    """The hand-maintained `data/ticker_currencies.json` map, memoised."""
    global _TICKER_CURRENCY_OVERRIDES
    if _TICKER_CURRENCY_OVERRIDES is None:
        path = get_config().ticker_currencies_path
        if path.exists():
            try:
                _TICKER_CURRENCY_OVERRIDES = json.loads(
                    path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                _TICKER_CURRENCY_OVERRIDES = {}
        else:
            _TICKER_CURRENCY_OVERRIDES = {}
    return _TICKER_CURRENCY_OVERRIDES


def _load_registry_currencies() -> dict[str, str]:
    """Vendor-captured `currency` per symbol from `data/tickers.json`, memoised.

    Entries predating the currency field (name + type only) are simply
    absent from the returned map, so they fall through to the heuristic
    rather than resolving to `None`.

    So is anything not shaped like a currency code. `engine.tickers` already
    refuses to write those, but the registry is a committed, hand-editable
    file and this is its consumer: a value like `"3.3"` (which the vendor
    really did return for ENX.AS on 2026-08-07) must not reach
    `engine.fx.convert`, where an unconvertible currency takes the entire
    book's valuation to `None`.
    """
    global _TICKER_REGISTRY_CURRENCIES
    if _TICKER_REGISTRY_CURRENCIES is None:
        path = get_config().tickers_path
        out: dict[str, str] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            for symbol, info in (raw or {}).items():
                if not isinstance(info, dict):
                    continue
                ccy = info.get("currency")
                if isinstance(ccy, str) and _CURRENCY_CODE.match(ccy.strip()):
                    out[symbol] = ccy.strip()
        _TICKER_REGISTRY_CURRENCIES = out
    return _TICKER_REGISTRY_CURRENCIES


def _reset_caches() -> None:
    """Invalidate both memoised maps (fired by `reset_config_cache`)."""
    global _TICKER_CURRENCY_OVERRIDES, _TICKER_REGISTRY_CURRENCIES
    _TICKER_CURRENCY_OVERRIDES = None
    _TICKER_REGISTRY_CURRENCIES = None


register_reset_callback(_reset_caches)


def _heuristic_unit(ticker: str) -> str:
    """Suffix-only guess. USD when nothing matches — the pre-existing default."""
    if ticker.endswith("-EUR"):
        return "EUR"
    if ticker.endswith("-USD"):
        return "USD"
    if ticker.endswith("-GBP"):
        return "GBP"
    _, dot, suffix = ticker.rpartition(".")
    if dot:
        return _SUFFIX_UNITS.get(suffix.upper(), "USD")
    return "USD"


def quote_currency(ticker: str) -> str:
    """The unit the store's prices for `ticker` are denominated in.

    May be a sub-unit code (`GBp`) — this is the raw answer, not an ISO
    code. Use `ticker_currency` for the ISO code and `normalise_quote` /
    `latest_price` to get a price you can pass to `engine.fx.convert`.
    """
    overrides = _load_ticker_currency_overrides()
    if ticker in overrides:
        return overrides[ticker]
    registry = _load_registry_currencies()
    if ticker in registry:
        return registry[ticker]
    return _heuristic_unit(ticker)


def _iso_and_scale(unit: str) -> tuple[str, float]:
    """Split a quote unit into (ISO currency, price multiplier).

    Case-sensitive on purpose — see `_SUB_UNITS`. Anything not a known
    sub-unit is passed through upper-cased, so a hand-written override of
    `"eur"` still resolves to `EUR` at scale 1.0.
    """
    if unit in _SUB_UNITS:
        return _SUB_UNITS[unit]
    return unit.upper(), 1.0


def ticker_currency(ticker: str) -> str:
    """Resolve ticker → ISO 4217 currency code.

    `GBp` (pence) resolves to `GBP`: the sub-unit is a quoting convention,
    not a currency, and `engine.fx` only knows ISO codes. The 1/100 that
    goes with it is applied to the *price*, by `normalise_quote`.
    """
    return _iso_and_scale(quote_currency(ticker))[0]


def normalise_quote(ticker: str, raw_price: float) -> Quote:
    """Convert a RAW store/vendor quote for `ticker` into its ISO currency.

    This is the only place the pence→pounds division exists. Every pricing
    path (the fill path in `engine.paper_broker`, `engine.valuation`,
    `engine.restatement`, `scripts.daily_session`) reaches a normalised
    price through this function or through `latest_price`, which wraps it —
    so the conversion cannot be applied by one caller and skipped by
    another, nor applied twice by two callers each assuming the other
    didn't.

    `raw_price` must be a quote as stored (`data/market/ohlcv/*.jsonl`) or
    as observed live by the trigger watcher — never the output of a previous
    `normalise_quote` call.
    """
    iso, scale = _iso_and_scale(quote_currency(ticker))
    return Quote(raw_price * scale, iso)


def latest_price(
    ticker: str, on: date | None = None, store: Path | None = None
) -> Quote | None:
    """Latest close on or before `on`, normalised to the ticker's ISO currency.

    Thin composition of `engine.ohlcv_store.latest_close_on_or_before` and
    `normalise_quote`. Returns `None` on exactly the same condition the raw
    reader does — ticker absent from the store, or no row on or before `on`.

    Prefer this over calling the raw reader plus `ticker_currency`
    separately: that pair is what silently dropped the pence conversion on
    three pricing paths at once.
    """
    raw = latest_close_on_or_before(ticker, on, store=store)
    if raw is None:
        return None
    return normalise_quote(ticker, raw)
