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
   book in pence. `GBp 116.60` is `GBP 1.1660`. Since 2026-08-07 the store
   is **ISO-denominated**: the pence→pounds division happens exactly once,
   on the *ingest* side, in `scripts.fetch_ohlcv._normalise_vendor_units`
   via `vendor_unit_scale`. A row in `data/market/ohlcv/LLOY.L.jsonl` is
   therefore already pounds.

   The split between `normalise_vendor_quote` (converts a **vendor** price)
   and `store_quote` (labels a **stored** price, scaling nothing) is what
   makes double-application structurally impossible. **A read path must
   never scale** — doing so divides every LSE price by 100 twice.

Resolution order (highest first):

  1. `data/ticker_currencies.json` — the hand-maintained override map. The
     only way to correct a ticker the vendor answers wrongly, or one it has
     stopped answering for (`WDFC.SW` is delisted at Yahoo and still needs a
     currency).
  2. `data/tickers.json` — the vendor-captured registry (`currency` field).
  3. The suffix heuristic below — and if that has no entry for the suffix,
     **`None`**, not a guess. Everything here can return `None`; the broker
     turns it into a `CURRENCY_UNRESOLVED` rejection rather than trading a
     position it cannot denominate.

Stdlib + engine only: this module is in the `midas-core` sync manifest, so
it must never import a vendor client. The vendor call lives in
`scripts/fetch_ohlcv.py`.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import NamedTuple

from engine.config import get_config, register_reset_callback
from engine.ohlcv_store import latest_close_on_or_before

logger = logging.getLogger(__name__)


class Quote(NamedTuple):
    """A price already denominated in an ISO currency.

    `price` is never a sub-unit quote: the pence→pounds division happened at
    ingest (`normalise_vendor_quote`, via `vendor_unit_scale`) before this
    exists. `currency` is therefore always an ISO 4217 code, never `GBp` —
    and never `None`, because a price whose currency is unresolvable is
    returned as `None` instead of as a Quote carrying an empty unit.
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
    "V": "CAD",  # TSX Venture
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
            except json.JSONDecodeError as exc:
                # Degrading to {} demotes every ticker to the layers below —
                # a whole-desk change of behaviour from one stray comma, and
                # invisible before this line existed (2026-08-07, W7.4). The
                # empty map is still the right fallback (fail-closed: an
                # unresolvable ticker becomes CURRENCY_UNRESOLVED at the
                # broker rather than trading at a guessed currency), but it
                # must not be quiet.
                logger.error(
                    "ticker-currency override map %s is unparseable (%s) — "
                    "EVERY ticker now falls through to the vendor registry "
                    "and the suffix heuristic. Fix the file.",
                    path,
                    exc,
                )
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
            except json.JSONDecodeError as exc:
                # Same failure shape as the override map above: an empty
                # registry silently demotes 1,000+ tickers to the suffix
                # heuristic, which is the layer the 2026-08-07 defect lived in.
                logger.error(
                    "ticker registry %s is unparseable (%s) — EVERY ticker "
                    "now falls through to the suffix heuristic. Fix the file.",
                    path,
                    exc,
                )
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


def _heuristic_unit(ticker: str) -> str | None:
    """Suffix-only guess, or `None` when the suffix is one we do not know.

    An **unenumerated suffix used to answer `USD`**, which is how 126 tickers
    in `world`'s universe alone (`.ST`, `.MC`, `.BR`, `.HE`, `.OL`, `.CO`,
    `.VI`, `.IR`, `.LS`, `.LU`, `.WA`, `.AT`) were silently valued as dollars
    — the fourth failure mode of the 2026-08-07 quote-currency defect, and
    the one that produced no visible symptom because a wrong currency still
    prices. Guessing is not better than refusing: a `None` here becomes a
    `CURRENCY_UNRESOLVED` rejection at the broker, which is loud.

    A bare ticker with no suffix keeps `USD`. That is not a guess about an
    unknown exchange — it is Yahoo's convention for a US listing, and the
    only shape it can take.

    **An FX pair (`…=X`) refuses.** A pair quotes in its *second* leg, which
    no suffix rule can see, and `…=X` carries no dot — so this function used
    to fall through to the bare-ticker branch and answer `USD` for every
    pair. Right for `EURUSD=X` by luck, wrong for `EURGBP=X`, `GBPJPY=X`,
    `USDJPY=X` and `EURJPY=X`, and wrong in the silent way: a mislabelled
    price still prices. The vendor registry answers all of these correctly
    (verified 2026-08-07 against the committed `data/tickers.json`), so the
    heuristic has nothing to add and refuses instead of guessing. This also
    puts it back in step with `site/src/lib/ohlcv.ts`, which already
    refused; `tests/test_quote_parity.py` now pins the two together.
    """
    if ticker.endswith("-EUR"):
        return "EUR"
    if ticker.endswith("-USD"):
        return "USD"
    if ticker.endswith("-GBP"):
        return "GBP"
    if ticker.endswith("=X"):
        return None
    _, dot, suffix = ticker.rpartition(".")
    if dot:
        return _SUFFIX_UNITS.get(suffix.upper())
    return "USD"


def vendor_quote_unit(ticker: str) -> str | None:
    """The unit the VENDOR quotes `ticker` in, or `None` if unresolvable.

    May be a sub-unit code (`GBp`) — this is the raw answer, not an ISO
    code. It describes what arrives from yfinance, **not** what is in the
    store: since 2026-08-07 the store is normalised to ISO at ingest (see
    the module docstring), so a stored LSE close is already pounds.

    Callers wanting a price to hand to `engine.fx.convert` want
    `latest_price` (store) — this function is for the ingest path and for
    the registry, which records the vendor's own answer verbatim.
    """
    overrides = _load_ticker_currency_overrides()
    if ticker in overrides:
        return overrides[ticker]
    registry = _load_registry_currencies()
    if ticker in registry:
        return registry[ticker]
    return _heuristic_unit(ticker)


def _iso_and_scale(unit: str | None) -> tuple[str | None, float]:
    """Split a quote unit into (ISO currency, price multiplier).

    Case-sensitive on purpose — see `_SUB_UNITS`. Anything not a known
    sub-unit is passed through upper-cased, so a hand-written override of
    `"eur"` still resolves to `EUR` at scale 1.0.

    An unresolved unit yields `(None, 1.0)`: no currency, and no scaling
    either. Scaling by anything other than 1.0 on an unknown unit would be
    inventing the very fact we just admitted we do not have.
    """
    if unit is None:
        return None, 1.0
    if unit in _SUB_UNITS:
        return _SUB_UNITS[unit]
    return unit.upper(), 1.0


def ticker_currency(ticker: str) -> str | None:
    """Resolve ticker → ISO 4217 currency code, or `None` if unresolvable.

    `GBp` (pence) resolves to `GBP`: the sub-unit is a quoting convention,
    not a currency, and `engine.fx` only knows ISO codes. The 1/100 that
    goes with it is applied to the *price* once, at ingest, by
    `normalise_vendor_quote`.

    `None` means neither map answered and the suffix is unknown — see
    `_heuristic_unit`. Every caller must decide explicitly what to do with
    that; the broker rejects the order (`CURRENCY_UNRESOLVED`) rather than
    trading a position it cannot denominate.
    """
    return _iso_and_scale(vendor_quote_unit(ticker))[0]


def vendor_unit_scale(ticker: str) -> float:
    """Multiplier taking a VENDOR price for `ticker` to its ISO currency.

    `0.01` for a pence-quoted LSE line, `1.0` for everything else. This is
    the ingest-side counterpart of `ticker_currency`, and the only reason a
    caller should ever need it is to normalise a freshly fetched frame —
    `scripts.fetch_ohlcv._fetch_symbol` does exactly that, which is what
    keeps the store ISO-denominated.
    """
    return _iso_and_scale(vendor_quote_unit(ticker))[1]


def normalise_vendor_quote(ticker: str, vendor_price: float) -> Quote | None:
    """Convert a raw VENDOR quote for `ticker` into its ISO currency.

    `None` when the currency is unresolvable — the price would be a number
    with no unit, which is what this module exists to stop.

    This is the only place the pence→pounds division exists, and since
    2026-08-07 it lives on the **ingest** side of the store rather than the
    read side. `vendor_price` must be a price as yfinance served it — never
    a value read back from `data/market/ohlcv/*.jsonl`, which is already
    normalised, and never the output of a previous call.

    Read paths want `store_quote` / `latest_price` instead. Splitting the
    two is what makes double-application structurally impossible: before,
    every reader had to remember to normalise, and three of them did not.
    """
    iso, scale = _iso_and_scale(vendor_quote_unit(ticker))
    if iso is None:
        return None
    return Quote(vendor_price * scale, iso)


def store_quote(ticker: str, stored_price: float) -> Quote | None:
    """Attach the ISO currency to a price read from the store.

    Deliberately applies **no** scaling: the store is ISO-denominated at
    ingest, so a stored value is already in `ticker_currency(ticker)`. This
    exists rather than having callers pair the raw reader with
    `ticker_currency` by hand, because that hand-pairing is exactly what
    dropped the conversion on three pricing paths at once.

    `None` when the currency is unresolvable. A price with no currency is
    not a quote, and returning one anyway is how an unlabelled number
    reaches `fx.convert` and comes back as if it were already in the book's
    own currency.
    """
    currency = ticker_currency(ticker)
    if currency is None:
        return None
    return Quote(stored_price, currency)


def latest_price(
    ticker: str, on: date | None = None, store: Path | None = None
) -> Quote | None:
    """Latest close on or before `on`, in the ticker's ISO currency.

    Thin composition of `engine.ohlcv_store.latest_close_on_or_before` and
    `store_quote`. Returns `None` when the raw reader does — ticker absent
    from the store, or no row on or before `on` — and also when the
    ticker's currency is unresolvable, since an unlabelled price is not a
    quote. Callers that must tell those two apart (the broker does: they
    are `NO_PRICE_DATA` and `CURRENCY_UNRESOLVED`) should ask
    `ticker_currency` first.

    No scaling happens here: the store is normalised at ingest. Prefer this
    over calling the raw reader plus `ticker_currency` separately — that
    pair is what silently dropped the pence conversion on three pricing
    paths at once, and it would now silently reintroduce it.
    """
    raw = latest_close_on_or_before(ticker, on, store=store)
    if raw is None:
        return None
    return store_quote(ticker, raw)
