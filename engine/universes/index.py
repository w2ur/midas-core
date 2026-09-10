"""Index universe resolvers.

US: S&P 500, Dow 30, Nasdaq 100.
EU: CAC 40, DAX, FTSE 100, STOXX Europe 600.

Universe lists live in `data/universes/{name}.json`, **committed** to the
repo. The cloud sandbox has no outbound HTTP, so resolvers must NEVER hit
Wikipedia, Slickcharts, DWS or Yahoo at runtime. File presence is
authoritative. Periodic refresh runs out-of-band (manual
`scripts/refresh_universes.py` or the GitHub Actions weekly cron
`refresh-universes.yml`) and commits the diff.

This module previously kept these files under `data/cache/universes/`
(gitignored) with a 24-hour TTL — that combination crashed every cloud
session whose cache was older than a day. Apr 29 incident.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import time
import urllib.request
from collections.abc import Callable

import pandas as pd

from engine.config import get_config

logger = logging.getLogger(__name__)

_WIKI_USER_AGENT = "midas-fund/0.1 (https://github.com/w2ur/midas; research)"


def _fetch_text(url: str, *, timeout: float = 15, encoding: str = "utf-8") -> str:
    """GET `url` with a descriptive User-Agent and decode the body.

    Wikipedia (and Slickcharts) reject pandas' default Python-urllib UA, so
    every upstream is fetched here before parsing.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _WIKI_USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode(encoding)


def _fetch_html_tables(url: str) -> list[pd.DataFrame]:
    """Fetch an HTML page and parse its tables — the Wikipedia index pages and
    the Slickcharts Nasdaq-100 source."""
    return pd.read_html(io.StringIO(_fetch_text(url)))


def _largest_table_with_column(
    tables: list[pd.DataFrame], column: str
) -> pd.DataFrame | None:
    """Return the largest table containing `column` in its columns.

    Robust against Wikipedia page layout changes: avoids picking small
    "examples" or "recent changes" tables that happen to share a column name.
    """
    candidates = [t for t in tables if column in [str(c) for c in t.columns]]
    if not candidates:
        return None
    return max(candidates, key=len)


def _read_data(name: str) -> list[str] | None:
    """Return committed tickers for `name`, or None if the file is missing.

    No TTL: the file is the source of truth. If you want to refresh from
    Wikipedia, call `refresh_<name>()` explicitly.
    """
    path = get_config().universes_dir / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_data(name: str, tickers: list[str]) -> None:
    """Persist tickers to `data/universes/{name}.json`."""
    data_dir = get_config().universes_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{name}.json").write_text(json.dumps(tickers), encoding="utf-8")


def _normalise(ticker: str) -> str:
    """Replace dots with hyphens for yfinance compatibility (BRK.B → BRK-B)."""
    return ticker.replace(".", "-").strip()


# ---------------------------------------------------------------------------
# S&P 500
# ---------------------------------------------------------------------------


def get_sp500_tickers() -> list[str]:
    """Return committed S&P 500 constituents.

    Reads `data/universes/sp500.json`. Falls back to a Wikipedia refresh ONLY
    when the file is missing — which should never happen in production since
    the file is committed.
    """
    cached = _read_data("sp500")
    if cached is not None:
        return cached
    return refresh_sp500()


def refresh_sp500() -> list[str]:
    """Re-fetch S&P 500 from Wikipedia and overwrite the committed file."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Symbol")
    if table is None:
        raise RuntimeError("S&P 500: no 'Symbol' column on Wikipedia page")
    tickers = sorted({_normalise(str(t)) for t in table["Symbol"].tolist()})
    if len(tickers) < 100:
        raise RuntimeError(
            f"S&P 500: {len(tickers)} tickers — Wikipedia layout may have changed"
        )
    _write_data("sp500", tickers)
    return tickers


# ---------------------------------------------------------------------------
# Dow 30
# ---------------------------------------------------------------------------


def get_dow30_tickers() -> list[str]:
    """Return committed Dow Jones Industrial Average constituents."""
    cached = _read_data("dow30")
    if cached is not None:
        return cached
    return refresh_dow30()


def refresh_dow30() -> list[str]:
    """Re-fetch Dow 30 from Wikipedia and overwrite the committed file."""
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Symbol")
    if table is None:
        raise RuntimeError("Dow 30: no 'Symbol' column on Wikipedia page")
    raw = [str(t) for t in table["Symbol"].dropna().tolist() if str(t) != "Symbol"]
    tickers = sorted({_normalise(t) for t in raw if t})
    if len(tickers) < 20:
        raise RuntimeError(
            f"Dow 30: {len(tickers)} tickers — Wikipedia layout may have changed"
        )
    _write_data("dow30", tickers)
    return tickers


# ---------------------------------------------------------------------------
# Nasdaq 100
# ---------------------------------------------------------------------------


def get_nasdaq100_tickers() -> list[str]:
    """Return committed Nasdaq-100 constituents."""
    cached = _read_data("nasdaq100")
    if cached is not None:
        return cached
    return refresh_nasdaq100()


# Candidate ticker-column header names on the Slickcharts Nasdaq-100 page,
# tried in order. Tolerant to a future header rename (Slickcharts has already
# forced one source change — see docstring below) without another crash.
_NASDAQ100_SYMBOL_COLUMNS: tuple[str, ...] = ("Symbol", "Ticker")


def refresh_nasdaq100() -> list[str]:
    """Re-fetch Nasdaq-100 from Slickcharts and overwrite the committed file.

    Source moved off Wikipedia on 2026-07-13: the en.wikipedia.org/wiki/Nasdaq-100
    article dropped its constituents table entirely (the "Components" section is
    now just an external link to nasdaq.com), so no column-name variant could
    recover it. Slickcharts publishes a clean weighted table with a "Symbol"
    column (~100 rows, dual-class shares like GOOGL/GOOG included). Column
    detection tries each of `_NASDAQ100_SYMBOL_COLUMNS` in turn so a future
    Slickcharts header rename (e.g. back to "Ticker") degrades gracefully
    instead of raising immediately.
    """
    url = "https://www.slickcharts.com/nasdaq100"
    tables = _fetch_html_tables(url)
    table = None
    symbol_col = None
    for candidate in _NASDAQ100_SYMBOL_COLUMNS:
        table = _largest_table_with_column(tables, candidate)
        if table is not None:
            symbol_col = candidate
            break
    if table is None:
        raise RuntimeError(
            "Nasdaq-100: no "
            f"{' or '.join(repr(c) for c in _NASDAQ100_SYMBOL_COLUMNS)} "
            "column on Slickcharts page"
        )
    raw = [str(t) for t in table[symbol_col].dropna().tolist() if str(t) != symbol_col]
    tickers = sorted({_normalise(t) for t in raw if t})
    if len(tickers) < 90:
        raise RuntimeError(
            f"Nasdaq-100: {len(tickers)} tickers — Slickcharts layout may have changed"
        )
    _write_data("nasdaq100", tickers)
    return tickers


# ---------------------------------------------------------------------------
# EU indices — CAC 40, DAX, FTSE 100, STOXX Europe 600
# ---------------------------------------------------------------------------

# Country → yfinance exchange suffix. For STOXX 600 this is a PREFERENCE among
# the listings Yahoo returns for an ISIN (see `_pick_symbol`), never something
# appended to a code: the export's country is a domicile (Prosus is "China",
# Airbus "Netherlands") and the home market can differ from it.
_STOXX_COUNTRY_SUFFIX: dict[str, str] = {
    "Austria": ".VI",
    "Belgium": ".BR",
    "Denmark": ".CO",
    "Finland": ".HE",
    "France": ".PA",
    "Germany": ".DE",
    "Greece": ".AT",
    "Ireland": ".IR",
    "Italy": ".MI",
    "Luxembourg": ".LU",
    "Netherlands": ".AS",
    "Norway": ".OL",
    "Poland": ".WA",
    "Portugal": ".LS",
    "Spain": ".MC",
    "Sweden": ".ST",
    "Switzerland": ".SW",
    "United Kingdom": ".L",
    # Jersey / Bermuda / Israel companies often list on LSE
    "Jersey": ".L",
    "Bermuda": ".L",
    "Israel": ".L",
}


def _clean_ticker(raw: object) -> str | None:
    s = str(raw).strip()
    if not s or s.lower() in ("ticker", "nan", "none", "—"):
        return None
    if "[" in s:
        s = s.split("[", 1)[0].strip()
    return s or None


def get_cac40_tickers() -> list[str]:
    """Return committed CAC 40 constituents."""
    cached = _read_data("cac40")
    if cached is not None:
        return cached
    return refresh_cac40()


def refresh_cac40() -> list[str]:
    url = "https://en.wikipedia.org/wiki/CAC_40"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Ticker")
    if table is None:
        raise RuntimeError("CAC 40: no 'Ticker' column on Wikipedia page")
    tickers = sorted({t for t in (_clean_ticker(v) for v in table["Ticker"]) if t})
    if len(tickers) < 30:
        raise RuntimeError(f"CAC 40: {len(tickers)} tickers — layout changed")
    _write_data("cac40", tickers)
    return tickers


def get_dax_tickers() -> list[str]:
    """Return committed DAX constituents."""
    cached = _read_data("dax")
    if cached is not None:
        return cached
    return refresh_dax()


def refresh_dax() -> list[str]:
    url = "https://en.wikipedia.org/wiki/DAX"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Ticker")
    if table is None:
        raise RuntimeError("DAX: no 'Ticker' column on Wikipedia page")
    tickers = sorted({t for t in (_clean_ticker(v) for v in table["Ticker"]) if t})
    if len(tickers) < 30:
        raise RuntimeError(f"DAX: {len(tickers)} tickers — layout changed")
    _write_data("dax", tickers)
    return tickers


def get_ftse100_tickers() -> list[str]:
    """Return committed FTSE 100 constituents (.L suffix appended for yfinance)."""
    cached = _read_data("ftse100")
    if cached is not None:
        return cached
    return refresh_ftse100()


def refresh_ftse100() -> list[str]:
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    tables = _fetch_html_tables(url)
    table = _largest_table_with_column(tables, "Ticker")
    if table is None:
        raise RuntimeError("FTSE 100: no 'Ticker' column on Wikipedia page")
    tickers: set[str] = set()
    for raw in table["Ticker"]:
        t = _clean_ticker(raw)
        if t is None:
            continue
        if not t.endswith(".L"):
            # Yahoo spells an LSE share-class suffix with a dash, not a dot:
            # Wikipedia's "BT.A" is "BT-A.L". Appending ".L" verbatim produced
            # "BT.A.L", which resolves to nothing — the store has no such file
            # and the ticker registry carried it as `unknown` forever, while
            # the ISIN-based STOXX 600 resolver had "BT-A.L" right all along.
            t = f"{_normalise(t)}.L"
        tickers.add(t)
    result = sorted(tickers)
    if len(result) < 80:
        raise RuntimeError(f"FTSE 100: {len(result)} tickers — layout changed")
    _write_data("ftse100", result)
    return result


def get_stoxx600_tickers() -> list[str]:
    """Return committed STOXX Europe 600 constituents (Yahoo symbols).

    No refresh fallback, unlike the other indexes: theirs is one 15 s page
    fetch, this one is ~605 throttled vendor lookups. A missing file on the
    session path must fail in a millisecond, not crawl Yahoo for seven
    minutes and overwrite the universe mid-session.
    """
    cached = _read_data("stoxx600")
    if cached is None:
        raise FileNotFoundError(
            "data/universes/stoxx600.json is missing — run "
            "`python scripts/refresh_universes.py` (network) to regenerate it"
        )
    return cached


# ---------------------------------------------------------------------------
# STOXX Europe 600 — keyed on ISIN, resolved through the vendor's own lookup
# ---------------------------------------------------------------------------
#
# Until 2026-09-09 this index was scraped from Wikipedia's "Ticker" column with
# a country suffix appended. That column holds Reuters-style codes (AIRP, BNPP,
# CAGR, "AMBU B", "ATCOa"), not Yahoo symbols, so 120 of the 463 entries it
# produced had never served a single row (issue #36): four agents were handed
# them as tradable, every order on one died at the broker with NO_PRICE_DATA,
# and every nightly fetch printed ~120 "Quote not found" lines. The list was
# also stale — it lacked 270 current constituents and carried names that had
# left the index. No per-exchange rewrite rule fixes a code the vendor does not
# route, and a hand-typed override map of 120 entries goes stale the same way.
#
# An ISIN is the one identifier both sides agree on. The constituent list comes
# from DWS's export for the Xtrackers STOXX Europe 600 UCITS ETF 1C
# (LU0328475792), the only free source found that carries an ISIN per line:
# STOXX's own components CSV answers 404, Wikipedia has no ISIN column, and the
# iShares EXSA holdings file omits ISIN in every locale. Each ISIN is then put
# to Yahoo's search endpoint, which answers with the listings it actually
# serves — so a symbol in the committed file is one the nightly fetch can
# fetch, by construction. The tradable universe is defined by what the vendor
# can price, which is the property the old list lacked.
#
# The export's "Constituent Currency ISO Code" is deliberately NOT used to
# choose a listing. Measured 2026-09-09 it says USD for Compass and IHG, EUR
# for Shell and Nordea (the ETF's own line, not the listing) — a rule keyed on
# it would have refused five correct London and Stockholm listings.
_STOXX600_CONSTITUENTS_URL = (
    "https://etf.dws.com/etfdata/export/LUX/ENG/csv/product/constituent/LU0328475792/"
)
_STOXX600_REQUIRED_COLUMNS = ("Constituent ISIN", "Constituent Name", "Constituent Country")
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

#: Seconds between two vendor lookups, and the back-off schedule after a rate
#: limit. Measured 2026-09-09 over 605 ISINs: unthrottled (~4/s) the endpoint
#: rate-limited 3 times; at 0.4 s spacing once, recovered by the first
#: back-off. 0.5 s puts a full pass at ~7 minutes (424 s measured).
ISIN_LOOKUP_SPACING_S = 0.5
ISIN_LOOKUP_RETRY_SLEEPS_S = (5.0, 15.0)
#: Consecutive lookups that fail on transport or rate limit before the run is
#: declared a vendor outage and abandoned. Bounds an outage's cost to
#: ~5 × 20 s instead of 605 × 20 s; the committed file is then left alone.
MAX_CONSECUTIVE_LOOKUP_FAILURES = 5
#: Wall-clock budget for the whole crawl. Neither abort rule above bounds
#: elapsed time when every lookup SUCCEEDS after a back-off: 605 lookups each
#: recovering on the first 5 s sleep is ~55 minutes of green answers, past the
#: workflow's timeout — and a cancelled job commits nothing, not even the six
#: indexes already refreshed to disk. 15 min is twice the measured pass.
ISIN_LOOKUP_BUDGET_S = 15 * 60

#: Refuse to overwrite the committed file when more than this share of the
#: export's ISINs did not resolve. Measured baseline 2026-09-09: 8 of 605
#: (1.3%) — loyalty/bonus-share ISINs (L'Oréal, Air Liquide), an Engie
#: preference line, a cash line with no name, two names Yahoo does not index by
#: ISIN (Kesko B, Vår Energi) and one only quoted on a German regional floor.
#: `resolve_isins` stops as soon as the count is exceeded rather than crawling
#: the remainder.
MAX_UNRESOLVED_ISIN_RATE = 0.05

#: Refuse to overwrite the committed file when the symmetric difference with
#: it exceeds this share of the previous list — the refresh commits to main
#: unattended, and a selection regression that swaps ~200 names for their
#: Frankfurt or US twins keeps the count near 600 and passes every other gate.
#: A real STOXX rebalance moves a few names per quarter (~3%). The first
#: ISIN-keyed refresh legitimately moved 486 of 463; that is what the
#: `MIDAS_ACCEPT_UNIVERSE_CHURN=1` override exists for, set by a human on a
#: deliberate local run, never by the workflow.
MAX_UNIVERSE_CHURN_RATE = 0.20
_ACCEPT_CHURN_ENV = "MIDAS_ACCEPT_UNIVERSE_CHURN"
_ENV_TRUE = {"1", "true"}  # engine.live_switch's convention

#: Yahoo exchange codes a symbol must not come from: the three US OTC tiers
#: (pink sheets, OTCQB, OTCQX — yfinance's own `const.py` lists all three).
#: An ADR or grey-market print, never the listing an EU desk trades, and a
#: dotless symbol the currency heuristic would happily call USD.
_OTC_EXCHANGES = frozenset({"PNK", "OQB", "OQX"})
#: German venues: Xetra plus the regional floors. Yahoo lists many foreign
#: names on them (SAGAX B of Stockholm answered only as EFE.F) with thin,
#: often stale daily bars, and a Xetra secondary line for a CHF or SEK name is
#: the wrong instrument in the right currency. Accepted only for a German
#: constituent, where they are the home market.
_GERMAN_REGIONAL_FLOORS = frozenset({"FRA", "STU", "MUN", "DUS", "BER", "HAM", "HAN"})
_GERMAN_EXCHANGES = _GERMAN_REGIONAL_FLOORS | {"GER"}
#: Suffixes of the European home markets, preferred over a dotless US listing
#: when a constituent's own domicile suffix is absent from the answer.
_EUROPEAN_SUFFIXES = frozenset(_STOXX_COUNTRY_SUFFIX.values())


def _fetch_stoxx600_constituents(url: str = _STOXX600_CONSTITUENTS_URL) -> list[dict[str, str]]:
    """Return the export's ISIN-bearing rows as dicts keyed by column name.

    The export also lists cash (`_CURRENCYEUR`) and index-future lines whose
    "ISIN" is not ISIN-shaped; those are dropped here. A missing column is a
    layout change and raises, like the Wikipedia scrapers do.
    """
    text = _fetch_text(url, timeout=30, encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    columns = [str(c).strip() for c in (reader.fieldnames or [])]
    missing = [c for c in _STOXX600_REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise RuntimeError(f"STOXX 600: export lacks column(s) {missing} — layout changed")
    rows: list[dict[str, str]] = []
    for row in reader:
        clean = {str(k).strip(): (v or "").strip() for k, v in row.items() if k is not None}
        if _ISIN_RE.match(clean.get("Constituent ISIN", "")):
            rows.append(clean)
    return rows


def _search_isin_quotes(
    isin: str, sleep: Callable[[float], None] | None = None
) -> list[dict] | None:
    """Ask Yahoo which listings it serves for `isin`.

    Returns the vendor's quotes (possibly empty: it knows no listing) or
    `None` when the lookup itself failed — a rate limit that outlasted
    `ISIN_LOOKUP_RETRY_SLEEPS_S`, or a transport error. Only those two are
    retried: anything else (a signature change in yfinance, Yahoo's own
    "currently down" exception) is a fact about this run, not this ISIN, and
    propagates so the refresh skips the index and keeps the committed file.
    """
    import yfinance as yf  # network-only; keep import cost off the session path
    from yfinance.config import YfConfig
    from yfinance.exceptions import YFRateLimitError

    sleep = sleep or time.sleep  # bound at call time so tests can patch it
    # yfinance's default swallows a non-JSON body (an HTML 5xx, a captcha page)
    # into an EMPTY quote list, which reads as "the vendor knows no listing"
    # and would drop a tradable name from the universe on a transient window.
    # With the flag off the body raises JSONDecodeError, which is retried and
    # then reported as a failed lookup — the distinction the abort rules need.
    hidden = YfConfig.debug.hide_exceptions
    YfConfig.debug.hide_exceptions = False
    try:
        for backoff in (*ISIN_LOOKUP_RETRY_SLEEPS_S, None):
            try:
                return list(
                    yf.Search(isin, max_results=10, news_count=0, lists_count=0).quotes
                )
            # OSError is every transport error; JSONDecodeError is a non-JSON body.
            except (YFRateLimitError, OSError, json.JSONDecodeError) as exc:
                if backoff is None:
                    logger.warning(
                        "STOXX 600: lookup for %s failed after retries — %s", isin, exc
                    )
                    return None
                sleep(backoff)
    finally:
        YfConfig.debug.hide_exceptions = hidden
    return None


def _pick_symbol(isin: str, country: str, quotes: list[dict]) -> str | None:
    """Choose the listing to trade among what the vendor returned for `isin`.

    A quote is a candidate only if it is an equity on a named exchange that is
    neither a US OTC tier nor (for a name domiciled elsewhere) a German venue
    — a regional floor is refused for any non-German domicile, Xetra only for
    a domicile the suffix map can place — is a real
    symbol rather than Stuttgart's `<ISIN>.SG` placeholder, and carries a
    suffix the currency layer can denominate (`engine.quotes`) — a symbol it
    cannot would reach the broker only to die with CURRENCY_UNRESOLVED. Among
    candidates the constituent's home-market suffix wins (Xetra over the
    Frankfurt floor for a German name), then any European home market over a
    dotless US listing, then the symbol itself — so two runs over the same
    answer pick the same listing, and the venue a EUR book is exposed to is
    never decided by sort order between a EUR and a USD line.
    """
    from engine.quotes import _heuristic_unit

    suffix = _STOXX_COUNTRY_SUFFIX.get(country)
    candidates: list[tuple[str, str]] = []
    for q in quotes:
        symbol, exchange = q.get("symbol"), q.get("exchange")
        if q.get("quoteType") != "EQUITY" or not symbol or not exchange:
            continue
        if exchange in _OTC_EXCHANGES or isin in symbol:
            continue
        # A German venue for a name whose domicile is KNOWN to be elsewhere is
        # a secondary line. For a domicile the map cannot place (the export
        # files Delivery Hero under Korea, Prosus under China) Xetra may well
        # be the home market, so only the regional floors are refused there.
        if exchange in _GERMAN_EXCHANGES and suffix not in (None, ".DE"):
            continue
        if exchange in _GERMAN_REGIONAL_FLOORS and suffix != ".DE":
            continue
        if _heuristic_unit(symbol) is None:
            continue
        candidates.append((symbol, exchange))
    if not candidates:
        return None

    def rank(candidate: tuple[str, str]) -> tuple[int, int, int, str]:
        symbol, exchange = candidate
        _, dot, tail = symbol.rpartition(".")
        return (
            0 if suffix and symbol.endswith(suffix) else 1,
            # A German name's Frankfurt floor is still its home market,
            # ahead of a Vienna or Milan secondary.
            0 if suffix == ".DE" and exchange in _GERMAN_EXCHANGES else 1,
            0 if dot and f".{tail}" in _EUROPEAN_SUFFIXES else 1,
            symbol,
        )

    return min(candidates, key=rank)[0]

def resolve_isins(
    constituents: list[dict[str, str]],
    *,
    max_unresolved: int | None = None,
    budget_s: float | None = ISIN_LOOKUP_BUDGET_S,
    lookup: Callable[[str], list[dict] | None] | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Map each constituent's ISIN to a Yahoo symbol.

    Returns `(resolved, unresolved)`: `resolved` is `{isin: symbol}`,
    `unresolved` lists `(isin, name)` for every line the vendor could not
    place. Lookups are spaced `ISIN_LOOKUP_SPACING_S` apart. Raises as soon
    as `unresolved` exceeds `max_unresolved`, after
    `MAX_CONSECUTIVE_LOOKUP_FAILURES` lookups in a row failed outright, or
    once `budget_s` of wall-clock has elapsed — all three mean the run cannot
    produce a list worth committing, and finishing the crawl would only cost
    the workflow its timeout.
    """
    # Resolved at call time, not bound as defaults: a default captures the
    # original `time.sleep`, which is how the first version of the test suite
    # sat through 250 s of real throttling.
    lookup = lookup or _search_isin_quotes
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    started = clock()
    resolved: dict[str, str] = {}
    unresolved: list[tuple[str, str]] = []
    failed: list[str] = []
    consecutive_failures = 0
    for i, row in enumerate(constituents):
        if i:
            sleep(ISIN_LOOKUP_SPACING_S)
        if budget_s is not None and clock() - started > budget_s:
            raise RuntimeError(
                f"STOXX 600: {budget_s:.0f} s budget exhausted after {i} of "
                f"{len(constituents)} lookups — committed file left at its last "
                "known-good value"
            )
        isin = row["Constituent ISIN"]
        quotes = lookup(isin)
        if quotes is None:
            consecutive_failures += 1
            failed.append(isin)
            if consecutive_failures >= MAX_CONSECUTIVE_LOOKUP_FAILURES:
                raise RuntimeError(
                    f"STOXX 600: {consecutive_failures} consecutive vendor lookups "
                    "failed — lookup unavailable; committed file left at its last "
                    "known-good value"
                )
            quotes = []
        else:
            consecutive_failures = 0
        symbol = _pick_symbol(isin, row.get("Constituent Country", ""), quotes)
        if symbol is None:
            unresolved.append((isin, row.get("Constituent Name", "")))
            if max_unresolved is not None and len(unresolved) > max_unresolved:
                raise RuntimeError(
                    f"STOXX 600: {len(unresolved)} of {len(constituents)} ISINs "
                    f"unresolved after {i + 1} lookups (limit {max_unresolved}) — "
                    "committed file left at its last known-good value"
                )
        else:
            resolved[isin] = symbol
    if failed:
        # Named separately from the vendor's genuine "no listing" answers, so a
        # blip and a delisting never read the same in the log.
        logger.warning(
            "STOXX 600: %d lookup(s) failed and count as unresolved this run: %s",
            len(failed),
            ", ".join(failed),
        )
    return resolved, unresolved

def _check_universe_churn(previous: list[str] | None, result: list[str]) -> None:
    """Refuse a result that differs from the committed list by more than
    `MAX_UNIVERSE_CHURN_RATE`, unless a human set `MIDAS_ACCEPT_UNIVERSE_CHURN`.
    Always logs what moved, so the weekly commit's diff is readable."""
    if not previous:
        return
    added = sorted(set(result) - set(previous))
    removed = sorted(set(previous) - set(result))
    if not added and not removed:
        return
    logger.warning(
        "STOXX 600: %d added (%s), %d removed (%s)",
        len(added),
        ", ".join(added),
        len(removed),
        ", ".join(removed),
    )
    churn = (len(added) + len(removed)) / len(previous)
    # Same truthiness convention as engine.live_switch: only "1"/"true" accept.
    # A bare truthiness test would read `=0` or `=false` as consent.
    accepted = os.environ.get(_ACCEPT_CHURN_ENV, "").strip().lower() in _ENV_TRUE
    if churn > MAX_UNIVERSE_CHURN_RATE and not accepted:
        raise RuntimeError(
            f"STOXX 600: {len(added)} added + {len(removed)} removed against "
            f"{len(previous)} committed ({churn:.0%}, limit "
            f"{MAX_UNIVERSE_CHURN_RATE:.0%}) — refusing to overwrite; a deliberate "
            f"rebuild sets {_ACCEPT_CHURN_ENV}=1"
        )


def refresh_stoxx600() -> list[str]:
    constituents = _fetch_stoxx600_constituents()
    if len(constituents) < 400:
        raise RuntimeError(
            f"STOXX 600: export lists {len(constituents)} ISINs — layout changed"
        )
    resolved, unresolved = resolve_isins(
        constituents,
        max_unresolved=int(len(constituents) * MAX_UNRESOLVED_ISIN_RATE),
    )
    if unresolved:
        logger.warning(
            "STOXX 600: %d of %d ISINs did not resolve to a tradable listing and "
            "are left out: %s",
            len(unresolved),
            len(constituents),
            ", ".join(f"{isin} ({name or 'unnamed'})" for isin, name in unresolved),
        )
    result = sorted(set(resolved.values()))
    if len(result) < 400:
        raise RuntimeError(f"STOXX 600: {len(result)} symbols — layout changed")
    _check_universe_churn(_read_data("stoxx600"), result)
    _write_data("stoxx600", result)
    return result


# ---------------------------------------------------------------------------
# Bulk refresh
# ---------------------------------------------------------------------------


# Canonical {name: refresher-function-name} mapping — the single source of
# truth for which indexes exist. `scripts/refresh_universes.py` derives its
# skip report from these keys, so adding an index here is the only change
# needed. Values are attribute names resolved at call time (late-bound) so
# tests can monkeypatch the individual refresh_* functions.
INDEX_REFRESHERS = {
    "sp500": "refresh_sp500",
    "dow30": "refresh_dow30",
    "nasdaq100": "refresh_nasdaq100",
    "cac40": "refresh_cac40",
    "dax": "refresh_dax",
    "ftse100": "refresh_ftse100",
    "stoxx600": "refresh_stoxx600",
}


def refresh_all_indexes() -> dict[str, int]:
    """Re-fetch every index universe from its upstream source and overwrite
    the committed files.

    Sources: Wikipedia for the S&P 500, Dow 30, CAC 40, DAX and FTSE 100;
    Slickcharts for the Nasdaq-100; for the STOXX 600, DWS's constituent
    export resolved ISIN by ISIN through Yahoo's lookup (see
    `refresh_stoxx600`), which is the slow one — a few minutes, throttled.

    Used by `scripts/refresh_universes.py` and the weekly GitHub Actions cron.
    Each index refreshes independently: a scraper that raises (e.g. an
    upstream layout change like the 2026-07-13 Nasdaq-100/Wikipedia break)
    logs a warning and is skipped rather than aborting the whole run — one
    broken source must never take the other six indexes down with it. The
    committed file for a skipped index is left untouched at its last known
    -good value. Returns {name: ticker_count} for indexes that succeeded
    only; a failed index is simply absent from the result (callers alert on
    the gap — the weekly workflow exits non-zero so the failure still emails).
    """
    results: dict[str, int] = {}
    for name, fn_name in INDEX_REFRESHERS.items():
        refresher = globals()[fn_name]
        try:
            results[name] = len(refresher())
        except Exception as exc:
            logger.warning("refresh_all_indexes: skipping %s — %s", name, exc)
    return results
