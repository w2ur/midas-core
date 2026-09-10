"""Tests for universe resolvers — index, asset class, and alternative data.

Tests must NEVER write to the real `data/universes/` directory. Every test
that exercises the cache-write path monkeypatches `_DATA_DIR` to a tmp_path.
A previous version of this file silently overwrote the real `sp500.json`
during pytest runs, dropping it from 503 tickers to 3 — the Apr 29 cloud
session aborted as a downstream consequence.
"""

from __future__ import annotations

import io
import json

from engine.config import get_config
from pathlib import Path

import pytest

from engine.universes.alternative import (
    get_congressional_tickers,
    get_high_short_tickers,
    get_insider_tickers,
)
from engine.universes.assets import (
    get_classic_60_40,
    get_crypto_tickers,
    get_forex_tickers,
    get_metals_tickers,
    get_voo_only,
)


# ---------------------------------------------------------------------------
# Index universe resolvers — never touch the network in tests
# ---------------------------------------------------------------------------


class TestSP500Tickers:
    def test_committed_data_present_and_valid(self):
        """The committed `data/universes/sp500.json` must contain a real S&P 500."""
        from engine.universes.index import get_sp500_tickers

        tickers = get_sp500_tickers()
        assert isinstance(tickers, list)
        # Real S&P 500 has ~500 constituents; <100 means the file is corrupt
        # (e.g. test pollution). Apr 29 incident.
        assert len(tickers) > 100, (
            f"sp500 file looks corrupt: only {len(tickers)} tickers"
        )
        assert all(isinstance(t, str) for t in tickers)

    def test_no_dots_in_committed_tickers(self):
        from engine.universes.index import get_sp500_tickers

        for ticker in get_sp500_tickers():
            assert "." not in ticker, f"{ticker!r} still contains a dot"

    def test_isolated_cache_returns_isolated_data(self, midas_data_root, monkeypatch):
        """Monkeypatch the data dir; verify reads come from the patched location."""
        import engine.universes.index as ix_mod

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)

        sample = ["AAPL", "MSFT"]
        (fake_dir / "sp500.json").write_text(json.dumps(sample))
        assert ix_mod.get_sp500_tickers() == sample

    def test_no_network_call_when_file_exists(self, midas_data_root, monkeypatch):
        import engine.universes.index as ix_mod

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)
        (fake_dir / "sp500.json").write_text(json.dumps(["AAPL"]))

        called = []

        def boom(*a, **kw):
            called.append(True)
            raise AssertionError("network must not be called when data file exists")

        monkeypatch.setattr(ix_mod, "_fetch_html_tables", boom)

        assert ix_mod.get_sp500_tickers() == ["AAPL"]
        assert not called


class TestDow30Tickers:
    def test_committed_data_present_and_valid(self):
        from engine.universes.index import get_dow30_tickers

        tickers = get_dow30_tickers()
        assert isinstance(tickers, list)
        assert len(tickers) >= 25, (
            f"dow30 file looks corrupt: only {len(tickers)} tickers"
        )

    def test_known_members_present(self):
        from engine.universes.index import get_dow30_tickers

        tickers = get_dow30_tickers()
        for t in ("AAPL", "MSFT"):
            assert t in tickers


class TestNasdaq100Tickers:
    def test_committed_data_present_and_valid(self):
        from engine.universes.index import get_nasdaq100_tickers

        tickers = get_nasdaq100_tickers()
        assert isinstance(tickers, list)
        assert len(tickers) >= 50, (
            f"nasdaq100 file looks corrupt: only {len(tickers)} tickers"
        )

    def test_known_members_present(self):
        from engine.universes.index import get_nasdaq100_tickers

        tickers = get_nasdaq100_tickers()
        for t in ("AAPL", "MSFT", "NVDA"):
            assert t in tickers


class TestEUIndices:
    def test_cac40_committed_and_paris_suffix(self):
        from engine.universes.index import get_cac40_tickers

        tickers = get_cac40_tickers()
        assert len(tickers) >= 30
        assert any(t.endswith(".PA") for t in tickers)

    def test_dax_committed(self):
        from engine.universes.index import get_dax_tickers

        assert len(get_dax_tickers()) >= 30

    def test_ftse100_all_lse_suffix(self):
        from engine.universes.index import get_ftse100_tickers

        tickers = get_ftse100_tickers()
        assert len(tickers) >= 80
        for t in tickers:
            assert t.endswith(".L"), f"{t!r} missing .L suffix"

    def test_no_committed_universe_carries_a_dotted_share_class(self):
        """Regression: the committed universes listed BT Group as "BT.A.L".

        `test_ftse100_all_lse_suffix` above passes on "BT.A.L" — it ends in
        ".L" — which is why it never caught this. Yahoo spells an LSE share
        class with a dash, so the symbol resolved to nothing: no OHLCV file in
        the committed store and an `unknown`/`null`-currency row in
        `data/tickers.json`. A dot anywhere but the final suffix is the
        signature.

        Both `.L`-bearing universes are checked, not just the one the scraper
        broke. `examples/demo-desk/data/universes/` is a second committed copy
        that no manifest compares against live, and it was carrying the bad
        spelling in *both* files — the FTSE 100 one from the same scraper bug,
        the STOXX 600 one frozen from before the ISIN resolver landed. A guard
        that reads whatever `MIDAS_DATA_DIR` points at covers both copies; one
        pinned to a single universe covers neither reliably.
        """
        from engine.universes.index import (
            get_ftse100_tickers,
            get_stoxx600_tickers,
        )

        for resolver in (get_ftse100_tickers, get_stoxx600_tickers):
            for t in resolver():
                assert t.count(".") <= 1, f"{t!r} has a dotted share class"

    def test_stoxx600_committed(self):
        from engine.universes.index import get_stoxx600_tickers

        assert len(get_stoxx600_tickers()) >= 400


class TestRefreshFunctions:
    def test_refresh_sp500_writes_to_data_dir(self, midas_data_root, monkeypatch):
        import engine.universes.index as ix_mod
        import pandas as pd

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)

        fresh = [f"T{i:03d}" for i in range(150)]

        def fake_fetch(url):
            return [pd.DataFrame({"Symbol": fresh})]

        monkeypatch.setattr(ix_mod, "_fetch_html_tables", fake_fetch)

        result = ix_mod.refresh_sp500()
        assert result == sorted(fresh)
        assert (fake_dir / "sp500.json").exists()
        assert json.loads((fake_dir / "sp500.json").read_text()) == sorted(fresh)

    def test_refresh_ftse100_dashes_a_share_class_before_the_lse_suffix(
        self, midas_data_root, monkeypatch
    ):
        """Regression: "BT.A" became "BT.A.L" instead of Yahoo's "BT-A.L"."""
        import engine.universes.index as ix_mod
        import pandas as pd

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)

        # 80 plain tickers to clear the layout-change floor, plus the two
        # shapes that matter: a dotted share class, and one already suffixed.
        rows = [f"T{i:03d}" for i in range(80)] + ["BT.A", "HSBA.L"]
        monkeypatch.setattr(
            ix_mod, "_fetch_html_tables", lambda url: [pd.DataFrame({"Ticker": rows})]
        )

        result = ix_mod.refresh_ftse100()
        assert "BT-A.L" in result
        assert "BT.A.L" not in result
        # An already-suffixed ticker is passed through, not re-suffixed.
        assert "HSBA.L" in result
        assert all(t.count(".") == 1 for t in result)

    def test_refresh_nasdaq100_reads_slickcharts_symbol_column(
        self, midas_data_root, monkeypatch
    ):
        """Source moved to Slickcharts on 2026-07-13 (Wikipedia dropped the
        constituents table). Refresh reads the largest 'Symbol' table, ignores
        stray header rows, and writes the committed file."""
        import engine.universes.index as ix_mod
        import pandas as pd

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)
        fresh = [f"N{i:03d}" for i in range(100)]

        # Slickcharts table: a "Symbol" column plus a stray repeated header row.
        def fake_fetch(url):
            assert "slickcharts" in url
            return [pd.DataFrame({"Symbol": ["Symbol", *fresh]})]

        monkeypatch.setattr(ix_mod, "_fetch_html_tables", fake_fetch)
        result = ix_mod.refresh_nasdaq100()
        assert result == sorted(fresh)

    def test_refresh_nasdaq100_falls_back_to_ticker_column(
        self, midas_data_root, monkeypatch
    ):
        """Tolerant column detection: if Slickcharts ever renames "Symbol" to
        "Ticker" (a real-world rename its Nasdaq-100 sibling pages already
        use), the refresher must still find the tickers instead of raising."""
        import engine.universes.index as ix_mod
        import pandas as pd

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)
        fresh = [f"N{i:03d}" for i in range(100)]

        def fake_fetch(url):
            # No "Symbol" column at all — only the renamed "Ticker" column.
            return [pd.DataFrame({"Company": fresh, "Ticker": fresh})]

        monkeypatch.setattr(ix_mod, "_fetch_html_tables", fake_fetch)
        result = ix_mod.refresh_nasdaq100()
        assert result == sorted(fresh)

    def test_refresh_nasdaq100_raises_when_no_known_column(
        self, midas_data_root, monkeypatch
    ):
        """Neither 'Symbol' nor 'Ticker' present — this is a genuine layout
        change the refresher cannot recover from and must surface loudly."""
        import engine.universes.index as ix_mod
        import pandas as pd

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)

        def fake_fetch(url):
            return [pd.DataFrame({"Company": ["Apple", "Microsoft"]})]

        monkeypatch.setattr(ix_mod, "_fetch_html_tables", fake_fetch)
        with pytest.raises(RuntimeError, match="Nasdaq-100"):
            ix_mod.refresh_nasdaq100()


class TestRefreshAllIndexesDegradesPerIndex:
    """The weekly refresh must not go dark for the other six indexes because
    one scraper's upstream layout changed — this is the class of bug that
    broke Nasdaq-100 on 2026-07-13 and, before the loop was hardened, would
    have aborted sp500/dow30/cac40/dax/ftse100/stoxx600 too."""

    def test_one_failing_refresher_does_not_abort_the_others(
        self, midas_data_root, monkeypatch
    ):
        import engine.universes.index as ix_mod

        def boom():
            raise RuntimeError(
                "Nasdaq-100: no 'Symbol' or 'Ticker' column on Slickcharts page"
            )

        monkeypatch.setattr(ix_mod, "refresh_nasdaq100", boom)
        monkeypatch.setattr(ix_mod, "refresh_sp500", lambda: ["AAPL", "MSFT"])
        monkeypatch.setattr(ix_mod, "refresh_dow30", lambda: ["AAPL"])
        monkeypatch.setattr(ix_mod, "refresh_cac40", lambda: ["MC.PA"])
        monkeypatch.setattr(ix_mod, "refresh_dax", lambda: ["SAP.DE"])
        monkeypatch.setattr(ix_mod, "refresh_ftse100", lambda: ["HSBA.L"])
        monkeypatch.setattr(ix_mod, "refresh_stoxx600", lambda: ["ASML.AS"])

        result = ix_mod.refresh_all_indexes()

        assert "nasdaq100" not in result
        assert result == {
            "sp500": 2,
            "dow30": 1,
            "cac40": 1,
            "dax": 1,
            "ftse100": 1,
            "stoxx600": 1,
        }

    def test_all_succeed_returns_all_seven(self, midas_data_root, monkeypatch):
        import engine.universes.index as ix_mod

        for name in (
            "refresh_sp500",
            "refresh_dow30",
            "refresh_nasdaq100",
            "refresh_cac40",
            "refresh_dax",
            "refresh_ftse100",
            "refresh_stoxx600",
        ):
            monkeypatch.setattr(ix_mod, name, lambda: ["X"])

        result = ix_mod.refresh_all_indexes()
        assert set(result) == {
            "sp500",
            "dow30",
            "nasdaq100",
            "cac40",
            "dax",
            "ftse100",
            "stoxx600",
        }
        assert all(count == 1 for count in result.values())


# ---------------------------------------------------------------------------
# Asset class universe resolvers (no I/O)
# ---------------------------------------------------------------------------


class TestCryptoTickers:
    def test_returns_20_tickers(self):
        assert len(get_crypto_tickers()) == 20

    def test_all_end_with_usd(self):
        for t in get_crypto_tickers():
            assert t.endswith("-USD")

    def test_contains_major_cryptos(self):
        result = get_crypto_tickers()
        for t in ("BTC-USD", "ETH-USD", "SOL-USD"):
            assert t in result


class TestForexTickers:
    def test_returns_at_least_8_pairs(self):
        assert len(get_forex_tickers()) >= 8

    def test_all_end_with_x(self):
        for t in get_forex_tickers():
            assert t.endswith("=X")

    def test_contains_major_pairs(self):
        result = get_forex_tickers()
        for t in ("EURUSD=X", "GBPUSD=X", "USDJPY=X"):
            assert t in result


class TestMetalsTickers:
    def test_contains_expected_tickers(self):
        result = get_metals_tickers()
        for t in ("GC=F", "SI=F", "PL=F", "CL=F", "HG=F", "GLD", "SLV", "USO"):
            assert t in result

    def test_returns_8_tickers(self):
        assert len(get_metals_tickers()) == 8


class TestVOOOnlyTickers:
    def test_returns_single_ticker(self):
        assert get_voo_only() == ["VOO"]


class TestClassic6040Tickers:
    def test_contains_voo_and_bnd(self):
        result = get_classic_60_40()
        assert "VOO" in result and "BND" in result

    def test_returns_two_tickers(self):
        assert len(get_classic_60_40()) == 2


# ---------------------------------------------------------------------------
# Alternative data universe resolvers
# ---------------------------------------------------------------------------


class TestCongressionalTickers:
    def test_committed_or_seeds_from_fallback(self):
        result = get_congressional_tickers()
        assert isinstance(result, list)
        assert len(result) >= 25
        assert "AAPL" in result and "MSFT" in result

    def test_no_dots_in_tickers(self):
        for t in get_congressional_tickers():
            assert "." not in t

    def test_result_is_sorted(self):
        result = get_congressional_tickers()
        assert result == sorted(result)

    def test_isolated_seed_writes_to_patched_dir(self, midas_data_root, monkeypatch):
        import engine.universes.alternative as alt_mod

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)
        cache_path = fake_dir / "congressional.json"
        assert not cache_path.exists()

        result = alt_mod.get_congressional_tickers()
        assert cache_path.exists()
        assert json.loads(cache_path.read_text()) == result


class TestInsiderTickers:
    def test_committed_or_seeds(self):
        result = get_insider_tickers()
        assert len(result) >= 20
        for t in ("AAPL", "MSFT", "JPM"):
            assert t in result

    def test_result_is_sorted(self):
        assert get_insider_tickers() == sorted(get_insider_tickers())


class TestHighShortTickers:
    def test_committed_or_seeds(self):
        result = get_high_short_tickers()
        # Floor lowered from 20 to 15 after 2026-04-17 delisting cleanup.
        assert len(result) >= 15

    def test_contains_known_meme_stocks(self):
        result = get_high_short_tickers()
        for t in ("GME", "AMC"):
            assert t in result

    def test_result_is_sorted(self):
        assert get_high_short_tickers() == sorted(get_high_short_tickers())


# ---------------------------------------------------------------------------
# STOXX 600 — ISIN-keyed resolution (issue #36)
# ---------------------------------------------------------------------------


def _quote(symbol: str, exchange: str | None, quote_type: str = "EQUITY") -> dict:
    q = {"symbol": symbol, "quoteType": quote_type}
    if exchange is not None:
        q["exchange"] = exchange
    return q


def _row(isin: str, name: str = "Some Co", country: str = "France") -> dict[str, str]:
    return {
        "Constituent ISIN": isin,
        "Constituent Name": name,
        "Constituent Country": country,
    }


class TestStoxx600PickSymbol:
    """Regression: issue #36 — Wikipedia's ticker column produced 120 symbols
    Yahoo had no route for. Resolution is now keyed on ISIN, and these pin the
    listing-selection rule against the vendor answers measured 2026-09-09."""

    def test_home_market_suffix_beats_first_returned(self):
        from engine.universes.index import _pick_symbol

        # Yahoo listed Stuttgart's placeholder first for Inchcape.
        quotes = [_quote("GB00B61TVQ02.SG", "STU"), _quote("INCH.L", "LSE")]
        assert _pick_symbol("GB00B61TVQ02", "United Kingdom", quotes) == "INCH.L"

    def test_xetra_beats_frankfurt_floor_for_a_german_name(self):
        from engine.universes.index import _pick_symbol

        quotes = [_quote("G1A.F", "FRA"), _quote("G1A.DE", "GER")]
        assert _pick_symbol("DE0006602006", "Germany", quotes) == "G1A.DE"

    def test_frankfurt_floor_accepted_for_a_german_name_over_a_foreign_venue(self):
        from engine.universes.index import _pick_symbol

        # Fresenius' new ISIN answered only with its Frankfurt quote; and when a
        # foreign venue is offered too, Frankfurt is still the home market.
        assert _pick_symbol("DE000FRE5EN2", "Germany", [_quote("FRE.F", "FRA")]) == "FRE.F"
        quotes = [_quote("FRE.VI", "VIE"), _quote("FRE.F", "FRA")]
        assert _pick_symbol("DE000FRE5EN2", "Germany", quotes) == "FRE.F"

    def test_german_venues_refused_for_a_foreign_name(self):
        from engine.universes.index import _pick_symbol

        # SAGAX B (Stockholm) answered only as a Frankfurt regional print.
        assert _pick_symbol("SE0005127818", "Sweden", [_quote("EFE.F", "FRA")]) is None
        # A Xetra secondary line for a CHF name is the wrong instrument in the
        # right currency — refused too, and the home listing wins when offered.
        assert _pick_symbol("CH0012032048", "Switzerland", [_quote("RHO5.DE", "GER")]) is None
        quotes = [_quote("RHO5.DE", "GER"), _quote("ROG.SW", "EBS")]
        assert _pick_symbol("CH0012032048", "Switzerland", quotes) == "ROG.SW"
        # Control: the same Xetra line IS the home market for a German name.
        assert _pick_symbol("DE0007236101", "Germany", [_quote("SIE.DE", "GER")]) == "SIE.DE"
        # A domicile the map cannot place (the export files Delivery Hero under
        # Korea) keeps its Xetra answer — but still not a regional floor.
        assert _pick_symbol("DE000A2E4K43", "Korea, Republic of", [_quote("DHER.DE", "GER")]) == "DHER.DE"
        assert _pick_symbol("DE000A2E4K43", "Korea, Republic of", [_quote("DHER.F", "FRA")]) is None

    def test_domicile_without_a_home_listing_takes_the_real_symbol(self):
        from engine.universes.index import _pick_symbol

        # Prosus is domiciled "China" in the export; no suffix preference applies.
        quotes = [_quote("PRX.AS", "AMS"), _quote("NL0013654783.SG", "STU")]
        assert _pick_symbol("NL0013654783", "China", quotes) == "PRX.AS"

    def test_placeholder_only_is_refused(self):
        from engine.universes.index import _pick_symbol

        quotes = [_quote("GB0000000001.SG", "STU")]
        assert _pick_symbol("GB0000000001", "United Kingdom", quotes) is None

    def test_otc_and_non_equity_are_refused(self):
        from engine.universes.index import _pick_symbol

        # NMC Health: the only answer was a pink-sheet line typed MUTUALFUND.
        quotes = [_quote("NMMCF", "PNK", "MUTUALFUND")]
        assert _pick_symbol("GB00B7FC0762", "United Kingdom", quotes) is None
        assert _pick_symbol("GB00B7FC0762", "United Kingdom", []) is None
        # The other two US OTC tiers are OTC as well: a dotless ADR symbol the
        # currency heuristic would otherwise call USD.
        for tier in ("OQX", "OQB"):
            assert _pick_symbol("SE0000000001", "Sweden", [_quote("XXXXY", tier)]) is None
        # Control: the same dotless shape on a real US exchange is accepted.
        assert _pick_symbol("SE0000000001", "Sweden", [_quote("XXXXY", "NMS")]) == "XXXXY"

    def test_quote_without_an_exchange_is_refused(self):
        from engine.universes.index import _pick_symbol

        # yfinance's Search.quotes only guarantees "symbol"; an unattributed
        # listing must not outrank a real one.
        quotes = [_quote("FRE.F", "FRA"), _quote("FREG.XX", None)]
        assert _pick_symbol("DE000FRE5EN2", "Germany", quotes) == "FRE.F"
        assert _pick_symbol("DE000FRE5EN2", "Germany", [_quote("FREG.XX", None)]) is None

    def test_unknown_suffix_is_refused(self):
        """A suffix `engine.quotes` cannot denominate would reach the broker
        only to die with CURRENCY_UNRESOLVED, so it never enters the file."""
        from engine.quotes import _heuristic_unit
        from engine.universes.index import _pick_symbol

        assert _heuristic_unit("HIK.JO") is None  # control: the suffix really is unknown
        assert _pick_symbol("GB00B0LCW083", "Jordan", [_quote("HIK.JO", "JNB")]) is None
        quotes = [_quote("HIK.JO", "JNB"), _quote("HIK.L", "LSE")]
        assert _pick_symbol("GB00B0LCW083", "Jordan", quotes) == "HIK.L"

    def test_us_primary_listing_without_suffix_is_accepted(self):
        from engine.universes.index import _pick_symbol

        assert _pick_symbol("NL0015002SN0", "Netherlands", [_quote("QGEN", "NYQ")]) == "QGEN"

    def test_european_venue_beats_a_us_listing_and_ties_break_on_the_symbol(self):
        """Without a home-market answer, the venue — and so the currency a EUR
        book is exposed to — must not be decided by sort order between a EUR
        and a USD line, nor by the order the vendor happened to list them."""
        from engine.universes.index import _pick_symbol

        a = [_quote("QGEN", "NYQ"), _quote("QIA.PA", "PAR")]
        assert _pick_symbol("NL0012169213", "Jersey", a) == "QIA.PA"
        assert _pick_symbol("NL0012169213", "Jersey", list(reversed(a))) == "QIA.PA"
        # Two European venues, neither the home market: the symbol decides,
        # identically in both orders.
        b = [_quote("CPG.MI", "MIL"), _quote("CPG.L", "LSE")]
        assert _pick_symbol("GB00BD6K4575", "Cayman Islands", b) == "CPG.L"
        assert _pick_symbol("GB00BD6K4575", "Cayman Islands", list(reversed(b))) == "CPG.L"
        # Control: with a home-market suffix the preference, not the tie-break, decides.
        assert _pick_symbol("GB00BD6K4575", "Italy", b) == "CPG.MI"


class TestStoxx600ResolveIsins:
    def test_maps_each_isin_and_lists_the_unresolved(self):
        from engine.universes.index import ISIN_LOOKUP_SPACING_S, resolve_isins

        answers = {
            "FR0000120073": [_quote("AI.PA", "PAR")],
            "FR0014010OO5": [],  # Air Liquide's bonus-share line
            "SE0011166610": [_quote("ATCO-A.ST", "STO")],
        }
        sleeps: list[float] = []
        rows = [
            _row("FR0000120073", "Air Liquide"),
            _row("FR0014010OO5", "L AIR LIQUIDE"),
            _row("SE0011166610", "Atlas Copco A", "Sweden"),
        ]
        resolved, unresolved = resolve_isins(
            rows, lookup=lambda isin: answers[isin], sleep=sleeps.append
        )
        assert resolved == {"FR0000120073": "AI.PA", "SE0011166610": "ATCO-A.ST"}
        assert unresolved == [("FR0014010OO5", "L AIR LIQUIDE")]
        # Throttled BETWEEN lookups: n-1 sleeps of the documented spacing.
        assert sleeps == [ISIN_LOOKUP_SPACING_S] * 2

    def test_stops_as_soon_as_the_unresolved_budget_is_exceeded(self):
        """An outage must not be crawled to the end: 605 x the retry schedule
        is longer than the workflow timeout, and a cancelled job commits
        nothing — not even the six indexes already refreshed to disk."""
        from engine.universes.index import resolve_isins

        rows = [_row(f"FR{i:010d}") for i in range(50)]
        calls: list[str] = []

        def lookup(isin: str) -> list[dict]:
            calls.append(isin)
            return []

        with pytest.raises(RuntimeError, match="unresolved"):
            resolve_isins(rows, max_unresolved=3, lookup=lookup, sleep=lambda s: None)
        assert len(calls) == 4

    def test_stops_after_consecutive_lookup_failures(self):
        from engine.universes.index import MAX_CONSECUTIVE_LOOKUP_FAILURES, resolve_isins

        rows = [_row(f"FR{i:010d}") for i in range(50)]
        calls: list[str] = []

        def lookup(isin: str) -> None:
            calls.append(isin)
            return None  # the lookup itself failed

        with pytest.raises(RuntimeError, match="lookup unavailable"):
            resolve_isins(rows, lookup=lookup, sleep=lambda s: None)
        assert len(calls) == MAX_CONSECUTIVE_LOOKUP_FAILURES

    def test_scattered_failures_reset_the_consecutive_count(self):
        """More failures in total than MAX_CONSECUTIVE_LOOKUP_FAILURES, never
        two in a row: the run completes. Deleting the reset makes this raise
        — verified, which is what makes the test evidence."""
        from engine.universes.index import MAX_CONSECUTIVE_LOOKUP_FAILURES, resolve_isins

        n = 2 * MAX_CONSECUTIVE_LOOKUP_FAILURES + 2
        answers = [None if i % 2 == 0 else [_quote(f"S{i}.PA", "PAR")] for i in range(n)]
        rows = [_row(f"FR{i:010d}", f"Co {i}") for i in range(n)]
        resolved, unresolved = resolve_isins(
            rows, lookup=lambda isin: answers.pop(0), sleep=lambda s: None
        )
        assert len(resolved) == n // 2
        assert len(unresolved) == n // 2
        assert all(int(isin[2:]) % 2 == 0 for isin, _ in unresolved)

    def test_stops_when_the_wall_clock_budget_is_spent(self):
        """Every lookup succeeding after a back-off bounds nothing above: the
        crawl must stop on elapsed time, or the job is cancelled and the six
        indexes already refreshed to disk die with the runner."""
        from engine.universes.index import resolve_isins

        rows = [_row(f"FR{i:010d}") for i in range(50)]
        ticks = iter(range(0, 10_000, 10))  # each lookup "costs" 10 s
        calls: list[str] = []

        def lookup(isin: str) -> list[dict]:
            calls.append(isin)
            return [_quote("A.PA", "PAR")]

        with pytest.raises(RuntimeError, match="budget exhausted"):
            resolve_isins(
                rows, budget_s=35, lookup=lookup, sleep=lambda s: None, clock=lambda: next(ticks)
            )
        # Clock reads 0 at start, then 10/20/30 before lookups 1-3, 40 > 35 stops.
        assert len(calls) == 3
        # Control: no budget, the same crawl completes.
        calls.clear()
        resolved, _ = resolve_isins(
            rows, budget_s=None, lookup=lookup, sleep=lambda s: None, clock=lambda: 0.0
        )
        assert len(calls) == 50 and len(resolved) == 50


class TestStoxx600SearchIsinQuotes:
    def test_retries_a_rate_limit_then_gives_up(self, monkeypatch):
        import yfinance
        from yfinance.exceptions import YFRateLimitError

        import engine.universes.index as ix_mod

        class FlakyOnce:
            calls = 0

            def __init__(self, query, **kwargs):
                FlakyOnce.calls += 1
                if FlakyOnce.calls == 1:
                    raise YFRateLimitError()
                self.quotes = [_quote("AI.PA", "PAR")]

        monkeypatch.setattr(yfinance, "Search", FlakyOnce)
        sleeps: list[float] = []
        assert ix_mod._search_isin_quotes("FR0000120073", sleep=sleeps.append) == [
            _quote("AI.PA", "PAR")
        ]
        assert sleeps == [ix_mod.ISIN_LOOKUP_RETRY_SLEEPS_S[0]]

        class AlwaysLimited:
            def __init__(self, query, **kwargs):
                raise YFRateLimitError()

        monkeypatch.setattr(yfinance, "Search", AlwaysLimited)
        sleeps.clear()
        assert ix_mod._search_isin_quotes("FR0000120073", sleep=sleeps.append) is None
        assert sleeps == list(ix_mod.ISIN_LOOKUP_RETRY_SLEEPS_S)

    def test_transport_errors_retry_and_other_errors_propagate(self, monkeypatch):
        """A signature change in yfinance or Yahoo's own "currently down"
        exception is a fact about the run, not the ISIN: it must surface at
        once instead of burning the retry schedule on all ~605 lines."""
        import yfinance

        import engine.universes.index as ix_mod

        class Unreachable:
            def __init__(self, query, **kwargs):
                raise ConnectionError("no route to host")

        monkeypatch.setattr(yfinance, "Search", Unreachable)
        sleeps: list[float] = []
        assert ix_mod._search_isin_quotes("FR0000120073", sleep=sleeps.append) is None
        assert sleeps == list(ix_mod.ISIN_LOOKUP_RETRY_SLEEPS_S)

        class HtmlBody:
            def __init__(self, query, **kwargs):
                raise json.JSONDecodeError("Expecting value", "<html>", 0)

        # An HTML 5xx or captcha page: yfinance would swallow it into an
        # empty answer by default, which reads as "no listing". Not here.
        monkeypatch.setattr(yfinance, "Search", HtmlBody)
        sleeps.clear()
        assert ix_mod._search_isin_quotes("FR0000120073", sleep=sleeps.append) is None
        assert sleeps == list(ix_mod.ISIN_LOOKUP_RETRY_SLEEPS_S)

        class SignatureChanged:
            def __init__(self, query, **kwargs):
                raise TypeError("unexpected keyword argument 'news_count'")

        monkeypatch.setattr(yfinance, "Search", SignatureChanged)
        sleeps.clear()
        with pytest.raises(TypeError):
            ix_mod._search_isin_quotes("FR0000120073", sleep=sleeps.append)
        assert sleeps == []

    def test_yfinance_is_told_not_to_hide_a_faulty_body(self, monkeypatch):
        import yfinance
        from yfinance.config import YfConfig

        import engine.universes.index as ix_mod

        seen: list[bool] = []

        class Recording:
            def __init__(self, query, **kwargs):
                seen.append(YfConfig.debug.hide_exceptions)
                self.quotes = []

        monkeypatch.setattr(yfinance, "Search", Recording)
        monkeypatch.setattr(YfConfig.debug, "hide_exceptions", True)
        assert ix_mod._search_isin_quotes("FR0000120073", sleep=lambda s: None) == []
        assert seen == [False]
        assert YfConfig.debug.hide_exceptions is True  # restored


class TestStoxx600Constituents:
    def test_keeps_only_isin_shaped_lines(self, monkeypatch):
        import engine.universes.index as ix_mod

        csv_text = "﻿" + "\n".join(
            [
                "ShareClass ISIN;Constituent ISIN;Constituent Name;Constituent Country;"
                "Constituent Currency ISO Code;Constituent Weighting",
                "LU0328475792;NL0010273215;ASML Holding NV;Netherlands;EUR;0.04",
                "LU0328475792;_CURRENCYEUR;EURO CURRENCY;;EUR;0.001",
                "LU0328475792;___ADI2V9YR9;STOXX EUROPE 600  SEP26;Germany;EUR;0.002",
                "LU0328475792;IE00BZ3FDF20;;Ireland;EUR;0.0001",
            ]
        )
        monkeypatch.setattr(
            ix_mod.urllib.request,
            "urlopen",
            lambda req, timeout: io.BytesIO(csv_text.encode("utf-8")),
        )
        rows = ix_mod._fetch_stoxx600_constituents()
        assert [r["Constituent ISIN"] for r in rows] == ["NL0010273215", "IE00BZ3FDF20"]
        assert rows[0]["Constituent Name"] == "ASML Holding NV"

    def test_missing_column_is_a_layout_change(self, monkeypatch):
        import engine.universes.index as ix_mod

        csv_text = "ShareClass ISIN;ISIN;Name\nLU0328475792;NL0010273215;ASML\n"
        monkeypatch.setattr(
            ix_mod.urllib.request,
            "urlopen",
            lambda req, timeout: io.BytesIO(csv_text.encode("utf-8")),
        )
        with pytest.raises(RuntimeError, match="layout changed"):
            ix_mod._fetch_stoxx600_constituents()


class TestRefreshStoxx600:
    @staticmethod
    def _rows(n: int) -> list[dict[str, str]]:
        return [_row(f"FR{i:010d}", f"Co {i}") for i in range(n)]

    @staticmethod
    def _install(monkeypatch, rows, lookup) -> None:
        import engine.universes.index as ix_mod

        monkeypatch.setattr(ix_mod, "_fetch_stoxx600_constituents", lambda: rows)
        monkeypatch.setattr(ix_mod, "_search_isin_quotes", lookup)
        monkeypatch.setattr(ix_mod.time, "sleep", lambda s: None)

    @staticmethod
    def _lookup_unresolving_first(n: int):
        def lookup(isin: str) -> list[dict]:
            i = int(isin[2:])
            return [] if i < n else [_quote(f"C{i:04d}.PA", "PAR")]

        return lookup

    def test_writes_sorted_deduped_symbols_and_tolerates_a_small_residual(
        self, midas_data_root, monkeypatch
    ):
        import engine.universes.index as ix_mod

        rows = self._rows(500)

        # Two ISINs answering the same symbol (a loyalty-share line that DOES
        # resolve) collapse to one; one line answers nothing.
        def lookup(isin: str) -> list[dict]:
            i = int(isin[2:])
            if i == 7:
                return []
            if i == 8:
                return [_quote("C0006.PA", "PAR")]
            return [_quote(f"C{i:04d}.PA", "PAR")]

        self._install(monkeypatch, rows, lookup)
        result = ix_mod.refresh_stoxx600()
        assert result == sorted(set(result))
        assert len(result) == 498
        assert "C0007.PA" not in result
        written = json.loads((get_config().universes_dir / "stoxx600.json").read_text())
        assert written == result

    def test_refuses_to_overwrite_when_the_vendor_lookup_is_down(
        self, midas_data_root, monkeypatch
    ):
        """A rate limit that outlasts the retry schedule, or an outage, must
        leave the committed file at its last known-good value rather than
        shrink it to whatever trickled through."""
        import engine.universes.index as ix_mod

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)
        (fake_dir / "stoxx600.json").write_text(json.dumps(["KEEP.PA"]))

        rows = self._rows(500)
        unresolved_n = int(len(rows) * ix_mod.MAX_UNRESOLVED_ISIN_RATE) + 1
        self._install(monkeypatch, rows, self._lookup_unresolving_first(unresolved_n))
        with pytest.raises(RuntimeError, match="unresolved"):
            ix_mod.refresh_stoxx600()
        assert json.loads((fake_dir / "stoxx600.json").read_text()) == ["KEEP.PA"]

    def test_control_the_unresolved_gate_can_pass(self, midas_data_root, monkeypatch):
        """Falsifying control for the test above: one fewer unresolved line and
        the same setup writes the file."""
        import engine.universes.index as ix_mod

        rows = self._rows(500)
        unresolved_n = int(len(rows) * ix_mod.MAX_UNRESOLVED_ISIN_RATE)
        self._install(monkeypatch, rows, self._lookup_unresolving_first(unresolved_n))
        assert len(ix_mod.refresh_stoxx600()) == 500 - unresolved_n

    def test_refuses_a_result_that_churns_the_committed_list(
        self, midas_data_root, monkeypatch
    ):
        """The refresh commits to main unattended. A selection regression that
        swaps ~200 names for their Frankfurt or US twins keeps the count near
        600 and passes every other gate; only a comparison with the file being
        overwritten can see it."""
        import engine.universes.index as ix_mod

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)
        previous = [f"C{i:04d}.PA" for i in range(500)]
        (fake_dir / "stoxx600.json").write_text(json.dumps(previous))

        rows = self._rows(500)
        moved = int(500 * ix_mod.MAX_UNIVERSE_CHURN_RATE / 2) + 1  # add + remove

        def lookup(isin: str) -> list[dict]:
            i = int(isin[2:])
            # The first `moved` lines resolve to a DIFFERENT symbol than the
            # committed one: each is one added plus one removed, the count is
            # unchanged, and every other gate passes.
            prefix = "X" if i < moved else "C"
            return [_quote(f"{prefix}{i:04d}.PA", "PAR")]

        self._install(monkeypatch, rows, lookup)
        monkeypatch.delenv(ix_mod._ACCEPT_CHURN_ENV, raising=False)
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            ix_mod.refresh_stoxx600()
        assert json.loads((fake_dir / "stoxx600.json").read_text()) == previous

        # "0" and "false" are refusals, not consent — engine.live_switch's
        # convention, which a bare truthiness test would have inverted.
        for value in ("0", "false"):
            monkeypatch.setenv(ix_mod._ACCEPT_CHURN_ENV, value)
            with pytest.raises(RuntimeError, match="refusing to overwrite"):
                ix_mod.refresh_stoxx600()
        # Control: the documented override, set by a human on a deliberate
        # rebuild, lets the same result through.
        monkeypatch.setenv(ix_mod._ACCEPT_CHURN_ENV, "1")
        result = ix_mod.refresh_stoxx600()
        assert json.loads((fake_dir / "stoxx600.json").read_text()) == result
        assert len(result) == 500

    def test_control_a_rebalance_sized_change_passes_the_churn_gate(
        self, midas_data_root, monkeypatch
    ):
        import engine.universes.index as ix_mod

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)
        previous = [f"C{i:04d}.PA" for i in range(500)]
        (fake_dir / "stoxx600.json").write_text(json.dumps(previous))
        # 10 names leave, 10 arrive: a quarter's rebalance, 4% churn.
        rows = self._rows(510)[10:]
        self._install(
            monkeypatch,
            rows,
            lambda isin: [_quote(f"C{int(isin[2:]):04d}.PA", "PAR")],
        )
        monkeypatch.delenv(ix_mod._ACCEPT_CHURN_ENV, raising=False)
        assert len(ix_mod.refresh_stoxx600()) == 500

    def test_missing_committed_file_fails_fast_instead_of_crawling(
        self, midas_data_root, monkeypatch
    ):
        """The other indexes fall back to a 15 s page fetch; this one would
        fall back to a seven-minute vendor crawl on the session path."""
        import engine.universes.index as ix_mod

        def boom(*a, **kw):
            raise AssertionError("network must not be called")

        monkeypatch.setattr(ix_mod, "_fetch_stoxx600_constituents", boom)
        monkeypatch.setattr(ix_mod, "_search_isin_quotes", boom)
        with pytest.raises(FileNotFoundError, match="refresh_universes"):
            ix_mod.get_stoxx600_tickers()

    def test_no_network_call_when_file_exists(self, midas_data_root, monkeypatch):
        import engine.universes.index as ix_mod

        fake_dir = get_config().universes_dir
        fake_dir.mkdir(parents=True, exist_ok=True)
        (fake_dir / "stoxx600.json").write_text(json.dumps(["AI.PA"]))

        def boom(*a, **kw):
            raise AssertionError("network must not be called when data file exists")

        monkeypatch.setattr(ix_mod, "_fetch_stoxx600_constituents", boom)
        monkeypatch.setattr(ix_mod, "_search_isin_quotes", boom)
        assert ix_mod.get_stoxx600_tickers() == ["AI.PA"]
