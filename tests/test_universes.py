"""Tests for universe resolvers — index, asset class, and alternative data.

Tests must NEVER write to the real `data/universes/` directory. Every test
that exercises the cache-write path monkeypatches `_DATA_DIR` to a tmp_path.
A previous version of this file silently overwrote the real `sp500.json`
during pytest runs, dropping it from 503 tickers to 3 — the Apr 29 cloud
session aborted as a downstream consequence.
"""

from __future__ import annotations

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
