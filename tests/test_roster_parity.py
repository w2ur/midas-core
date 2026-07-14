"""Snapshot tests — roster.yaml must produce the expected cast (default env).

Inline expected values so the test is not tautological (not config vs config).
Covers display_name, post_time, home_currency, max_positions, benchmark, universe
names, Oracle role, and trading_roster order.
"""

from __future__ import annotations

import pytest

from engine.config import get_config, reset_config_cache

pytestmark = pytest.mark.live_cast


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv("MIDAS_DATA_DIR", raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


class TestRosterSnapshot:
    # Inline expected values — the migration is locked in; these values come
    # from the roster.yaml committed in Task 1, not from any live dict.

    EXPECTED_TRADERS = {
        "monsieur-forex": {
            "display_name": "Monsieur Forex",
            "post_time": "07:00",
            "home_currency": "EUR",
            "max_positions": 6,
            "benchmark_ticker": "EUR_CASH_FLAT",
            "universe": ["forex-majors"],
        },
        "steady-eddie-eur": {
            "display_name": "Steady Eddie EUR",
            "post_time": "08:00",
            "home_currency": "EUR",
            "max_positions": 10,
            "benchmark_ticker": "VGK",
            "universe": ["stoxx-600", "cac40", "dax", "ftse100"],
        },
        "steady-eddie-usd": {
            "display_name": "Steady Eddie USD",
            "post_time": "08:15",
            "home_currency": "USD",
            "max_positions": 10,
            "benchmark_ticker": "SPY",
            "universe": ["sp500"],
        },
        "sharp-shooter-eur": {
            "display_name": "Sharp Shooter EUR",
            "post_time": "09:35",
            "home_currency": "EUR",
            "max_positions": 8,
            "benchmark_ticker": "VGK",
            "universe": ["stoxx-600", "cac40", "dax", "ftse100", "bearish-etfs-ucits"],
        },
        "sharp-shooter-usd": {
            "display_name": "Sharp Shooter USD",
            "post_time": "09:45",
            "home_currency": "USD",
            "max_positions": 8,
            "benchmark_ticker": "SPY",
            "universe": ["sp500", "bearish-etfs-ucits"],
        },
        "world": {
            "display_name": "World",
            "post_time": "10:00",
            "home_currency": "EUR",
            "max_positions": 12,
            "benchmark_ticker": "URTH",
            "universe": [
                "sp500",
                "stoxx-600",
                "cac40",
                "dax",
                "ftse100",
                "crypto-top20-eur",
                "crypto-top20",
                "forex-majors",
                "commodities-eur",
                "bearish-etfs-ucits",
            ],
        },
        "goldfinger": {
            "display_name": "Goldfinger",
            "post_time": "11:00",
            "home_currency": "EUR",
            "max_positions": 6,
            "benchmark_ticker": "4GLD.DE",
            "universe": ["commodities-eur"],
        },
        "yolo-sapiens-eur": {
            "display_name": "YOLO Sapiens EUR",
            "post_time": "random",
            "home_currency": "EUR",
            "max_positions": 5,
            "benchmark_ticker": "VGK",
            "universe": [
                "stoxx-600",
                "cac40",
                "dax",
                "ftse100",
                "crypto-top20-eur",
                "commodities-eur",
                "bearish-etfs-ucits",
            ],
        },
        "yolo-sapiens-usd": {
            "display_name": "YOLO Sapiens USD",
            "post_time": "random",
            "home_currency": "USD",
            "max_positions": 5,
            "benchmark_ticker": "SPY",
            "universe": [
                "sp500",
                "crypto-top20",
                "forex-majors",
                "metals-commodities",
                "bearish-etfs-ucits",
            ],
        },
        "satoshi": {
            "display_name": "Satoshi",
            "post_time": "23:00",
            "home_currency": "EUR",
            "max_positions": 8,
            "benchmark_ticker": "BTC-EUR",
            "universe": ["crypto-top20-eur"],
        },
    }

    # Trading roster order is load-bearing (matches post-time ordering in roster.yaml).
    EXPECTED_TRADING_ROSTER = (
        "monsieur-forex",
        "steady-eddie-eur",
        "steady-eddie-usd",
        "sharp-shooter-eur",
        "sharp-shooter-usd",
        "world",
        "goldfinger",
        "yolo-sapiens-eur",
        "yolo-sapiens-usd",
        "satoshi",
    )

    def test_trading_roster_order(self):
        cfg = get_config()
        assert cfg.trading_roster == self.EXPECTED_TRADING_ROSTER

    def test_trader_display_names(self):
        cfg = get_config()
        for aid, expected in self.EXPECTED_TRADERS.items():
            assert cfg.roster[aid].display_name == expected["display_name"], aid

    def test_trader_post_times(self):
        cfg = get_config()
        for aid, expected in self.EXPECTED_TRADERS.items():
            assert cfg.roster[aid].post_time == expected["post_time"], aid

    def test_trader_home_currencies(self):
        cfg = get_config()
        for aid, expected in self.EXPECTED_TRADERS.items():
            assert cfg.roster[aid].home_currency == expected["home_currency"], aid

    def test_trader_max_positions(self):
        cfg = get_config()
        for aid, expected in self.EXPECTED_TRADERS.items():
            assert cfg.roster[aid].max_positions == expected["max_positions"], aid

    def test_trader_benchmark_tickers(self):
        cfg = get_config()
        for aid, expected in self.EXPECTED_TRADERS.items():
            bench = cfg.roster[aid].benchmark
            assert bench is not None, f"{aid} missing benchmark"
            assert bench.ticker == expected["benchmark_ticker"], aid

    def test_trader_universe_names(self):
        """Each trader's universe field matches the expected list of universe names.

        Does NOT resolve the tickers — resolving 500+ tickers belongs in
        test_universes_resolve.py. This only checks the names stored in config.
        """
        cfg = get_config()
        for aid, expected in self.EXPECTED_TRADERS.items():
            raw = cfg.roster[aid].universe
            # Normalise to list for comparison
            actual = [raw] if isinstance(raw, str) else list(raw or [])
            assert actual == expected["universe"], aid

    def test_oracle_is_narrator(self):
        cfg = get_config()
        assert "the-oracle" in cfg.roster
        assert cfg.roster["the-oracle"].role == "narrator"
        assert cfg.roster["the-oracle"].display_name == "The Oracle"

    def test_oracle_not_in_trading_roster(self):
        cfg = get_config()
        assert "the-oracle" not in cfg.trading_roster

    def test_allocators_is_the_manager(self):
        cfg = get_config()
        assert cfg.allocators == ("the-manager",)
        assert "the-manager" not in cfg.trading_roster
        assert cfg.roster["the-manager"].role == "allocator"

    def test_day_one_and_initial_capital(self):
        from datetime import date

        cfg = get_config()
        assert cfg.day_one == date(2026, 4, 17)
        assert cfg.initial_capital == 10_000.0

    def test_global_reference_is_msci_world(self):
        cfg = get_config()
        assert cfg.global_reference.ticker == "URTH"
        assert cfg.global_reference.currency == "EUR"
