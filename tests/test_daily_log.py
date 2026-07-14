"""Tests for daily log generator."""

from datetime import date
from pathlib import Path

import pytest

from engine.daily_log import generate_daily_log


class TestPositionsDualShape:
    """Positions may be a list of ticker strings OR a list of {ticker, shares} dicts.

    Regression (2026-04-17): the leaderboard silently returned cash-only MTM when
    positions were passed as strings because portfolio_mtm couldn't extract shares.
    The per-agent display line then TypeError'd when positions were dicts.
    Both shapes must now work.
    """

    def test_positions_as_strings_renders(self, midas_data_root: Path) -> None:
        summaries = {
            "satoshi": {
                "cash": 5000.0,
                "deployed": 5000.0,
                "positions": ["BTC-EUR", "ETH-EUR"],
                "currency": "EUR",
            },
        }
        path = generate_daily_log(
            date(2026, 4, 17),
            {},
            {"satoshi": {"commentary": "", "trades": []}},
            summaries,
        )
        c = path.read_text()
        assert "**Positions (2):** BTC-EUR, ETH-EUR" in c

    def test_positions_as_dicts_renders(self, midas_data_root: Path) -> None:
        summaries = {
            "satoshi": {
                "cash": 5000.0,
                "deployed": 5000.0,
                "positions": [
                    {"ticker": "BTC-EUR", "shares": 0.05},
                    {"ticker": "ETH-EUR", "shares": 1.5},
                ],
                "currency": "EUR",
            },
        }
        path = generate_daily_log(
            date(2026, 4, 17),
            {},
            {"satoshi": {"commentary": "", "trades": []}},
            summaries,
        )
        c = path.read_text()
        assert "**Positions (2):** BTC-EUR, ETH-EUR" in c


class TestDailyLog:
    pytestmark = pytest.mark.live_cast

    def test_generates_markdown_file(self, midas_data_root: Path) -> None:
        market = {"sp500": 6967.38, "gold": 4825.0, "btc": 74181.61}
        agent_results = {
            "steady-eddie-eur": {
                "commentary": "Markets choppy. Staying defensive.",
                "trades": [
                    {
                        "action": "BUY",
                        "ticker": "JNJ",
                        "shares": 5,
                        "reasoning": "Defensive healthcare play.",
                    },
                ],
            },
            "yolo-sapiens-usd": {
                "commentary": "YOLO into leveraged tech.",
                "trades": [
                    {
                        "action": "BUY",
                        "ticker": "TQQQ",
                        "shares": 25,
                        "reasoning": "3x Nasdaq for max beta.",
                    },
                ],
            },
        }
        portfolio_summaries = {
            "steady-eddie-eur": {
                "cash": 1965.0,
                "deployed": 8035.0,
                "positions": ["JNJ", "XOM", "PG"],
            },
            "yolo-sapiens-usd": {
                "cash": 156.0,
                "deployed": 9844.0,
                "positions": ["TQQQ", "SOXL"],
            },
        }

        path = generate_daily_log(
            date(2026, 4, 14), market, agent_results, portfolio_summaries
        )

        assert path.exists()
        assert path.name == "2026-04-14.md"

        content = path.read_text()
        assert "# Midas Daily Log" in content
        assert "6,967.38" in content
        assert "Steady Eddie EUR" in content
        assert "Markets choppy" in content
        assert "JNJ" in content
        assert "YOLO Sapiens USD" in content
        assert "$1,965.00" in content
        assert "TQQQ, SOXL" in content

    def test_no_trades_day(self, midas_data_root: Path) -> None:
        agent_results = {
            "steady-eddie": {
                "commentary": "No opportunities today. Holding positions.",
                "trades": [],
            },
        }
        path = generate_daily_log(date(2026, 4, 15), {}, agent_results, {})
        content = path.read_text()
        assert "No trades today" in content
        assert "No opportunities" in content

    def test_all_agents_present(self, midas_data_root: Path) -> None:
        agents = [
            "steady-eddie-eur",
            "sharp-shooter-usd",
            "satoshi",
            "monsieur-forex",
            "goldfinger",
            "yolo-sapiens-eur",
        ]
        agent_results = {
            a: {"commentary": f"{a} commentary", "trades": []} for a in agents
        }

        path = generate_daily_log(date(2026, 4, 14), {}, agent_results, {})
        content = path.read_text()

        for display in [
            "Steady Eddie EUR",
            "Sharp Shooter USD",
            "Satoshi",
            "Monsieur Forex",
            "Goldfinger",
            "YOLO Sapiens EUR",
        ]:
            assert display in content
