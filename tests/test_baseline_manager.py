"""Tests for engine.baseline_manager — deterministic baseline-manager twin.

TDD: these tests were written before the implementation.

The baseline-manager is an internal Gate C benchmark portfolio, NOT a public
trading agent. It is excluded from AGENT_POST_TIMES, AGENT_DISPLAY_NAMES,
and roster.ts.

Rules under test:
- eligible_tickers: ≥2 distinct agents mark a ticker strong_buy/buy
- Ranking: count desc, then strong_buy-weight desc (strong_buy=2/buy=1), then ticker alpha
- Cap: at most 6 positions
- is_rebalance_day: first weekday (Mon-Fri) of each calendar month
- rebalance: equal-weight EUR 300/position, sells exits, respects cash, applies fees
- Missing-price tickers excluded
- First run: initializes EUR 2000 portfolio
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.baseline_manager import eligible_tickers, is_rebalance_day, rebalance
from engine.research_note import ResearchNote


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _note(action_bias: str, tickers: list[str]) -> ResearchNote:
    """Minimal valid ResearchNote factory."""
    return ResearchNote(
        thesis="Test thesis.",
        conviction=7,
        tickers=tickers,
        action_bias=action_bias,
        horizon="weeks",
        catalysts="Test catalysts.",
        currency="EUR",
    )


def _price_lookup(prices: dict[str, float]):
    """Return a price-lookup callable backed by a fixed dict."""

    def lookup(ticker: str, on: date) -> float | None:
        return prices.get(ticker)

    return lookup


# ---------------------------------------------------------------------------
# eligible_tickers — threshold and ranking
# ---------------------------------------------------------------------------


class TestEligibleTickers:
    def test_ticker_with_two_buy_agents_included(self) -> None:
        notes = [
            ("agent-a", _note("buy", ["AAPL"])),
            ("agent-b", _note("buy", ["AAPL"])),
        ]
        result = eligible_tickers(notes)
        assert "AAPL" in result

    def test_ticker_with_one_agent_excluded(self) -> None:
        notes = [
            ("agent-a", _note("buy", ["AAPL"])),
            ("agent-b", _note("buy", ["MSFT"])),  # AAPL only has 1 agent
        ]
        result = eligible_tickers(notes)
        assert "AAPL" not in result
        assert "MSFT" not in result

    def test_strong_buy_counts_as_buy(self) -> None:
        notes = [
            ("agent-a", _note("strong_buy", ["BTC-EUR"])),
            ("agent-b", _note("buy", ["BTC-EUR"])),
        ]
        result = eligible_tickers(notes)
        assert "BTC-EUR" in result

    def test_hold_does_not_count(self) -> None:
        notes = [
            ("agent-a", _note("hold", ["AAPL"])),
            ("agent-b", _note("buy", ["AAPL"])),
        ]
        result = eligible_tickers(notes)
        assert "AAPL" not in result

    def test_same_agent_twice_only_counts_once(self) -> None:
        """Two notes from the same agent for the same ticker count as 1."""
        notes = [
            ("agent-a", _note("buy", ["AAPL"])),
            ("agent-a", _note("strong_buy", ["AAPL"])),  # same agent!
        ]
        result = eligible_tickers(notes)
        assert "AAPL" not in result

    def test_ranking_by_count_desc(self) -> None:
        """Ticker with more agent votes comes first."""
        notes = [
            ("agent-a", _note("buy", ["AAPL"])),
            ("agent-b", _note("buy", ["AAPL"])),
            ("agent-c", _note("buy", ["AAPL"])),  # AAPL: 3 agents
            ("agent-d", _note("buy", ["MSFT"])),
            ("agent-e", _note("buy", ["MSFT"])),  # MSFT: 2 agents
        ]
        result = eligible_tickers(notes)
        assert result[0] == "AAPL"
        assert result[1] == "MSFT"

    def test_ranking_strong_buy_weight_tiebreak(self) -> None:
        """When count ties, strong_buy-weight (strong_buy=2/buy=1) breaks tie."""
        notes = [
            ("agent-a", _note("strong_buy", ["MSFT"])),
            ("agent-b", _note("buy", ["MSFT"])),  # MSFT: count=2, weight=3
            ("agent-c", _note("buy", ["AAPL"])),
            ("agent-d", _note("buy", ["AAPL"])),  # AAPL: count=2, weight=2
        ]
        result = eligible_tickers(notes)
        assert result[0] == "MSFT"
        assert result[1] == "AAPL"

    def test_ranking_ticker_alpha_tiebreak(self) -> None:
        """When count AND weight tie, alphabetical order is used."""
        notes = [
            ("agent-a", _note("buy", ["AAPL"])),
            ("agent-b", _note("buy", ["AAPL"])),  # AAPL: count=2, weight=2
            ("agent-c", _note("buy", ["MSFT"])),
            ("agent-d", _note("buy", ["MSFT"])),  # MSFT: count=2, weight=2
        ]
        result = eligible_tickers(notes)
        assert result[0] == "AAPL"
        assert result[1] == "MSFT"

    def test_capped_at_six(self) -> None:
        """At most 6 tickers returned."""
        tickers = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
        notes = []
        for t in tickers:
            notes.append(("agent-a", _note("buy", [t])))
            notes.append(("agent-b", _note("buy", [t])))
        result = eligible_tickers(notes)
        assert len(result) <= 6

    def test_fewer_than_six_eligible_returns_all(self) -> None:
        """If fewer than 6 eligible, return all of them."""
        notes = [
            ("agent-a", _note("buy", ["AAPL"])),
            ("agent-b", _note("buy", ["AAPL"])),
        ]
        result = eligible_tickers(notes)
        assert result == ["AAPL"]

    def test_empty_notes_returns_empty(self) -> None:
        assert eligible_tickers([]) == []

    def test_ticker_in_multiple_notes_from_same_agent(self) -> None:
        """Ticker appears in multiple tickers lists for the same agent — deduplicated."""
        notes = [
            ("agent-a", _note("buy", ["AAPL", "MSFT"])),
            ("agent-a", _note("buy", ["AAPL"])),  # same agent again
            ("agent-b", _note("buy", ["AAPL"])),
        ]
        # agent-a gives 2 signals for AAPL, but they're the same agent → count=2 (a+b)
        result = eligible_tickers(notes)
        assert "AAPL" in result


# ---------------------------------------------------------------------------
# is_rebalance_day
# ---------------------------------------------------------------------------


class TestIsRebalanceDay:
    def test_first_weekday_when_1st_is_monday(self) -> None:
        # 2026-06-01 is a Monday
        assert is_rebalance_day(date(2026, 6, 1)) is True

    def test_first_weekday_when_1st_is_saturday(self) -> None:
        # 2026-08-01 is a Saturday → first weekday is Monday 2026-08-03
        assert is_rebalance_day(date(2026, 8, 3)) is True

    def test_first_of_month_that_is_saturday_is_not_rebalance(self) -> None:
        # 2026-08-01 is a Saturday — not the first weekday
        assert is_rebalance_day(date(2026, 8, 1)) is False

    def test_first_weekday_when_1st_is_sunday(self) -> None:
        # 2027-08-01 is a Sunday → first weekday is Monday 2027-08-02
        assert is_rebalance_day(date(2027, 8, 2)) is True

    def test_mid_month_is_not_rebalance(self) -> None:
        assert is_rebalance_day(date(2026, 6, 15)) is False

    def test_second_weekday_is_not_rebalance(self) -> None:
        # 2026-06-01 is Monday (rebalance), 2026-06-02 is Tuesday (not)
        assert is_rebalance_day(date(2026, 6, 2)) is False

    def test_first_weekday_when_1st_is_friday(self) -> None:
        # 2026-05-01 is a Friday — it IS the first weekday
        assert is_rebalance_day(date(2026, 5, 1)) is True

    def test_first_weekday_when_1st_is_wednesday(self) -> None:
        # 2026-04-01 is a Wednesday — it IS the first weekday
        assert is_rebalance_day(date(2026, 4, 1)) is True


# ---------------------------------------------------------------------------
# rebalance — trade generation
# ---------------------------------------------------------------------------


class TestRebalance:
    def test_buy_target_positions_from_cash(self) -> None:
        """Starting from all-cash, rebalance buys target tickers at EUR 300 each."""
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        lookup = _price_lookup(prices)
        portfolio = {"cash": 2000.0, "positions": []}
        target = ["AAPL", "MSFT"]

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=target,
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning="baseline-manager monthly rebalance: ≥2-agent buy consensus",
        )

        buy_trades = [t for t in trades if t.action == "BUY"]
        buy_tickers = {t.ticker for t in buy_trades}
        assert "AAPL" in buy_tickers
        assert "MSFT" in buy_tickers

    def test_sell_removed_position(self) -> None:
        """Rebalance sells a position that dropped out of the target set."""
        prices = {"AAPL": 160.0, "MSFT": 300.0}  # AAPL price needed to sell it
        lookup = _price_lookup(prices)
        portfolio = {
            "cash": 1700.0,
            "positions": [{"ticker": "AAPL", "shares": 2.0, "avg_cost": 150.0}],
        }
        target = ["MSFT"]

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=target,
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning="baseline-manager monthly rebalance: ≥2-agent buy consensus",
        )

        sell_trades = [t for t in trades if t.action == "SELL"]
        assert any(t.ticker == "AAPL" for t in sell_trades)

    def test_missing_price_ticker_excluded(self) -> None:
        """Ticker with no price in store is skipped; no trade emitted for it."""
        prices = {"MSFT": 300.0}  # AAPL has no price
        lookup = _price_lookup(prices)
        portfolio = {"cash": 2000.0, "positions": []}
        target = ["AAPL", "MSFT"]

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=target,
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning="baseline-manager monthly rebalance: ≥2-agent buy consensus",
        )

        buy_tickers = {t.ticker for t in trades if t.action == "BUY"}
        assert "AAPL" not in buy_tickers
        assert "MSFT" in buy_tickers

    def test_fees_applied_on_buys(self) -> None:
        """Every BUY trade has fees > 0."""
        prices = {"AAPL": 150.0}
        lookup = _price_lookup(prices)
        portfolio = {"cash": 2000.0, "positions": []}
        target = ["AAPL"]

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=target,
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning="baseline-manager monthly rebalance: ≥2-agent buy consensus",
        )

        buy_trades = [t for t in trades if t.action == "BUY"]
        assert buy_trades
        assert all(t.fees > 0 for t in buy_trades)

    def test_fees_applied_on_sells(self) -> None:
        """Every SELL trade has fees > 0."""
        prices = {"AAPL": 160.0, "MSFT": 300.0}  # AAPL price needed to sell it
        lookup = _price_lookup(prices)
        portfolio = {
            "cash": 1700.0,
            "positions": [{"ticker": "AAPL", "shares": 2.0, "avg_cost": 150.0}],
        }
        target = ["MSFT"]

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=target,
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning="baseline-manager monthly rebalance: ≥2-agent buy consensus",
        )

        sell_trades = [t for t in trades if t.action == "SELL"]
        assert sell_trades
        assert all(t.fees > 0 for t in sell_trades)

    def test_all_trades_have_reasoning(self) -> None:
        """Project rule: every trade must have a reasoning field."""
        prices = {"AAPL": 150.0}
        lookup = _price_lookup(prices)
        portfolio = {"cash": 2000.0, "positions": []}
        target = ["AAPL"]
        reason = "baseline-manager monthly rebalance: ≥2-agent buy consensus"

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=target,
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning=reason,
        )

        assert all(t.reasoning == reason for t in trades)

    def test_no_trades_when_no_target_and_no_positions(self) -> None:
        """Empty target + empty portfolio → no trades."""
        prices = {}
        lookup = _price_lookup(prices)
        portfolio = {"cash": 2000.0, "positions": []}

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=[],
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning="baseline-manager monthly rebalance: ≥2-agent buy consensus",
        )

        assert trades == []

    def test_already_held_position_not_bought_again(self) -> None:
        """Position already in the target set is not doubled up with another buy."""
        prices = {"AAPL": 150.0}
        lookup = _price_lookup(prices)
        # Already holds AAPL at roughly target sizing
        portfolio = {
            "cash": 1700.0,
            "positions": [{"ticker": "AAPL", "shares": 2.0, "avg_cost": 150.0}],
        }
        target = ["AAPL"]

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=target,
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning="baseline-manager monthly rebalance: ≥2-agent buy consensus",
        )

        # No sell of AAPL (it's in target), no re-buy of AAPL (already held)
        aapl_trades = [t for t in trades if t.ticker == "AAPL"]
        assert len(aapl_trades) == 0

    def test_trade_ids_are_unique(self) -> None:
        """All returned trade IDs are unique."""
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        lookup = _price_lookup(prices)
        portfolio = {"cash": 2000.0, "positions": []}
        target = ["AAPL", "MSFT"]

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=target,
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning="test",
        )

        ids = [t.id for t in trades]
        assert len(ids) == len(set(ids))

    def test_buy_notional_approximately_position_size(self) -> None:
        """BUY notional (total) is approximately position_size."""
        prices = {"AAPL": 150.0}
        lookup = _price_lookup(prices)
        portfolio = {"cash": 2000.0, "positions": []}
        target = ["AAPL"]

        trades = rebalance(
            portfolio=portfolio,
            target_tickers=target,
            price_lookup=lookup,
            on=date(2026, 6, 1),
            position_size=300.0,
            reasoning="test",
        )

        buy = next(t for t in trades if t.action == "BUY" and t.ticker == "AAPL")
        # Fractional shares guarantee: shares = position_size / price, total = shares * price
        # so total == position_size exactly (within float precision).
        assert buy.total == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# step_build_baseline_manager integration test — initialization path
# ---------------------------------------------------------------------------


class TestStepBuildBaselineManager:
    @pytest.mark.live_cast
    def test_initializes_portfolio_on_first_run(
        self, tmp_path: Path, monkeypatch, midas_data_root
    ) -> None:
        """step_build_baseline_manager creates portfolio dir with EUR 2000 on first call."""
        from engine.config import get_config

        # Write minimal OHLCV data for a ticker that will appear in notes
        ohlcv_dir = get_config().ohlcv_dir
        ohlcv_dir.mkdir(parents=True, exist_ok=True)
        (ohlcv_dir / "AAPL.jsonl").write_text(
            '{"date": "2026-06-01", "close": 150.0, "adj_close": 150.0}\n',
            encoding="utf-8",
        )

        # No store patch needed: step_build_baseline_manager reads
        # baseline_manager._OHLCV_STORE, which now lazily resolves to
        # get_config().ohlcv_dir — the same redirected dir seeded above.

        portfolios_dir = tmp_path / "portfolios"
        portfolios_dir.mkdir()

        # Build agent_results with enough research notes to trigger a rebalance signal
        agent_results = {
            "agent-a": {
                "research_note": {
                    "thesis": "Test thesis.",
                    "conviction": 8,
                    "tickers": ["AAPL"],
                    "action_bias": "strong_buy",
                    "horizon": "weeks",
                    "catalysts": "Test.",
                    "currency": "EUR",
                }
            },
            "agent-b": {
                "research_note": {
                    "thesis": "Test thesis.",
                    "conviction": 7,
                    "tickers": ["AAPL"],
                    "action_bias": "buy",
                    "horizon": "weeks",
                    "catalysts": "Test.",
                    "currency": "EUR",
                }
            },
        }

        from scripts.daily_session import step_build_baseline_manager

        # 2026-06-01 is the first weekday of June 2026 → rebalance day
        step_build_baseline_manager(
            agent_results,
            trade_date=date(2026, 6, 1),
            portfolios_dir=portfolios_dir,
            ohlcv_store=ohlcv_dir,
        )

        portfolio_json = portfolios_dir / "baseline-manager" / "portfolio.json"
        assert portfolio_json.exists(), (
            "baseline-manager portfolio.json was not created"
        )

        import json

        data = json.loads(portfolio_json.read_text())
        # Initial cash was 2000 EUR, rebalance may have spent some on AAPL
        assert data["currency"] == "EUR"
        assert data["cash"] <= 2000.0

    def test_no_op_on_non_rebalance_day(self, tmp_path: Path) -> None:
        """On non-rebalance days (portfolio already initialized), cash is unchanged."""
        import json

        from engine.portfolio import PortfolioManager
        from scripts.daily_session import step_build_baseline_manager

        portfolios_dir = tmp_path / "portfolios"
        portfolios_dir.mkdir()

        # Pre-initialize so it's not a first-run scenario
        manager = PortfolioManager(base_dir=portfolios_dir)
        manager.initialize("baseline-manager", initial_capital=2000.0, currency="EUR")
        original = json.loads(
            (portfolios_dir / "baseline-manager" / "portfolio.json").read_text()
        )

        agent_results: dict = {}
        # 2026-06-15 is mid-month — not a rebalance day, portfolio already exists
        step_build_baseline_manager(
            agent_results,
            trade_date=date(2026, 6, 15),
            portfolios_dir=portfolios_dir,
            ohlcv_store=tmp_path / "ohlcv",
        )

        current = json.loads(
            (portfolios_dir / "baseline-manager" / "portfolio.json").read_text()
        )
        assert current["cash"] == original["cash"]


# ---------------------------------------------------------------------------
# Public surface exclusion guard
# ---------------------------------------------------------------------------


class TestPublicExclusion:
    def test_baseline_manager_not_in_trading_roster(self) -> None:
        from engine.config import get_config

        assert "baseline-manager" not in get_config().trading_roster

    def test_baseline_manager_not_in_roster(self) -> None:
        from engine.config import get_config

        assert "baseline-manager" not in get_config().roster

    def test_build_portfolio_summaries_excludes_baseline_manager(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """build_portfolio_summaries keys off AGENT_POST_TIMES — baseline-manager never appears."""
        import scripts.daily_session as ds_module

        monkeypatch.setattr(ds_module, "_PROJECT_ROOT", tmp_path)

        portfolios_dir = tmp_path / "portfolios"
        portfolios_dir.mkdir()

        # Create both a real agent portfolio and the baseline-manager portfolio
        from engine.portfolio import PortfolioManager

        manager = PortfolioManager(base_dir=portfolios_dir)
        manager.initialize("steady-eddie-eur", initial_capital=10000.0, currency="EUR")
        manager.initialize("baseline-manager", initial_capital=2000.0, currency="EUR")

        from scripts.daily_session import build_portfolio_summaries

        summaries = build_portfolio_summaries()

        assert "baseline-manager" not in summaries
