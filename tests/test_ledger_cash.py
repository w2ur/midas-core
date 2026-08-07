"""Cash-replay reconciliation: does the trade ledger add up to the cash on disk?

The sibling `test_ledger_integrity.py` asks an *existence* question — every
filled inbox row has a matching trade, and vice versa. This file asks the
*arithmetic* one:

    initial_capital + (net cash effect of replaying every trade) == live cash

Both can fail independently. The existence check would not notice a trade
booked at the wrong notional, which is exactly what the 2026-08-07
quote-currency defect did to 24 fills: the rows were all present and joined
cleanly, and the cash was wrong by EUR 2,057.65 across four books.

Why this belongs in CI rather than in a script: the check already existed —
`scripts/restate_valuations._final_reconciliation_diff` is the hard gate that
restatement runs before it will touch anything. But it only ran when a human
ran a restatement, which over the experiment's life meant a handful of times.
The nightly attestation that *did* run computed a digest and asserted nothing
about it, so a ledger that stopped adding up produced a green run and a
different hash.

The arithmetic is imported, not reimplemented. A second copy of a
reconciliation is a second thing to drift.
"""

from __future__ import annotations

import inspect
import json

import pytest

from engine.config import get_config
from engine.portfolio import PortfolioManager
from scripts.restate_valuations import (
    _CASH_TOLERANCE,
    _final_reconciliation_diff,
    _initial_capital,
)


def _real_books() -> list[str]:
    """Books backed by a real order/fill flow.

    `baseline-manager` is excluded for the same reason `test_ledger_integrity`
    excludes it: it is a synthetic passive book written by the baselines step,
    never by `apply_trade`, so it has no ledger to reconcile.
    """
    cfg = get_config()
    return sorted(set(cfg.trading_roster) | set(cfg.allocators))


@pytest.mark.live_cast
class TestLiveLedgerReconciles:
    """Reconcile every committed book. Cast-coupled: reads the live ledger.

    midas-core ships no `data/portfolios/`, so on the demo desk there is
    nothing here to reconcile and the marker skips the class — while
    `TestReconciliationCanFail` below stays hermetic and runs everywhere,
    which is what keeps the arithmetic itself under test in both repos.
    """

    def test_every_book_cash_matches_its_replayed_ledger(self):
        manager = PortfolioManager(get_config().portfolios_dir)
        books = _real_books()
        assert books, "no books found — this test would pass vacuously"

        diverged = []
        for agent_id in books:
            trades = manager.load_trades(agent_id)
            snapshots = manager.load_snapshots(agent_id)
            initial = _initial_capital(agent_id, trades, snapshots)
            diff = _final_reconciliation_diff(agent_id, trades, initial, manager)
            if abs(diff) > _CASH_TOLERANCE:
                diverged.append((agent_id, round(diff, 4)))

        assert diverged == [], (
            "these books' cash does not equal initial capital plus their own "
            f"replayed trade ledger (agent, diff): {diverged}"
        )


class TestReconciliationCanFail:
    """The check must produce the opposite answer on a broken ledger.

    Without this the test above is a check that has never failed, which this
    project's own rule says is not evidence. Hermetic — no live data.
    """

    @staticmethod
    def _seed(tmp_path, cash: float, trades: list[dict]) -> PortfolioManager:
        manager = PortfolioManager(tmp_path)
        manager.initialize("book", initial_capital=1_000.0, currency="EUR")
        portfolio = manager.load("book")
        portfolio.cash = cash
        (tmp_path / "book" / "portfolio.json").write_text(
            json.dumps(portfolio.to_dict()), encoding="utf-8"
        )
        (tmp_path / "book" / "trades.json").write_text(
            json.dumps(trades), encoding="utf-8"
        )
        return manager

    _TRADE = {
        "id": "t1",
        "timestamp": "2026-04-17T20:00:00+00:00",
        "action": "BUY",
        "ticker": "AAPL",
        "shares": 1.0,
        "price": 100.0,
        "total": 100.0,
        "fees": 1.0,
        "reasoning": "seed",
    }

    def test_a_consistent_ledger_reconciles(self, tmp_path):
        # 1000 - (100 + 1) = 899
        manager = self._seed(tmp_path, cash=899.0, trades=[self._TRADE])
        diff = _final_reconciliation_diff(
            "book", manager.load_trades("book"), 1_000.0, manager
        )
        assert abs(diff) < _CASH_TOLERANCE

    def test_cash_that_does_not_match_the_ledger_is_caught(self, tmp_path):
        """The shape the quote-currency defect took: right rows, wrong amount."""
        manager = self._seed(tmp_path, cash=899.0, trades=[self._TRADE])
        doctored = dict(self._TRADE, total=10_000.0)
        diff = _final_reconciliation_diff("book", [doctored], 1_000.0, manager)
        assert abs(diff) > _CASH_TOLERANCE

    def test_a_trade_missing_from_the_ledger_is_caught(self, tmp_path):
        """And the shape the lost-fill defect took: cash moved, row vanished."""
        manager = self._seed(tmp_path, cash=899.0, trades=[self._TRADE])
        diff = _final_reconciliation_diff("book", [], 1_000.0, manager)
        assert abs(diff) > _CASH_TOLERANCE

    def test_the_tolerance_is_tight_enough_to_matter(self, tmp_path):
        """A one-cent slip passes; a one-euro slip does not.

        Pinning both sides stops the tolerance being quietly widened until
        the check stops meaning anything.
        """
        manager = self._seed(tmp_path, cash=899.0 + 0.005, trades=[self._TRADE])
        assert (
            abs(
                _final_reconciliation_diff(
                    "book", manager.load_trades("book"), 1_000.0, manager
                )
            )
            < _CASH_TOLERANCE
        )

        manager = self._seed(tmp_path, cash=898.0, trades=[self._TRADE])
        assert (
            abs(
                _final_reconciliation_diff(
                    "book", manager.load_trades("book"), 1_000.0, manager
                )
            )
            > _CASH_TOLERANCE
        )


def test_replay_is_evaluated_as_of_today_not_a_snapshot_date():
    """Guard the one assumption the book-level gate rests on.

    `_final_reconciliation_diff` replays with `as_of=date.today()`, which is
    what makes it unambiguous — a date-bounded per-row compare produces false
    positives from session-timing artifacts in legacy rows. If that ever
    became a bounded replay, this file would start passing for the wrong
    reason.
    """
    source = inspect.getsource(_final_reconciliation_diff)
    assert "date.today()" in source
