"""Portfolio state manager — disk-backed portfolio with trade execution and snapshots.

Manages portfolio directories under a base directory, each identified by a strategy ID.

Directory layout:
    {base_dir}/{strategy_id}/
        portfolio.json   — current cash + positions
        trades.json      — full trade log (append-only)
        snapshots.json   — daily portfolio snapshots (append-only)
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from engine.types import Portfolio, Position, Trade


class PortfolioManager:
    """Manages portfolio state on disk for one or more strategies.

    Parameters
    ----------
    base_dir:
        Base directory that holds all portfolio subdirectories.
        Each strategy gets its own subdirectory: {base_dir}/{strategy_id}/
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _portfolio_dir(self, strategy_id: str) -> Path:
        return self._base_dir / strategy_id

    def _portfolio_path(self, strategy_id: str) -> Path:
        return self._portfolio_dir(strategy_id) / "portfolio.json"

    def _trades_path(self, strategy_id: str) -> Path:
        return self._portfolio_dir(strategy_id) / "trades.json"

    def _snapshots_path(self, strategy_id: str) -> Path:
        return self._portfolio_dir(strategy_id) / "snapshots.json"

    def _read_json(self, path: Path) -> list | dict:
        with path.open() as f:
            return json.load(f)

    def _write_json(self, path: Path, data: list | dict) -> None:
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    def _append_json_list(self, path: Path, item: dict) -> None:
        """Append a dict to a JSON file containing a list."""
        records: list[dict] = self._read_json(path)  # type: ignore[assignment]
        records.append(item)
        self._write_json(path, records)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        strategy_id: str,
        initial_capital: float,
        currency: str = "USD",
    ) -> None:
        """Create portfolio directory with empty portfolio, trades, and snapshots files.

        Does nothing if the portfolio directory already exists.
        """
        portfolio_dir = self._portfolio_dir(strategy_id)
        portfolio_dir.mkdir(parents=True, exist_ok=True)

        portfolio_path = self._portfolio_path(strategy_id)
        if not portfolio_path.exists():
            portfolio = Portfolio(
                cash=initial_capital,
                positions=[],
                last_updated=date.today(),
                currency=currency,
            )
            self._write_json(portfolio_path, portfolio.to_dict())

        trades_path = self._trades_path(strategy_id)
        if not trades_path.exists():
            self._write_json(trades_path, [])

        snapshots_path = self._snapshots_path(strategy_id)
        if not snapshots_path.exists():
            self._write_json(snapshots_path, [])

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, strategy_id: str) -> Portfolio:
        """Read and return the current portfolio from disk."""
        path = self._portfolio_path(strategy_id)
        data = self._read_json(path)
        return Portfolio.from_dict(data)  # type: ignore[arg-type]

    def load_trades(self, strategy_id: str) -> list[dict]:
        """Read and return the full trade log from disk."""
        return self._read_json(self._trades_path(strategy_id))  # type: ignore[return-value]

    def load_snapshots(self, strategy_id: str) -> list[dict]:
        """Read and return all daily snapshots from disk."""
        return self._read_json(self._snapshots_path(strategy_id))  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def apply_trade(self, strategy_id: str, trade: Trade) -> None:
        """Apply a trade to the portfolio and append it to the trade log.

        BUY:
            - Deduct total + fees from cash.
            - If the ticker already has an open position, update avg_cost
              using a weighted average (average-in semantics).
            - Otherwise, create a new position.

        SELL:
            - Add total - fees to cash.
            - Reduce shares from the existing position.
            - Remove the position entirely when shares reach zero.

        Raises
        ------
        ValueError
            If action is not "BUY" or "SELL", or if trying to sell more
            shares than currently held.
        """
        portfolio = self.load(strategy_id)

        action = trade.action.upper()

        if action == "BUY":
            cost = trade.total + trade.fees
            if cost > portfolio.cash:
                raise ValueError(
                    f"Insufficient cash for {trade.ticker}: "
                    f"trade costs ${cost:,.2f} but only ${portfolio.cash:,.2f} available"
                )
            portfolio.cash -= cost

            # Find existing position for this ticker.
            existing = next(
                (p for p in portfolio.positions if p.ticker == trade.ticker), None
            )
            if existing is not None:
                # Weighted average cost.
                total_shares = existing.shares + trade.shares
                existing.avg_cost = (
                    existing.shares * existing.avg_cost + trade.shares * trade.price
                ) / total_shares
                existing.shares = total_shares
            else:
                new_position = Position(
                    ticker=trade.ticker,
                    shares=trade.shares,
                    avg_cost=trade.price,
                    date_opened=trade.timestamp.date(),
                    grid_level=0,
                )
                portfolio.positions.append(new_position)

        elif action == "SELL":
            existing = next(
                (p for p in portfolio.positions if p.ticker == trade.ticker), None
            )
            if existing is None:
                raise ValueError(
                    f"Cannot sell {trade.ticker}: no open position for strategy {strategy_id!r}"
                )
            if trade.shares > existing.shares:
                raise ValueError(
                    f"Cannot sell {trade.shares} shares of {trade.ticker}: "
                    f"only {existing.shares} shares held"
                )

            portfolio.cash += trade.total - trade.fees

            existing.shares -= trade.shares
            if existing.shares == 0:
                portfolio.positions = [
                    p for p in portfolio.positions if p.ticker != trade.ticker
                ]

        else:
            raise ValueError(
                f"Invalid trade action: {trade.action!r}. Expected 'BUY' or 'SELL'."
            )

        portfolio.last_updated = trade.timestamp.date()

        # Persist updated portfolio.
        self._write_json(self._portfolio_path(strategy_id), portfolio.to_dict())

        # Append trade to log.
        trade_record = {
            "id": trade.id,
            "timestamp": trade.timestamp.isoformat(),
            "action": trade.action,
            "ticker": trade.ticker,
            "shares": trade.shares,
            "price": trade.price,
            "total": trade.total,
            "fees": trade.fees,
            "reasoning": trade.reasoning,
        }
        self._append_json_list(self._trades_path(strategy_id), trade_record)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def add_snapshot(
        self,
        strategy_id: str,
        snapshot_date: date,
        portfolio_value: float,
        cash: float,
        positions_value: float,
        benchmarks: dict,
        session_date: date | None = None,
    ) -> bool:
        """Append a daily snapshot to snapshots.json. History is immutable.

        The equity curve is keyed on ``snapshot_date`` — the *market* date the
        valuation was priced at — because that is the axis the baseline series
        lives on (``engine.baselines`` dates every point from the price series,
        so a session-dated agent curve would drift off its own benchmark).

        ``session_date`` records *when the row was observed*, which is a
        different thing and the reason this is not a plain upsert. A resumed or
        re-run session must be able to correct its own row; a *later* session
        must never rewrite an earlier one's. Without that distinction a Monday
        session whose OHLCV store has not advanced past Friday silently
        overwrites the weekend's already-published value with a composition that
        includes trades made days afterwards.

        The write is therefore append-or-refuse: an existing row is replaced
        only when it carries the *same* ``session_date``. Anything else — an
        older session, a newer one, or a legacy row predating this field — is
        refused rather than clobbered, so the failure is loud and history stands.

        Parameters
        ----------
        strategy_id:
            Target portfolio.
        snapshot_date:
            Market date the valuation was priced at. Keys the row.
        portfolio_value:
            Total portfolio value (cash + positions_value).
        cash:
            Cash component.
        positions_value:
            Market value of all open positions.
        benchmarks:
            Dict of benchmark values, e.g. {"sp500": 5200.0, "btc": 65000.0}.
        session_date:
            The session that observed this valuation. Defaults to
            ``snapshot_date``, which is the honest reading for a backfill
            replaying history one market day at a time.

        Returns
        -------
        bool
            True if the snapshot was written, False if it was refused because
            an earlier session already owns that market date. Callers should
            surface a refusal rather than swallow it.
        """
        observed = session_date or snapshot_date
        snapshot = {
            "date": snapshot_date.isoformat(),
            "session_date": observed.isoformat(),
            "portfolio_value": portfolio_value,
            "cash": cash,
            "positions_value": positions_value,
            "benchmarks": benchmarks,
        }
        path = self._snapshots_path(strategy_id)
        records: list[dict] = self._read_json(path)  # type: ignore[assignment]
        date_key = snapshot["date"]
        for i, existing in enumerate(records):
            if existing.get("date") != date_key:
                continue
            # Same session correcting itself is the one legitimate overwrite.
            # A missing session_date is a legacy row: never equal, so refused.
            if existing.get("session_date") != snapshot["session_date"]:
                return False
            records[i] = snapshot
            self._write_json(path, records)
            return True
        records.append(snapshot)
        self._write_json(path, records)
        return True
