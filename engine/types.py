"""Core data types for the Midas trading system."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import ClassVar

# ---------------------------------------------------------------------------
# Valid value sets for StrategySpec validation
# ---------------------------------------------------------------------------

VALID_UNIVERSES: frozenset[str] = frozenset(
    {
        "dow30",
        "sp500",
        "nasdaq100",
        "dividend-aristocrats",
        "congress",
        "insiders",
        "13f-whales",
        "high-short",
        "etf-sectors",
        "etf-broad",
        "crypto-top20",
        "crypto-top20-eur",
        "forex-majors",
        "metals-commodities",
        "commodities-eur",
        "single-voo",
        "classic-60-40",
        "bearish-etfs",
        "bearish-etfs-ucits",
        # EU indices
        "cac40",
        "dax",
        "ftse100",
        "stoxx-600",
    }
)

VALID_SELECTORS: frozenset[str] = frozenset(
    {
        "golden-cross",
        "rsi-oversold",
        "dip-entry",
        "earnings-beat",
        "sector-cycle",
        "fear-greed",
        "data-follow",
        "claude-analysis",
        "random",
        "buy-and-hold",
    }
)

VALID_MANAGERS: frozenset[str] = frozenset(
    {
        "equal-weight",
        "grid-conservative",
        "grid-aggressive",
        "scaled-exit",
        "trailing-stop",
        "time-boxed",
        "rebalance-monthly",
        "volatility-sized",
        "fixed-60-40",
    }
)


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """A single executed trade."""

    id: str
    timestamp: datetime
    action: str  # "BUY" | "SELL"
    ticker: str
    shares: float
    price: float
    total: float
    fees: float
    reasoning: str


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """An open position in a portfolio."""

    ticker: str
    shares: float
    avg_cost: float
    date_opened: date
    grid_level: int

    def __post_init__(self) -> None:
        if self.shares <= 0:
            raise ValueError(f"shares must be positive, got {self.shares}")

    @property
    def cost_basis(self) -> float:
        """Total cost basis: shares × avg_cost."""
        return self.shares * self.avg_cost


# ---------------------------------------------------------------------------
# BenchmarkValues
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkValues:
    """Benchmark index/asset values for a given day."""

    sp500: float
    msci_world: float
    gold: float
    btc: float


# ---------------------------------------------------------------------------
# DailySnapshot
# ---------------------------------------------------------------------------


@dataclass
class DailySnapshot:
    """End-of-day snapshot of portfolio state relative to benchmarks."""

    date: date
    portfolio_value: float
    cash: float
    positions_value: float
    benchmarks: BenchmarkValues


# ---------------------------------------------------------------------------
# FundingConfig
# ---------------------------------------------------------------------------


@dataclass
class FundingConfig:
    """Capital funding configuration for a strategy.

    NOTE: only ``initial`` is consumed by the bt backtest engine. The DCA
    fields ``monthly_addition`` / ``weekly_addition`` are parsed and preserved
    for forward compatibility but are NOT yet wired into bt — periodic
    contributions do not affect backtest results today.
    """

    initial: float = 10_000.0
    monthly_addition: float = 0.0  # UNUSED by the backtest engine (parsed only)
    weekly_addition: float = 0.0  # UNUSED by the backtest engine (parsed only)


# ---------------------------------------------------------------------------
# StrategyRules
# ---------------------------------------------------------------------------


@dataclass
class StrategyRules:
    """Risk and position management rules for a strategy.

    NOTE: ``max_positions`` and ``max_position_pct`` are enforced in the bt
    pipeline (SelectN / LimitWeights). ``min_hold_days`` is parsed and exposed
    in the backtester API form but is NOT enforced by the bt engine today.
    """

    max_positions: int = 10
    max_position_pct: float = 20.0
    min_hold_days: int = 3  # UNUSED by the backtest engine (parsed only)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


@dataclass
class Portfolio:
    """Live portfolio state: cash + open positions + base currency."""

    cash: float
    positions: list[Position]
    last_updated: date
    currency: str = "USD"  # ISO 4217 code; legacy portfolios default to USD

    @property
    def cost_basis(self) -> float:
        """Sum of cost basis across all open positions."""
        return sum(p.cost_basis for p in self.positions)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "cash": self.cash,
            "currency": self.currency,
            "last_updated": self.last_updated.isoformat(),
            "positions": [
                {
                    "ticker": p.ticker,
                    "shares": p.shares,
                    "avg_cost": p.avg_cost,
                    "date_opened": p.date_opened.isoformat(),
                    "grid_level": p.grid_level,
                }
                for p in self.positions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        """Deserialize from a dictionary (inverse of to_dict)."""
        positions = [
            Position(
                ticker=p["ticker"],
                shares=p["shares"],
                avg_cost=p["avg_cost"],
                date_opened=date.fromisoformat(p["date_opened"]),
                grid_level=p["grid_level"],
            )
            for p in data.get("positions", [])
        ]
        return cls(
            cash=data["cash"],
            positions=positions,
            last_updated=date.fromisoformat(data["last_updated"]),
            currency=data.get("currency", "USD"),
        )


# ---------------------------------------------------------------------------
# StrategySpec
# ---------------------------------------------------------------------------


@dataclass
class StrategySpec:
    """Full specification for a trading strategy."""

    id: str
    name: str
    universe: str
    selector: str
    manager: str
    funding: FundingConfig
    # "reinvest" | "cash" — parsed and preserved, but the bt backtest engine
    # does not currently model dividend handling. UNUSED downstream today.
    dividends: str
    rules: StrategyRules

    @classmethod
    def from_dict(cls, data: dict) -> "StrategySpec":
        """Construct and validate a StrategySpec from a plain dictionary."""
        universe = data["universe"]
        selector = data["selector"]
        manager = data["manager"]

        if universe not in VALID_UNIVERSES:
            raise ValueError(
                f"Invalid universe {universe!r}. Valid options: {sorted(VALID_UNIVERSES)}"
            )
        if selector not in VALID_SELECTORS:
            raise ValueError(
                f"Invalid selector {selector!r}. Valid options: {sorted(VALID_SELECTORS)}"
            )
        if manager not in VALID_MANAGERS:
            raise ValueError(
                f"Invalid manager {manager!r}. Valid options: {sorted(VALID_MANAGERS)}"
            )

        funding_data = data.get("funding", {})
        funding = FundingConfig(
            initial=funding_data.get("initial", FundingConfig.initial),
            monthly_addition=funding_data.get(
                "monthly_addition", FundingConfig.monthly_addition
            ),
            weekly_addition=funding_data.get(
                "weekly_addition", FundingConfig.weekly_addition
            ),
        )

        rules_data = data.get("rules", {})
        rules = StrategyRules(
            max_positions=rules_data.get("max_positions", StrategyRules.max_positions),
            max_position_pct=rules_data.get(
                "max_position_pct", StrategyRules.max_position_pct
            ),
            min_hold_days=rules_data.get("min_hold_days", StrategyRules.min_hold_days),
        )

        return cls(
            id=data["id"],
            name=data["name"],
            universe=universe,
            selector=selector,
            manager=manager,
            funding=funding,
            dividends=data.get("dividends", "reinvest"),
            rules=rules,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "StrategySpec":
        """Load and parse a strategy spec from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
