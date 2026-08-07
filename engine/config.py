"""Runtime configuration — single source of truth for paths, globals, and the roster.

Replaces the former module-level ``Path(__file__).resolve().parents[1]`` data-root
assumptions (which break once the package is pip-installed) and the hardcoded cast
dicts scattered across engine.posts / engine.baselines / scripts.backfill_baselines.

The project root is resolved from ``MIDAS_DATA_DIR`` (default: the repo root two
levels up — legacy behaviour). The cast + globals load from ``<root>/roster.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

_LEGACY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkSpec:
    label: str
    ticker: str
    currency: str


@dataclass(frozen=True)
class SafetyRails:
    """Per-agent limits the broker enforces. The persona is aspirational; this binds.

    ``max_order_notional_pct`` expresses the per-order cap as a percentage of
    the book's **current** value rather than as an absolute amount, and takes
    precedence over ``max_order_notional`` when set. A fixed cap has to be
    chosen against a book size, and it silently stops binding as soon as the
    book moves away from that size — the ten traders ran from day one with
    `1_000_000` against €10,000 books, a cap 100x larger than the entire
    portfolio, which is a rail that cannot fire. A percentage re-scales with
    the book and needs no maintenance.

    The absolute field stays for forks and for the allocator, whose cap is
    deliberately a fixed small number.
    """

    max_order_notional: float = 500.0
    max_order_notional_pct: float | None = None
    max_orders_per_day: int = 5
    daily_drawdown_halt_pct: float = -5.0
    allowed_universe: tuple[str, ...] = ()
    dry_run: bool = False


@dataclass(frozen=True)
class RiskBudget:
    max_positions: int = 6
    per_position_cap: float = 400.0
    cash_floor: float = 150.0
    max_trades_per_week: int = 2
    min_conviction: int = 6


@dataclass(frozen=True)
class AllocatorSpec:
    channels_prefix: str = "manager"
    outcome_resolution_days: int = 10
    outcome_memory_same_max: int = 5
    outcome_memory_other_max: int = 3
    baseline_enabled: bool = True
    risk_budget: RiskBudget = field(default_factory=RiskBudget)
    blocklist: tuple[str, ...] = ()
    policy_prose_override: str | None = None


@dataclass(frozen=True)
class JurisdictionSpec:
    tax_rate_pct: float = 0.0  # core no-op default; FR sets 30.0
    fees: dict = field(default_factory=dict)  # empty => fees.py module defaults


@dataclass(frozen=True)
class AgentSpec:
    id: str
    display_name: str
    voice: str
    post_time: str
    home_currency: str
    initial_capital: float
    max_positions: int
    universe: str | list[str] | None  # universe name, list of names, or unset
    benchmark: BenchmarkSpec | None
    persona: str
    role: str = "trader"
    safety: SafetyRails = field(default_factory=SafetyRails)
    allocator: "AllocatorSpec | None" = None


@dataclass(frozen=True)
class MidasConfig:
    data_dir: Path
    day_one: date
    currencies: tuple[str, ...]
    initial_capital: float
    global_reference: BenchmarkSpec
    agents_dir: Path
    roster: dict[str, AgentSpec]
    jurisdiction: JurisdictionSpec

    @property
    def _data(self) -> Path:
        return self.data_dir / "data"

    @property
    def posts_dir(self) -> Path:
        return self._data / "posts"

    @property
    def baselines_dir(self) -> Path:
        return self._data / "baselines"

    @property
    def ohlcv_dir(self) -> Path:
        return self._data / "market" / "ohlcv"

    @property
    def journal_dir(self) -> Path:
        return self._data / "agent_memory"

    @property
    def logs_dir(self) -> Path:
        return self._data / "logs"

    @property
    def blog_dir(self) -> Path:
        return self._data / "blog"

    @property
    def output_dir(self) -> Path:
        return self._data / "output"

    @property
    def universes_dir(self) -> Path:
        return self._data / "universes"

    @property
    def agent_config_dir(self) -> Path:
        return self._data / "agent_config"

    @property
    def orders_dir(self) -> Path:
        return self._data / "orders"

    @property
    def tickers_path(self) -> Path:
        return self._data / "tickers.json"

    @property
    def ticker_currencies_path(self) -> Path:
        return self._data / "ticker_currencies.json"

    @property
    def portfolios_dir(self) -> Path:
        return self._data / "portfolios"

    @property
    def leaderboard_dir(self) -> Path:
        return self._data / "leaderboard"

    @property
    def tax_shadow_dir(self) -> Path:
        return self._data / "tax_shadow"

    @property
    def session_state_dir(self) -> Path:
        return self._data / "session_state"

    @property
    def trading_roster(self) -> tuple[str, ...]:
        return tuple(aid for aid, spec in self.roster.items() if spec.role == "trader")

    @property
    def allocators(self) -> tuple[str, ...]:
        return tuple(
            aid for aid, spec in self.roster.items() if spec.role == "allocator"
        )

    def allocator_spec(self, agent_id: str) -> "AllocatorSpec":
        spec = self.roster[agent_id].allocator
        if spec is None:
            raise ValueError(f"{agent_id!r} has no allocator config")
        return spec


def _benchmark(raw: dict | None) -> BenchmarkSpec | None:
    if not raw:
        return None
    return BenchmarkSpec(
        label=raw["label"], ticker=raw["ticker"], currency=raw["currency"]
    )


def _safety(raw: dict | None) -> SafetyRails:
    raw = raw or {}
    pct = raw.get("max_order_notional_pct")
    return SafetyRails(
        max_order_notional=float(raw.get("max_order_notional", 500.0)),
        max_order_notional_pct=None if pct is None else float(pct),
        max_orders_per_day=int(raw.get("max_orders_per_day", 5)),
        daily_drawdown_halt_pct=float(raw.get("daily_drawdown_halt_pct", -5.0)),
        allowed_universe=tuple(raw.get("allowed_universe", []) or []),
        dry_run=bool(raw.get("dry_run", False)),
    )


def _risk_budget(raw: dict | None) -> RiskBudget:
    raw = raw or {}
    return RiskBudget(
        max_positions=int(raw.get("max_positions", 6)),
        per_position_cap=float(raw.get("per_position_cap", 400.0)),
        cash_floor=float(raw.get("cash_floor", 150.0)),
        max_trades_per_week=int(raw.get("max_trades_per_week", 2)),
        min_conviction=int(raw.get("min_conviction", 6)),
    )


def _allocator(raw: dict | None) -> AllocatorSpec | None:
    if not raw:
        return None
    policy = raw.get("policy") or {}
    mem = raw.get("outcome_memory") or {}
    baseline = raw.get("baseline") or {}
    return AllocatorSpec(
        channels_prefix=str(raw.get("channels_prefix", "manager")),
        outcome_resolution_days=int(raw.get("outcome_resolution_days", 10)),
        outcome_memory_same_max=int(mem.get("same_ticker_max", 5)),
        outcome_memory_other_max=int(mem.get("other_ticker_max", 3)),
        baseline_enabled=bool(baseline.get("enabled", True)),
        risk_budget=_risk_budget(raw.get("risk_budget")),
        blocklist=tuple(policy.get("blocklist", []) or []),
        policy_prose_override=policy.get("prose_override"),
    )


def _jurisdiction(raw: dict | None) -> JurisdictionSpec:
    raw = raw or {}
    return JurisdictionSpec(
        tax_rate_pct=float(raw.get("tax_rate_pct", 0.0)),
        fees=dict(raw.get("fees") or {}),
    )


def _agent(agent_id: str, raw: dict, default_capital: float) -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        display_name=raw["display_name"],
        voice=raw.get("voice", ""),
        post_time=raw.get("post_time", ""),
        home_currency=raw.get("home_currency", "EUR"),
        initial_capital=float(raw.get("initial_capital", default_capital)),
        max_positions=int(raw.get("max_positions", 5)),
        universe=raw.get("universe"),
        benchmark=_benchmark(raw.get("benchmark")),
        persona=raw.get("persona", f"{agent_id}.md"),
        role=raw.get("role", "trader"),
        safety=_safety(raw.get("safety")),
        allocator=_allocator(raw.get("allocator")),
    )


def _resolve_data_dir() -> Path:
    env = os.environ.get("MIDAS_DATA_DIR")
    return Path(env).expanduser().resolve() if env else _LEGACY_ROOT


def _load(data_dir: Path) -> MidasConfig:
    raw = yaml.safe_load((data_dir / "roster.yaml").read_text(encoding="utf-8"))
    g = raw["globals"]
    day_one = g["day_one"]
    if not isinstance(day_one, date):
        day_one = date.fromisoformat(str(day_one))
    default_capital = float(g.get("initial_capital", 10000.0))
    roster = {aid: _agent(aid, a, default_capital) for aid, a in raw["agents"].items()}
    return MidasConfig(
        data_dir=data_dir,
        day_one=day_one,
        currencies=tuple(g.get("currencies", ["EUR", "USD"])),
        initial_capital=default_capital,
        global_reference=_benchmark(g["global_reference"]),
        agents_dir=data_dir / g.get("agents_dir", ".claude/agents"),
        roster=roster,
        jurisdiction=_jurisdiction(g.get("jurisdiction")),
    )


@lru_cache(maxsize=1)
def get_config() -> MidasConfig:
    return _load(_resolve_data_dir())


# Modules that derive their own caches from config paths (e.g. paper_broker's
# ticker-currency override map) register a callback here so they are invalidated
# in lockstep with the config cache. Without this, a MIDAS_DATA_DIR switch +
# reset_config_cache() would leave those module caches pinned to the old tree.
_RESET_CALLBACKS: list = []


def register_reset_callback(fn) -> None:
    """Register a zero-arg callback invoked after every reset_config_cache()."""
    _RESET_CALLBACKS.append(fn)


def reset_config_cache() -> None:
    """Clear the cached config — for tests that change MIDAS_DATA_DIR between cases.

    Also fires every registered reset callback so module-level caches keyed on
    config paths are invalidated together.
    """
    get_config.cache_clear()
    for fn in _RESET_CALLBACKS:
        fn()


def resolve_agent_universe(spec: AgentSpec) -> list[str]:
    """Resolve an agent's universe (a name or list of names) to a ticker list.

    ``spec.universe`` is a single universe name (str) or a list of names
    (list[str]), each resolved via engine.universes.resolve_universe. The
    composition rules replicate today's behaviour in scripts.backfill_baselines
    verbatim:
      - empty / None       -> []
      - exactly one name   -> resolve_universe(name) (native order, NOT sorted)
      - two or more names  -> sorted({t for n in names for t in resolve_universe(n)})

    Inline-ticker fallback: if ANY name raises KeyError (i.e. it is not a
    registered universe name), the whole list is treated as literal ticker
    symbols and returned as-is. This lets forkers supply e.g.
    ``universe: [SPY, QQQ, IWM]`` without registering a universe name.
    William's live agents use only registered names, so they never hit this
    path.
    """
    names = spec.universe
    if not names:
        return []
    if isinstance(names, str):
        names = [names]
    from engine.universes import resolve_universe

    try:
        resolved = [resolve_universe(n) for n in names]
    except KeyError:
        return list(names)  # not registry names → treat as literal tickers
    if len(names) == 1:
        return resolved[0]  # single registered name → native order (unchanged)
    return sorted({t for lst in resolved for t in lst})  # multi → sorted union
