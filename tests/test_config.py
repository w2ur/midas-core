"""Tests for engine.config — the runtime configuration singleton."""

from datetime import date

import pytest

from engine.config import (
    AgentSpec,
    get_config,
    reset_config_cache,
    resolve_agent_universe,
)
from engine.universes import resolve_universe


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_config_cache()
    yield
    reset_config_cache()


class TestConfig:
    def test_default_data_dir_is_repo_root(self, monkeypatch):
        monkeypatch.delenv("MIDAS_DATA_DIR", raising=False)
        reset_config_cache()
        cfg = get_config()
        # repo root is two levels up from engine/config.py
        assert (cfg.data_dir / "engine" / "config.py").exists()

    def test_env_overrides_data_dir(self, monkeypatch, tmp_path):
        (tmp_path / "roster.yaml").write_text(_MINIMAL_ROSTER, encoding="utf-8")
        monkeypatch.setenv("MIDAS_DATA_DIR", str(tmp_path))
        reset_config_cache()
        cfg = get_config()
        assert cfg.data_dir == tmp_path.resolve()
        assert cfg.posts_dir == tmp_path.resolve() / "data" / "posts"

    def test_roster_parses_globals_and_agent(self, monkeypatch, tmp_path):
        (tmp_path / "roster.yaml").write_text(_MINIMAL_ROSTER, encoding="utf-8")
        monkeypatch.setenv("MIDAS_DATA_DIR", str(tmp_path))
        reset_config_cache()
        cfg = get_config()
        assert cfg.day_one == date(2026, 4, 17)
        assert cfg.initial_capital == 10000.0
        assert cfg.global_reference.ticker == "URTH"
        assert "demo-one" in cfg.roster
        spec = cfg.roster["demo-one"]
        assert spec.display_name == "Demo One"
        assert spec.benchmark.ticker == "SPY"
        assert spec.safety.daily_drawdown_halt_pct == -5.0
        assert cfg.trading_roster == ("demo-one",)

    def test_paths_redirect_under_env(self, monkeypatch, tmp_path):
        (tmp_path / "roster.yaml").write_text(_MINIMAL_ROSTER, encoding="utf-8")
        monkeypatch.setenv("MIDAS_DATA_DIR", str(tmp_path))
        reset_config_cache()
        root = tmp_path.resolve()
        cfg = get_config()
        assert cfg.ohlcv_dir == root / "data" / "market" / "ohlcv"
        assert cfg.journal_dir == root / "data" / "agent_memory"
        assert cfg.agents_dir == root / ".claude" / "agents"

        # Regression guard for the whole laziness property: real engine functions
        # must WRITE under the redirected root. If any module froze its path at
        # import time (the bug this task fixed), these would land in the legacy
        # tree and the assertions would fail.
        import engine.agent_memory as agent_memory
        import engine.posts as posts

        jpath = agent_memory.save_journal("demo-one", "redirect proof")
        assert jpath == root / "data" / "agent_memory" / "demo-one.md"
        assert jpath.read_text(encoding="utf-8").startswith("redirect proof")

        ppath = posts.save_daily_posts(date(2026, 4, 17), {"demo-one": []})
        assert ppath == root / "data" / "posts" / "2026-04-17.json"
        assert ppath.exists()

    def test_reset_clears_registered_module_caches_across_env_switch(
        self, monkeypatch, tmp_path
    ):
        """reset_config_cache() must fire registered callbacks so module-level
        caches keyed on config paths (e.g. paper_broker's ticker-currency
        override map) are invalidated across a MIDAS_DATA_DIR switch."""
        import engine.paper_broker as pb  # registers its reset callback on import

        def _make_root(name: str, overrides: str) -> object:
            r = tmp_path / name
            (r / "data").mkdir(parents=True)
            (r / "roster.yaml").write_text(_MINIMAL_ROSTER, encoding="utf-8")
            (r / "data" / "ticker_currencies.json").write_text(
                overrides, encoding="utf-8"
            )
            return r

        root1 = _make_root("r1", '{"FOO": "USD"}')
        monkeypatch.setenv("MIDAS_DATA_DIR", str(root1))
        reset_config_cache()
        assert pb._load_ticker_currency_overrides() == {"FOO": "USD"}

        root2 = _make_root("r2", '{"BAR": "EUR"}')
        monkeypatch.setenv("MIDAS_DATA_DIR", str(root2))
        reset_config_cache()  # fires the callback that nils pb._TICKER_CURRENCY_OVERRIDES
        assert pb._load_ticker_currency_overrides() == {"BAR": "EUR"}

    def test_resolve_agent_universe_accepts_bare_string(self):
        # A single universe name given as a bare str (not a list) resolves in
        # native order — same as a one-element list.
        spec = AgentSpec(
            id="x",
            display_name="X",
            voice="",
            post_time="",
            home_currency="USD",
            initial_capital=10000.0,
            max_positions=5,
            universe="sp500",
            benchmark=None,
            persona="x.md",
        )
        assert resolve_agent_universe(spec) == resolve_universe("sp500")

    def test_resolve_agent_universe_literal_tickers(self):
        # Inline-ticker fallback: if universe items are NOT registered universe
        # names, they are treated as literal ticker symbols and returned as-is.
        # This lets forkers write universe: [SPY, QQQ, IWM] without registering
        # a universe name. William's live agents always use registered names and
        # never hit this path.
        spec = AgentSpec(
            id="y",
            display_name="Y",
            voice="",
            post_time="",
            home_currency="USD",
            initial_capital=10000.0,
            max_positions=5,
            universe=["SPY", "QQQ"],
            benchmark=None,
            persona="y.md",
        )
        assert resolve_agent_universe(spec) == ["SPY", "QQQ"]


_MINIMAL_ROSTER = """
globals:
  day_one: 2026-04-17
  currencies: [EUR, USD]
  initial_capital: 10000
  global_reference: { label: "MSCI World", ticker: URTH, currency: EUR }
  agents_dir: .claude/agents
agents:
  demo-one:
    display_name: "Demo One"
    voice: "Test voice."
    post_time: "09:00"
    home_currency: USD
    initial_capital: 10000
    max_positions: 5
    universe: [SPY, QQQ]
    benchmark: { label: "S&P 500", ticker: SPY, currency: USD }
    persona: demo-one.md
    role: trader
    safety: { max_order_notional: 500, max_orders_per_day: 100, daily_drawdown_halt_pct: -5, allowed_universe: [], dry_run: false }
"""
