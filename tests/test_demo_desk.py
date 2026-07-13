"""Proof of reusability: a different cast flows through the deterministic pipeline.

This test copies examples/demo-desk into a tmp_path, sets MIDAS_DATA_DIR to
the tmp copy, and runs the deterministic Hands pipeline (baselines) on the
demo cast. It proves that:

  1. A forker-supplied roster.yaml with literal-ticker universes loads cleanly.
  2. resolve_agent_universe falls back to literal tickers (not registered names).
  3. build_all_baselines writes per-agent files under the redirected data dir.
  4. No artefact from William's cast (e.g. data/baselines/satoshi) is produced.

Note: run-session dispatches LLM persona subagents and is NOT headless-testable.
This test exercises only the deterministic pipeline.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from engine.config import get_config, reset_config_cache, resolve_agent_universe

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo-desk"


@pytest.fixture
def demo_home(tmp_path, monkeypatch):
    """Copy the demo-desk example into tmp_path and redirect MIDAS_DATA_DIR."""
    home = tmp_path / "demo"
    shutil.copytree(DEMO, home)

    ohlcv_dir = home / "data" / "market" / "ohlcv"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)

    # Seed two trading days of SPY closes — needed for benchmark.json to be
    # non-empty. SPY is also demo-momentum's benchmark ticker.
    (ohlcv_dir / "SPY.jsonl").write_text(
        '{"date": "2026-04-17", "close": 500.0, "adj_close": 500.0}\n'
        '{"date": "2026-04-18", "close": 505.0, "adj_close": 505.0}\n',
        encoding="utf-8",
    )
    # Seed QQQ and IWM so the coin-flip for demo-momentum has price data.
    (ohlcv_dir / "QQQ.jsonl").write_text(
        '{"date": "2026-04-17", "close": 430.0, "adj_close": 430.0}\n'
        '{"date": "2026-04-18", "close": 435.0, "adj_close": 435.0}\n',
        encoding="utf-8",
    )
    (ohlcv_dir / "IWM.jsonl").write_text(
        '{"date": "2026-04-17", "close": 195.0, "adj_close": 195.0}\n'
        '{"date": "2026-04-18", "close": 197.0, "adj_close": 197.0}\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("MIDAS_DATA_DIR", str(home))
    reset_config_cache()
    yield home
    reset_config_cache()


def test_demo_roster_loads(demo_home):
    """The demo roster is read cleanly and the correct roles are assigned."""
    cfg = get_config()
    assert cfg.trading_roster == ("demo-momentum", "demo-value")
    assert "demo-oracle" in cfg.roster
    assert cfg.roster["demo-oracle"].role == "narrator"


def test_demo_universe_resolves_as_literal_tickers(demo_home):
    """resolve_agent_universe returns literal tickers when they are not registry names."""
    cfg = get_config()
    universe = resolve_agent_universe(cfg.roster["demo-momentum"])
    assert universe == ["SPY", "QQQ", "IWM"]

    universe_value = resolve_agent_universe(cfg.roster["demo-value"])
    assert universe_value == ["VTV", "SCHD", "BRK-B"]


def test_demo_baselines_build_for_demo_cast(demo_home):
    """build_all_baselines writes per-agent files under the demo data dir.

    Uses the canonical path to resolve universes — exercises the inline-ticker
    fallback introduced in this task. Asserts that:
    - demo-momentum/benchmark.json exists and is non-empty with currency=="USD"
    - demo-momentum/coinflip.json exists
    - No satoshi/ dir is created (proves only the demo cast ran, not William's)
    """
    from engine.baselines import build_all_baselines

    cfg = get_config()

    # Canonical path: resolve universes via resolve_agent_universe (exercises
    # the inline-ticker fallback for demo-momentum and demo-value).
    universes = {
        aid: resolve_agent_universe(cfg.roster[aid]) for aid in cfg.trading_roster
    }

    build_all_baselines(
        universes,
        cfg.day_one,
        date(2026, 4, 18),
        {a: cfg.roster[a].max_positions for a in cfg.trading_roster},
    )

    bench = demo_home / "data" / "baselines" / "demo-momentum" / "benchmark.json"
    assert bench.exists(), "benchmark.json was not written for demo-momentum"
    series = json.loads(bench.read_text(encoding="utf-8"))
    assert series, "benchmark.json is empty — SPY OHLCV was not found"
    assert series[0]["currency"] == "USD"

    coinflip = demo_home / "data" / "baselines" / "demo-momentum" / "coinflip.json"
    assert coinflip.exists(), "coinflip.json was not written for demo-momentum"

    # The baselines ran on the demo cast, not William's — satoshi must not exist.
    assert not (demo_home / "data" / "baselines" / "satoshi").exists(), (
        "satoshi baseline dir found — build_all_baselines must have used "
        "the default (William's) config instead of the demo one"
    )
