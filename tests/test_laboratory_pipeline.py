"""End-to-end test for the Ring 1 laboratory pipeline.

Drives the full daily flow: author orders → paper broker fill (with all 9 rails
exercised) → build post+oracle prompts → parse mock responses → save content.
Asserts every rejection code appears, output artifacts are on disk, and retry
is idempotent.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine import orders as orders_module
from engine.blog import BlogDraft, parse_oracle_response
from engine.config import get_config
from engine.output_bundle import get_day_number, save_output_bundle
from engine.paper_broker import fill_day
from engine.portfolio import PortfolioManager
from engine.posts import PostPayload, parse_post_response
from scripts.daily_session import (
    step_author_orders,
    step_build_oracle_prompt,
    step_build_post_prompts,
    step_fill_orders,
    step_save_content,
)

pytestmark = pytest.mark.live_cast


TRADE_DATE = date(2026, 4, 17)


@pytest.fixture
def lab_env(midas_data_root: Path, monkeypatch):
    """Wire every module's writable path to a tmp directory so nothing leaks."""
    cfg = get_config()
    ohlcv = cfg.ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)
    config_dir = cfg.agent_config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    ticker_ccy = cfg.ticker_currencies_path
    ticker_ccy.write_text(json.dumps({"MC.PA": "EUR"}), encoding="utf-8")
    outbox = cfg.orders_dir / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    inbox = cfg.orders_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    pm_base = midas_data_root / "portfolios"
    pm_base.mkdir()
    posts_dir = cfg.posts_dir
    blog_dir = cfg.blog_dir
    output_dir = cfg.output_dir

    monkeypatch.setattr("engine.quotes._TICKER_CURRENCY_OVERRIDES", None)

    # Seed OHLCV: yesterday's close present (simulates cron-before-OHLCV-refresh).
    def seed_ohlcv(ticker: str, price: float, on_date: date) -> None:
        (ohlcv / f"{ticker}.jsonl").write_text(
            json.dumps({"date": on_date.isoformat(), "close": price}) + "\n",
            encoding="utf-8",
        )

    # Prices in each ticker's natural currency.
    seed_ohlcv("MC.PA", 700.0, date(2026, 4, 16))  # EUR via override
    seed_ohlcv("BTC-EUR", 50000.0, date(2026, 4, 16))
    seed_ohlcv("MSFT", 400.0, date(2026, 4, 16))

    # Seed per-agent safety rails via roster.yaml (Task 4: no more agent_config/*.json).
    import yaml
    from engine.config import reset_config_cache

    roster_path = cfg.data_dir / "roster.yaml"
    roster_data = yaml.safe_load(roster_path.read_text(encoding="utf-8"))

    _base_safety = {
        "max_order_notional": 1000.0,
        "max_orders_per_day": 5,
        "daily_drawdown_halt_pct": -5.0,
        "allowed_universe": [],
        "dry_run": False,
    }
    for agent_id in ["satoshi", "funded-buyer", "steady-eddie-eur", "yolo-sapiens-usd"]:
        if agent_id not in roster_data["agents"]:
            roster_data["agents"][agent_id] = {
                "display_name": agent_id,
                "role": "trader",
            }
        roster_data["agents"][agent_id]["safety"] = dict(_base_safety)

    # Dedicated agent with drawdown halt triggered.
    roster_data["agents"]["halted-agent"] = {
        "display_name": "halted-agent",
        "role": "trader",
        "safety": dict(_base_safety),
    }

    # Dedicated agent with universe restriction.
    roster_data["agents"]["gated-agent"] = {
        "display_name": "gated-agent",
        "role": "trader",
        "safety": {**_base_safety, "allowed_universe": ["single-voo"]},
    }

    roster_path.write_text(yaml.dump(roster_data), encoding="utf-8")
    reset_config_cache()

    pm = PortfolioManager(base_dir=pm_base)
    pm.initialize("satoshi", 10000.0, currency="EUR")
    pm.initialize("funded-buyer", 10000.0, currency="EUR")
    pm.initialize("steady-eddie-eur", 10000.0, currency="EUR")
    pm.initialize("yolo-sapiens-usd", 10000.0, currency="USD")
    pm.initialize("halted-agent", 10000.0, currency="EUR")
    pm.initialize("gated-agent", 10000.0, currency="EUR")

    # Seed a big prior snapshot for halted-agent so current MTM (10k cash) is a massive drawdown.
    pm.add_snapshot(
        strategy_id="halted-agent",
        snapshot_date=date(2026, 4, 16),
        portfolio_value=100000.0,  # current cash = 10k → drawdown = -90%
        cash=100000.0,
        positions_value=0.0,
        benchmarks={},
    )

    return {
        "pm": pm,
        "ohlcv": ohlcv,
        "outbox": outbox,
        "inbox": inbox,
        "posts_dir": posts_dir,
        "blog_dir": blog_dir,
        "output_dir": output_dir,
        "config_dir": config_dir,
    }


class TestLaboratoryPipeline:
    def test_full_pipeline_end_to_end_with_every_rail(self, lab_env, monkeypatch):
        pm = lab_env["pm"]

        # ---- Author orders (covers many rails) ----

        # Valid BUY → fills (dedicated agent so INSUFFICIENT_CASH scenario doesn't conflict)
        step_author_orders(
            "funded-buyer",
            [
                {
                    "action": "BUY",
                    "ticker": "BTC-EUR",
                    "shares": 0.01,
                    "reasoning": "Dip",
                },
            ],
            TRADE_DATE,
            currency="EUR",
        )

        # MAX_ORDER_NOTIONAL: 10 * 700 = 7000 > 1000 cap
        step_author_orders(
            "steady-eddie-eur",
            [
                {
                    "action": "BUY",
                    "ticker": "MC.PA",
                    "shares": 10,
                    "reasoning": "Luxury overweight",
                },
            ],
            TRADE_DATE,
            currency="EUR",
        )

        # TICKER_NOT_IN_UNIVERSE: MSFT isn't in single-voo
        step_author_orders(
            "gated-agent",
            [
                {
                    "action": "BUY",
                    "ticker": "MSFT",
                    "shares": 1,
                    "reasoning": "Out of scope",
                },
            ],
            TRADE_DATE,
            currency="EUR",
        )

        # NO_POSITION_TO_SELL: long-only enforcement via rail
        step_author_orders(
            "yolo-sapiens-usd",
            [
                {
                    "action": "SELL",
                    "ticker": "MSFT",
                    "shares": 1,
                    "reasoning": "Short-attempt guard",
                },
            ],
            TRADE_DATE,
            currency="USD",
        )

        # DAILY_DRAWDOWN_HALT
        step_author_orders(
            "halted-agent",
            [
                {
                    "action": "BUY",
                    "ticker": "BTC-EUR",
                    "shares": 0.001,
                    "reasoning": "Will halt",
                },
            ],
            TRADE_DATE,
            currency="EUR",
        )

        # Hand-write a malformed outbox line to exercise INVALID_SHARES
        outbox_file = lab_env["outbox"] / f"{TRADE_DATE.isoformat()}.jsonl"
        with outbox_file.open("a", encoding="utf-8") as f:
            f.write(
                '{"order_id": "malformed_1", "agent_id": "satoshi", "action": "BUY", "shares": 0, "ticker": "BTC-EUR", "currency": "EUR", "ts": "2026-04-17T20:00:00Z", "reasoning": "zero"}\n'
            )
            f.write('{"not valid json\n')  # corrupt

        # NO_PRICE_DATA: author a valid order for a ticker with no store row
        step_author_orders(
            "satoshi",
            [
                {
                    "action": "BUY",
                    "ticker": "NONEXISTENT-EUR",
                    "shares": 0.01,
                    "reasoning": "No price",
                },
            ],
            TRADE_DATE,
            currency="EUR",
        )

        # INSUFFICIENT_CASH: bump the agent's cash low, then submit an expensive buy.
        satoshi_json = pm._portfolio_path("satoshi")
        d = json.loads(satoshi_json.read_text(encoding="utf-8"))
        d["cash"] = 10.0
        satoshi_json.write_text(json.dumps(d), encoding="utf-8")
        step_author_orders(
            "satoshi",
            [
                {
                    "action": "BUY",
                    "ticker": "MC.PA",
                    "shares": 1,
                    "reasoning": "No money",
                },
            ],
            TRADE_DATE,
            currency="EUR",
        )

        # MAX_ORDERS_PER_DAY: submit 5 (cap) + 1 more for yolo. First we need valid cheap trades.
        for i in range(6):
            step_author_orders(
                "yolo-sapiens-usd",
                [
                    {
                        "action": "BUY",
                        "ticker": "MSFT",
                        "shares": 0.5,
                        "reasoning": f"order {i}",
                    },
                ],
                TRADE_DATE,
                currency="USD",
            )

        # APPLY_TRADE_FAILED: monkeypatch apply_trade to raise for one specific order.
        # We'll use fail-agent with a trade that WOULD pass all rails otherwise.
        # Seed a SELL that will pass pre-checks (give the agent a position first).
        pm.initialize("fail-agent", 1000.0, currency="EUR")
        # Seed fail-agent safety rails via roster.yaml (Task 4: no more agent_config/*.json).
        import yaml as _yaml
        from engine.config import get_config as _gc, reset_config_cache as _rcc

        _roster_path = _gc().data_dir / "roster.yaml"
        _roster_data = _yaml.safe_load(_roster_path.read_text(encoding="utf-8"))
        _roster_data["agents"]["fail-agent"] = {
            "display_name": "fail-agent",
            "role": "trader",
            "safety": {
                "max_order_notional": 1000.0,
                "max_orders_per_day": 5,
                "daily_drawdown_halt_pct": -5.0,
                "allowed_universe": [],
                "dry_run": False,
            },
        }
        _roster_path.write_text(_yaml.dump(_roster_data), encoding="utf-8")
        _rcc()

        # Give fail-agent a tiny BTC-EUR position. Shares chosen so that:
        # - APPLY_TRADE_FAILED order: 0.00001 shares (notional=0.5 EUR < 1000 cap, ≤ held)
        # - INSUFFICIENT_SHARES order: 0.001 shares (notional=50 EUR < 1000 cap, > held=0.0001)
        fa_json = pm._portfolio_path("fail-agent")
        fa_state = json.loads(fa_json.read_text(encoding="utf-8"))
        fa_state["positions"] = [
            {
                "ticker": "BTC-EUR",
                "shares": 0.0001,
                "avg_cost": 50000.0,
                "date_opened": "2026-04-16",
                "grid_level": 0,
            }
        ]
        fa_json.write_text(json.dumps(fa_state), encoding="utf-8")

        # Now monkeypatch apply_trade to raise on fail-agent only
        original_apply = PortfolioManager.apply_trade

        def flaky_apply(self, strategy_id, trade):
            if strategy_id == "fail-agent":
                raise ValueError("simulated broker race")
            return original_apply(self, strategy_id, trade)

        monkeypatch.setattr(PortfolioManager, "apply_trade", flaky_apply)

        # APPLY_TRADE_FAILED: 0.00001 ≤ 0.0001 held, notional=0.5 EUR < 1000 cap → passes all
        # pre-checks, then apply_trade raises.
        step_author_orders(
            "fail-agent",
            [
                {
                    "action": "SELL",
                    "ticker": "BTC-EUR",
                    "shares": 0.00001,
                    "reasoning": "Race-induced fail",
                },
            ],
            TRADE_DATE,
            currency="EUR",
        )

        # INSUFFICIENT_SHARES: 0.001 > 0.0001 held, notional=50 EUR < 1000 cap → caught before apply.
        step_author_orders(
            "fail-agent",
            [
                {
                    "action": "SELL",
                    "ticker": "BTC-EUR",
                    "shares": 0.001,
                    "reasoning": "Too many",
                },
            ],
            TRADE_DATE,
            currency="EUR",
        )

        # ---- Fill ----
        fills = step_fill_orders(TRADE_DATE, pm)
        reasons = {f.reason for f in fills if f.status == "rejected"}

        # Assert every expected rejection code is present
        expected_reasons = {
            "INVALID_SHARES",
            "MAX_ORDER_NOTIONAL",
            "MAX_ORDERS_PER_DAY",
            "TICKER_NOT_IN_UNIVERSE",
            "NO_PRICE_DATA",
            "INSUFFICIENT_CASH",
            "NO_POSITION_TO_SELL",
            "INSUFFICIENT_SHARES",
            "DAILY_DRAWDOWN_HALT",
            "APPLY_TRADE_FAILED",
        }
        missing = expected_reasons - reasons
        assert not missing, f"Rejection codes not exercised: {missing}. Got: {reasons}"

        # Assert at least one filled fill occurred (the valid BTC-EUR buy)
        filled_ids = [f.order_id for f in fills if f.status == "filled"]
        assert len(filled_ids) > 0, "No successful fills — pipeline is all-reject"

        # ---- Build & parse post prompts ----
        agent_results = {
            "satoshi": {"commentary": "Loading the dip.", "trades": []},
            "steady-eddie-eur": {"commentary": "Holding.", "trades": []},
            "yolo-sapiens-usd": {"commentary": "YOLO.", "trades": []},
        }
        prompts = step_build_post_prompts(agent_results)
        assert set(prompts.keys()) == {
            "satoshi",
            "steady-eddie-eur",
            "yolo-sapiens-usd",
        }
        assert "Loading the dip." in prompts["satoshi"]

        # Parse mock post responses
        agent_posts: dict[str, list[PostPayload]] = {}
        for aid in agent_results:
            mock_resp = json.dumps(
                [{"text": f"{aid} posted.", "mentions": [], "kind": "trade"}]
            )
            agent_posts[aid] = parse_post_response(aid, mock_resp)

        # ---- Build & parse Oracle prompt ----
        leaderboard = [
            {"agent": "satoshi", "return_pct": 1.2, "rank": 1},
            {"agent": "steady-eddie-eur", "return_pct": 0.3, "rank": 2},
            {"agent": "yolo-sapiens-usd", "return_pct": -0.8, "rank": 3},
        ]
        market_data = {"sp500": 7000.0, "eur_usd": 1.1784}
        oracle_prompt = step_build_oracle_prompt(
            market_data=market_data,
            agent_results=agent_results,
            agent_posts={
                aid: [p.to_dict() for p in posts] for aid, posts in agent_posts.items()
            },
            leaderboard=leaderboard,
        )
        assert "Day 1" in oracle_prompt
        assert "Satoshi" in oracle_prompt

        mock_oracle_resp = json.dumps(
            {
                "blog_draft": {
                    "title": "Day 1: Opening",
                    "body_md": "# Day 1",
                    "slug": "day-1",
                },
                "posts": [
                    {"text": "Scoreboard.", "mentions": [], "kind": "scoreboard"}
                ],
            }
        )
        blog_draft, oracle_posts = parse_oracle_response(mock_oracle_resp)

        # ---- Idempotency check: get_day_number BEFORE save ----
        day_before = get_day_number(for_date=TRADE_DATE)

        # ---- Save content ----
        portfolio_summaries = {
            aid: {
                "cash": pm.load(aid).cash,
                "positions": [p.ticker for p in pm.load(aid).positions],
            }
            for aid in agent_results
        }
        bundle = step_save_content(
            bundle_date=TRADE_DATE,
            market_data=market_data,
            agent_results=agent_results,
            agent_posts=agent_posts,
            portfolio_summaries=portfolio_summaries,
            leaderboard=leaderboard,
            blog_draft=blog_draft,
            oracle_posts=oracle_posts,
        )

        # ---- Idempotency check: get_day_number AFTER save must be unchanged ----
        day_after = get_day_number(for_date=TRADE_DATE)
        assert day_before == day_after, (
            f"Day number changed: {day_before} → {day_after}"
        )

        # Call save again (retry scenario)
        step_save_content(
            bundle_date=TRADE_DATE,
            market_data=market_data,
            agent_results=agent_results,
            agent_posts=agent_posts,
            portfolio_summaries=portfolio_summaries,
            leaderboard=leaderboard,
            blog_draft=blog_draft,
            oracle_posts=oracle_posts,
        )
        day_after_retry = get_day_number(for_date=TRADE_DATE)
        assert day_after_retry == day_before, (
            f"Day number changed on retry: {day_before} → {day_after_retry}"
        )

        # ---- Artifact assertions ----
        assert (lab_env["posts_dir"] / f"{TRADE_DATE.isoformat()}.json").exists()
        assert (lab_env["blog_dir"] / f"{TRADE_DATE.isoformat()}.md").exists()
        bundle_path = lab_env["output_dir"] / f"{TRADE_DATE.isoformat()}.json"
        assert bundle_path.exists()

        loaded = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert loaded["date"] == TRADE_DATE.isoformat()
        assert loaded["market_snapshot"] == market_data
        # Bundle always carries the full 10-agent roster (cadence-invariant).
        # Running agents in this test: 3. Non-runners get null commentary.
        from engine.output_bundle import ROSTER

        assert set(loaded["agents"].keys()) == set(ROSTER)
        running = {"satoshi", "steady-eddie-eur", "yolo-sapiens-usd"}
        for aid in running:
            assert loaded["agents"][aid]["commentary"] is not None
        for aid in set(ROSTER) - running:
            assert loaded["agents"][aid]["commentary"] is None
            assert loaded["agents"][aid]["trades"] == []
            assert loaded["agents"][aid]["posts"] == []
        assert loaded["narrator"]["blog_draft"]["title"] == "Day 1: Opening"
        assert len(loaded["narrator"]["posts"]) == 1
        assert loaded["leaderboard"] == leaderboard
