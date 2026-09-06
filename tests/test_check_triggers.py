"""Tests for scripts.check_triggers — the conditional-order watcher."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.config import get_config

from engine.orders import Order, read_inbox
from engine.portfolio import PortfolioManager
from engine.triggers import list_pending, save_pending


# ---------------------------------------------------------------------------
# Fixtures (replicated from test_paper_broker — cross-file fixture sharing
# via conftest is the alternative; inline keeps this test file self-contained).
# ---------------------------------------------------------------------------


@pytest.fixture
def broker_env(midas_data_root, monkeypatch):
    cfg = get_config()
    ohlcv = cfg.ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)
    config_dir = cfg.agent_config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    ticker_ccy_path = cfg.ticker_currencies_path
    outbox = cfg.orders_dir / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    inbox = cfg.orders_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    pm_base = midas_data_root / "portfolios"
    pm_base.mkdir()
    pending_dir = cfg.orders_dir / "pending"
    cancels_dir = cfg.orders_dir / "cancels"
    manager_inbox = cfg.orders_dir / "manager-inbox"
    manager_pending = cfg.orders_dir / "manager-pending"
    monkeypatch.setattr("engine.quotes._TICKER_CURRENCY_OVERRIDES", None)
    return {
        "ohlcv": ohlcv,
        "config_dir": config_dir,
        "ticker_ccy": ticker_ccy_path,
        "outbox": outbox,
        "inbox": inbox,
        "pm_base": pm_base,
        "pending": pending_dir,
        "cancels": cancels_dir,
        "manager_inbox": manager_inbox,
        "manager_pending": manager_pending,
    }


def _write_config(config_dir: Path, agent_id: str, **overrides) -> None:
    """Seed per-agent safety rails via roster.yaml in the tmp MIDAS_DATA_DIR.

    As of Task 4, AgentConfig.load() reads from get_config().roster.
    ``config_dir`` is kept in the signature for call-site compatibility.
    """
    import yaml
    from engine.config import get_config, reset_config_cache

    safety = {
        "max_order_notional": 10_000.0,
        "max_orders_per_day": 10,
        "daily_drawdown_halt_pct": -50.0,
        "allowed_universe": [],
        "dry_run": False,
    }
    safety.update(overrides)
    cfg = get_config()
    roster_path = cfg.data_dir / "roster.yaml"
    data = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
    if agent_id not in data["agents"]:
        data["agents"][agent_id] = {"display_name": agent_id, "role": "trader"}
    data["agents"][agent_id]["safety"] = safety
    roster_path.write_text(yaml.dump(data), encoding="utf-8")
    reset_config_cache()


def _init_portfolio(
    pm_base: Path, agent_id: str, cash: float = 10_000.0, currency: str = "EUR"
) -> PortfolioManager:
    pm = PortfolioManager(pm_base)
    pm.initialize(agent_id, initial_capital=cash, currency=currency)
    return pm


def _seed_pending(broker_env, **overrides) -> Order:
    defaults = dict(
        order_id="ord_2026-05-10_satoshi_003",
        ts=datetime(2026, 5, 10, 20, 2, tzinfo=timezone.utc),
        agent_id="satoshi",
        action="SELL",
        ticker="BTC-EUR",
        shares=0.01,
        reasoning="trim at 85k",
        currency="EUR",
        trigger={"op": ">=", "level": 85000.0},
        expires="2026-06-10",
    )
    defaults.update(overrides)
    o = Order(**defaults)
    save_pending(o)
    return o


# ---------------------------------------------------------------------------
# Blackout window
# ---------------------------------------------------------------------------


class TestBlackoutWindow:
    @pytest.mark.parametrize(
        "hh,mm",
        [
            (19, 55),
            (20, 0),
            (20, 15),
            (20, 30),
            # Extended 20:30 → 21:00 on 2026-08-07: the measured session tail
            # (auto-merge as late as 20:45) fell outside the old window, and a
            # fire in the tail discards the whole session via StaleSessionError.
            (20, 31),
            (20, 45),
            (21, 0),
        ],
    )
    def test_blackout_skips_processing(self, broker_env, hh, mm) -> None:
        from scripts import check_triggers

        _seed_pending(broker_env)
        fake_now = datetime(2026, 5, 17, hh, mm, tzinfo=timezone.utc)
        result = check_triggers.run(now=fake_now, portfolio_manager=None)
        assert result["blacked_out"] is True
        assert len(list_pending()) == 1  # untouched

    # 21:01 is the first minute outside the window (21:31 while the session
    # ran at 20:30 UTC, 2026-08-10..11). Both edges are pinned, so a blackout
    # that tracks the session start cannot silently become an all-day one.
    @pytest.mark.parametrize("hh,mm", [(19, 54), (21, 1), (3, 0), (14, 30)])
    def test_normal_hours_do_run(self, broker_env, monkeypatch, hh, mm) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        # Force trigger NOT to fire so the run is a no-op besides the blackout check.
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 70000.0)
        fake_now = datetime(2026, 5, 17, hh, mm, tzinfo=timezone.utc)
        result = check_triggers.run(now=fake_now, portfolio_manager=pm)
        assert result["blacked_out"] is False


# ---------------------------------------------------------------------------
# Trigger fire / no-fire / unavailable price
# ---------------------------------------------------------------------------


class TestTriggerFire:
    def test_fire_when_price_meets_trigger(self, broker_env, monkeypatch) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        o = _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        # Seed a position so the SELL can be filled (rails check NO_POSITION_TO_SELL).
        # Cash must cover the seed BUY (0.1 BTC × 70000 = 7000 EUR).
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=8000.0, currency="EUR"
        )
        from engine.types import Trade

        pm.apply_trade(
            "satoshi",
            Trade(
                id="seed_001",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                action="BUY",
                ticker="BTC-EUR",
                shares=0.1,
                price=70000.0,
                total=7000.0,
                fees=0.0,
                reasoning="seed",
            ),
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert list_pending() == []
        fills = read_inbox(fake_now.date())
        triggered = [f for f in fills if f.order_id == o.order_id]
        assert len(triggered) == 1
        assert triggered[0].status == "filled"
        assert triggered[0].trigger_fired is True
        assert triggered[0].fill_price == 85123.45

    def test_no_fire_when_price_doesnt_meet_trigger(
        self, broker_env, monkeypatch
    ) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 80000.0)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert len(list_pending()) == 1
        assert read_inbox(fake_now.date()) == []

    def test_price_unavailable_carries_forward(self, broker_env, monkeypatch) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: None)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert len(list_pending()) == 1
        assert read_inbox(fake_now.date()) == []


# ---------------------------------------------------------------------------
# Expiry takes precedence over firing
# ---------------------------------------------------------------------------


class TestExpiry:
    def test_expired_order_removed_and_rejection_logged(
        self, broker_env, monkeypatch
    ) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        o = _seed_pending(broker_env, expires="2026-04-01")  # already expired
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )  # would fire if not expired
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert list_pending() == []
        fills = read_inbox(fake_now.date())
        expired = [f for f in fills if f.order_id == o.order_id]
        assert len(expired) == 1
        assert expired[0].status == "rejected"
        assert expired[0].reason == "TRIGGER_EXPIRED"
        assert expired[0].trigger_fired is True


# ---------------------------------------------------------------------------
# Safety rails apply at fire time, not declaration time
# ---------------------------------------------------------------------------


class TestBrokerRailsApplyOnFire:
    def test_insufficient_cash_at_fire_time_logged_as_rejection(
        self, broker_env, monkeypatch
    ) -> None:
        """The agent's portfolio has 0 cash when the trigger fires — rejection, not fill."""
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(broker_env, action="BUY", trigger={"op": "<=", "level": 90000.0})
        _write_config(broker_env["config_dir"], "satoshi")
        # Initialize with 0 cash so a BUY will fail INSUFFICIENT_CASH.
        pm = _init_portfolio(broker_env["pm_base"], "satoshi", cash=0.0, currency="EUR")
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 80000.0)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert list_pending() == []  # pending removed regardless of outcome
        fills = read_inbox(fake_now.date())
        assert any(
            f.status == "rejected" and f.reason == "INSUFFICIENT_CASH" for f in fills
        )


# ---------------------------------------------------------------------------
# Manager channel isolation: the watcher fires Manager pending orders into the
# Manager's OWN inbox, never the public one the site joins by order_id.
# ---------------------------------------------------------------------------


class TestManagerChannelIsolation:
    @pytest.mark.live_cast
    def test_manager_pending_fires_into_manager_inbox(
        self, broker_env, monkeypatch
    ) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod
        from engine.triggers import MANAGER_PENDING_DIR, save_pending

        # A Manager conditional BUY whose trigger is already satisfied at fire price.
        order = Order(
            order_id="ord_2026-05-10_the-manager_001",
            ts=datetime(2026, 5, 10, 20, 2, tzinfo=timezone.utc),
            agent_id="the-manager",
            action="BUY",
            ticker="BTC-EUR",
            shares=0.01,
            reasoning="accumulate on dip",
            currency="EUR",
            trigger={"op": "<=", "level": 90000.0},
            expires="2026-06-10",
        )
        save_pending(order, pending_dir=MANAGER_PENDING_DIR)

        _write_config(broker_env["config_dir"], "the-manager")
        pm = _init_portfolio(
            broker_env["pm_base"], "the-manager", cash=10_000.0, currency="EUR"
        )
        cash_before = pm.load("the-manager").cash

        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 80000.0)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        # (a) Filled fill lands in the MANAGER inbox.
        manager_fills = read_inbox(
            fake_now.date(), inbox_dir=broker_env["manager_inbox"]
        )
        triggered = [f for f in manager_fills if f.order_id == order.order_id]
        assert len(triggered) == 1
        assert triggered[0].status == "filled"
        assert triggered[0].trigger_fired is True
        assert triggered[0].fill_price == 80000.0

        # (b) NOTHING leaks into the public inbox.
        assert read_inbox(fake_now.date()) == []

        # (c) the-manager book mutated: cash down, position opened.
        book = pm.load("the-manager")
        assert book.cash < cash_before
        assert any(p.ticker == "BTC-EUR" and p.shares > 0 for p in book.positions)

        # (d) Manager pending file deleted; public pending untouched (empty).
        assert list_pending(pending_dir=MANAGER_PENDING_DIR) == []
        assert list_pending() == []

    def test_public_channel_unchanged(self, broker_env, monkeypatch) -> None:
        """Regression guard: a public pending order still fires into the public inbox."""
        from scripts import check_triggers
        from engine import triggers as triggers_mod
        from engine.triggers import MANAGER_PENDING_DIR

        o = _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=8000.0, currency="EUR"
        )
        from engine.types import Trade

        pm.apply_trade(
            "satoshi",
            Trade(
                id="seed_001",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                action="BUY",
                ticker="BTC-EUR",
                shares=0.1,
                price=70000.0,
                total=7000.0,
                fees=0.0,
                reasoning="seed",
            ),
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        # Public fill lands in the public inbox exactly as before.
        fills = read_inbox(fake_now.date())
        triggered = [f for f in fills if f.order_id == o.order_id]
        assert len(triggered) == 1
        assert triggered[0].status == "filled"
        assert triggered[0].fill_price == 85123.45
        assert list_pending() == []
        # Manager inbox stays empty for a public fire.
        assert read_inbox(fake_now.date(), inbox_dir=broker_env["manager_inbox"]) == []
        assert list_pending(pending_dir=MANAGER_PENDING_DIR) == []


# ---------------------------------------------------------------------------
# crypto-only cadence split
# ---------------------------------------------------------------------------


class TestCryptoOnly:
    """Pins the --crypto-only sweep introduced when the watcher cadence split.

    Context: check-triggers ran */15 24/7 on the belief (stated in the old
    workflow comment) that midas was public and Actions minutes free. It was
    private at the time — private from creation until 2026-08-19 — and that one
    workflow burned 923 of 2000 monthly minutes. (The repo is public now, so
    minutes are free and the cadence decision is worth revisiting; the split
    below is unchanged.) The
    split gives crypto an hourly 24/7 pass and everything else a daily one,
    because non-crypto prices only move once a day when fetch-ohlcv lands.
    """

    def test_crypto_only_skips_non_crypto(self, broker_env, monkeypatch) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(broker_env)  # BTC-EUR
        _seed_pending(
            broker_env,
            order_id="ord_2026-05-10_satoshi_004",
            ticker="AAPL",
            currency="USD",
        )
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(broker_env["pm_base"], "satoshi", cash=10_000.0)
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 1.0)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)

        result = check_triggers.run(
            now=fake_now, portfolio_manager=pm, crypto_only=True
        )
        assert result["checked"] == 1  # only BTC-EUR

        # Default stays the full sweep — the flag must be strictly opt-in.
        result_all = check_triggers.run(now=fake_now, portfolio_manager=pm)
        assert result_all["checked"] == 2

    def test_crypto_only_does_not_expire_non_crypto(
        self, broker_env, monkeypatch
    ) -> None:
        """The safety property: an hourly crypto pass must not retire an equity.

        Expiry is date-based, so had filtering been done inside _process_channel
        (after the is_expired branch) the hourly crypto job would have expired
        every stale non-crypto order and written TRIGGER_EXPIRED rejections into
        the ledger 24x a day. Filtering happens in run() precisely to stop that.
        """
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(
            broker_env,
            order_id="ord_2026-05-10_satoshi_005",
            ticker="AAPL",
            currency="USD",
            expires="2026-05-11",  # long past relative to fake_now
        )
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(broker_env["pm_base"], "satoshi", cash=10_000.0)
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 1.0)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)

        result = check_triggers.run(
            now=fake_now, portfolio_manager=pm, crypto_only=True
        )
        assert result["expired"] == 0
        assert len(list_pending()) == 1  # still pending, untouched
        assert read_inbox(fake_now.date()) == []  # no rejection written

        # The daily full sweep remains the sole owner of expiry.
        result_all = check_triggers.run(now=fake_now, portfolio_manager=pm)
        assert result_all["expired"] == 1
        assert list_pending() == []


# ---------------------------------------------------------------------------
# A failed push must turn the run red
# ---------------------------------------------------------------------------


class TestFailedPushExitsNonZero:
    """A watcher push that fails is a fact nobody was told.

    Both push sites logged a warning and returned, inside a run that exited 0.
    The self-healing story is real — the fill is local-only, but the pending
    file still exists on origin and inbox-scoped idempotency means the next
    evaluation legitimately re-fires it — so what was missing is not recovery,
    it is VISIBILITY. With the watchers now wired to `.github/actions/
    failure-issue`, a non-zero exit is what reaches a human.

    Semantics deliberately preserved, and asserted below: every order is still
    evaluated and every push still attempted before the process exits; a
    blackout run and a no-op run both stay green.
    """

    def _fake_git(
        self, monkeypatch, *, push_rc: int, is_ancestor_rc: int | None = None
    ):
        """Intercept EVERY git call the watcher makes.

        Not just push: the fixtures live under a tmp MIDAS_DATA_DIR, so a real
        `git add` would either fail with "outside repository" or — worse, if
        the paths ever did resolve — stage something in the actual checkout.
        The watcher's own control flow is left intact, which is the point:
        `diff --cached --quiet` answers 1 (there ARE staged changes) so the run
        reaches commit and push exactly as it would in CI.
        """
        import subprocess as sp
        import types

        from scripts import check_triggers

        seen: list[list[str]] = []

        def fake(cmd, *a, **kw):
            assert isinstance(cmd, (list, tuple)) and cmd[0] == "git", cmd
            seen.append(list(cmd))
            if cmd[1] == "diff":
                return sp.CompletedProcess(cmd, 1)  # something is staged
            if cmd[1] in {"push", "pull"}:
                return sp.CompletedProcess(cmd, push_rc)
            if cmd[1] == "merge-base":
                # Models reality rather than answering 0 to everything: HEAD is
                # an ancestor of origin/main exactly when a push landed. The
                # default keeps the two in step; pass is_ancestor_rc to model
                # the self-healed case, where an earlier failure's commit was
                # carried to main by a later order's successful push.
                rc = push_rc if is_ancestor_rc is None else is_ancestor_rc
                return sp.CompletedProcess(cmd, rc)
            return sp.CompletedProcess(cmd, 0)  # add, commit, fetch, rebase

        # Replace the module's OWN `subprocess` reference, not `subprocess.run`
        # itself: the latter is global and would also intercept
        # paper_broker._current_commit_sha, which shells out to `git rev-parse`
        # and would then stamp every fill with a None SHA.
        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(
                run=fake,
                CompletedProcess=sp.CompletedProcess,
                CalledProcessError=sp.CalledProcessError,
            ),
        )
        return seen

    def _always_fail_push(self, monkeypatch):
        return self._fake_git(monkeypatch, push_rc=1)

    def _fireable(self, broker_env, monkeypatch):
        from datetime import datetime as dt

        from engine import triggers as triggers_mod
        from engine.types import Trade

        order = _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=8000.0, currency="EUR"
        )
        pm.apply_trade(
            "satoshi",
            Trade(
                id="seed_001",
                timestamp=dt(2026, 5, 1, tzinfo=timezone.utc),
                action="BUY",
                ticker="BTC-EUR",
                shares=0.1,
                price=70000.0,
                total=7000.0,
                fees=0.0,
                reasoning="seed",
            ),
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        return order, pm

    def _run_main(self, monkeypatch, pm, now, argv=("check_triggers.py",)):
        from datetime import datetime as dt

        from scripts import check_triggers

        monkeypatch.setattr(check_triggers.sys, "argv", list(argv))
        monkeypatch.setattr(check_triggers, "PortfolioManager", lambda base_dir: pm)

        class _Now(dt):
            @classmethod
            def now(cls, tz=None):
                return now

        # main() also asks origin whether `main` has moved under this
        # checkout before evaluating (main_tip_ahead_of_checkout). Stubbed
        # by name here for the same reason as the branch list above; the
        # guard itself is covered by TestRefusesAStaleCheckout.
        monkeypatch.setattr(check_triggers, "main_tip_ahead_of_checkout", lambda: "")
        monkeypatch.setattr(check_triggers, "datetime", _Now)
        return check_triggers.main

    def test_a_failed_push_exits_one(self, broker_env, monkeypatch) -> None:
        """Regression: 2026-08-18 — a failed watcher push was a warning inside
        a green run, so `failure-issue` never fired on the money path."""
        from scripts import check_triggers

        order, pm = self._fireable(broker_env, monkeypatch)
        seen = self._always_fail_push(monkeypatch)
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        main = self._run_main(monkeypatch, pm, now)

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

        # (a) the batch was still fully processed before exiting, and the
        # retry really was attempted — an early abort would lose the rest.
        pushes = [c for c in seen if c[1] == "push"]
        assert len(pushes) >= 2, f"no rebase-retry was attempted: {seen}"
        assert any(c[1] == "pull" for c in seen), "the rebase retry never ran"
        fills = read_inbox(now.date())
        assert [f.order_id for f in fills] == [order.order_id]
        assert fills[0].status == "filled"
        assert list_pending() == []

    def test_a_successful_push_still_exits_zero(self, broker_env, monkeypatch) -> None:
        """The control. Without it, `exit 1` on every run would pass the test
        above and break every watcher run in production."""
        _order, pm = self._fireable(broker_env, monkeypatch)
        self._fake_git(monkeypatch, push_rc=0)
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        self._run_main(monkeypatch, pm, now)()  # must not raise SystemExit

    def test_a_blackout_run_stays_green(self, broker_env, monkeypatch) -> None:
        """A blackout is a deliberate no-op, not a failure."""
        _order, pm = self._fireable(broker_env, monkeypatch)
        self._always_fail_push(monkeypatch)
        now = datetime(2026, 5, 17, 20, 15, tzinfo=timezone.utc)
        self._run_main(monkeypatch, pm, now)()

    def test_a_no_op_run_stays_green(self, broker_env, monkeypatch) -> None:
        """Nothing pending, nothing fired, nothing pushed — nothing to report."""
        from engine import triggers as triggers_mod

        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=8000.0, currency="EUR"
        )
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 1.0)
        self._always_fail_push(monkeypatch)
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        self._run_main(monkeypatch, pm, now)()

    def test_a_self_healed_push_does_not_cry_wolf(
        self, broker_env, monkeypatch
    ) -> None:
        """A push pushes the whole branch, so a later order's successful push
        carries an earlier order's rejected commit. Reporting "stranded" then
        is noise on the one alert channel that has to stay trustworthy.

        The verdict is taken from the published state, not from the counter:
        every local commit is on origin/main, so nothing is stranded.
        """
        _order, pm = self._fireable(broker_env, monkeypatch)
        self._fake_git(monkeypatch, push_rc=1, is_ancestor_rc=0)
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        self._run_main(monkeypatch, pm, now)()  # must not raise SystemExit

    def test_a_failed_commit_is_not_silent(self, broker_env, monkeypatch) -> None:
        """`_git_add_commit` runs git-add and git-commit under check=True, and
        process_fired_order swallows the exception so the batch continues. That
        left a failed COMMIT invisible while a failed push was not — and it is
        the worse case: the fill is on disk with no commit at all."""
        import subprocess as sp
        import types

        from scripts import check_triggers

        _order, pm = self._fireable(broker_env, monkeypatch)

        def fake(cmd, *a, **kw):
            if cmd[1] == "diff":
                return sp.CompletedProcess(cmd, 1)
            # ONLY the per-order commit fails. The tail commit_and_push must
            # succeed, or it would set the failure flag by itself and this test
            # would pass with the per-order counting removed — which is exactly
            # what happened on the first attempt.
            if cmd[1] == "commit" and "execute ord_" in " ".join(cmd):
                raise sp.CalledProcessError(1, cmd)
            if cmd[1] == "merge-base":
                return sp.CompletedProcess(cmd, 1)
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(
                run=fake,
                CompletedProcess=sp.CompletedProcess,
                CalledProcessError=sp.CalledProcessError,
            ),
        )
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        with pytest.raises(SystemExit) as exc:
            self._run_main(monkeypatch, pm, now)()
        assert exc.value.code == 1

    def test_an_order_that_raises_turns_the_run_red(
        self, broker_env, monkeypatch
    ) -> None:
        """`errors` is execute_triggered_order raising on a FIRED trigger — the
        worst failure on this path, and it used to exit 0 while the Worker
        re-dispatched the same order every hour."""
        from scripts import check_triggers

        _order, pm = self._fireable(broker_env, monkeypatch)
        self._fake_git(monkeypatch, push_rc=0)

        def boom(*a, **kw):
            raise RuntimeError("broker exploded")

        monkeypatch.setattr(check_triggers, "execute_triggered_order", boom)
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        with pytest.raises(SystemExit) as exc:
            self._run_main(monkeypatch, pm, now)()
        assert exc.value.code == 1

    def test_a_systemic_commit_failure_is_not_masked_by_the_ancestor_check(
        self, broker_env, monkeypatch
    ) -> None:
        """Regression: 2026-08-18 — the self-heal check masked the failure it
        sat next to.

        `_nothing_is_stranded` asked "is anything committed-but-unpushed". When
        git-commit fails outright NOTHING is committed, so HEAD is trivially an
        ancestor of origin/main, the flag was cleared and the run exited 0 —
        while the fill sat on disk, in no commit, and died with the runner.

        `merge-base` answers 0 here on purpose: that is the TRUTHFUL answer
        when nothing was committed, and the previous test's stub hard-coded 1,
        which is why it could not see this.
        """
        import subprocess as sp
        import types

        from scripts import check_triggers

        _order, pm = self._fireable(broker_env, monkeypatch)

        def fake(cmd, *a, **kw):
            if cmd[1] == "diff":
                return sp.CompletedProcess(cmd, 1)
            if cmd[1] == "commit":
                raise sp.CalledProcessError(1, cmd)
            if cmd[1] == "merge-base":
                return sp.CompletedProcess(cmd, 0)  # truthful: nothing committed
            if cmd[1] == "status":
                return sp.CompletedProcess(cmd, 0, stdout="")
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(
                run=fake,
                CompletedProcess=sp.CompletedProcess,
                CalledProcessError=sp.CalledProcessError,
            ),
        )
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        with pytest.raises(SystemExit) as exc:
            self._run_main(monkeypatch, pm, now)()
        assert exc.value.code == 1

    def test_uncommitted_watcher_paths_mean_something_is_stranded(
        self, broker_env, monkeypatch
    ) -> None:
        """The second lock: "unpushed" is only half of "stranded".

        Even with the commit-failure counter cleared, a dirty worktree under
        the paths the watcher writes must refuse to report all-clear.
        """
        import subprocess as sp
        import types

        from scripts import check_triggers

        def fake(cmd, *a, **kw):
            if cmd[1] == "status":
                return sp.CompletedProcess(
                    cmd, 0, stdout=" M data/orders/inbox/x.jsonl\n"
                )
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(run=fake, CompletedProcess=sp.CompletedProcess),
        )
        assert check_triggers._nothing_is_stranded() is False


# ---------------------------------------------------------------------------
# --dry-run must actually be dry
# ---------------------------------------------------------------------------


def _init_tmp_git_repo(root: Path) -> None:
    """Make the tmp MIDAS_DATA_DIR a real git repo.

    The dry-run assertions are about the git index and the commit history, and
    those are only real if the git commands the watcher runs are real. The
    alternative — stubbing `check_triggers.subprocess`, as the push tests must
    — would assert about a mock, and a mock cannot tell you whether a *real*
    `git add` was reached. `_PROJECT_ROOT` is repointed here so nothing can
    touch the actual checkout.
    """
    import subprocess as sp

    sp.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    for key, value in (
        ("user.email", "watcher-test@example.invalid"),
        ("user.name", "watcher test"),
        ("commit.gpgsign", "false"),
    ):
        sp.run(["git", "config", key, value], cwd=root, check=True)
    # Seed commit, so the fixtures are TRACKED. `_git_add_commit` stages the
    # pending file by path after deleting it, and `git add` on an untracked
    # path that no longer exists fails outright — the real repo tracks it, so
    # the rehearsal repo has to as well or the control below tests a different
    # failure than the one production would hit.
    sp.run(["git", "add", "-A"], cwd=root, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=root, check=True)


def _staged(root: Path) -> list[str]:
    import subprocess as sp

    out = sp.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def _commit_subjects(root: Path) -> list[str]:
    import subprocess as sp

    out = sp.run(
        ["git", "log", "--format=%s"], cwd=root, capture_output=True, text=True
    )
    if out.returncode != 0:  # no commits yet — `git log` fails on an empty repo
        return []
    return [line for line in out.stdout.splitlines() if line]


class TestDryRunIsActuallyDry:
    """Regression: 2026-09-05 — `--dry-run` filled, committed and pushed.

    `main()` read `args.dry_run` only AFTER `run()` had returned, and only to
    skip the tail `commit_and_push()`. `run()` took no such parameter, so
    `_process_channel` still called `execute_triggered_order` (book mutation +
    inbox append + pending delete) and the per-order `_commit` closure, which
    commits and pushes `HEAD:main`. A local `--dry-run` at 07:07 UTC that day
    pushed three real fills to main — `7a4a1fb57`, `d91e0e014`, `ba3f4c9a5`
    (the fix commit cannot cite itself; those are the evidence).

    The assertions below are on files and on the real git index, not on mocks:
    a stub can only tell you which calls were made, and the claim under test is
    that the pending file, the inbox, the book and the index are unchanged.
    `test_the_same_fixture_does_mutate_without_dry_run` is the falsifying
    control — without it, a fixture that simply never fires would pass every
    assertion here.
    """

    def _fireable(self, broker_env, midas_data_root, monkeypatch):
        """A pending SELL whose trigger is hit, on a book that can fill it."""
        from engine import triggers as triggers_mod
        from engine.types import Trade

        order = _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        # The book lives under get_config().portfolios_dir (not broker_env's
        # pm_base) because the per-fire `git add` stages exactly that path.
        pm_base = get_config().portfolios_dir
        pm_base.mkdir(parents=True, exist_ok=True)
        pm = _init_portfolio(pm_base, "satoshi", cash=8000.0, currency="EUR")
        pm.apply_trade(
            "satoshi",
            Trade(
                id="seed_001",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                action="BUY",
                ticker="BTC-EUR",
                shares=0.1,
                price=70000.0,
                total=7000.0,
                fees=0.0,
                reasoning="seed",
            ),
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        _init_tmp_git_repo(midas_data_root)
        from scripts import check_triggers

        monkeypatch.setattr(check_triggers, "_PROJECT_ROOT", midas_data_root)
        return order, pm, pm_base

    def test_dry_run_writes_nothing_and_still_reports_the_fire(
        self, broker_env, midas_data_root, monkeypatch, caplog
    ) -> None:
        import logging

        from scripts import check_triggers

        order, pm, pm_base = self._fireable(broker_env, midas_data_root, monkeypatch)

        pending_path = broker_env["pending"] / f"{order.order_id}.json"
        pending_before = pending_path.read_bytes()
        portfolio_path = pm_base / "satoshi" / "portfolio.json"
        portfolio_before = portfolio_path.read_bytes()
        trades_before = (pm_base / "satoshi" / "trades.json").read_bytes()

        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        with caplog.at_level(logging.INFO, logger="scripts.check_triggers"):
            summary = check_triggers.run(now=now, portfolio_manager=pm, dry_run=True)

        # (a) The rehearsal is informative: the fire is counted and labelled.
        assert summary["dry_run"] is True
        assert summary["fired"] == 1
        assert summary["checked"] == 1

        # (b) The pending file survives byte-for-byte.
        assert pending_path.read_bytes() == pending_before
        assert [o.order_id for o in list_pending()] == [order.order_id]

        # (c) Nothing reached the inbox — not even an empty day file.
        assert read_inbox(now.date()) == []
        assert not (broker_env["inbox"] / f"{now.date().isoformat()}.jsonl").exists()

        # (d) The book did not move.
        assert portfolio_path.read_bytes() == portfolio_before
        assert (pm_base / "satoshi" / "trades.json").read_bytes() == trades_before

        # (e) The git index is untouched and no commit was created.
        assert _staged(midas_data_root) == []
        assert _commit_subjects(midas_data_root) == ["seed"]  # only the fixture

        # (f) The log names what a reader needs to act on: ticker, op, level,
        #     observed price, would-be notional (0.01 × 85123.45).
        report = caplog.text
        assert "would FIRE" in report
        assert order.order_id in report
        assert "BTC-EUR" in report
        assert ">=" in report and "85000.0" in report
        assert "85123.45" in report
        assert "851.23" in report

    def test_the_same_fixture_does_mutate_without_dry_run(
        self, broker_env, midas_data_root, monkeypatch
    ) -> None:
        """The falsifying control for every assertion above.

        Same fixture, `dry_run=False`: the pending file goes, the fill lands,
        the book moves and a per-fire commit exists. Without this, a fixture
        whose trigger silently stopped firing would pass the dry-run test while
        proving nothing. (The push fails — the tmp repo has no remote — which
        is counted and is not this test's subject.)
        """
        from scripts import check_triggers

        order, pm, pm_base = self._fireable(broker_env, midas_data_root, monkeypatch)
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)

        summary = check_triggers.run(now=now, portfolio_manager=pm)

        assert summary["dry_run"] is False
        assert summary["fired"] == 1
        assert list_pending() == []
        fills = read_inbox(now.date())
        assert [f.order_id for f in fills] == [order.order_id]
        assert fills[0].status == "filled"
        assert any(t.get("id") != "seed_001" for t in pm.load_trades("satoshi")), (
            "the fill did not reach the trade log"
        )
        assert _commit_subjects(midas_data_root) == [
            f"chore(triggers): execute {order.order_id} {now.date().isoformat()}",
            "seed",
        ]

    def test_dry_run_evaluates_expiry_without_retiring_the_order(
        self, broker_env, midas_data_root, monkeypatch
    ) -> None:
        """Expiry is the other write path in `_process_channel`.

        It writes a TRIGGER_EXPIRED rejection and deletes the pending file
        before any price is read, so it needs its own guard — and its own
        assertion that the order is still counted, or a dry run would under-
        report exactly the orders about to be retired.
        """
        from engine import triggers as triggers_mod
        from scripts import check_triggers

        order = _seed_pending(broker_env, expires="2026-04-01")
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)

        summary = check_triggers.run(now=now, portfolio_manager=pm, dry_run=True)

        assert summary["expired"] == 1
        assert [o.order_id for o in list_pending()] == [order.order_id]
        assert read_inbox(now.date()) == []

    def test_process_fired_order_refuses_to_write_under_dry_run(
        self, broker_env, monkeypatch
    ) -> None:
        """The second lock.

        `_process_channel` stops before the broker, so this helper is not
        reached in a dry run today. The guard exists so a future caller cannot
        route a dry run back into the inbox append / pending delete / commit,
        and this is its consumer — without a test it would be a check that can
        never fail.
        """
        from engine.orders import Fill
        from scripts.check_triggers import process_fired_order

        order = _seed_pending(broker_env)
        pending_path = broker_env["pending"] / f"{order.order_id}.json"
        fill = Fill(
            order_id=order.order_id,
            ts_filled=datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc),
            status="filled",
            fill_price=85123.45,
            fill_currency="EUR",
            notional_base=851.23,
            fees=0.0,
            reason=None,
            trigger_fired=True,
        )

        def committer(*_a, **_k):
            raise AssertionError("a dry run must not commit")

        process_fired_order(order, fill, date(2026, 5, 17), committer, dry_run=True)

        assert pending_path.exists()
        assert read_inbox(date(2026, 5, 17)) == []


class TestDryRunMainSkipsTheGitHelpers:
    """`main(--dry-run)` must not reach a single git write or the leaderboard.

    Bug 1 was reported as "--dry-run pushes": the flag was checked after the
    push had already happened. Monkeypatching the write helpers to raise pins
    the end-to-end path, not just `run()`.
    """

    def _run_main(self, monkeypatch, pm, now, argv):
        from datetime import datetime as dt

        from scripts import check_triggers

        monkeypatch.setattr(check_triggers.sys, "argv", list(argv))
        monkeypatch.setattr(check_triggers, "PortfolioManager", lambda base_dir: pm)
        # main() asks origin for unmerged `triggers/*` branches before it
        # evaluates anything. This class does not stub `subprocess` (the
        # write helpers are stubbed by name instead), so without this the
        # test would `git ls-remote` the REAL remote from the real checkout.
        monkeypatch.setattr(check_triggers, "unmerged_fallback_branches", lambda: [])

        class _Now(dt):
            @classmethod
            def now(cls, tz=None):
                return now

        # main() also asks origin whether `main` has moved under this
        # checkout before evaluating (main_tip_ahead_of_checkout). Stubbed
        # by name here for the same reason as the branch list above; the
        # guard itself is covered by TestRefusesAStaleCheckout.
        monkeypatch.setattr(check_triggers, "main_tip_ahead_of_checkout", lambda: "")
        monkeypatch.setattr(check_triggers, "datetime", _Now)
        return check_triggers.main

    def _fireable(self, broker_env, monkeypatch):
        from engine import triggers as triggers_mod
        from engine.types import Trade

        order = _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=8000.0, currency="EUR"
        )
        pm.apply_trade(
            "satoshi",
            Trade(
                id="seed_001",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                action="BUY",
                ticker="BTC-EUR",
                shares=0.1,
                price=70000.0,
                total=7000.0,
                fees=0.0,
                reasoning="seed",
            ),
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        return order, pm

    def _forbid_writes(self, monkeypatch):
        from scripts import check_triggers

        def boom(name):
            def _raise(*_a, **_k):
                raise AssertionError(f"a dry run must not call {name}")

            return _raise

        monkeypatch.setattr(check_triggers, "_git_add_commit", boom("_git_add_commit"))
        monkeypatch.setattr(check_triggers, "commit_and_push", boom("commit_and_push"))
        monkeypatch.setattr(
            check_triggers,
            "refresh_leaderboard_artifact",
            boom("refresh_leaderboard_artifact"),
        )

    def test_dry_run_main_never_commits_pushes_or_refreshes(
        self, broker_env, monkeypatch
    ) -> None:
        order, pm = self._fireable(broker_env, monkeypatch)
        self._forbid_writes(monkeypatch)
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)

        self._run_main(
            monkeypatch, pm, now, ("check_triggers.py", "--dry-run")
        )()  # must not raise, and must not exit non-zero

        assert [o.order_id for o in list_pending()] == [order.order_id]
        assert read_inbox(now.date()) == []

    def test_the_same_stubs_do_fire_without_dry_run(
        self, broker_env, monkeypatch
    ) -> None:
        """Control: those stubs really are on the path a real run takes.

        Drop `--dry-run` and all three are called, in order. Without this the
        test above would pass against a watcher that had simply stopped firing,
        or against stubs patched onto names nothing calls.
        """
        from scripts import check_triggers

        _order, pm = self._fireable(broker_env, monkeypatch)
        called: list[str] = []
        monkeypatch.setattr(
            check_triggers,
            "_git_add_commit",
            lambda *a, **k: (
                called.append("_git_add_commit") or check_triggers.COMMIT_OK
            ),
        )
        monkeypatch.setattr(
            check_triggers,
            "commit_and_push",
            lambda *a, **k: (
                called.append("commit_and_push") or check_triggers.COMMIT_OK
            ),
        )
        monkeypatch.setattr(
            check_triggers,
            "refresh_leaderboard_artifact",
            lambda *a, **k: called.append("refresh_leaderboard_artifact"),
        )
        now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)

        self._run_main(monkeypatch, pm, now, ("check_triggers.py",))()

        assert called == [
            "_git_add_commit",
            "refresh_leaderboard_artifact",
            "commit_and_push",
        ]


# ---------------------------------------------------------------------------
# A refused push to main falls back to a per-run `triggers/*` branch
# ---------------------------------------------------------------------------


class TestRefusedPushFallsBack:
    """Regression: 2026-08-24..09-04 — a fill main refused died with the runner.

    A branch-protection rule rejected every bot push to `main` with GH006 for
    eleven days. The watcher's per-fire commit existed only on the runner, so
    when the run ended the fill was gone, the pending file was back on the
    next checkout, and the next run re-fired the same order at a new price.
    Ten orders were lost that way (METHODOLOGY `#watcher-outage-2026-08-24`);
    the fix commit cannot cite itself, so the cancels ledger
    (`data/orders/cancels/2026-08-31.jsonl` … `2026-09-04.jsonl`) is the
    evidence.

    Now a commit that main refuses is pushed to `triggers/<date>-<run id>`
    and `.github/workflows/auto-merge-session.yml` merges it. Three verdicts
    are pinned here, because collapsing any two of them is the defect:

    - reached main → success, as before;
    - reached only the branch → durable, so a WARNING: exit 0, a
      `fallback_pushes` count, and a `::warning::` line naming the branch;
    - reached neither → exit 1, exactly as `TestFailedPushExitsNonZero` pins.

    The git stub below models a protected main: `HEAD:main` is refused, the
    rebase-retry is refused, the branch push is not. `_fake_git` in the class
    above refuses BOTH (its push_rc applies to every push), which is why that
    class's exit-1 assertions still hold — they are the double-failure case.
    """

    RUN_ID = "424242"
    BRANCH = "triggers/2026-05-17-424242"

    def _fake_git(
        self,
        monkeypatch,
        *,
        branch_push_rcs: list[int] | None = None,
        remote_branch_sha: str = "",
        fallback_branches: str = "",
    ):
        """Every git call the watcher makes, with main refusing HEAD:main.

        ``branch_push_rcs`` is consumed one per branch push (last value
        repeats); ``remote_branch_sha`` is what `ls-remote` reports for the
        run's own branch (empty → absent); ``fallback_branches`` is the
        `ls-remote` answer for `triggers/*`, i.e. a previous run's unmerged
        branch. `merge-base --is-ancestor` answers 1 after a fetch of main and
        0 after a fetch of the fallback branch, modelling a commit that is on
        the branch and not on main.
        """
        import subprocess as sp
        import types

        from scripts import check_triggers

        seen: list[list[str]] = []
        state = {"last_fetch": None, "branch_pushes": 0}
        rcs = list(branch_push_rcs or [0])

        def fake(cmd, *a, **kw):
            assert isinstance(cmd, (list, tuple)) and cmd[0] == "git", cmd
            seen.append(list(cmd))
            if cmd[1] == "diff":
                return sp.CompletedProcess(cmd, 1)  # something is staged
            if cmd[1] == "push":
                if cmd[-1] == "HEAD:main":
                    return sp.CompletedProcess(cmd, 1)  # GH006
                assert cmd[-1].startswith("HEAD:refs/heads/triggers/"), cmd
                i = min(state["branch_pushes"], len(rcs) - 1)
                state["branch_pushes"] += 1
                return sp.CompletedProcess(cmd, rcs[i])
            if cmd[1] == "ls-remote":
                pattern = cmd[-1]
                if pattern.endswith("*"):
                    return sp.CompletedProcess(cmd, 0, stdout=fallback_branches)
                out = f"{remote_branch_sha}\t{pattern}\n" if remote_branch_sha else ""
                return sp.CompletedProcess(cmd, 0, stdout=out)
            if cmd[1] == "fetch":
                state["last_fetch"] = cmd[-1]
                return sp.CompletedProcess(cmd, 0)
            if cmd[1] == "merge-base":
                on_branch = state["last_fetch"] != "main"
                return sp.CompletedProcess(cmd, 0 if on_branch else 1)
            if cmd[1] == "status":
                return sp.CompletedProcess(cmd, 0, stdout="")
            return sp.CompletedProcess(cmd, 0)  # add, commit, pull, rebase

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(
                run=fake,
                CompletedProcess=sp.CompletedProcess,
                CalledProcessError=sp.CalledProcessError,
            ),
        )
        monkeypatch.setenv("GITHUB_RUN_ID", self.RUN_ID)
        monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
        monkeypatch.setattr(check_triggers, "_fallback", None)
        return seen

    def _fireable(self, broker_env, monkeypatch):
        from engine import triggers as triggers_mod
        from engine.types import Trade

        order = _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=8000.0, currency="EUR"
        )
        pm.apply_trade(
            "satoshi",
            Trade(
                id="seed_001",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                action="BUY",
                ticker="BTC-EUR",
                shares=0.1,
                price=70000.0,
                total=7000.0,
                fees=0.0,
                reasoning="seed",
            ),
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        return order, pm

    def _run_main(self, monkeypatch, pm, now, argv=("check_triggers.py",)):
        """main(), with the summary dict main() mutates captured for assertion."""
        from datetime import datetime as dt

        from scripts import check_triggers

        monkeypatch.setattr(check_triggers.sys, "argv", list(argv))
        monkeypatch.setattr(check_triggers, "PortfolioManager", lambda base_dir: pm)

        class _Now(dt):
            @classmethod
            def now(cls, tz=None):
                return now

        # main() also asks origin whether `main` has moved under this
        # checkout before evaluating (main_tip_ahead_of_checkout). Stubbed
        # by name here for the same reason as the branch list above; the
        # guard itself is covered by TestRefusesAStaleCheckout.
        monkeypatch.setattr(check_triggers, "main_tip_ahead_of_checkout", lambda: "")
        monkeypatch.setattr(check_triggers, "datetime", _Now)

        captured: dict = {}
        real_run = check_triggers.run

        def run_and_capture(*a, **kw):
            summary = real_run(*a, **kw)
            captured["summary"] = summary  # same object main() keeps counting into
            return summary

        monkeypatch.setattr(check_triggers, "run", run_and_capture)
        return check_triggers.main, captured

    NOW = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)

    def test_a_refused_main_push_lands_on_the_branch_and_the_run_stays_green(
        self, broker_env, monkeypatch, capsys, caplog
    ) -> None:
        import logging

        order, pm = self._fireable(broker_env, monkeypatch)
        seen = self._fake_git(monkeypatch)
        main, captured = self._run_main(monkeypatch, pm, self.NOW)

        with caplog.at_level(logging.WARNING, logger="scripts.check_triggers"):
            main()  # must NOT raise SystemExit: the fill is durable

        summary = captured["summary"]
        # Per-fire commit AND the tail (leaderboard/expired) commit both fell
        # back; neither is a failure.
        assert summary["fallback_pushes"] == 2
        assert summary["push_failures"] == 0
        assert summary["commit_failures"] == 0
        assert summary["fired"] == 1

        # (a) main was tried first, and the rebase-retry too — ONCE. After the
        #     first fallback the run stays on the branch: no second attempt on
        #     main and no second `pull --rebase`, because the branch's shas are
        #     published `executed_sha` provenance by then (see
        #     TestFallbackBranchAgainstRealGit for the sha itself).
        main_pushes = [c for c in seen if c[1] == "push" and c[-1] == "HEAD:main"]
        assert len(main_pushes) == 2, seen  # 1st commit only: push + retry
        assert sum(1 for c in seen if c[1] == "pull") == 1
        first_fallback = next(
            i for i, c in enumerate(seen) if c[1] == "push" and c[-1] != "HEAD:main"
        )
        after = seen[first_fallback + 1 :]
        assert not any(c[1] == "pull" for c in after), after
        assert not any(c[1] == "push" and c[-1] == "HEAD:main" for c in after), after

        # (b) The branch push names the per-run branch and is leased on
        #     "must not exist" for the first push (empty expectation).
        branch_pushes = [c for c in seen if c[1] == "push" and c[-1] != "HEAD:main"]
        assert [c[-1] for c in branch_pushes] == [
            f"HEAD:refs/heads/{self.BRANCH}",
            f"HEAD:refs/heads/{self.BRANCH}",
        ]
        assert branch_pushes[0][2] == f"--force-with-lease=refs/heads/{self.BRANCH}:"

        # (c) The fill itself went through exactly as on a normal run.
        fills = read_inbox(self.NOW.date())
        assert [f.order_id for f in fills] == [order.order_id]
        assert fills[0].status == "filled"
        assert list_pending() == []

        # (d) A human can see it without opening the log: the run annotates
        #     itself with the branch, and the log says the fill is durable.
        out = capsys.readouterr().out
        assert "::warning::" in out and self.BRANCH in out
        assert "Nothing is lost" in out
        assert "durable" in caplog.text and self.BRANCH in caplog.text

    def test_the_branch_push_is_leased_on_what_origin_holds(
        self, broker_env, monkeypatch
    ) -> None:
        """The second push of a run must not clobber the first blindly: the
        lease is whatever `ls-remote` reports right before the push. (Since
        the run stays on the branch after its first fallback, later pushes
        are fast-forwards and the lease is a tripwire, not a mechanism.)"""
        _order, pm = self._fireable(broker_env, monkeypatch)
        seen = self._fake_git(monkeypatch, remote_branch_sha="abc123")
        main, captured = self._run_main(monkeypatch, pm, self.NOW)

        main()

        branch_pushes = [c for c in seen if c[1] == "push" and c[-1] != "HEAD:main"]
        assert branch_pushes, seen
        assert all(
            c[2] == f"--force-with-lease=refs/heads/{self.BRANCH}:abc123"
            for c in branch_pushes
        ), branch_pushes
        assert captured["summary"]["fallback_pushes"] == 2

    def test_a_commit_that_reaches_neither_main_nor_the_branch_exits_one(
        self, broker_env, monkeypatch
    ) -> None:
        """The double failure is the ONLY case that is still a failure."""
        _order, pm = self._fireable(broker_env, monkeypatch)
        self._fake_git(monkeypatch, branch_push_rcs=[1])
        main, captured = self._run_main(monkeypatch, pm, self.NOW)

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        assert captured["summary"]["fallback_pushes"] == 0
        assert captured["summary"]["push_failures"] >= 1

    def test_a_later_branch_push_carries_an_earlier_stranded_commit(
        self, broker_env, monkeypatch
    ) -> None:
        """`_nothing_is_stranded` must count the fallback branch as origin.

        The per-fire push reaches neither; the tail push reaches the branch,
        carrying the fire's commit with it (a push pushes the whole branch).
        Before the fallback existed this could only self-heal via main; the
        stranded check now asks the branch too, so this run is green.
        """
        _order, pm = self._fireable(broker_env, monkeypatch)
        seen = self._fake_git(monkeypatch, branch_push_rcs=[1, 0])
        main, captured = self._run_main(monkeypatch, pm, self.NOW)

        main()  # no SystemExit

        summary = captured["summary"]
        assert summary["push_failures"] == 1
        assert summary["fallback_pushes"] == 1
        # The verdict came from the published state: the branch was fetched
        # and HEAD tested against it, not against main alone.
        fetches = [c[-1] for c in seen if c[1] == "fetch"]
        assert fetches[-2:] == ["main", self.BRANCH], fetches

    def test_a_failed_fallback_push_also_ends_this_runs_right_to_rebase(
        self, broker_env, monkeypatch
    ) -> None:
        """Regression (round-5 review, 2026-09-05): step 0 keyed on `pushed`.

        `_push_head`'s "once this run has fallen back, never rebase again"
        guard read `_fallback.pushed`, which only flips on a SUCCESSFUL branch
        push. A run whose fallback push also failed therefore rebased on the
        next order — and that is the worst case for a rebase, not the exempt
        one: the fill executed after the first failure carries
        `executed_sha` = the sha of the previous order's LOCAL commit
        (`engine.paper_broker` stamps `git rev-parse HEAD` at execution time),
        so `pull --rebase` rewrote the commit that row names. The retry push
        then landed the rewritten twins on main, `_nothing_is_stranded` found
        HEAD an ancestor of origin/main and cleared `push_failed`, and the run
        exited 0 having published a fill whose `executed_sha` is reachable
        from nothing on origin — the exact provenance `git checkout
        <executed_sha>` promises in METHODOLOGY.

        Same stub as the test above (branch push 1 fails, 2 succeeds), read
        for the other half of the story: what the second push site does.
        """
        _order, pm = self._fireable(broker_env, monkeypatch)
        seen = self._fake_git(monkeypatch, branch_push_rcs=[1, 0])
        main, captured = self._run_main(monkeypatch, pm, self.NOW)

        main()  # no SystemExit — the tail push reached the branch

        first_fallback = next(
            i for i, c in enumerate(seen) if c[1] == "push" and c[-1] != "HEAD:main"
        )
        after = seen[first_fallback + 1 :]
        assert not any(c[1] == "pull" for c in after), after
        assert not any(c[1] == "rebase" for c in after), after
        assert not any(c[1] == "push" and c[-1] == "HEAD:main" for c in after), after
        # ...and the whole run rebased exactly once, before it ever fell back.
        assert sum(1 for c in seen if c[1] == "pull") == 1, seen
        assert [c[-1] for c in seen if c[1] == "push" and c[-1] == "HEAD:main"] == [
            "HEAD:main",
            "HEAD:main",
        ], seen
        # The tail commit still went out — refusing to rebase is not refusing
        # to publish.
        assert captured["summary"]["fallback_pushes"] == 1

    def test_the_stranded_check_does_not_trust_a_branch_that_was_never_pushed(
        self, monkeypatch
    ) -> None:
        """A branch NAME proves nothing: only a branch this run pushed to
        (and re-fetched) may clear a stranded commit."""
        import subprocess as sp
        import types

        from scripts import check_triggers

        seen: list[list[str]] = []

        def fake(cmd, *a, **kw):
            seen.append(list(cmd))
            if cmd[1] == "status":
                return sp.CompletedProcess(cmd, 0, stdout="")
            if cmd[1] == "merge-base":
                return sp.CompletedProcess(cmd, 1)  # not on main
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(run=fake, CompletedProcess=sp.CompletedProcess),
        )
        fb = check_triggers._FallbackBranch("triggers/2026-05-17-x")
        monkeypatch.setattr(check_triggers, "_fallback", fb)  # named, never pushed

        assert check_triggers._nothing_is_stranded() is False
        assert not any(c[1] == "fetch" and c[-1] == fb.name for c in seen)

    def test_an_unmerged_fallback_branch_on_origin_refuses_to_evaluate(
        self, broker_env, monkeypatch
    ) -> None:
        """A previous run's branch that has not reached main makes main a
        stale ledger: the pending file is back and the inbox row is absent,
        so evaluating would re-fire the same order at a new price — the exact
        damage of the outage, one failed merge away. Refuse, exit 1."""
        from scripts import check_triggers

        order, pm = self._fireable(broker_env, monkeypatch)
        self._fake_git(
            monkeypatch,
            fallback_branches="deadbeef\trefs/heads/triggers/2026-05-16-111\n",
        )
        main, captured = self._run_main(monkeypatch, pm, self.NOW)

        def never(*a, **kw):
            raise AssertionError("run() must not be reached with an unmerged branch")

        monkeypatch.setattr(check_triggers, "run", never)

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        # Nothing was evaluated, so nothing moved.
        assert [o.order_id for o in list_pending()] == [order.order_id]
        assert read_inbox(self.NOW.date()) == []

    def test_an_unreadable_remote_refuses_too(self, broker_env, monkeypatch) -> None:
        """Fail closed: "could not list branches" is unknown, not clean."""
        import subprocess as sp

        from scripts import check_triggers

        _order, pm = self._fireable(broker_env, monkeypatch)
        self._fake_git(monkeypatch)
        real = check_triggers.subprocess.run

        def ls_remote_fails(cmd, *a, **kw):
            if cmd[1] == "ls-remote" and cmd[-1].endswith("*"):
                return sp.CompletedProcess(cmd, 128, stdout="")
            return real(cmd, *a, **kw)

        monkeypatch.setattr(check_triggers.subprocess, "run", ls_remote_fails)
        main, _captured = self._run_main(monkeypatch, pm, self.NOW)
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_a_dry_run_reports_an_unmerged_branch_without_refusing(
        self, broker_env, monkeypatch, caplog
    ) -> None:
        """A dry run writes nothing, so it cannot re-fire anything; it warns
        and still rehearses — the rehearsal is the point of asking for one."""
        import logging

        _order, pm = self._fireable(broker_env, monkeypatch)
        self._fake_git(
            monkeypatch,
            fallback_branches="deadbeef\trefs/heads/triggers/2026-05-16-111\n",
        )
        main, captured = self._run_main(
            monkeypatch, pm, self.NOW, ("check_triggers.py", "--dry-run")
        )
        with caplog.at_level(logging.WARNING, logger="scripts.check_triggers"):
            main()  # no SystemExit
        assert "triggers/2026-05-16-111" in caplog.text
        assert captured["summary"]["dry_run"] is True
        assert captured["summary"]["fired"] == 1

    def test_the_guard_parses_what_ls_remote_prints(self, monkeypatch) -> None:
        """Both the empty answer (clean) and the tab-separated lines."""
        import subprocess as sp
        import types

        from scripts import check_triggers

        answers = {"out": ""}

        def fake(cmd, *a, **kw):
            assert cmd[1] == "ls-remote" and cmd[-1] == "refs/heads/triggers/*", cmd
            return sp.CompletedProcess(cmd, 0, stdout=answers["out"])

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(run=fake, CompletedProcess=sp.CompletedProcess),
        )
        assert check_triggers.unmerged_fallback_branches() == []
        answers["out"] = (
            "1111111\trefs/heads/triggers/2026-05-16-111\n"
            "2222222\trefs/heads/triggers/2026-05-17-222\n"
        )
        assert check_triggers.unmerged_fallback_branches() == [
            "triggers/2026-05-16-111",
            "triggers/2026-05-17-222",
        ]

    def test_the_branch_name_is_unique_per_run_and_per_attempt(
        self, monkeypatch
    ) -> None:
        from scripts import check_triggers

        d = date(2026, 5, 17)
        monkeypatch.setenv("GITHUB_RUN_ID", "9")
        monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
        assert check_triggers.fallback_branch_name(d) == "triggers/2026-05-17-9"
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
        assert check_triggers.fallback_branch_name(d) == "triggers/2026-05-17-9"
        # A re-run keeps the run id; the first attempt's branch may still exist.
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
        assert check_triggers.fallback_branch_name(d) == "triggers/2026-05-17-9-r2"
        # Outside Actions: still unique, still under the prefix the workflow
        # triggers on.
        monkeypatch.delenv("GITHUB_RUN_ID")
        monkeypatch.delenv("GITHUB_RUN_ATTEMPT")
        a = check_triggers.fallback_branch_name(d)
        b = check_triggers.fallback_branch_name(d)
        assert a != b
        assert a.startswith(check_triggers.FALLBACK_BRANCH_PREFIX + "2026-05-17-local-")


class TestFallbackBranchAgainstRealGit:
    """The same fallback, driven through real git against a bare remote whose
    `pre-receive` hook refuses `refs/heads/main` — the GH006 shape exactly.

    The stubbed tests above assert which commands were issued; this one
    asserts that those commands WORK: that `--force-with-lease=<ref>:` with an
    empty expectation creates the branch, that a second fire in the same run
    fast-forwards the branch WITHOUT rewriting the first fill's sha even after
    main moved underneath, and that `_nothing_is_stranded` and
    `unmerged_fallback_branches` read the result back correctly.
    """

    HOOK = (
        "#!/bin/sh\n"
        "while read old new ref; do\n"
        '  if [ "$ref" = "refs/heads/main" ]; then\n'
        '    echo "GH006: Protected branch update failed for refs/heads/main" >&2\n'
        "    exit 1\n"
        "  fi\n"
        "done\n"
        "exit 0\n"
    )

    def _repos(self, tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
        import subprocess as sp

        from tests.test_watcher_ordering import _init_git_repo

        from scripts import check_triggers

        repo, bare = tmp_path / "repo", tmp_path / "bare.git"
        _init_git_repo(repo, bare)  # seeds main on both, BEFORE the hook
        hooks = bare / "hooks"
        hooks.mkdir(exist_ok=True)
        hook = hooks / "pre-receive"
        hook.write_text(self.HOOK)
        hook.chmod(0o755)
        # The hook is real: prove it refuses main before relying on it.
        (repo / "probe.txt").write_text("probe\n")
        sp.run(["git", "add", "probe.txt"], cwd=repo, check=True, capture_output=True)
        sp.run(["git", "commit", "-q", "-m", "probe"], cwd=repo, check=True)
        refused = sp.run(
            ["git", "push", "origin", "HEAD:main"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert refused.returncode != 0 and "GH006" in refused.stderr
        sp.run(["git", "reset", "-q", "--hard", "HEAD~1"], cwd=repo, check=True)

        monkeypatch.setattr(check_triggers, "_PROJECT_ROOT", repo)
        monkeypatch.setattr(check_triggers, "_fallback", None)
        monkeypatch.setenv("GITHUB_RUN_ID", "777")
        monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
        return repo, bare

    @staticmethod
    def _bare_ref(bare: Path, ref: str) -> str | None:
        import subprocess as sp

        out = sp.run(
            ["git", "rev-parse", "--verify", "-q", ref],
            cwd=bare,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None

    @staticmethod
    def _head(repo: Path) -> str:
        import subprocess as sp

        return sp.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def test_two_fires_reach_the_branch_even_after_main_moves_underneath(
        self, tmp_path, monkeypatch
    ) -> None:
        import subprocess as sp

        from scripts import check_triggers as ct

        repo, bare = self._repos(tmp_path, monkeypatch)
        main_before = self._bare_ref(bare, "refs/heads/main")
        branch = "refs/heads/triggers/" + date(2026, 5, 17).isoformat() + "-777"
        # `_fallback_branch` dates the name from the clock; pin it.
        monkeypatch.setattr(
            ct, "fallback_branch_name", lambda today: "triggers/2026-05-17-777"
        )

        # Fire 1: refused by main, created on the branch.
        (repo / "fill1.txt").write_text("fill 1\n")
        assert (
            ct._git_add_commit("ord_1", date(2026, 5, 17), [str(repo / "fill1.txt")])
            == ct.PUSHED_FALLBACK
        )
        first = self._bare_ref(bare, branch)
        assert first == self._head(repo)
        assert self._bare_ref(bare, "refs/heads/main") == main_before

        # Main moves underneath (another writer landed; the hook only bites
        # pushes, so land it through a fetch into the bare).
        other = tmp_path / "other"
        sp.run(["git", "clone", "-q", str(bare), str(other)], check=True)
        sp.run(["git", "-C", str(other), "config", "user.email", "o@x"], check=True)
        sp.run(["git", "-C", str(other), "config", "user.name", "o"], check=True)
        (other / "competing.txt").write_text("competing\n")
        sp.run(["git", "-C", str(other), "add", "competing.txt"], check=True)
        sp.run(["git", "-C", str(other), "commit", "-q", "-m", "competing"], check=True)
        sp.run(
            ["git", "-C", str(bare), "fetch", "-q", str(other), "main:main"], check=True
        )
        competing = self._bare_ref(bare, "refs/heads/main")
        assert competing != main_before

        # Fire 2 stamps its `executed_sha` from HEAD — which is fill 1's
        # commit, already on origin. The run must therefore stay on the
        # branch: no attempt on main, NO `pull --rebase`, and fill 1's sha
        # unchanged on origin afterwards. Until 2026-09-05 this fire rebased
        # onto the moved main and the leased push replaced the branch with
        # rewritten history, so fill 2's row named a sha reachable from
        # nothing on origin — `git checkout <executed_sha>` failed for the
        # very fill the fallback existed to keep.
        real_run = sp.run
        rebases: list[list[str]] = []

        def spy(cmd, *a, **kw):
            if isinstance(cmd, (list, tuple)) and cmd[:2] in (
                ["git", "pull"],
                ["git", "rebase"],
            ):
                rebases.append(list(cmd))
            return real_run(cmd, *a, **kw)

        monkeypatch.setattr(ct.subprocess, "run", spy)
        (repo / "fill2.txt").write_text("fill 2\n")
        assert (
            ct._git_add_commit("ord_2", date(2026, 5, 17), [str(repo / "fill2.txt")])
            == ct.PUSHED_FALLBACK
        )
        assert rebases == [], rebases
        tip = self._bare_ref(bare, branch)
        assert tip == self._head(repo) and tip != first
        parent = sp.run(
            ["git", "-C", str(bare), "rev-parse", f"{tip}^"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert parent == first, "fill 1's sha was rewritten after it reached origin"
        on_branch = sp.run(
            ["git", "-C", str(bare), "ls-tree", "--name-only", tip],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert {"fill1.txt", "fill2.txt"} <= set(on_branch)
        assert "competing.txt" not in on_branch, "the branch was rebased onto main"
        assert (
            sp.run(
                ["git", "-C", str(bare), "merge-base", "--is-ancestor", competing, tip]
            ).returncode
            != 0
        ), "the branch must NOT have been rewritten onto the moved main"
        assert self._bare_ref(bare, "refs/heads/main") == competing

        # The run reads its own result back: nothing stranded, one unmerged
        # branch on origin that the NEXT run must refuse to evaluate behind.
        assert ct._nothing_is_stranded() is True
        assert ct.unmerged_fallback_branches() == ["triggers/2026-05-17-777"]

        # The merge workflow's effect, by hand: main takes the branch (a real
        # merge commit, since main moved), the branch is deleted, and the
        # guard clears.
        sp.run(["git", "-C", str(other), "fetch", "-q", "origin", branch], check=True)
        sp.run(
            ["git", "-C", str(other), "merge", "-q", "--no-edit", "FETCH_HEAD"],
            check=True,
        )
        merged = sp.run(
            ["git", "-C", str(other), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        sp.run(
            ["git", "-C", str(bare), "fetch", "-q", str(other), "main:main"], check=True
        )
        assert self._bare_ref(bare, "refs/heads/main") == merged
        sp.run(["git", "-C", str(bare), "update-ref", "-d", branch], check=True)
        assert ct.unmerged_fallback_branches() == []
        assert ct._nothing_is_stranded() is True

    def test_the_first_push_refuses_to_replace_a_branch_that_already_exists(
        self, tmp_path, monkeypatch
    ) -> None:
        """The uniqueness assumption has a tripwire: a first push carries an
        EMPTY lease, which git reads as "must not exist". If the name is ever
        reused, the push is refused rather than silently replacing a fill."""
        import subprocess as sp

        from scripts import check_triggers as ct

        repo, bare = self._repos(tmp_path, monkeypatch)
        monkeypatch.setattr(
            ct, "fallback_branch_name", lambda today: "triggers/2026-05-17-777"
        )
        # Somebody else's commit already sits on that name.
        stale = sp.run(
            ["git", "-C", str(bare), "rev-parse", "main"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        sp.run(
            [
                "git",
                "-C",
                str(bare),
                "update-ref",
                "refs/heads/triggers/2026-05-17-777",
                stale,
            ],
            check=True,
        )
        # ...and `ls-remote` is made to miss it, so the lease says "absent".
        monkeypatch.setattr(ct, "_remote_branch_sha", lambda name: None)

        (repo / "fill.txt").write_text("fill\n")
        outcome = ct._git_add_commit(
            "ord_1", date(2026, 5, 17), [str(repo / "fill.txt")]
        )
        assert outcome == ct.PUSH_FAILED
        assert self._bare_ref(bare, "refs/heads/triggers/2026-05-17-777") == stale


# ---------------------------------------------------------------------------
# Run report — Task A3, 2026-09-05: the failure-issue used to tell the reader
# to "check whether any pending order's trigger was hit". These pin the two
# building blocks (the per-order dict, the commit-label vocabulary) and the
# writer that turns them into the JSON/markdown artifacts the workflows read.
# ---------------------------------------------------------------------------


class TestRunReportEntries:
    """`_report_entry` / `_report_commit_label` in isolation from git and I/O."""

    def _order(self, **overrides) -> Order:
        defaults = dict(
            order_id="ord_1",
            ts=datetime(2026, 9, 5, tzinfo=timezone.utc),
            agent_id="satoshi",
            action="SELL",
            ticker="BTC-EUR",
            shares=0.5,
            reasoning="x",
            currency="EUR",
            trigger={"op": ">=", "level": 90000.0},
            expires="2026-10-01",
        )
        defaults.update(overrides)
        return Order(**defaults)

    def test_a_fired_entry_carries_the_fill(self) -> None:
        from scripts import check_triggers as ct
        from engine.orders import Fill

        fill = Fill(
            order_id="ord_1",
            ts_filled=datetime(2026, 9, 5, tzinfo=timezone.utc),
            status="filled",
            fill_price=91000.0,
            fill_currency="EUR",
            notional_base=45500.0,
            fees=1.5,
            reason=None,
            trigger_fired=True,
        )
        entry = ct._report_entry(self._order(), "fired", 91000.0, fill)
        assert entry == {
            "order_id": "ord_1",
            "agent_id": "satoshi",
            "ticker": "BTC-EUR",
            "action": "SELL",
            "shares": 0.5,
            "op": ">=",
            "level": 90000.0,
            "observed_price": 91000.0,
            "fill_price": 91000.0,
            "notional": 45500.0,
            "kind": "fired",
            "error": None,
            "commit": None,
        }

    def test_an_expired_entry_has_no_price_or_fill(self) -> None:
        """The rejection Fill `_process_channel` writes for TRIGGER_EXPIRED
        already carries fill_price=None; the report must not invent one."""
        from scripts import check_triggers as ct
        from engine.orders import Fill

        rejection = Fill(
            order_id="ord_1",
            ts_filled=datetime(2026, 9, 5, tzinfo=timezone.utc),
            status="rejected",
            fill_price=None,
            fill_currency=None,
            notional_base=None,
            fees=None,
            reason="TRIGGER_EXPIRED",
            trigger_fired=True,
        )
        entry = ct._report_entry(self._order(), "expired", None, rejection)
        assert entry["observed_price"] is None
        assert entry["fill_price"] is None
        assert entry["notional"] is None
        assert entry["kind"] == "expired"

    def test_a_zombie_cleanup_has_no_fill_either(self) -> None:
        """`fill=None` is the idempotency-skip case (already filled in a prior
        run) — distinct from a genuine rejection, but reported the same way:
        this run produced no new fill_price/notional either way."""
        from scripts import check_triggers as ct

        entry = ct._report_entry(self._order(), "fired", 91000.0, None)
        assert entry["fill_price"] is None
        assert entry["notional"] is None
        assert entry["observed_price"] == 91000.0  # the price WAS read

    def test_commit_label_maps_main_and_stranded(self) -> None:
        from scripts import check_triggers as ct

        assert ct._report_commit_label(ct.COMMIT_OK) == "main"
        assert ct._report_commit_label(ct.PUSH_FAILED) == "stranded"
        # COMMIT_FAILED collapses into the same label as PUSH_FAILED
        # deliberately — see `_report_commit_label`'s own docstring.
        assert ct._report_commit_label(ct.COMMIT_FAILED) == "stranded"

    def test_commit_label_names_the_fallback_branch(self, monkeypatch) -> None:
        from scripts import check_triggers as ct

        fb = ct._FallbackBranch("triggers/2026-09-05-42")
        fb.pushed = True
        monkeypatch.setattr(ct, "_fallback", fb)
        assert ct._report_commit_label(ct.PUSHED_FALLBACK) == "triggers/2026-09-05-42"


class TestRunReportTable:
    """`_report_markdown_table` — the $GITHUB_STEP_SUMMARY half."""

    def test_no_orders_says_so_rather_than_an_empty_table(self) -> None:
        from scripts import check_triggers as ct

        table = ct._report_markdown_table([])
        assert "No orders fired, expired or failed this run." in table
        assert "|" not in table  # no header row over zero rows

    def test_a_fired_row_renders_every_field(self) -> None:
        from scripts import check_triggers as ct

        entries = [
            {
                "order_id": "ord_1",
                "agent_id": "satoshi",
                "ticker": "BTC-EUR",
                "action": "SELL",
                "shares": 0.5,
                "op": ">=",
                "level": 90000.0,
                "observed_price": 91000.0,
                "fill_price": 91000.0,
                "notional": 45500.0,
                "kind": "fired",
                "commit": "main",
            }
        ]
        table = ct._report_markdown_table(entries)
        for needle in (
            "ord_1",
            "satoshi",
            "BTC-EUR",
            "SELL",
            ">= 90000.0",
            "91000.0",
            "45500.0",
            "main",
        ):
            assert needle in table, needle

    def test_missing_values_render_as_an_em_dash_not_none(self) -> None:
        """An expired order has no price/fill; the reader should see a dash,
        not the string "None" — which reads as a value, not an absence."""
        from scripts import check_triggers as ct

        entries = [
            {
                "order_id": "ord_2",
                "agent_id": "goldfinger",
                "ticker": "AAPL",
                "action": "BUY",
                "shares": 3,
                "op": "<=",
                "level": 150.0,
                "observed_price": None,
                "fill_price": None,
                "notional": None,
                "kind": "expired",
                "commit": "stranded",
            }
        ]
        table = ct._report_markdown_table(entries)
        assert "—" in table
        assert "None" not in table


class TestWriteRunReport:
    """The env-var contract: `WATCHER_REPORT_PATH`, `$RUNNER_TEMP` as its
    default AND as the home of the markdown twin the workflows read back,
    `$GITHUB_STEP_SUMMARY` for this step's own summary tab, and "neither
    set" as a genuine no-op — the case every local pytest run is actually
    exercising."""

    ENTRIES = [
        {
            "order_id": "ord_1",
            "agent_id": "satoshi",
            "ticker": "BTC-EUR",
            "action": "SELL",
            "shares": 0.5,
            "op": ">=",
            "level": 90000.0,
            "observed_price": 91000.0,
            "fill_price": 91000.0,
            "notional": 45500.0,
            "kind": "fired",
            "commit": "main",
        }
    ]
    NOW = datetime(2026, 9, 5, 13, 0, tzinfo=timezone.utc)

    def _clear_env(self, monkeypatch) -> None:
        from scripts import check_triggers as ct

        for name in (ct.REPORT_PATH_ENV, ct.RUNNER_TEMP_ENV, ct.STEP_SUMMARY_ENV):
            monkeypatch.delenv(name, raising=False)

    def test_writes_json_to_the_named_path(self, tmp_path, monkeypatch) -> None:
        from scripts import check_triggers as ct

        self._clear_env(monkeypatch)
        path = tmp_path / "report.json"
        monkeypatch.setenv(ct.REPORT_PATH_ENV, str(path))

        ct.write_run_report(self.ENTRIES, now=self.NOW)

        payload = json.loads(path.read_text())
        assert payload["orders"] == self.ENTRIES
        assert payload["generated_at"] == "2026-09-05T13:00:00Z"

    def test_falls_back_to_a_path_under_runner_temp(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts import check_triggers as ct

        self._clear_env(monkeypatch)
        monkeypatch.setenv(ct.RUNNER_TEMP_ENV, str(tmp_path))

        ct.write_run_report([], now=self.NOW)

        payload = json.loads((tmp_path / "watcher-report.json").read_text())
        assert payload["orders"] == []

    def test_the_named_path_wins_over_the_runner_temp_default(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts import check_triggers as ct

        self._clear_env(monkeypatch)
        monkeypatch.setenv(ct.RUNNER_TEMP_ENV, str(tmp_path))
        named = tmp_path / "elsewhere.json"
        monkeypatch.setenv(ct.REPORT_PATH_ENV, str(named))

        ct.write_run_report(self.ENTRIES, now=self.NOW)

        assert named.exists()
        assert not (tmp_path / "watcher-report.json").exists()

    def test_neither_env_var_set_is_a_genuine_no_op(
        self, tmp_path, monkeypatch
    ) -> None:
        """The falsifiable control: without the guard clause in
        `write_run_report`, every local pytest run of the watcher would write
        a report file into whatever the current directory happened to be."""
        from scripts import check_triggers as ct

        self._clear_env(monkeypatch)
        monkeypatch.chdir(tmp_path)

        ct.write_run_report(self.ENTRIES, now=self.NOW)

        assert list(tmp_path.iterdir()) == []

    def test_writes_the_markdown_twin_under_runner_temp(
        self, tmp_path, monkeypatch
    ) -> None:
        """Regression (round-1 review, 2026-09-05): the table went ONLY to
        `$GITHUB_STEP_SUMMARY`, which is per-step, so the workflow step that
        read it back got an empty file and every issue shipped without the
        table. The reader now reads this file; its basename is pinned to the
        constant the workflows are pinned to in tests/test_ci_guards.py."""
        from scripts import check_triggers as ct

        self._clear_env(monkeypatch)
        monkeypatch.setenv(ct.RUNNER_TEMP_ENV, str(tmp_path))

        ct.write_run_report(self.ENTRIES, now=self.NOW)

        md = tmp_path / ct.REPORT_MD_FILENAME
        assert md.read_text() == ct._report_markdown_table(self.ENTRIES)
        assert "ord_1" in md.read_text()
        # ...and it does not depend on $GITHUB_STEP_SUMMARY being set at all.
        assert (tmp_path / ct.REPORT_JSON_FILENAME).exists()

    def test_the_named_json_path_does_not_move_the_markdown(
        self, tmp_path, monkeypatch
    ) -> None:
        """`WATCHER_REPORT_PATH` overrides the JSON only; the reader step
        looks for the markdown under `$RUNNER_TEMP` regardless."""
        from scripts import check_triggers as ct

        self._clear_env(monkeypatch)
        monkeypatch.setenv(ct.RUNNER_TEMP_ENV, str(tmp_path))
        monkeypatch.setenv(ct.REPORT_PATH_ENV, str(tmp_path / "elsewhere.json"))

        ct.write_run_report(self.ENTRIES, now=self.NOW)

        assert (tmp_path / ct.REPORT_MD_FILENAME).exists()

    def test_appends_a_table_to_the_step_summary_without_overwriting_it(
        self, tmp_path, monkeypatch
    ) -> None:
        """This step's own summary tab. Nothing reads it back (that file is
        per-step), so it is display; it must still append rather than
        truncate whatever this step wrote before it."""
        from scripts import check_triggers as ct

        self._clear_env(monkeypatch)
        summary = tmp_path / "summary.md"
        summary.write_text("### An earlier step's summary\n\n")
        monkeypatch.setenv(ct.STEP_SUMMARY_ENV, str(summary))

        ct.write_run_report(self.ENTRIES, now=self.NOW)

        text = summary.read_text()
        assert text.startswith("### An earlier step's summary")
        assert "ord_1" in text and "satoshi" in text


class TestRunReportThroughMain:
    """The report's `commit` field end-to-end through `main()` — the whole
    point of the feature, not just the label-mapping unit above. Reuses the
    fireable-order fixture shape from `TestFailedPushExitsNonZero`."""

    def _fireable(self, broker_env, monkeypatch):
        from engine import triggers as triggers_mod
        from engine.types import Trade

        order = _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=8000.0, currency="EUR"
        )
        pm.apply_trade(
            "satoshi",
            Trade(
                id="seed_001",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                action="BUY",
                ticker="BTC-EUR",
                shares=0.1,
                price=70000.0,
                total=7000.0,
                fees=0.0,
                reasoning="seed",
            ),
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        return order, pm

    def _run_main(self, monkeypatch, pm, now):
        from datetime import datetime as dt
        from scripts import check_triggers

        monkeypatch.setattr(check_triggers.sys, "argv", ["check_triggers.py"])
        monkeypatch.setattr(check_triggers, "PortfolioManager", lambda base_dir: pm)

        class _Now(dt):
            @classmethod
            def now(cls, tz=None):
                return now

        # main() also asks origin whether `main` has moved under this
        # checkout before evaluating (main_tip_ahead_of_checkout). Stubbed
        # by name here for the same reason as the branch list above; the
        # guard itself is covered by TestRefusesAStaleCheckout.
        monkeypatch.setattr(check_triggers, "main_tip_ahead_of_checkout", lambda: "")
        monkeypatch.setattr(check_triggers, "datetime", _Now)
        return check_triggers.main

    NOW = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)

    def _clear_report_env(self, monkeypatch) -> None:
        from scripts import check_triggers as ct

        for name in (ct.REPORT_PATH_ENV, ct.RUNNER_TEMP_ENV, ct.STEP_SUMMARY_ENV):
            monkeypatch.delenv(name, raising=False)

    def test_a_successful_push_is_labelled_main(
        self, broker_env, monkeypatch, tmp_path
    ) -> None:
        import subprocess as sp
        import types

        from scripts import check_triggers

        order, pm = self._fireable(broker_env, monkeypatch)

        def fake(cmd, *a, **kw):
            if cmd[1] == "diff":
                return sp.CompletedProcess(cmd, 1)
            if cmd[1] == "ls-remote":
                return sp.CompletedProcess(cmd, 0, stdout="")
            return sp.CompletedProcess(cmd, 0)  # every push/merge-base succeeds

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(
                run=fake,
                CompletedProcess=sp.CompletedProcess,
                CalledProcessError=sp.CalledProcessError,
            ),
        )
        self._clear_report_env(monkeypatch)
        report_path = tmp_path / "report.json"
        summary_path = tmp_path / "summary.md"
        monkeypatch.setenv("WATCHER_REPORT_PATH", str(report_path))
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

        main = self._run_main(monkeypatch, pm, self.NOW)
        main()  # must not raise SystemExit

        payload = json.loads(report_path.read_text())
        assert len(payload["orders"]) == 1
        fired = payload["orders"][0]
        assert fired["order_id"] == order.order_id
        assert fired["kind"] == "fired"
        assert fired["commit"] == "main"
        assert fired["fill_price"] == 85123.45
        assert order.order_id in summary_path.read_text()
        assert "main" in summary_path.read_text()

    def test_a_doubly_refused_push_is_labelled_stranded(
        self, broker_env, monkeypatch, tmp_path
    ) -> None:
        """The double-failure case `TestFailedPushExitsNonZero` also covers —
        here the report itself is the thing under test."""
        import subprocess as sp
        import types

        from scripts import check_triggers

        order, pm = self._fireable(broker_env, monkeypatch)

        def fake(cmd, *a, **kw):
            if cmd[1] == "diff":
                return sp.CompletedProcess(cmd, 1)
            if cmd[1] in {"push", "pull"}:
                return sp.CompletedProcess(cmd, 1)
            if cmd[1] == "merge-base":
                return sp.CompletedProcess(
                    cmd, 1
                )  # not an ancestor — genuinely stranded
            if cmd[1] == "ls-remote":
                return sp.CompletedProcess(cmd, 0, stdout="")
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(
                run=fake,
                CompletedProcess=sp.CompletedProcess,
                CalledProcessError=sp.CalledProcessError,
            ),
        )
        self._clear_report_env(monkeypatch)
        report_path = tmp_path / "report.json"
        monkeypatch.setenv("WATCHER_REPORT_PATH", str(report_path))

        main = self._run_main(monkeypatch, pm, self.NOW)
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

        payload = json.loads(report_path.read_text())
        assert payload["orders"][0]["order_id"] == order.order_id
        assert payload["orders"][0]["commit"] == "stranded"

    def test_a_fallback_branch_push_is_labelled_with_the_branch_name(
        self, broker_env, monkeypatch, tmp_path
    ) -> None:
        import subprocess as sp
        import types

        from scripts import check_triggers

        order, pm = self._fireable(broker_env, monkeypatch)

        def fake(cmd, *a, **kw):
            if cmd[1] == "diff":
                return sp.CompletedProcess(cmd, 1)
            if cmd[1] == "push":
                if cmd[-1] == "HEAD:main":
                    return sp.CompletedProcess(cmd, 1)  # main refuses
                return sp.CompletedProcess(cmd, 0)  # the fallback branch does not
            if cmd[1] == "ls-remote":
                return sp.CompletedProcess(cmd, 0, stdout="")
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(
                run=fake,
                CompletedProcess=sp.CompletedProcess,
                CalledProcessError=sp.CalledProcessError,
            ),
        )
        monkeypatch.setenv("GITHUB_RUN_ID", "999")
        monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
        monkeypatch.setattr(check_triggers, "_fallback", None)

        self._clear_report_env(monkeypatch)
        report_path = tmp_path / "report.json"
        monkeypatch.setenv("WATCHER_REPORT_PATH", str(report_path))

        main = self._run_main(monkeypatch, pm, self.NOW)
        main()  # durable fallback — must not raise SystemExit

        payload = json.loads(report_path.read_text())
        assert payload["orders"][0]["order_id"] == order.order_id
        assert payload["orders"][0]["commit"] == "triggers/2026-05-17-999"


# ---------------------------------------------------------------------------
# Round-2 review (2026-09-05)
# ---------------------------------------------------------------------------


class TestSessionStartIsTheOneConstant:
    """`BLACKOUT_START` and the auto-merge deferral are both functions of the
    session start, and they are NOT the same function.

    Regression: the merge deferred over `in_blackout`, whose front edge sits
    five minutes early for the watcher's own multi-minute fire path. A merge is
    one fetch-merge-push, and those five minutes trapped branches — a fire at
    19:54 pushed its fallback branch, the ~19:55 dispatch deferred, the 20:00
    session appended its own fills to the same dated inbox file, and the stale
    check refused that branch permanently while every watcher run refused to
    evaluate behind it.
    """

    def test_the_evaluation_blackout_opens_five_minutes_before_the_session(self):
        from datetime import datetime, timedelta

        from scripts import check_triggers as ct

        start = datetime.combine(date(2026, 9, 5), ct.SESSION_START)
        assert datetime.combine(
            date(2026, 9, 5), ct.BLACKOUT_START
        ) == start - timedelta(minutes=5)

    def test_the_blackout_end_is_after_the_session_start(self):
        from scripts import check_triggers as ct

        assert ct.SESSION_START < ct.BLACKOUT_END

    @pytest.mark.parametrize(
        "hh,mm,evaluation,merge",
        [
            (19, 54, False, False),
            # The trap: the watcher stops evaluating, the merge must NOT stop.
            (19, 55, True, False),
            (19, 59, True, False),
            (20, 0, True, True),
            (21, 0, True, True),
            (21, 1, False, False),
            (13, 0, False, False),
        ],
    )
    def test_the_two_windows_differ_exactly_over_the_lead(
        self, hh, mm, evaluation, merge
    ):
        from scripts import check_triggers as ct

        now = datetime(2026, 9, 5, hh, mm, tzinfo=timezone.utc)
        assert ct.in_blackout(now) is evaluation
        assert ct.merge_deferred(now) is merge


class TestRefusesAStaleCheckout:
    """Regression (round-2 review, 2026-09-05): `unmerged_fallback_branches()`
    asks origin about BRANCHES at process start, but the run evaluates the
    worktree `actions/checkout` produced at job start (`fetch-depth: 1`, never
    refreshed).

    Nothing serialises that against auto-merge-session — its concurrency group
    is per branch, the watchers share `check-triggers` — so a merge that lands
    and DELETES the branch in between leaves a clean origin and a stale
    worktree: the fill absent, the pending file present, and the run about to
    fire the same order again at that moment's price. The rebase conflict on
    `portfolio.json` was the only thing keeping the duplicate off main.
    """

    def _stub(self, monkeypatch, *, remote: str, head: str, ls_rc: int = 0):
        import subprocess as sp
        import types

        from scripts import check_triggers

        def fake(cmd, *a, **kw):
            if cmd[1] == "ls-remote":
                out = f"{remote}\trefs/heads/main\n" if remote else ""
                return sp.CompletedProcess(cmd, ls_rc, stdout=out)
            if cmd[1] == "rev-parse":
                return sp.CompletedProcess(cmd, 0, stdout=head + "\n")
            raise AssertionError(f"unexpected git call: {cmd}")

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(run=fake, CompletedProcess=sp.CompletedProcess),
        )

    def test_a_matching_tip_is_clean(self, monkeypatch) -> None:
        from scripts import check_triggers as ct

        self._stub(monkeypatch, remote="abc123", head="abc123")
        assert ct.main_tip_ahead_of_checkout() == ""

    def test_a_moved_tip_is_named(self, monkeypatch) -> None:
        from scripts import check_triggers as ct

        self._stub(monkeypatch, remote="newsha", head="oldsha")
        assert ct.main_tip_ahead_of_checkout() == "newsha"

    @pytest.mark.parametrize(
        "remote,ls_rc", [("abc123", 128), ("", 0)], ids=["ls-remote-fails", "no-main"]
    )
    def test_an_unreadable_origin_fails_closed(
        self, monkeypatch, remote, ls_rc
    ) -> None:
        """ "Could not read origin" is unknown, not clean — same posture as
        `unmerged_fallback_branches`."""
        from scripts import check_triggers as ct

        self._stub(monkeypatch, remote=remote, head="abc123", ls_rc=ls_rc)
        assert ct.main_tip_ahead_of_checkout() is None

    def test_it_never_fetches(self, monkeypatch) -> None:
        """One `ls-remote`, no fetch: `git fetch origin main` into a depth-1
        checkout deepens it, and this repo's history is gigabytes. The stub
        above raises on any other git call, so this is the assertion."""
        from scripts import check_triggers as ct

        self._stub(monkeypatch, remote="abc123", head="abc123")
        assert ct.main_tip_ahead_of_checkout() == ""

    def test_main_refuses_to_evaluate_behind_a_moved_tip(
        self, broker_env, monkeypatch
    ) -> None:
        from scripts import check_triggers

        order, pm = TestRefusedPushFallsBack()._fireable(broker_env, monkeypatch)
        TestRefusedPushFallsBack()._fake_git(monkeypatch)
        monkeypatch.setattr(check_triggers, "unmerged_fallback_branches", lambda: [])
        main, _captured = TestRefusedPushFallsBack()._run_main(
            monkeypatch, pm, TestRefusedPushFallsBack.NOW
        )
        # AFTER _run_main: that helper stubs this guard clean for every other
        # test in the file, and this one is about the guard firing.
        monkeypatch.setattr(
            check_triggers, "main_tip_ahead_of_checkout", lambda: "newsha"
        )

        def never(*a, **kw):
            raise AssertionError("run() must not be reached behind a moved tip")

        monkeypatch.setattr(check_triggers, "run", never)
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
        # Nothing was evaluated, so nothing moved.
        assert [o.order_id for o in list_pending()] == [order.order_id]
        assert read_inbox(TestRefusedPushFallsBack.NOW.date()) == []

    def test_a_dry_run_reports_a_moved_tip_without_refusing(
        self, broker_env, monkeypatch, caplog
    ) -> None:
        """A dry run writes nothing, so it cannot re-fire anything."""
        import logging

        from scripts import check_triggers

        _order, pm = TestRefusedPushFallsBack()._fireable(broker_env, monkeypatch)
        TestRefusedPushFallsBack()._fake_git(monkeypatch)
        monkeypatch.setattr(check_triggers, "unmerged_fallback_branches", lambda: [])
        main, captured = TestRefusedPushFallsBack()._run_main(
            monkeypatch,
            pm,
            TestRefusedPushFallsBack.NOW,
            ("check_triggers.py", "--dry-run"),
        )
        monkeypatch.setattr(  # after _run_main — see the sibling test
            check_triggers, "main_tip_ahead_of_checkout", lambda: "newsha"
        )
        with caplog.at_level(logging.WARNING, logger="scripts.check_triggers"):
            main()  # no SystemExit
        assert "newsha" in caplog.text
        assert captured["summary"]["fired"] == 1


class TestABrokerCrashReachesTheReport:
    """Regression (round-2 review, 2026-09-05): `execute_triggered_order`
    raising on a FIRED trigger incremented `summary["errors"]` and appended no
    report entry at all.

    A run whose only event was that crash therefore wrote
    `_report_markdown_table([])` — "No orders fired or expired this run." — to
    `$RUNNER_TEMP/watcher-report.md`, and the workflow handed that to
    `.github/actions/failure-issue` as `details` under a body naming the broker
    crash. The reader of the one alert channel this whole change exists to make
    trustworthy was told the run was a no-op.
    """

    def _order(self) -> Order:
        return Order(
            order_id="ord_1",
            ts=datetime(2026, 9, 5, tzinfo=timezone.utc),
            agent_id="satoshi",
            action="BUY",
            ticker="BTC-EUR",
            shares=0.5,
            reasoning="x",
            currency="EUR",
            trigger={"op": ">=", "level": 90000.0},
            expires="2026-10-01",
        )

    def test_an_error_entry_names_the_exception_and_no_commit(self) -> None:
        from scripts import check_triggers as ct

        entry = ct._report_entry(
            self._order(), "error", 91000.0, None, error="ValueError: NO_FX_RATE"
        )
        assert entry["kind"] == "error"
        assert entry["error"] == "ValueError: NO_FX_RATE"
        # The price WAS read — the trigger was evaluated against it.
        assert entry["observed_price"] == 91000.0
        assert entry["fill_price"] is None and entry["notional"] is None
        # Nothing was written, so nothing is stranded: a distinct fact.
        assert entry["commit"] == ct.REPORT_COMMIT_NONE
        assert entry["commit"] != ct.REPORT_COMMIT_STRANDED

    def test_the_table_contradicts_nothing(self) -> None:
        from scripts import check_triggers as ct

        table = ct._report_markdown_table(
            [
                ct._report_entry(
                    self._order(),
                    "error",
                    91000.0,
                    None,
                    error="ValueError: NO_FX_RATE",
                )
            ]
        )
        assert "No orders fired" not in table
        assert "ord_1" in table and "error" in table
        assert "NO_FX_RATE" in table, table

    def test_a_raising_broker_produces_an_entry(self, broker_env, monkeypatch) -> None:
        """End to end through `_process_channel`, which is where the entry was
        missing."""
        from engine import triggers as triggers_mod
        from scripts import check_triggers

        _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 95_000.0
        )

        def boom(*a, **kw):
            raise RuntimeError("NO_PRICE_DATA")

        monkeypatch.setattr(check_triggers, "execute_triggered_order", boom)
        summary = check_triggers.run(
            now=datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc),
            portfolio_manager=pm,
        )
        assert summary["errors"] == 1
        assert summary["fired"] == 0
        errors = [e for e in summary["report"] if e["kind"] == "error"]
        assert len(errors) == 1, summary["report"]
        assert "NO_PRICE_DATA" in errors[0]["error"]
        table = check_triggers._report_markdown_table(summary["report"])
        assert "No orders fired" not in table
