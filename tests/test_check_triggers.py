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

    def _fake_git(self, monkeypatch, *, push_rc: int, is_ancestor_rc: int | None = None):
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

    def test_a_self_healed_push_does_not_cry_wolf(self, broker_env, monkeypatch) -> None:
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
                return sp.CompletedProcess(cmd, 0, stdout=" M data/orders/inbox/x.jsonl\n")
            return sp.CompletedProcess(cmd, 0)

        monkeypatch.setattr(
            check_triggers,
            "subprocess",
            types.SimpleNamespace(run=fake, CompletedProcess=sp.CompletedProcess),
        )
        assert check_triggers._nothing_is_stranded() is False
