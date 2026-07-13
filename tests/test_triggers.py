"""Tests for engine.triggers — pending I/O, cancel I/O, evaluation, expiry."""

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.orders import Order
from engine.triggers import (
    CancelRequest,
    append_cancel,
    delete_pending,
    list_pending,
    read_cancels,
    save_pending,
)


def _make_order(order_id: str = "ord_2026-05-17_satoshi_001", **overrides) -> Order:
    defaults = dict(
        order_id=order_id,
        ts=datetime(2026, 5, 17, 20, 2, 0, tzinfo=timezone.utc),
        agent_id="satoshi",
        action="SELL",
        ticker="BTC-EUR",
        shares=0.01,
        reasoning="trim at resistance",
        currency="EUR",
        trigger={"op": ">=", "level": 85000.0},
        expires="2026-06-17",
    )
    defaults.update(overrides)
    return Order(**defaults)


class TestPendingStorage:
    def test_save_and_list(self, tmp_path: Path) -> None:
        o = _make_order()
        save_pending(o, pending_dir=tmp_path)
        pending = list_pending(pending_dir=tmp_path)
        assert len(pending) == 1
        assert pending[0].order_id == o.order_id
        assert pending[0].trigger == {"op": ">=", "level": 85000.0}

    def test_save_overwrites_same_id(self, tmp_path: Path) -> None:
        save_pending(
            _make_order(trigger={"op": ">=", "level": 85000.0}), pending_dir=tmp_path
        )
        save_pending(
            _make_order(trigger={"op": ">=", "level": 90000.0}), pending_dir=tmp_path
        )  # same id
        pending = list_pending(pending_dir=tmp_path)
        assert len(pending) == 1
        assert pending[0].trigger["level"] == 90000.0

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        o = _make_order()
        save_pending(o, pending_dir=tmp_path)
        assert delete_pending(o.order_id, pending_dir=tmp_path) is True
        assert list_pending(pending_dir=tmp_path) == []

    def test_delete_missing_returns_false(self, tmp_path: Path) -> None:
        assert delete_pending("ord_does_not_exist", pending_dir=tmp_path) is False

    def test_list_empty_when_dir_empty(self, tmp_path: Path) -> None:
        assert list_pending(pending_dir=tmp_path) == []

    def test_list_empty_when_dir_missing(self, tmp_path: Path) -> None:
        assert list_pending(pending_dir=tmp_path / "nonexistent") == []

    def test_list_skips_gitkeep_and_non_json(self, tmp_path: Path) -> None:
        (tmp_path / ".gitkeep").write_text("")
        (tmp_path / "README.md").write_text("not a pending order")
        save_pending(_make_order(), pending_dir=tmp_path)
        assert len(list_pending(pending_dir=tmp_path)) == 1

    def test_save_requires_trigger(self, tmp_path: Path) -> None:
        market_order = _make_order(trigger=None, expires=None)
        with pytest.raises(ValueError, match="save_pending requires order.trigger"):
            save_pending(market_order, pending_dir=tmp_path)


class TestCancelStorage:
    def test_append_and_read(self, tmp_path: Path) -> None:
        d = date(2026, 5, 17)
        c = CancelRequest(
            request_id="cnl_2026-05-17_satoshi_001",
            ts=datetime(2026, 5, 17, 20, 5, tzinfo=timezone.utc),
            agent_id="satoshi",
            target_order_id="ord_2026-05-10_satoshi_003",
            reasoning="thesis changed, no longer want to sell at 85k",
        )
        append_cancel(d, c, cancels_dir=tmp_path)
        back = read_cancels(d, cancels_dir=tmp_path)
        assert len(back) == 1
        assert back[0].target_order_id == "ord_2026-05-10_satoshi_003"
        assert back[0].ts.tzinfo is not None

    def test_empty_when_missing(self, tmp_path: Path) -> None:
        assert read_cancels(date(2026, 5, 17), cancels_dir=tmp_path) == []

    def test_read_cancels_skips_malformed_lines(self, tmp_path: Path) -> None:
        d = date(2026, 5, 17)
        # Mix: one good line, one truncated JSON, one good line.
        path = tmp_path / f"{d.isoformat()}.jsonl"
        path.write_text(
            '{"request_id":"cnl_a","ts":"2026-05-17T20:05:00Z","agent_id":"satoshi",'
            '"target_order_id":"ord_a","reasoning":"ok"}\n'
            '{"request_id":"cnl_b","ts":"2026-05-17T20:06:0\n'  # truncated
            '{"request_id":"cnl_c","ts":"2026-05-17T20:07:00Z","agent_id":"satoshi",'
            '"target_order_id":"ord_c","reasoning":"ok"}\n'
        )
        cancels = read_cancels(d, cancels_dir=tmp_path)
        assert [c.request_id for c in cancels] == ["cnl_a", "cnl_c"]


from engine.triggers import evaluate_trigger, is_expired


class TestEvaluateTrigger:
    def test_gte_fires_when_price_above_level(self) -> None:
        assert evaluate_trigger(85100.0, {"op": ">=", "level": 85000.0}) is True

    def test_gte_fires_at_exact_level(self) -> None:
        assert evaluate_trigger(85000.0, {"op": ">=", "level": 85000.0}) is True

    def test_gte_does_not_fire_when_below(self) -> None:
        assert evaluate_trigger(84999.99, {"op": ">=", "level": 85000.0}) is False

    def test_lte_fires_when_price_below_level(self) -> None:
        assert evaluate_trigger(2999.0, {"op": "<=", "level": 3000.0}) is True

    def test_lte_fires_at_exact_level(self) -> None:
        assert evaluate_trigger(3000.0, {"op": "<=", "level": 3000.0}) is True

    def test_lte_does_not_fire_when_above(self) -> None:
        assert evaluate_trigger(3000.01, {"op": "<=", "level": 3000.0}) is False

    def test_unknown_op_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown trigger op"):
            evaluate_trigger(100.0, {"op": "between", "level": 50.0})


class TestIsExpired:
    def test_expires_in_future_is_not_expired(self) -> None:
        o = _make_order(expires="2026-06-17")
        assert is_expired(o, today=date(2026, 5, 17)) is False

    def test_expires_today_is_expired(self) -> None:
        """Inclusive: on the expiry date, the order is considered expired."""
        o = _make_order(expires="2026-05-17")
        assert is_expired(o, today=date(2026, 5, 17)) is True

    def test_expires_in_past_is_expired(self) -> None:
        o = _make_order(expires="2026-04-01")
        assert is_expired(o, today=date(2026, 5, 17)) is True

    def test_no_expiry_never_expires(self) -> None:
        """Defensive: an order with no expires set shouldn't expire. (Authoring step enforces expiry, but be safe.)"""
        o = _make_order(trigger=None, expires=None)
        assert is_expired(o, today=date(2030, 1, 1)) is False


from unittest.mock import MagicMock

from engine.triggers import get_current_price, is_crypto_ticker


class TestIsCryptoTicker:
    @pytest.mark.parametrize("t", ["BTC-EUR", "ETH-USD", "SOL-EUR", "DOGE-USD"])
    def test_known_crypto(self, t: str) -> None:
        assert is_crypto_ticker(t) is True

    @pytest.mark.parametrize(
        "t", ["MSFT", "AAPL", "SPY", "EURUSD=X", "GLD", "TLT", "AAPL.PA", ""]
    )
    def test_non_crypto(self, t: str) -> None:
        assert is_crypto_ticker(t) is False

    def test_unknown_base_is_not_crypto(self) -> None:
        # Looks like crypto-shape, but PEPE isn't in the allowlist.
        assert is_crypto_ticker("PEPE-EUR") is False


class TestChannelScopedPending:
    """Channel-scoped pending dir: manager orders must not leak into global channel."""

    def test_pending_roundtrip_manager_dir(self, tmp_path: Path) -> None:
        from engine import triggers

        manager_dir = tmp_path / "manager-pending"
        o = _make_order(order_id="ord_2026-06-27_manager_001")

        # Save to manager dir
        save_pending(o, pending_dir=manager_dir)

        # File must exist under manager_dir
        assert (manager_dir / f"{o.order_id}.json").exists()

        # Global PENDING_DIR must NOT contain the file
        assert not (triggers.PENDING_DIR / f"{o.order_id}.json").exists()

        # list_pending with manager dir must return it
        listed = list_pending(pending_dir=manager_dir)
        assert len(listed) == 1
        assert listed[0].order_id == o.order_id

        # Global list_pending() must NOT see it
        global_listed = [x for x in list_pending() if x.order_id == o.order_id]
        assert global_listed == []

        # delete_pending with manager dir must return True and remove
        assert delete_pending(o.order_id, pending_dir=manager_dir) is True
        assert not (manager_dir / f"{o.order_id}.json").exists()

    def test_cancels_roundtrip_manager_dir(self, tmp_path: Path) -> None:
        from engine import triggers

        manager_dir = tmp_path / "manager-cancels"
        d = date(2026, 6, 27)
        c = CancelRequest(
            request_id="cnl_2026-06-27_manager_001",
            ts=datetime(2026, 6, 27, 20, 5, tzinfo=timezone.utc),
            agent_id="manager",
            target_order_id="ord_2026-06-20_manager_003",
            reasoning="strategy rebalance",
        )

        # Append to manager cancels dir
        append_cancel(d, c, cancels_dir=manager_dir)

        # Read back from manager dir must return it
        back = read_cancels(d, cancels_dir=manager_dir)
        assert len(back) == 1
        assert back[0].request_id == c.request_id

        # Global read_cancels must NOT see it
        global_back = [x for x in read_cancels(d) if x.request_id == c.request_id]
        assert global_back == []


class TestGetCurrentPrice:
    def test_equity_falls_through_to_ohlcv(self, monkeypatch) -> None:
        from engine import triggers

        called = {}

        def fake_latest(ticker, on, store=None):
            called["args"] = (ticker, on)
            return 420.42

        monkeypatch.setattr(triggers, "latest_close_on_or_before", fake_latest)
        price = get_current_price("MSFT", today=date(2026, 5, 17))
        assert price == 420.42
        assert called["args"] == ("MSFT", date(2026, 5, 17))

    def test_equity_missing_returns_none(self, monkeypatch) -> None:
        from engine import triggers

        monkeypatch.setattr(triggers, "latest_close_on_or_before", lambda *a, **k: None)
        assert get_current_price("MSFT", today=date(2026, 5, 17)) is None

    def test_crypto_uses_ccxt(self, monkeypatch) -> None:
        from engine import triggers

        fake_exchange = MagicMock()
        fake_exchange.fetch_ticker.return_value = {"last": 85123.45}
        monkeypatch.setattr(triggers, "_get_crypto_exchange", lambda: fake_exchange)
        price = get_current_price("BTC-EUR", today=date(2026, 5, 17))
        assert price == 85123.45
        fake_exchange.fetch_ticker.assert_called_once_with("BTC/EUR")

    def test_crypto_exchange_exception_returns_none(self, monkeypatch) -> None:
        from engine import triggers

        fake_exchange = MagicMock()
        fake_exchange.fetch_ticker.side_effect = Exception("network down")
        monkeypatch.setattr(triggers, "_get_crypto_exchange", lambda: fake_exchange)
        assert get_current_price("BTC-EUR", today=date(2026, 5, 17)) is None

    def test_crypto_missing_last_returns_none(self, monkeypatch) -> None:
        from engine import triggers

        fake_exchange = MagicMock()
        fake_exchange.fetch_ticker.return_value = {}  # no 'last' key
        monkeypatch.setattr(triggers, "_get_crypto_exchange", lambda: fake_exchange)
        assert get_current_price("BTC-EUR", today=date(2026, 5, 17)) is None
