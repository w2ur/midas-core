"""Pending-order paths and cancel ownership: agent-authored ids never name a file
outside the pending folder, and one agent cannot cancel another's order.

Regression: fixed in the commit that adds this file (midas-core 0.1.2). Under
0.1.1 a cancel whose ``target_order_id`` was ``../../portfolios/<agent>/portfolio``
unlinked that agent's portfolio file, and any agent could cancel any other
agent's armed order by id. Each guard here was seen failing with it removed.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone

import pytest

from engine.config import get_config
from engine.orders import (
    Order,
    append_order,
    inbox_order_ids,
    is_valid_order_id,
    read_inbox,
)
from engine.paper_broker import REJECTION_REASON_CODES, fill_day
from engine.triggers import (
    CancelRequest,
    append_cancel,
    delete_pending,
    list_pending,
    save_pending,
)
from tests.test_paper_broker import _init_portfolio, _write_config

TRADE_DATE = date(2026, 5, 17)


def _conditional(order_id: str, agent_id: str = "satoshi") -> Order:
    return Order(
        order_id=order_id,
        ts=datetime(2026, 5, 10, 20, 2, tzinfo=timezone.utc),
        agent_id=agent_id,
        action="SELL",
        ticker="BTC-EUR",
        shares=0.01,
        reasoning="trim",
        currency="EUR",
        trigger={"op": ">=", "level": 85000.0},
        expires="2026-06-10",
    )


def _cancel(request_id: str, agent_id: str, target: str) -> CancelRequest:
    return CancelRequest(
        request_id=request_id,
        ts=datetime(2026, 5, 17, 20, 5, tzinfo=timezone.utc),
        agent_id=agent_id,
        target_order_id=target,
        reasoning="test",
    )


@pytest.fixture
def desk(midas_data_root):
    cfg = get_config()
    pm_base = midas_data_root / "portfolios"
    pm_base.mkdir()
    _write_config(cfg.agent_config_dir, "satoshi")
    _write_config(cfg.agent_config_dir, "other")
    pm = _init_portfolio(pm_base, "satoshi", cash=10_000.0, currency="EUR")
    _init_portfolio(pm_base, "other", cash=10_000.0, currency="EUR")
    pending = cfg.orders_dir / "pending"
    pending.mkdir(parents=True)
    portfolio_file = pm_base / "satoshi" / "portfolio.json"
    assert portfolio_file.is_file()
    traversal = os.path.relpath(pm_base / "satoshi" / "portfolio", pending)
    assert traversal.startswith("..")
    return {"pm": pm, "pending": pending, "portfolio_file": portfolio_file, "traversal": traversal}


@pytest.mark.parametrize(
    "order_id",
    ["ord_2026-05-10_satoshi_003", "ord_2026-07-30_001", "ord_x", "ord_demo_001", "o1"],
)
def test_ids_in_the_minted_alphabet_are_valid(order_id):
    assert is_valid_order_id(order_id)


@pytest.mark.parametrize(
    "order_id",
    [
        "../../portfolios/satoshi/portfolio",
        "ord_../x",
        "ord_a/b",
        "ord_a\\b",
        "ord_x\n",
        "ord_x.json",
        "_ord",
        "ord_" + "a" * 200,
        "",
        "..",
        "ord x",
        None,
        7,
    ],
)
def test_ids_that_could_name_a_path_are_invalid(order_id):
    assert not is_valid_order_id(order_id)


def test_delete_pending_refuses_a_path_outside_its_folder(desk):
    with pytest.raises(ValueError, match="escapes"):
        delete_pending(desk["traversal"], pending_dir=desk["pending"])
    assert desk["portfolio_file"].is_file()


def test_save_pending_refuses_a_path_outside_its_folder(desk):
    escape = _conditional("../escaped")
    with pytest.raises(ValueError, match="escapes"):
        save_pending(escape, pending_dir=desk["pending"])
    assert not (desk["pending"].parent / "escaped.json").exists()


def test_delete_pending_refuses_a_symlink_pointing_out_of_its_folder(desk):
    (desk["pending"] / "ord_link.json").symlink_to(desk["portfolio_file"])
    with pytest.raises(ValueError, match="escapes"):
        delete_pending("ord_link", pending_dir=desk["pending"])
    assert desk["portfolio_file"].is_file()


def test_fill_day_rejects_a_traversal_cancel_and_leaves_the_portfolio(desk):
    append_cancel(
        TRADE_DATE, _cancel("cnl_2026-05-17_satoshi_001", "satoshi", desk["traversal"])
    )

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert desk["portfolio_file"].is_file()
    assert [f.reason for f in fills] == ["CANCEL_TARGET_NOT_FOUND"]
    # Filed under the request id: a junk target never enters the ledger.
    assert [f.order_id for f in fills] == ["cnl_2026-05-17_satoshi_001"]


def test_fill_day_rejects_a_cancel_whose_target_is_a_planted_symlink(desk):
    (desk["pending"] / "ord_link.json").symlink_to(desk["portfolio_file"])
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_satoshi_001", "satoshi", "ord_link"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert desk["portfolio_file"].is_file()
    assert [f.reason for f in fills] == ["CANCEL_TARGET_NOT_FOUND"]


def test_an_outbox_order_with_a_traversal_id_is_refused_and_never_armed(desk):
    append_order(TRADE_DATE, _conditional("ord_../../escaped"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert [f.reason for f in fills] == ["INVALID_ORDER_ID"]
    assert list_pending(pending_dir=desk["pending"]) == []
    assert not list(desk["pending"].parent.parent.glob("escaped*"))


def test_cancel_by_another_agent_is_rejected_not_owner(desk):
    save_pending(_conditional("ord_2026-05-10_satoshi_003"), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_other_001", "other", "ord_2026-05-10_satoshi_003"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert [f.reason for f in fills] == ["CANCEL_NOT_OWNER"]
    assert [o.order_id for o in list_pending(pending_dir=desk["pending"])] == [
        "ord_2026-05-10_satoshi_003"
    ]
    # The refusal must not mark the TARGET as processed: the watcher skips any
    # order id already in the inbox, so filing it there would disarm the order.
    assert "ord_2026-05-10_satoshi_003" not in inbox_order_ids(None)
    assert read_inbox(TRADE_DATE)[0].order_id == "cnl_2026-05-17_other_001"


def test_the_owner_can_still_cancel_its_own_order(desk):
    save_pending(_conditional("ord_2026-05-10_satoshi_003"), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_satoshi_001", "satoshi", "ord_2026-05-10_satoshi_003"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert [f.reason for f in fills] == ["CANCELLED_BY_AGENT"]
    assert list_pending(pending_dir=desk["pending"]) == []


def test_a_refused_cancel_is_not_replayed_on_a_second_run(desk):
    save_pending(_conditional("ord_2026-05-10_satoshi_003"), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_other_001", "other", "ord_2026-05-10_satoshi_003"))

    fill_day(TRADE_DATE, desk["pm"])
    fill_day(TRADE_DATE, desk["pm"])

    assert len(read_inbox(TRADE_DATE)) == 1


def test_the_new_reason_codes_are_declared():
    assert {"CANCEL_NOT_OWNER", "INVALID_ORDER_ID"} <= set(REJECTION_REASON_CODES)


def test_the_portfolio_file_is_real_json(desk):
    """Control for the assertions above: the file they protect is a real one."""
    assert json.loads(desk["portfolio_file"].read_text(encoding="utf-8"))
