"""Identity fields in agent-writable files are untrusted: cancels, ids, agent ids.

Regression: see the ``Regression:`` line above each test for the commit that fixed
it. Each guard was seen failing against the code before that commit.
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timezone

import pytest

from engine import triggers
from engine.orders import (
    Order,
    append_order,
    inbox_order_ids,
    is_valid_agent_id,
    is_valid_id,
    is_valid_order_id,
    is_valid_request_id,
    read_inbox,
)
from engine.paper_broker import REJECTION_REASON_CODES, execute_triggered_order, fill_day
from engine.triggers import CancelRequest, append_cancel, list_pending, save_pending
from scripts.daily_session import step_author_cancels
from tests.test_pending_path_safety import (  # noqa: F401  (desk is a fixture)
    TRADE_DATE,
    _cancel,
    _conditional,
    desk,
)

FIRE_DAY = date(2026, 5, 20)


def _inbox_rows(trade_date=TRADE_DATE):
    return read_inbox(trade_date)


def _market(order_id: str, agent_id: str = "satoshi", **kw) -> Order:
    return Order(
        order_id=order_id,
        ts=datetime(2026, 5, 17, 20, tzinfo=timezone.utc),
        agent_id=agent_id,
        action="BUY",
        ticker="BTC-EUR",
        shares=0.0001,
        reasoning="r",
        currency="EUR",
        **kw,
    )


def _fires(order: Order, desk):
    return execute_triggered_order(
        order, FIRE_DAY, desk["pm"], 90000.0, fire_as_of=FIRE_DAY
    )


# Regression: a255a2cc8 — a cancel that found nothing was filed under the TARGET
# order id; ids are predictable, so agent A could pre-file a row under agent B's
# future id and B's stop-loss was later skipped as "already processed".
def test_a_prefiled_not_found_cancel_does_not_disarm_a_future_order(desk):
    victim_id = "ord_2026-05-17_satoshi_001"
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_other_001", "other", victim_id))
    append_order(TRADE_DATE, _conditional(victim_id))

    fill_day(TRADE_DATE, desk["pm"])

    assert [o.order_id for o in list_pending(pending_dir=desk["pending"])] == [victim_id]
    assert victim_id not in inbox_order_ids(None)
    assert _fires(_conditional(victim_id), desk) is not None


@pytest.mark.parametrize("outcome", ["not_found", "not_owner", "junk_target"])
def test_every_refused_cancel_is_filed_under_its_request_id(desk, outcome):
    # Regression: a255a2cc8
    target = {
        "not_found": "ord_2026-05-17_satoshi_001",
        "not_owner": "ord_2026-05-10_satoshi_003",
        "junk_target": "../../x",
    }[outcome]
    save_pending(_conditional("ord_2026-05-10_satoshi_003"), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_other_001", "other", target))

    fill_day(TRADE_DATE, desk["pm"])

    ids = inbox_order_ids(TRADE_DATE)
    assert "cnl_2026-05-17_other_001" in ids
    assert target not in ids


def test_a_successful_cancel_is_still_filed_under_the_target(desk):
    victim_id = "ord_2026-05-10_satoshi_003"
    save_pending(_conditional(victim_id), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_satoshi_001", "satoshi", victim_id))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert [(f.order_id, f.reason) for f in fills] == [(victim_id, "CANCELLED_BY_AGENT")]
    assert fill_day(TRADE_DATE, desk["pm"]) == []  # a resume files nothing twice


def test_a_refused_cancel_is_not_refiled_on_resume(desk):
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_other_001", "other", "ord_gone"))
    assert len(fill_day(TRADE_DATE, desk["pm"])) == 1
    assert fill_day(TRADE_DATE, desk["pm"]) == []


# Regression: a255a2cc8 — a malformed outbox line carrying a foreign order_id filed
# an INVALID_SHARES row under that id, the same pre-filing vector as a cancel.
def test_a_malformed_outbox_line_never_files_under_its_claimed_order_id(desk):
    victim_id = "ord_2026-05-17_satoshi_001"
    bad = _market(victim_id, agent_id="other").to_dict()
    bad["shares"] = 0
    append_raw_outbox(bad)

    fill_day(TRADE_DATE, desk["pm"])

    assert victim_id not in inbox_order_ids(TRADE_DATE)
    assert [r.reason for r in _inbox_rows()] == ["INVALID_SHARES"]


def append_raw_outbox(raw: dict) -> None:
    from engine import orders

    path = orders.OUTBOX_DIR / f"{TRADE_DATE.isoformat()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(raw) + "\n")


# Regression: 520aed90d — `target in already_processed` raised TypeError for a dict
# or list target before the shape check ran, so one malformed cancel crashed
# fill_day for every agent, on every resume.
@pytest.mark.parametrize("target", [{"a": 1}, ["x"], 7, None])
def test_a_non_string_cancel_target_is_refused_without_crashing_the_batch(desk, target):
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_other_001", "other", target))
    append_order(TRADE_DATE, _market("ord_2026-05-17_satoshi_002"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    by_id = {f.order_id: f for f in fills}
    assert by_id["cnl_2026-05-17_other_001"].reason == "CANCEL_TARGET_NOT_FOUND"
    assert "ord_2026-05-17_satoshi_002" in by_id  # the other agent's order still ran
    assert fill_day(TRADE_DATE, desk["pm"]) == []  # and a resume is clean too


@pytest.mark.parametrize("target", [{"a": 1}, ["x"], 7, None])
def test_cancel_request_from_dict_never_carries_a_non_string_target(target):
    # Regression: 520aed90d
    cancel = CancelRequest.from_dict(
        {
            "request_id": "cnl_1",
            "ts": "2026-05-17T20:05:00Z",
            "agent_id": "other",
            "target_order_id": target,
            "reasoning": "x",
        }
    )
    assert isinstance(cancel.target_order_id, str)
    assert not is_valid_order_id(cancel.target_order_id)


@pytest.mark.parametrize("target", [{"a": 1}, ["x"], 7, None, "../x"])
def test_step_author_cancels_writes_a_refusable_row_for_a_bad_target(
    midas_data_root, target
):
    # Regression: 520aed90d — the author step indexed c["target_order_id"] blindly.
    n = step_author_cancels("satoshi", [{"target_order_id": target}], TRADE_DATE)
    assert n == 1
    (cancel,) = triggers.read_cancels(TRADE_DATE)
    assert isinstance(cancel.target_order_id, str)


def test_step_author_cancels_survives_a_missing_target_key(midas_data_root):
    # Regression: 520aed90d
    assert step_author_cancels("satoshi", [{}, "junk"], TRADE_DATE) == 2


# Regression: 269dcab68 — Order.agent_id became a path component of
# PortfolioManager with no check, so "../elsewhere" traded a book outside
# portfolios/.
def test_an_agent_id_that_names_a_path_is_refused_and_no_book_is_touched(
    desk, midas_data_root, monkeypatch
):
    shutil.copytree(midas_data_root / "portfolios" / "other", midas_data_root / "elsewhere")
    book = midas_data_root / "elsewhere" / "portfolio.json"
    before = book.read_bytes()
    touched: list[str] = []
    real = desk["pm"]._read_json
    monkeypatch.setattr(
        desk["pm"], "_read_json", lambda p: (touched.append(str(p)), real(p))[1]
    )
    append_order(TRADE_DATE, _market("ord_x_001", agent_id="../elsewhere"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert [f.reason for f in fills] == ["INVALID_AGENT_ID"]
    assert not [p for p in touched if "elsewhere" in p]
    assert book.read_bytes() == before
    assert (midas_data_root / "elsewhere" / "trades.json").read_text().strip() == "[]"


def test_a_conditional_order_with_a_path_agent_id_is_never_armed(desk):
    # Regression: 269dcab68
    append_order(TRADE_DATE, _conditional("ord_x_002", agent_id="../elsewhere"))
    fills = fill_day(TRADE_DATE, desk["pm"])
    assert [f.reason for f in fills] == ["INVALID_AGENT_ID"]
    assert list_pending(pending_dir=desk["pending"]) == []


def test_an_invalid_agent_id_row_is_not_filed_under_the_claimed_order_id(desk):
    # Regression: 269dcab68 — the order id is as untrusted as the agent id beside it.
    append_order(TRADE_DATE, _market("ord_2026-05-17_satoshi_001", agent_id="a/b"))
    fill_day(TRADE_DATE, desk["pm"])
    assert "ord_2026-05-17_satoshi_001" not in inbox_order_ids(TRADE_DATE)


def test_a_cancel_whose_request_id_is_the_victims_order_id_does_not_disarm_it(desk):
    # Regression: 269dcab68 — request_id was trusted, so a driver that let the agent
    # author it could file a row under the victim's id and skip the order forever.
    victim_id = "ord_2026-05-10_satoshi_003"
    save_pending(_conditional(victim_id), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel(victim_id, "other", victim_id))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert [f.reason for f in fills] == ["INVALID_REQUEST_ID"]
    assert victim_id not in inbox_order_ids(None)
    assert [o.order_id for o in list_pending(pending_dir=desk["pending"])] == [victim_id]
    assert _fires(_conditional(victim_id), desk) is not None


@pytest.mark.parametrize("agent_id", ["../x", "", "a b", None, 7, "x" * 200])
def test_a_cancel_with_a_bad_agent_id_is_refused_under_a_derived_id(desk, agent_id):
    # Regression: 269dcab68 (refused, not crashed); filed under a derived id since
    # the identity-trust follow-up, so the README's claim holds for this code too.
    victim_id = "ord_2026-05-10_satoshi_003"
    save_pending(_conditional(victim_id), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_x_001", agent_id, victim_id))

    fills = fill_day(TRADE_DATE, desk["pm"])

    (row,) = fills
    assert row.reason == "INVALID_AGENT_ID"
    assert row.order_id != "cnl_2026-05-17_x_001"
    assert is_valid_request_id(row.order_id)
    assert [o.order_id for o in list_pending(pending_dir=desk["pending"])] == [victim_id]
    assert fill_day(TRADE_DATE, desk["pm"]) == []  # derived id keeps resume idempotent


def test_request_ids_and_order_ids_live_in_disjoint_namespaces():
    # Regression: 269dcab68
    assert is_valid_request_id("cnl_2026-05-17_satoshi_001")
    assert not is_valid_request_id("ord_2026-05-17_satoshi_001")
    assert not is_valid_request_id("../cnl_x")
    assert not is_valid_order_id("cnl_2026-05-17_satoshi_001")
    assert is_valid_agent_id("the-manager") and not is_valid_agent_id("../x")


def test_the_new_reason_codes_are_declared():
    # Regression: 269dcab68
    assert {"INVALID_AGENT_ID", "INVALID_REQUEST_ID"} <= REJECTION_REASON_CODES


# Regression: d823b714f — INVALID_ORDER_ID wrote the junk id into the inbox, which
# the code comment said it must not.
@pytest.mark.parametrize("junk", ["../../x", "ord x", "ord_x\n", "ord_" + "a" * 200])
def test_an_invalid_order_id_never_enters_the_inbox(desk, junk):
    append_order(TRADE_DATE, _market(junk))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert [f.reason for f in fills] == ["INVALID_ORDER_ID"]
    (row,) = fills
    assert row.order_id != junk
    assert is_valid_order_id(row.order_id)
    assert fill_day(TRADE_DATE, desk["pm"]) == []  # derived id keeps resume idempotent


def test_a_non_string_order_id_is_refused_without_crashing_the_batch(desk):
    # Regression: d823b714f
    raw = _market("ord_x").to_dict()
    raw["order_id"] = {"a": 1}
    append_raw_outbox(raw)
    append_order(TRADE_DATE, _market("ord_2026-05-17_satoshi_002"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert "INVALID_ORDER_ID" in [f.reason for f in fills]
    assert "ord_2026-05-17_satoshi_002" in {f.order_id for f in fills}


@pytest.mark.parametrize("request_id", [{"a": 1}, ["x"], 7, None, "", "../cnl_x"])
def test_a_cancel_with_a_bad_request_id_is_refused_without_crashing_the_batch(
    desk, request_id
):
    # Regression: 269dcab68 — an unhashable request id raised in the membership test.
    victim_id = "ord_2026-05-10_satoshi_003"
    save_pending(_conditional(victim_id), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel(request_id, "satoshi", victim_id))
    append_order(TRADE_DATE, _market("ord_2026-05-17_satoshi_002"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    reasons = {f.order_id: f.reason for f in fills}
    assert "INVALID_REQUEST_ID" in reasons.values()
    assert "ord_2026-05-17_satoshi_002" in reasons
    assert [o.order_id for o in list_pending(pending_dir=desk["pending"])] == [victim_id]
    assert fill_day(TRADE_DATE, desk["pm"]) == []


# ---- round 2: one malformed line must not crash the batch ----


def _raw_cancel(raw: dict) -> None:
    from engine.config import get_config

    path = get_config().orders_dir / "cancels" / f"{TRADE_DATE.isoformat()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(raw) + "\n")


@pytest.mark.parametrize("ts", [7, ["2026-05-17"], {"a": 1}, None])
def test_a_cancel_with_a_non_string_ts_does_not_crash_fill_day(desk, ts):
    # Regression: 2751bff7b — CancelRequest.from_dict raised AttributeError on
    # ``ts.replace``, which read_cancels did not catch, so one line crashed fill_day.
    _raw_cancel(
        {
            "request_id": "cnl_2026-05-17_other_001",
            "ts": ts,
            "agent_id": "other",
            "target_order_id": "ord_x",
        }
    )
    append_order(TRADE_DATE, _market("ord_2026-05-17_satoshi_002"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert "ord_2026-05-17_satoshi_002" in {f.order_id for f in fills}
    assert fill_day(TRADE_DATE, desk["pm"]) == []


@pytest.mark.parametrize("field", ["ts", "ticker", "currency"])
@pytest.mark.parametrize("value", [7, ["BTC-EUR"], {"a": 1}, None])
def test_an_outbox_line_with_a_non_string_field_is_refused_not_fatal(desk, field, value):
    # Regression: 2751bff7b — a non-string ts raised AttributeError past the outbox
    # reader's except clause; a list ticker reached ``.strip()`` in pass 3.
    from engine.config import get_config

    ohlcv = get_config().ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)
    (ohlcv / "BTC-EUR.jsonl").write_text(
        json.dumps({"date": "2026-05-17", "close": 80000.0, "adj_close": 80000.0}) + "\n"
    )
    victim_id = "ord_2026-05-17_other_001"
    raw = _market(victim_id, agent_id="other").to_dict()
    raw[field] = value
    append_raw_outbox(raw)
    append_order(TRADE_DATE, _market("ord_2026-05-17_satoshi_002"))

    fills = fill_day(TRADE_DATE, desk["pm"])

    by_id = {f.order_id: f for f in fills}
    assert by_id["ord_2026-05-17_satoshi_002"].status == "filled"
    assert victim_id not in by_id
    assert [f.reason for f in fills if f.order_id != "ord_2026-05-17_satoshi_002"] == [
        "INVALID_SHARES"
    ]
    assert fill_day(TRADE_DATE, desk["pm"]) == []


@pytest.mark.parametrize("field", ["ticker", "currency"])
def test_order_refuses_a_non_string_identifier_field(field):
    # Regression: 2751bff7b
    kwargs = {"ticker": "BTC-EUR", "currency": "EUR", field: ["x"]}
    with pytest.raises(ValueError, match=field):
        Order(
            order_id="ord_x",
            ts=datetime(2026, 5, 17, 20, tzinfo=timezone.utc),
            agent_id="satoshi",
            action="BUY",
            shares=1,
            reasoning="r",
            **kwargs,
        )


# Regression: 3c96128d1 — a request id is predictable, and any row filed under it
# blocks the cancel that owns it. Agent "other" used satoshi's future id with a
# junk target, ahead of satoshi's real cancel in the file, and satoshi's was then
# skipped as already processed.
def test_one_agent_cannot_preempt_another_agents_cancel_by_request_id(desk):
    victim = "ord_2026-05-10_satoshi_003"
    save_pending(_conditional(victim), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_satoshi_001", "other", "ord_nothing"))
    append_cancel(TRADE_DATE, _cancel("cnl_2026-05-17_satoshi_001", "satoshi", victim))

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert sorted(f.reason for f in fills) == ["CANCELLED_BY_AGENT", "INVALID_REQUEST_ID"]
    assert list_pending(pending_dir=desk["pending"]) == []
    assert "cnl_2026-05-17_satoshi_001" not in inbox_order_ids(TRADE_DATE)


@pytest.mark.parametrize(
    "request_id",
    [
        "cnl_2026-05-18_satoshi_001",  # another day's id
        "cnl_2026-05-17_other_001",  # another agent's id
        "cnl_2026-05-17_satoshi_",  # no sequence
        "cnl_2026-05-17_satoshi_x1",  # sequence is not digits
        "cnl_2026-05-17_satoshi_1_001",  # extra segment
        "cnl_1",
    ],
)
def test_a_cancel_is_only_accepted_under_the_id_the_driver_mints_for_it(desk, request_id):
    # Regression: 3c96128d1
    victim = "ord_2026-05-10_satoshi_003"
    save_pending(_conditional(victim), pending_dir=desk["pending"])
    append_cancel(TRADE_DATE, _cancel(request_id, "satoshi", victim))

    (row,) = fill_day(TRADE_DATE, desk["pm"])

    assert row.reason == "INVALID_REQUEST_ID"
    assert row.order_id != request_id
    assert [o.order_id for o in list_pending(pending_dir=desk["pending"])] == [victim]


def test_a_prefix_agent_cannot_claim_a_longer_agents_request_id():
    # Regression: 3c96128d1 — a bare startswith would accept this.
    from engine.orders import is_minted_request_id

    assert is_minted_request_id("cnl_2026-05-17_a_001", TRADE_DATE, "a")
    assert not is_minted_request_id("cnl_2026-05-17_a_b_001", TRADE_DATE, "a")
    assert is_minted_request_id("cnl_2026-05-17_a_b_001", TRADE_DATE, "a_b")


def test_the_driver_mints_ids_the_broker_accepts(desk):
    # The consumer of the rule: step_author_cancels must produce ids it accepts.
    victim = "ord_2026-05-10_satoshi_003"
    save_pending(_conditional(victim), pending_dir=desk["pending"])
    step_author_cancels("satoshi", [{"target_order_id": victim}], TRADE_DATE)

    fills = fill_day(TRADE_DATE, desk["pm"])

    assert [(f.order_id, f.reason) for f in fills] == [(victim, "CANCELLED_BY_AGENT")]


# Regression: b5dea41da — a fired order with a bad agent id was filed under its own
# order id, contradicting "a refused row is filed under a derived id".
def test_a_fired_order_with_a_bad_agent_id_is_filed_under_a_derived_id(desk):
    order = _conditional("ord_2026-05-10_satoshi_003", agent_id="../elsewhere")

    fill = _fires(order, desk)

    assert fill.reason == "INVALID_AGENT_ID"
    assert fill.order_id != order.order_id
    assert is_valid_id(fill.order_id)


# Regression: 56bc16008 — delete_pending raises ValueError for a planted symlink;
# on the expiry path that happened after the inbox row was written and killed the
# watcher for every later order.
def test_the_watcher_survives_a_planted_symlink_on_an_expired_order(desk):
    from scripts.check_triggers import _process_channel

    (desk["pending"] / "ord_link.json").symlink_to(desk["portfolio_file"])
    link = _conditional("ord_link")
    second = _conditional("ord_2026-05-10_satoshi_004")
    save_pending(second, pending_dir=desk["pending"])
    summary = {"expired": 0, "carried": 0, "fired": 0, "errors": 0, "report": []}

    _process_channel(
        [link, second],
        datetime(2026, 7, 1, 13, tzinfo=timezone.utc),
        date(2026, 7, 1),
        None,
        summary,
        pending_dir=desk["pending"],
    )

    assert summary["expired"] == 2
    assert desk["portfolio_file"].is_file()
    assert not (desk["pending"] / "ord_2026-05-10_satoshi_004.json").exists()
