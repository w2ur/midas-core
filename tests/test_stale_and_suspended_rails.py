"""INSTRUMENT_SUSPENDED and STALE_PRICE (market data plan 2026-10-03, 1.3).

Until these rails existed every read path was blind to a price's age. A
CTVA order on 2026-10-02 would have filled at its 2026-09-30 close of 77.65
(the real price was 11.92; a what-if against the store at 0f981dd99, since no
CTVA order was ever placed), and PRICE_IMPLAUSIBLE could not object: its BUY reference
is the prior close from the same frozen file, so the ratio was exactly 1.0.

The fixtures are the real incidents, rebuilt as small stores: CTVA at
``0f981dd99`` (store at 09-30, one quarantined row for 10-01), MNST in August
(frozen at its 08-10 close through an unrestated 2:1 split), 4GLD.DE (a UCITS
fund that lands about a day after its exchange, and must keep trading) and
Xetra's 2026-05-01 holiday (a whole bucket closed, which must stay green).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from engine.config import get_config
from engine.orders import append_order, read_inbox
from tests.test_paper_broker import (  # noqa: F401 — broker_env is a fixture
    _init_portfolio,
    _make_order,
    _seed_ohlcv,
    _write_config,
    broker_env,
)

# Five peers is the smallest bucket the rail judges (MIN_BUCKET_POPULATION).
US_PEERS = ("AAPL", "MSFT", "KO", "PEP", "JNJ", "XOM")
DE_PEERS = ("SAP.DE", "SIE.DE", "ALV.DE", "BMW.DE", "BAS.DE", "DTE.DE")

# CTVA's quarantined row, verbatim from data/market/quarantine/CTVA.jsonl at
# 0f981dd99 (embedded: CI checks out at fetch-depth 1).
CTVA_QUARANTINE = {
    "symbol": "CTVA",
    "date": "2026-10-01",
    "kind": "new-row",
    "stored_close": 77.6500015258789,
    "incoming_close": 12.569999694824219,
    "ratio": 0.16188022469819185,
}
CTVA_STORE = [
    ("2026-09-28", 77.73999786376953),
    ("2026-09-29", 77.87000274658203),
    ("2026-09-30", 77.6500015258789),
]


def _weekdays(start: date, end: date, skip: frozenset[date] = frozenset()) -> list[str]:
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in skip:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _seed_bucket(ohlcv: Path, tickers, dates: list[str], close: float = 100.0) -> None:
    for t in tickers:
        _seed_ohlcv(ohlcv, t, [(d, close) for d in dates])


def _seed_registry_from_quarantine(rows_by_symbol: dict[str, list[dict]]) -> None:
    """Build the registry the way 1.2 seeds it: from unadjudicated quarantine rows."""
    from engine import instrument_status

    market = get_config().ohlcv_dir.parent
    qdir = market / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    for symbol, rows in rows_by_symbol.items():
        (qdir / f"{symbol}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    entries = instrument_status.seed_entries(
        qdir, market / "corporate_actions.jsonl", get_config().ohlcv_dir
    )
    instrument_status._save(entries, instrument_status.registry_path())


def _hold(pm, agent_id: str, ticker: str, shares: float, price: float) -> None:
    from datetime import datetime, timezone

    from engine.types import Trade

    pm.apply_trade(
        agent_id,
        Trade(
            id=f"seed_{agent_id}_{ticker}",
            timestamp=datetime(2026, 9, 1, 20, 0, tzinfo=timezone.utc),
            action="BUY",
            ticker=ticker,
            shares=shares,
            price=price,
            total=shares * price,
            fees=0.0,
            reasoning="seed a holding",
        ),
    )


# ---------------------------------------------------------------------------
# CTVA, 2026-10-02
# ---------------------------------------------------------------------------


@pytest.fixture
def ctva_store(broker_env):
    """The store as it stood at 0f981dd99: US peers through 10-02, CTVA at 09-30."""
    _seed_bucket(
        broker_env["ohlcv"], US_PEERS, _weekdays(date(2026, 9, 14), date(2026, 10, 2))
    )
    _seed_ohlcv(broker_env["ohlcv"], "CTVA", CTVA_STORE)
    _write_config(broker_env["config_dir"], "agent1")
    return broker_env


def test_ctva_buy_on_2026_10_02_is_refused_suspended(ctva_store):
    """Regression: the CTVA what-if of 2026-10-02 (plan 2026-10-03, 1.3). The
    registry is seeded from CTVA's real quarantine row, so this also holds the
    chain tripwire -> registry -> broker together."""
    from engine.paper_broker import fill_day

    _seed_registry_from_quarantine({"CTVA": [CTVA_QUARANTINE]})
    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    append_order(date(2026, 10, 2), _make_order("o_ctva", "agent1", "BUY", "CTVA", 10))

    fills = fill_day(date(2026, 10, 2), pm)
    assert [(f.status, f.reason) for f in fills] == [("rejected", "INSTRUMENT_SUSPENDED")]
    assert pm.load("agent1").positions == []


def test_ctva_without_a_registry_entry_is_refused_stale(ctva_store):
    """The two rails are independent: with no status recorded, the price's own
    date (09-30, two US sessions behind 10-02) still refuses it."""
    from engine.paper_broker import fill_day

    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    append_order(date(2026, 10, 2), _make_order("o_ctva", "agent1", "BUY", "CTVA", 10))

    assert [f.reason for f in fill_day(date(2026, 10, 2), pm)] == ["STALE_PRICE"]


def test_the_band_alone_would_have_filled_ctva(ctva_store):
    """Why the new rails are needed: the band compares the frozen close with
    the frozen close before it, and passes."""
    from engine.paper_broker import _price_out_of_band
    from engine.quotes import latest_price

    on = date(2026, 10, 2)
    quote = latest_price("CTVA", on)
    previous = latest_price("CTVA", on - timedelta(days=1))
    assert quote.as_of == date(2026, 9, 30)
    assert not _price_out_of_band(quote.price, previous.price)


def test_a_peer_in_the_same_store_still_fills(ctva_store):
    # Control: the fixture is not refusing everything.
    from engine.paper_broker import fill_day

    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    append_order(date(2026, 10, 2), _make_order("o_aapl", "agent1", "BUY", "AAPL", 10))

    assert [f.status for f in fill_day(date(2026, 10, 2), pm)] == ["filled"]


def test_a_sell_of_a_suspended_holding_is_refused_and_names_the_holders(ctva_store):
    """A refused SELL traps the holder (intended: no price, no fill), so the
    session's concern trailer names every book holding the ticker."""
    from engine.paper_broker import fill_day, instrument_refusal_concerns

    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    _init_portfolio(ctva_store["pm_base"], "agent2", cash=10_000.0)
    _init_portfolio(ctva_store["pm_base"], "agent3", cash=10_000.0)
    _hold(pm, "agent1", "CTVA", 5, 70.0)
    _hold(pm, "agent2", "CTVA", 3, 70.0)
    _seed_registry_from_quarantine({"CTVA": [CTVA_QUARANTINE]})
    on = date(2026, 10, 2)
    append_order(on, _make_order("o_out", "agent1", "SELL", "CTVA", 5))

    fills = fill_day(on, pm)
    assert [f.reason for f in fills] == ["INSTRUMENT_SUSPENDED"]
    assert pm.load("agent1").positions[0].shares == 5

    concerns = instrument_refusal_concerns(on, portfolios_dir=ctva_store["pm_base"])
    assert len(concerns) == 1
    assert "o_out" in concerns[0] and "SELL CTVA" in concerns[0]
    assert "agent1, agent2" in concerns[0] and "agent3" not in concerns[0]


@pytest.mark.parametrize("channel", ["pending", "manager-pending"])
def test_an_armed_order_on_a_suspended_ticker_is_named_every_session(ctva_store, channel):
    """Regression (review of feat/stage1-asof-reads, 2026-10-03): the concern
    read only the session date's inboxes, so a SELL fired by a weekend crypto
    pass, or any watcher fire between sessions, was never named. The watcher
    now holds such a fire armed (HELD_ON_FIRE) and writes no inbox row, so the
    concern reads the state instead: every armed order, in every channel,
    on a suspended instrument, whatever date it fired or was authored."""
    from engine.paper_broker import instrument_refusal_concerns
    from engine.triggers import save_pending

    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    _init_portfolio(ctva_store["pm_base"], "agent2", cash=10_000.0)
    _hold(pm, "agent1", "CTVA", 5, 70.0)
    _hold(pm, "agent2", "CTVA", 3, 70.0)
    order = _make_order("o_sat", "agent1", "SELL", "CTVA", 5)
    order.trigger = {"op": ">=", "level": 70.0}
    order.expires = "2026-10-30"
    save_pending(order, pending_dir=get_config().orders_dir / channel)
    calm = _make_order("o_calm", "agent1", "SELL", "AAPL", 1)
    calm.trigger = {"op": ">=", "level": 70.0}
    calm.expires = "2026-10-30"
    save_pending(calm, pending_dir=get_config().orders_dir / channel)
    _seed_registry_from_quarantine({"CTVA": [CTVA_QUARANTINE]})

    # Monday's session: nothing in its own inbox.
    concerns = instrument_refusal_concerns(
        date(2026, 10, 5), portfolios_dir=ctva_store["pm_base"]
    )
    assert len(concerns) == 1
    assert "o_sat" in concerns[0] and "SELL CTVA" in concerns[0]
    assert "agent1, agent2" in concerns[0]
    assert "o_calm" not in concerns[0]


def test_an_armed_order_on_a_stale_ticker_is_named_every_session(ctva_store):
    """Regression (review of feat/stage1-asof-reads, finding 2): a ticker the
    vendor stops serving without a quarantine row (SGLN.MI in September) is
    never suspended, so an armed stop on it fired at every sweep, was refused
    STALE_PRICE, re-armed with no inbox row and a green run, and no channel
    named it. The session's concern now does, from state."""
    from engine.paper_broker import instrument_refusal_concerns
    from engine.triggers import save_pending

    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    _hold(pm, "agent1", "CTVA", 5, 70.0)
    stop = _make_order("o_stop", "agent1", "SELL", "CTVA", 5)
    stop.trigger = {"op": "<=", "level": 60.0}
    stop.expires = "2026-10-30"
    save_pending(stop, pending_dir=get_config().orders_dir / "pending")
    calm = _make_order("o_calm", "agent1", "SELL", "AAPL", 1)
    calm.trigger = {"op": ">=", "level": 70.0}
    calm.expires = "2026-10-30"
    save_pending(calm, pending_dir=get_config().orders_dir / "pending")

    # No registry entry: only the price's own date (09-30) can see it.
    concerns = instrument_refusal_concerns(
        date(2026, 10, 2), portfolios_dir=ctva_store["pm_base"]
    )
    assert len(concerns) == 1
    assert concerns[0].startswith("STALE_PRICE holds armed order o_stop")
    assert "SELL CTVA" in concerns[0] and "2026-09-30" in concerns[0]

    # Control: on 10-01 CTVA trails by one session, inside the tolerance.
    assert instrument_refusal_concerns(
        date(2026, 10, 1), portfolios_dir=ctva_store["pm_base"]
    ) == []


def test_no_refusal_means_no_concern(ctva_store):
    from engine.paper_broker import fill_day, instrument_refusal_concerns

    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    append_order(date(2026, 10, 2), _make_order("o_aapl", "agent1", "BUY", "AAPL", 1))
    fill_day(date(2026, 10, 2), pm)
    assert instrument_refusal_concerns(date(2026, 10, 2)) == []


def test_a_conditional_on_a_suspended_ticker_never_arms(ctva_store):
    from engine.paper_broker import fill_day
    from engine.triggers import list_pending

    _seed_registry_from_quarantine({"CTVA": [CTVA_QUARANTINE]})
    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    order = _make_order("o_trig", "agent1", "BUY", "CTVA", 5)
    order.trigger = {"op": "<=", "level": 70.0}
    order.expires = "2026-10-30"
    append_order(date(2026, 10, 2), order)

    fill_day(date(2026, 10, 2), pm)
    assert [f.reason for f in read_inbox(date(2026, 10, 2))] == ["INSTRUMENT_SUSPENDED"]
    assert list_pending() == []


def test_a_conditional_on_a_merely_stale_ticker_still_arms(ctva_store):
    """Staleness is transient and is checked when the order fires, not at intake."""
    from engine.paper_broker import fill_day
    from engine.triggers import list_pending

    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    order = _make_order("o_trig", "agent1", "BUY", "CTVA", 5)
    order.trigger = {"op": "<=", "level": 70.0}
    order.expires = "2026-10-30"
    append_order(date(2026, 10, 2), order)

    fill_day(date(2026, 10, 2), pm)
    assert [o.order_id for o in list_pending()] == ["o_trig"]


@pytest.mark.parametrize("registry", [True, False])
def test_a_fire_on_ctva_is_refused(ctva_store, registry):
    from engine.paper_broker import execute_triggered_order

    if registry:
        _seed_registry_from_quarantine({"CTVA": [CTVA_QUARANTINE]})
    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    order = _make_order("o_fire", "agent1", "BUY", "CTVA", 5)
    order.trigger = {"op": "<=", "level": 78.0}
    order.expires = "2026-10-30"

    fill = execute_triggered_order(
        order, date(2026, 10, 2), pm, fire_price=77.65, fire_as_of=date(2026, 9, 30)
    )
    assert fill is not None and fill.trigger_fired
    expected = "INSTRUMENT_SUSPENDED" if registry else "STALE_PRICE"
    assert (fill.status, fill.reason) == ("rejected", expected)


def test_a_fire_on_a_current_price_fills(ctva_store):
    # Control for the fire path: same order shape on a peer at today's close.
    from engine.paper_broker import execute_triggered_order

    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    order = _make_order("o_fire", "agent1", "BUY", "AAPL", 5)
    order.trigger = {"op": "<=", "level": 101.0}
    order.expires = "2026-10-30"
    fill = execute_triggered_order(
        order, date(2026, 10, 2), pm, fire_price=100.0, fire_as_of=date(2026, 10, 2)
    )
    assert fill is not None and fill.status == "filled"


def test_an_unreadable_registry_refuses_every_ticker(ctva_store):
    """Fail closed: a registry that cannot say which instruments are broken
    cannot vouch for any of them."""
    from engine import instrument_status
    from engine.paper_broker import fill_day

    instrument_status.registry_path().write_text("{not json", encoding="utf-8")
    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    append_order(date(2026, 10, 2), _make_order("o_aapl", "agent1", "BUY", "AAPL", 1))

    assert [f.reason for f in fill_day(date(2026, 10, 2), pm)] == ["INSTRUMENT_SUSPENDED"]


# ---------------------------------------------------------------------------
# MNST, August 2026
# ---------------------------------------------------------------------------

MNST_QUARANTINE = [
    {"symbol": "MNST", "date": "2026-08-11", "kind": "new-row",
     "stored_close": 91.43000030517578, "incoming_close": 45.529998779296875,
     "ratio": 0.49797657910233495},
    {"symbol": "MNST", "date": "2026-08-12", "kind": "new-row",
     "stored_close": 91.43000030517578, "incoming_close": 45.97999954223633,
     "ratio": 0.5028983855273316},
]


@pytest.fixture
def mnst_store(broker_env):
    """Store as of the 2026-08-13 session (20:00 UTC then, so the store held
    the previous day): US peers through 08-12, MNST frozen at its 08-10 close
    of 91.43 while the vendor served 45.53 for 08-11 on the split basis."""
    _seed_bucket(
        broker_env["ohlcv"], US_PEERS, _weekdays(date(2026, 7, 20), date(2026, 8, 12))
    )
    _seed_ohlcv(
        broker_env["ohlcv"],
        "MNST",
        [(d, 91.43) for d in _weekdays(date(2026, 7, 20), date(2026, 8, 10))],
    )
    _write_config(broker_env["config_dir"], "agent1")
    return broker_env


@pytest.mark.parametrize(
    "registry, expected",
    [(True, "INSTRUMENT_SUSPENDED"), (False, "STALE_PRICE")],
)
def test_mnst_august_buy_is_refused(mnst_store, registry, expected):
    """Regression: MNST sat buyable at twice its price from 2026-08-11 to the
    08-17 adjudication (skill midas-market-data, "A quarantined row is
    adjudicated")."""
    from engine.paper_broker import fill_day

    if registry:
        _seed_registry_from_quarantine({"MNST": MNST_QUARANTINE})
    pm = _init_portfolio(mnst_store["pm_base"], "agent1", cash=10_000.0)
    append_order(date(2026, 8, 13), _make_order("o_mnst", "agent1", "BUY", "MNST", 10))

    assert [f.reason for f in fill_day(date(2026, 8, 13), pm)] == [expected]


def test_mnst_one_session_behind_is_the_suspension_rails_to_catch(mnst_store):
    """On 08-12 MNST trailed the US bucket by one session only, inside the
    stale tolerance: the stale rail alone fills it at 91.43, which is why the
    suspension rail is not optional. Pinned so nobody reads the stale rail as
    covering a split on its first night."""
    from engine.paper_broker import fill_day

    pm = _init_portfolio(mnst_store["pm_base"], "agent1", cash=10_000.0)
    append_order(date(2026, 8, 12), _make_order("o1", "agent1", "BUY", "MNST", 1))
    # The store as of the 08-12 session held the bucket through 08-11.
    for t in US_PEERS:
        path = mnst_store["ohlcv"] / f"{t}.jsonl"
        rows = [r for r in path.read_text().splitlines() if '"2026-08-12"' not in r]
        path.write_text("\n".join(rows) + "\n")
    assert [f.status for f in fill_day(date(2026, 8, 12), pm)] == ["filled"]

    _seed_registry_from_quarantine({"MNST": MNST_QUARANTINE[:1]})
    append_order(date(2026, 8, 12), _make_order("o2", "agent1", "BUY", "MNST", 1))
    assert [f.reason for f in fill_day(date(2026, 8, 12), pm)] == ["INSTRUMENT_SUSPENDED"]


# ---------------------------------------------------------------------------
# 4GLD.DE one day late; Xetra closed on 2026-05-01
# ---------------------------------------------------------------------------


def test_a_fund_one_day_behind_its_exchange_still_fills(broker_env):
    """4GLD.DE lands about a day after .DE on most nights (measured
    2026-10-03); a one-session tolerance keeps it tradable."""
    from engine.paper_broker import fill_day

    on = date(2026, 10, 2)
    _seed_bucket(broker_env["ohlcv"], DE_PEERS, _weekdays(date(2026, 9, 14), on))
    _seed_ohlcv(
        broker_env["ohlcv"],
        "4GLD.DE",
        [(d, 120.0) for d in _weekdays(date(2026, 9, 14), on - timedelta(days=1))],
    )
    _write_config(broker_env["config_dir"], "agent1")
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0, currency="EUR")
    append_order(on, _make_order("o_gld", "agent1", "BUY", "4GLD.DE", 1, "EUR"))

    fills = fill_day(on, pm)
    assert [(f.status, f.fill_price) for f in fills] == [("filled", 120.0)]


def test_two_days_behind_its_exchange_is_refused(broker_env):
    # The same fund one more session late: the tolerance is one day, not two.
    from engine.paper_broker import fill_day

    on = date(2026, 10, 2)
    _seed_bucket(broker_env["ohlcv"], DE_PEERS, _weekdays(date(2026, 9, 14), on))
    _seed_ohlcv(
        broker_env["ohlcv"],
        "4GLD.DE",
        [(d, 120.0) for d in _weekdays(date(2026, 9, 14), on - timedelta(days=2))],
    )
    _write_config(broker_env["config_dir"], "agent1")
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0, currency="EUR")
    append_order(on, _make_order("o_gld", "agent1", "BUY", "4GLD.DE", 1, "EUR"))

    assert [f.reason for f in fill_day(on, pm)] == ["STALE_PRICE"]


def test_a_market_sell_refused_stale_names_the_holders(broker_env):
    """Regression (review of feat/stage1-asof-reads, 2026-10-04): the concern
    covered INSTRUMENT_SUSPENDED inbox rows and armed orders, but not a
    same-session STALE_PRICE refusal of a market SELL. For a symbol frozen
    without tripping the tripwire (SGLN.MI) that refusal is the only one there
    is, so the holder could not exit night after night and only an inbox row
    said so. A refused BUY traps no one and stays unnamed."""
    from engine.paper_broker import fill_day, instrument_refusal_concerns

    on = date(2026, 10, 2)
    _seed_bucket(broker_env["ohlcv"], DE_PEERS, _weekdays(date(2026, 9, 14), on))
    _seed_ohlcv(
        broker_env["ohlcv"],
        "4GLD.DE",
        [(d, 120.0) for d in _weekdays(date(2026, 9, 14), on - timedelta(days=3))],
    )
    _write_config(broker_env["config_dir"], "agent1")
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0, currency="EUR")
    _init_portfolio(broker_env["pm_base"], "agent2", cash=10_000.0, currency="EUR")
    _hold(pm, "agent1", "4GLD.DE", 4, 120.0)
    append_order(on, _make_order("o_exit", "agent1", "SELL", "4GLD.DE", 4, "EUR"))
    append_order(on, _make_order("o_entry", "agent2", "BUY", "4GLD.DE", 1, "EUR"))

    fills = fill_day(on, pm)
    assert sorted((f.order_id, f.reason) for f in fills) == [
        ("o_entry", "STALE_PRICE"),
        ("o_exit", "STALE_PRICE"),
    ]

    concerns = instrument_refusal_concerns(on, portfolios_dir=broker_env["pm_base"])
    assert len(concerns) == 1
    assert concerns[0].startswith("STALE_PRICE refused o_exit")
    assert "SELL 4GLD.DE" in concerns[0] and "2026-09-29" in concerns[0]
    assert "held by agent1;" in concerns[0]


def test_a_whole_exchange_holiday_fills(broker_env):
    """Xetra was closed on 2026-05-01 (Labour Day): a session that day sees
    every .DE name at 04-30. Nobody is behind anybody: the bucket did not
    trade, so its reference date does not move."""
    from engine.paper_broker import fill_day

    holiday = frozenset({date(2026, 5, 1)})
    dates = _weekdays(date(2026, 4, 6), date(2026, 5, 1), skip=holiday)
    assert dates[-1] == "2026-04-30"
    _seed_bucket(broker_env["ohlcv"], DE_PEERS + ("4GLD.DE",), dates)
    _write_config(broker_env["config_dir"], "agent1")
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0, currency="EUR")
    append_order(date(2026, 5, 1), _make_order("o_hol", "agent1", "BUY", "SAP.DE", 1, "EUR"))

    assert [f.status for f in fill_day(date(2026, 5, 1), pm)] == ["filled"]


def test_a_bucket_too_small_to_judge_abstains(broker_env):
    """Below MIN_BUCKET_POPULATION the rail says nothing rather than reading a
    one-file bucket as agreeing with itself; existing single-ticker fixtures
    across the suite rely on that."""
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-09-01", 500.0)])
    _write_config(broker_env["config_dir"], "agent1")
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)
    append_order(date(2026, 10, 2), _make_order("o", "agent1", "BUY", "VOO", 1))
    assert [f.status for f in fill_day(date(2026, 10, 2), pm)] == ["filled"]


# ---------------------------------------------------------------------------
# The watcher holds a fire the rails call transient; it does not consume it
# ---------------------------------------------------------------------------


def _arm(order_id: str, ticker: str, level: float):
    from engine.triggers import save_pending

    order = _make_order(order_id, "agent1", "BUY", ticker, 5)
    order.trigger = {"op": "<=", "level": level}
    order.expires = "2026-10-30"
    save_pending(order)
    return order


@pytest.mark.parametrize(
    "registry, reason",
    [(False, "STALE_PRICE"), (True, "INSTRUMENT_SUSPENDED")],
)
def test_a_fire_refused_stale_or_suspended_keeps_the_order_armed(
    ctva_store, monkeypatch, registry, reason
):
    """Regression (review of feat/stage1-asof-reads, 2026-10-03): the watcher
    used to append the STALE_PRICE / INSTRUMENT_SUSPENDED rejection and delete
    the pending file, so a one-night store hole destroyed an armed order the
    intake path had deliberately let arm ("staleness is transient"). The order
    must be carried, like an unavailable quote, and reported as held."""
    from datetime import datetime, timezone

    from engine import triggers as triggers_mod
    from engine.ohlcv_store import DatedClose
    from engine.triggers import list_pending
    from scripts import check_triggers

    monkeypatch.setattr(check_triggers, "_git_add_commit", lambda *a, **k: "ok")
    if registry:
        _seed_registry_from_quarantine({"CTVA": [CTVA_QUARANTINE]})
    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=10_000.0)
    _arm("o_held", "CTVA", 78.0)
    monkeypatch.setattr(
        triggers_mod,
        "get_current_quote",
        lambda t, today: DatedClose(77.65, date(2026, 9, 30)),
    )

    summary = check_triggers.run(
        now=datetime(2026, 10, 2, 13, 0, tzinfo=timezone.utc), portfolio_manager=pm
    )

    assert [o.order_id for o in list_pending()] == ["o_held"]
    assert read_inbox(date(2026, 10, 2)) == []
    assert (summary["fired"], summary["carried"]) == (0, 1)
    [entry] = summary["report"]
    assert (entry["kind"], entry["error"]) == ("held", reason)
    assert entry["commit"] == check_triggers.REPORT_COMMIT_NONE
    assert pm.load("agent1").cash == 10_000.0


def test_a_fire_refused_for_cash_is_still_consumed(ctva_store, monkeypatch):
    # Control: a refusal that is NOT transient still retires the order.
    from datetime import datetime, timezone

    from engine import triggers as triggers_mod
    from engine.ohlcv_store import DatedClose
    from engine.triggers import list_pending
    from scripts import check_triggers

    monkeypatch.setattr(check_triggers, "_git_add_commit", lambda *a, **k: "ok")
    pm = _init_portfolio(ctva_store["pm_base"], "agent1", cash=0.0)
    _arm("o_poor", "AAPL", 101.0)
    monkeypatch.setattr(
        triggers_mod,
        "get_current_quote",
        lambda t, today: DatedClose(100.0, date(2026, 10, 2)),
    )

    check_triggers.run(
        now=datetime(2026, 10, 2, 13, 0, tzinfo=timezone.utc), portfolio_manager=pm
    )

    assert list_pending() == []
    assert [f.reason for f in read_inbox(date(2026, 10, 2))] == ["INSUFFICIENT_CASH"]


# ---------------------------------------------------------------------------
# The baseline-manager book trades outside the broker; the rails reach it too
# ---------------------------------------------------------------------------


def _bullish(*tickers: str) -> dict:
    note = {
        "thesis": "t",
        "conviction": 8,
        "tickers": list(tickers),
        "action_bias": "strong_buy",
        "horizon": "weeks",
        "catalysts": "c",
        "currency": "EUR",
    }
    return {"agent-a": {"research_note": note}, "agent-b": {"research_note": note}}


@pytest.mark.live_cast
@pytest.mark.parametrize("registry", [True, False])
def test_the_baseline_manager_never_buys_a_suspended_or_stale_ticker(
    broker_env, registry
):
    """Regression (review of feat/stage1-asof-reads, 2026-10-03):
    step_build_baseline_manager priced its rebalance with a bare
    latest_close_on_or_before and booked it with apply_trade, so the one
    trading path outside the broker would still buy CTVA at its frozen
    pre-separation close. Registry seeded: CTVA as the store stood on
    2026-10-01 (one session behind, suspended). No registry: CTVA two
    sessions behind its bucket (stale). AAPL is the control and must fill."""
    from scripts.daily_session import step_build_baseline_manager

    on = date(2026, 10, 1)  # first weekday of October: a rebalance day
    _seed_bucket(broker_env["ohlcv"], US_PEERS, _weekdays(date(2026, 9, 14), on))
    if registry:
        _seed_ohlcv(broker_env["ohlcv"], "CTVA", CTVA_STORE)
        _seed_registry_from_quarantine({"CTVA": [CTVA_QUARANTINE]})
    else:
        _seed_ohlcv(broker_env["ohlcv"], "CTVA", CTVA_STORE[:2])

    step_build_baseline_manager(
        _bullish("CTVA", "AAPL"),
        trade_date=on,
        portfolios_dir=broker_env["pm_base"],
        ohlcv_store=broker_env["ohlcv"],
    )

    book = json.loads(
        (broker_env["pm_base"] / "baseline-manager" / "portfolio.json").read_text()
    )
    assert sorted(p["ticker"] for p in book["positions"]) == ["AAPL"]


def test_an_armed_order_on_a_live_priced_crypto_pair_is_not_named_stale(broker_env):
    """Regression (review of feat/stage1-asof-reads): the watcher never fires
    an allowlisted crypto pair on the store. `engine.triggers.get_current_quote`
    fetches it live from ccxt and dates the quote today, so `_stale` reads lag 0
    on the fire path. Judging the armed order against the store's frozen close
    told a human a live stop was dead. A pair outside the allowlist (HBAR-USD)
    is priced from the store by the watcher too, so it is still judged."""
    from engine.paper_broker import instrument_refusal_concerns
    from engine.triggers import is_crypto_ticker, save_pending

    every_day = [
        (date(2026, 9, 14) + timedelta(days=i)).isoformat() for i in range(18)
    ]  # 09-14 .. 10-01
    _seed_bucket(
        broker_env["ohlcv"],
        ("ETH-EUR", "SOL-EUR", "XRP-EUR", "ADA-EUR", "LTC-EUR"),
        every_day,
    )
    frozen = [d for d in every_day if d <= "2026-09-27"]
    _seed_ohlcv(broker_env["ohlcv"], "BTC-EUR", [(d, 100.0) for d in frozen])
    _seed_ohlcv(broker_env["ohlcv"], "HBAR-USD", [(d, 100.0) for d in frozen])
    _write_config(broker_env["config_dir"], "agent1")
    assert is_crypto_ticker("BTC-EUR") and not is_crypto_ticker("HBAR-USD")

    for order_id, ticker in (("o_btc", "BTC-EUR"), ("o_hbar", "HBAR-USD")):
        stop = _make_order(order_id, "agent1", "SELL", ticker, 1)
        stop.trigger = {"op": "<=", "level": 50.0}
        stop.expires = "2026-10-30"
        save_pending(stop, pending_dir=get_config().orders_dir / "pending")

    concerns = instrument_refusal_concerns(
        date(2026, 10, 2), portfolios_dir=broker_env["pm_base"]
    )
    assert len(concerns) == 1
    assert concerns[0].startswith("STALE_PRICE holds armed order o_hbar")
