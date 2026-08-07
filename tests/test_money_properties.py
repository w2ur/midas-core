"""Property tests for the money path (reliability review W5.3).

CLAUDE.md mandates property tests for pure transforms, with financial-math
arbitraries that exclude NaN and infinity, run >=1000 examples, and compare to
1e-6 rather than 1e-10 (too tight at portfolio scale). Three such tests
existed, at 200, 300 and 400 examples; the transforms that actually carry money
— unit normalisation, FX conversion, order serialisation, fees — had none.

The example count comes from the shared `midas` profile in conftest.py, so
nothing here restates it.

Why these four. Every one sits on the path a euro travels: the vendor's number
becomes a stored price (`_normalise_vendor_units`), the stored price becomes a
book-currency amount (`fx.convert`), the amount becomes an order on disk
(serde), and the order costs something to execute (`fees`). Three of the four
have already been the site of a real defect.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from engine.fees import fee_for
from engine.orders import Order
from engine.quotes import _SUB_UNITS, normalise_vendor_quote

#: Financial-math arbitrary per CLAUDE.md: no NaN, no infinity, and bounded to
#: a range a real book can reach. Unbounded floats would only ever exercise
#: float64's edges, which is a different (and uninteresting) question.
prices = st.floats(
    min_value=1e-4,
    max_value=1e7,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
shares = st.floats(
    min_value=1e-6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)

TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# 1. Vendor unit normalisation
# ---------------------------------------------------------------------------


@given(vendor_price=prices)
def test_normalising_a_whole_unit_quote_is_the_identity(midas_data_root, vendor_price):
    """A USD-quoted ticker must come back bit-for-bit unchanged.

    This is the property the 2026-08-07 defect violated in the other
    direction: the scaling has to be inert everywhere it is not needed, or a
    single misapplied factor silently re-denominates a whole book.
    """
    quote = normalise_vendor_quote("AAPL", vendor_price)
    assert quote is not None
    assert quote.price == vendor_price
    assert quote.currency == "USD"


@given(vendor_price=prices)
def test_a_pence_quote_is_divided_exactly_once(midas_data_root, vendor_price):
    quote = normalise_vendor_quote("LLOY.L", vendor_price)
    assert quote is not None
    assert quote.price == pytest.approx(vendor_price / 100.0, rel=TOLERANCE)
    assert quote.currency == "GBP"


@given(vendor_price=prices)
def test_normalisation_is_never_idempotent_in_price_for_a_sub_unit(
    midas_data_root, vendor_price
):
    """Feeding the output back in must NOT be a no-op — and that is the point.

    A function that were idempotent here would be indistinguishable from one
    that had already been applied, which is exactly why the store carries a
    marker file rather than trusting inspection. Pinning non-idempotence stops
    anyone "fixing" it into a guard-by-shape that cannot work.
    """
    once = normalise_vendor_quote("LLOY.L", vendor_price)
    twice = normalise_vendor_quote("LLOY.L", once.price)
    assert twice.price == pytest.approx(once.price / 100.0, rel=TOLERANCE)
    assert twice.currency == once.currency  # the CURRENCY is idempotent


def test_every_declared_sub_unit_scales_by_a_hundred():
    """All four sub-units are 100:1. A future 1000:1 unit needs new thinking,
    not a new dict entry."""
    for unit, (iso, scale) in _SUB_UNITS.items():
        assert scale == 0.01, unit
        assert iso.isupper() and len(iso) == 3, unit


# ---------------------------------------------------------------------------
# 2. FX conversion
# ---------------------------------------------------------------------------


@given(amount=prices)
def test_converting_a_currency_to_itself_is_the_identity(midas_data_root, amount):
    from engine.fx import convert

    assert convert(amount, "EUR", "EUR", date(2026, 6, 1)) == amount


@given(
    amount=prices,
    rate=st.floats(
        min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
def test_convert_round_trips_within_tolerance(midas_data_root, amount, rate):
    """EUR -> USD -> EUR must return the original.

    Seeded with a synthetic rate rather than the live store so the property is
    about the arithmetic, not about which day's data happens to be committed.
    """
    from engine.config import get_config
    from engine.fx import convert
    import json

    store = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    (store / "EURUSD=X.jsonl").write_text(
        json.dumps({"date": "2026-06-01", "close": rate}) + "\n"
    )

    on = date(2026, 6, 1)
    forward = convert(amount, "EUR", "USD", on)
    assume(forward is not None)
    back = convert(forward, "USD", "EUR", on)
    assume(back is not None)
    assert back == pytest.approx(amount, rel=TOLERANCE)


# ---------------------------------------------------------------------------
# 3. Order serde
# ---------------------------------------------------------------------------


@given(
    share_count=shares,
    action=st.sampled_from(["BUY", "SELL"]),
    ticker=st.sampled_from(["AAPL", "LLOY.L", "BTC-EUR", "EURUSD=X"]),
    currency=st.sampled_from(["EUR", "USD"]),
)
def test_an_order_survives_a_disk_round_trip(share_count, action, ticker, currency):
    """to_dict -> from_dict must be lossless on every field that moves money.

    The outbox is the Brain/Hands contract: an order that changes shape in
    transit is a trade the agent did not author.
    """
    order = Order(
        order_id="ord_2026-06-01_agent_001",
        ts=datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc),
        agent_id="agent",
        action=action,
        ticker=ticker,
        shares=share_count,
        reasoning="property",
        currency=currency,
    )

    restored = Order.from_dict(order.to_dict())

    assert restored.shares == order.shares
    assert restored.action == order.action
    assert restored.ticker == order.ticker
    assert restored.currency == order.currency
    assert restored.order_id == order.order_id
    assert restored.ts == order.ts


@given(
    share_count=shares,
    level=prices,
    op=st.sampled_from([">=", "<="]),
)
def test_a_conditional_order_survives_a_disk_round_trip(share_count, level, op):
    """The trigger is the field that sat on disk for days before firing."""
    order = Order(
        order_id="ord_2026-06-01_agent_002",
        ts=datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc),
        agent_id="agent",
        action="SELL",
        ticker="LLOY.L",
        shares=share_count,
        reasoning="property",
        currency="EUR",
        trigger={"op": op, "level": level},
        expires="2026-07-01",
    )

    restored = Order.from_dict(order.to_dict())

    assert restored.trigger == order.trigger
    assert restored.expires == order.expires


# ---------------------------------------------------------------------------
# 4. Fees
# ---------------------------------------------------------------------------


@given(
    notional=prices, ticker=st.sampled_from(["AAPL", "LLOY.L", "BTC-EUR", "EURUSD=X"])
)
def test_a_fee_is_never_negative_and_never_nan(notional, ticker):
    fee = fee_for(ticker, notional)
    assert fee >= 0.0
    assert fee == fee  # NaN is the one value that fails this


@given(smaller=prices, delta=prices, ticker=st.sampled_from(["AAPL", "BTC-EUR"]))
def test_a_fee_is_monotonic_in_notional(smaller, delta, ticker):
    """A bigger trade never costs less. Obvious, and exactly the kind of thing
    a floor or a tier boundary quietly breaks."""
    larger = smaller + delta
    assert fee_for(ticker, larger) >= fee_for(ticker, smaller) - TOLERANCE
