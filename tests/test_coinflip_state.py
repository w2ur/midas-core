"""The coin flip advances from a persisted state over new dates only.

Background (plan subtask 1.6, owner decision D4, 2026-10-05). The coin flip
was a `bt` backtest recomputed from day one on every session, with one
`random.Random(seed)` reused across every day. A change in the candidate list
on any day (a universe refresh) or in any close (a corporate-action rescale)
changed every later pick, and integer shares made the series sensitive to the
scale a price is quoted in. The append-only merge kept each published row as
its session computed it, so the published curve became a splice of paths with
seams between rows written by different sessions (the largest measured at
+25.95% in one day, `yolo-sapiens-usd` 2026-09-20 -> 09-21).

Now each agent's coin flip keeps a state (`data/baselines/<agent>/state/
coinflip.json`): the date it was last advanced to, its cash and its holdings,
each `{shares, mark_date, mark_close}`. A session values the holdings by the
ratio of the current store's closes to the recorded mark, repicks with a seed
of `(agent, date)` over the sorted candidates of that date, and appends the new
dates. No published row is ever recomputed.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from engine.baselines import (
    CoinFlipHolding,
    CoinFlipState,
    advance_coin_flip,
    build_all_baselines,
    coin_flip_state_path,
    compute_coin_flip,
    init_coin_flip_state,
    load_coin_flip_state,
)
from engine.config import get_config
from engine.selectors.random_seeded import make_seed

_AGENT = "probe-agent"


def _store(ticker: str, rows: list[tuple[str, float]]) -> None:
    ohlcv = get_config().ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"date": d, "close": c}) for d, c in rows]
    (ohlcv / f"{ticker}.jsonl").write_text("\n".join(lines) + "\n")


def _days(start: date, n: int) -> list[str]:
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


_START = date(2026, 1, 1)
_BASES = {"AAA": 116.0, "BBB": 240.0, "CCC": 37.0, "DDD": 8.5, "EEE": 61.0}


def _seed_store(n_days: int = 12, scale: dict[str, float] | None = None) -> None:
    scale = scale or {}
    for k, (ticker, base) in enumerate(_BASES.items()):
        _store(
            ticker,
            [
                (d, base * (1 + 0.04 * (((i * (k + 1)) % 7) - 3)) * scale.get(ticker, 1.0))
                for i, d in enumerate(_days(_START, n_days))
            ],
        )


def _series_path() -> Path:
    return get_config().baselines_dir / _AGENT / "coinflip.json"


def _advance(to: date, tickers=None, max_positions: int = 2, currency: str = "USD"):
    # Bare test tickers resolve to USD, so a USD book needs no FX.
    return advance_coin_flip(
        agent_id=_AGENT,
        tickers=list(_BASES if tickers is None else tickers),
        currency=currency,
        max_positions=max_positions,
        series_path=_series_path(),
        from_date=_START,
        to_date=to,
    )


def _rows() -> list[dict]:
    return json.loads(_series_path().read_text())


# ---------------------------------------------------------------------------
# Fresh path and the semantics kept from the bt pipeline
# ---------------------------------------------------------------------------


def test_a_fresh_path_starts_at_initial_capital_and_holds_whole_shares(midas_data_root):
    _seed_store()
    rows = compute_coin_flip(_AGENT, list(_BASES), "USD", 2, _START, _START + timedelta(days=5))
    assert [r["date"] for r in rows] == _days(_START, 6)
    assert rows[0]["portfolio_value"] == pytest.approx(get_config().initial_capital)
    for r in rows:
        assert r["portfolio_value"] == pytest.approx(r["cash"] + r["positions_value"])
        assert r["currency"] == "USD"
    assert all(r["positions_value"] > 0 for r in rows)


def test_each_day_repicks_with_a_seed_of_agent_and_date(midas_data_root):
    """Picks are independent of earlier draws and of universe history: the pick
    on a day is the `(agent, date)`-seeded sample of that day's sorted
    candidates, whatever order the universe lists them in."""
    _seed_store()
    orders = [list(_BASES), list(reversed(list(_BASES))), ["CCC", "AAA", "EEE", "BBB", "DDD"]]
    for offset in range(10):
        day = _START + timedelta(days=offset)
        expected = random.Random(make_seed(_AGENT, day.isoformat())).sample(sorted(_BASES), 2)
        for order in orders:
            state = init_coin_flip_state(_AGENT, order, 2, day, 10_000.0, "USD")
            assert sorted(state.holdings) == sorted(expected), (day, order)


def test_weights_are_equal_and_capped_at_one_over_n(midas_data_root):
    """Fewer candidates than max_positions leaves the residue in cash (LimitWeights(1/n))."""
    _store("AAA", [(d, 10.0) for d in _days(_START, 3)])
    state = init_coin_flip_state(_AGENT, ["AAA"], 4, _START, 10_000.0, "USD")
    assert state.holdings["AAA"].shares == 250  # 10k / 4 / 10
    assert state.cash == pytest.approx(7_500.0)


def test_a_ticker_is_a_candidate_only_from_its_first_close(midas_data_root):
    _store("AAA", [(d, 10.0) for d in _days(_START, 5)])
    _store("LATE", [(d, 10.0) for d in _days(_START + timedelta(days=3), 2)])
    state = init_coin_flip_state(_AGENT, ["AAA", "LATE"], 2, _START, 10_000.0, "USD")
    assert set(state.holdings) == {"AAA"}


# ---------------------------------------------------------------------------
# Advancing: new dates only, published rows never move
# ---------------------------------------------------------------------------


def test_the_first_build_writes_the_series_then_the_state(midas_data_root):
    _seed_store()
    result = _advance(_START + timedelta(days=4))
    assert result.appended == 5 and result.concerns == []
    state = load_coin_flip_state(coin_flip_state_path(_series_path()))
    assert state.date == _rows()[-1]["date"]
    assert state.portfolio_value == _rows()[-1]["portfolio_value"]


def test_an_advance_appends_new_dates_and_leaves_published_rows_byte_identical(
    midas_data_root,
):
    _seed_store()
    _advance(_START + timedelta(days=4))
    before = _series_path().read_text()
    result = _advance(_START + timedelta(days=7))
    assert result.appended == 3
    after = _series_path().read_text()
    assert json.loads(after)[:5] == json.loads(before)


def test_a_same_day_rerun_appends_nothing_and_writes_nothing(midas_data_root):
    _seed_store()
    _advance(_START + timedelta(days=4))
    series = _series_path().read_text()
    state = coin_flip_state_path(_series_path()).read_text()
    result = _advance(_START + timedelta(days=4))
    assert result.appended == 0 and result.concerns == []
    assert _series_path().read_text() == series
    assert coin_flip_state_path(_series_path()).read_text() == state


def test_a_universe_swap_changes_no_published_row(midas_data_root):
    _seed_store()
    _advance(_START + timedelta(days=5))
    published = _rows()
    swapped = ["AAA", "CCC", "EEE", "ZZZ-NEW"]
    _store("ZZZ-NEW", [(d, 3.0) for d in _days(_START, 12)])
    result = _advance(_START + timedelta(days=8), tickers=swapped)
    assert result.concerns == []
    assert _rows()[: len(published)] == published


@pytest.mark.parametrize("book", ["USD", "EUR"])
def test_a_store_rescale_changes_no_published_row_and_not_the_next_value(
    midas_data_root, book
):
    """Review 2 MUST 2: holdings are valued by a ratio, so a constant rescale of a
    held symbol's whole history cancels. The first new row matches the
    un-rescaled run to 1e-9; only the repick on it sizes at the new scale.
    In a EUR book of USD names (the conversion path) the rescale still cancels:
    the rate multiplies the native price, it does not enter the ratio."""
    import shutil

    if book == "EUR":
        _store("EURUSD=X", [(d, 1.05 + 0.01 * i) for i, d in enumerate(_days(_START, 12))])

    def run(rescale: bool) -> tuple[list[dict], list[dict]]:
        shutil.rmtree(get_config().baselines_dir, ignore_errors=True)
        _seed_store()
        _advance(_START + timedelta(days=4), currency=book)
        published = _rows()
        held = sorted(load_coin_flip_state(coin_flip_state_path(_series_path())).holdings)
        assert held, "the fixture must hold something for the rescale to bite"
        if rescale:
            _seed_store(scale={t: 2.0 for t in held})
        _advance(_START + timedelta(days=5), currency=book)
        return published, _rows()

    published, plain = run(rescale=False)
    published_again, rescaled = run(rescale=True)
    assert published_again == published
    assert rescaled[: len(published)] == published
    assert rescaled[-1]["portfolio_value"] == pytest.approx(
        plain[-1]["portfolio_value"], rel=1e-9, abs=0
    )


# ---------------------------------------------------------------------------
# Missing prices (review 2 MUST 3)
# ---------------------------------------------------------------------------


def _held_state(ticker: str, shares: int, mark_date: str, mark_close: float, cash=0.0):
    return CoinFlipState(
        date=mark_date,
        portfolio_value=cash + shares * mark_close,
        cash=cash,
        holdings={ticker: CoinFlipHolding(shares, mark_date, mark_close, "USD", 1.0)},
    )


def _seed_state(state: CoinFlipState) -> None:
    from engine.baselines import write_coin_flip_state

    path = _series_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "date": state.date,
                    "portfolio_value": state.portfolio_value,
                    "cash": state.cash,
                    "positions_value": state.portfolio_value - state.cash,
                    "currency": "USD",
                }
            ]
        )
    )
    write_coin_flip_state(coin_flip_state_path(path), state, _AGENT)


def test_a_missing_close_on_the_day_values_at_the_last_close_before_it(midas_data_root):
    """(a) No close at d: the last close on or before d in the current store."""
    _store("AAA", [("2026-01-01", 10.0), ("2026-01-02", 12.0)])  # nothing on 01-03
    _seed_state(_held_state("AAA", 100, "2026-01-01", 10.0))
    result = _advance(date(2026, 1, 3), tickers=["AAA"], max_positions=1)
    assert result.concerns == []
    assert _rows()[-1]["portfolio_value"] == pytest.approx(1_200.0)


def test_a_held_file_that_is_gone_holds_at_its_mark_and_says_so(midas_data_root, capsys):
    """(b) The file is gone: hold at mark_close, one concern naming agent and
    ticker, and the holding stays in the book."""
    _store("BBB", [(d, 5.0) for d in _days(_START, 5)])
    _seed_state(_held_state("GONE", 100, "2026-01-01", 10.0, cash=500.0))
    result = _advance(date(2026, 1, 4), tickers=["BBB"], max_positions=1)
    warns = [l for l in capsys.readouterr().out.splitlines() if "[WARN]" in l]
    assert len(warns) == 1 and _AGENT in warns[0] and "GONE" in warns[0]
    assert len(result.concerns) == 1
    assert [r["portfolio_value"] for r in _rows()[1:]] == [1_500.0] * 3
    state = load_coin_flip_state(coin_flip_state_path(_series_path()))
    assert state.holdings["GONE"] == CoinFlipHolding(100, "2026-01-01", 10.0, "USD", 1.0)
    # The rest of the book is still repicked: BBB bought from the cash.
    assert state.holdings["BBB"].shares == 100


def test_a_held_file_with_no_close_before_its_mark_holds_at_its_mark(midas_data_root, capsys):
    """(b), second shape: the file survives but its history now starts after the mark."""
    _store("AAA", [("2026-01-03", 99.0), ("2026-01-04", 99.0)])
    _seed_state(_held_state("AAA", 10, "2026-01-01", 10.0))
    _advance(date(2026, 1, 4), tickers=["AAA"], max_positions=1)
    assert "[WARN]" in capsys.readouterr().out
    assert _rows()[-1]["portfolio_value"] == pytest.approx(100.0)


def test_a_mark_row_withdrawn_from_the_store_holds_at_its_mark_and_says_so(
    midas_data_root, capsys
):
    """Review fix round 1: the close a holding was marked at must still be in
    the store on its own date. If that row is withdrawn between two advances,
    the last close before it is a different price, and valuing from it would
    mis-value the holding by the move between the two dates with no concern.
    It is case (b): held at its mark, one concern naming agent and ticker.

    Round 5 (2026-10-06): AAA is the whole universe, so the date has no
    candidate while the book holds only that carried position; that is a
    second concern now (it was an [INFO] "stepped in cash" of a book that
    was not in cash)."""
    _store("AAA", [("2026-01-01", 10.0), ("2026-01-02", 20.0)])
    _seed_state(_held_state("AAA", 100, "2026-01-02", 20.0))
    # The 01-02 row is withdrawn; 01-03 lands at 21 (the old code would value
    # 100 x 20 x 21 / 10 = 4,200, a +110% move that never happened).
    _store("AAA", [("2026-01-01", 10.0), ("2026-01-03", 21.0)])
    result = _advance(date(2026, 1, 3), tickers=["AAA"], max_positions=1)
    warns = [l for l in capsys.readouterr().out.splitlines() if "[WARN]" in l]
    frozen = [l for l in warns if "NO_PRICE_DATA —" in l]
    assert len(frozen) == 1 and _AGENT in frozen[0] and "AAA" in frozen[0]
    assert len(result.concerns) == 2
    assert any("held only carried positions (AAA)" in c for c in result.concerns)
    assert _rows()[-1]["portfolio_value"] == pytest.approx(2_000.0)
    state = load_coin_flip_state(coin_flip_state_path(_series_path()))
    assert state.holdings["AAA"] == CoinFlipHolding(100, "2026-01-02", 20.0, "USD", 1.0)


def _suspend(symbol: str, status: str = "suspended") -> None:
    path = get_config().ohlcv_dir.parent / "instrument_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "instruments": {
                    symbol: {
                        "status": status,
                        "since": "2026-01-01",
                        "source": "human",
                        "reason": "test",
                    }
                },
            }
        )
    )


@pytest.mark.parametrize("status", ["suspended", "delisted"])
def test_a_suspended_or_delisted_symbol_is_never_a_candidate(midas_data_root, status):
    """(c) Excluded from candidates."""
    for t in ("AAA", "BAD"):
        _store(t, [(d, 10.0) for d in _days(_START, 3)])
    _suspend("BAD", status)
    for offset in range(3):
        state = init_coin_flip_state(
            _AGENT, ["AAA", "BAD"], 2, _START + timedelta(days=offset), 1e4, "USD"
        )
        assert set(state.holdings) == {"AAA"}


def test_a_held_suspended_symbol_is_valued_and_kept(midas_data_root):
    """(c) A held one is valued per (a)/(b) and stays in the book: it cannot be traded."""
    _store("BAD", [("2026-01-01", 10.0), ("2026-01-02", 11.0)])
    _store("AAA", [(d, 5.0) for d in _days(_START, 4)])
    _suspend("BAD")
    _seed_state(_held_state("BAD", 100, "2026-01-01", 10.0))
    _advance(date(2026, 1, 3), tickers=["AAA", "BAD"], max_positions=2)
    assert [r["portfolio_value"] for r in _rows()[1:]] == pytest.approx([1_100.0, 1_100.0])
    state = load_coin_flip_state(coin_flip_state_path(_series_path()))
    assert state.holdings["BAD"].shares == 100
    assert state.cash == pytest.approx(0.0)


def test_an_unreadable_registry_fails_closed_and_says_so(midas_data_root, capsys):
    _store("AAA", [(d, 10.0) for d in _days(_START, 3)])
    (get_config().ohlcv_dir.parent / "instrument_status.json").write_text("{broken")
    result = _advance(date(2026, 1, 2), tickers=["AAA"], max_positions=1)
    assert any("instrument status" in c for c in result.concerns)
    assert "[WARN]" in capsys.readouterr().out
    assert all(r["positions_value"] == 0.0 for r in _rows())


# ---------------------------------------------------------------------------
# Currency: every amount is in the book currency
# ---------------------------------------------------------------------------


def _currencies(mapping: dict[str, str]) -> None:
    """Seed the override map (layer 1 of `engine.quotes.ticker_currency`)."""
    from engine.config import reset_config_cache

    path = get_config().ticker_currencies_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping))
    reset_config_cache()


def test_a_mixed_currency_book_is_sized_and_valued_in_the_book_currency(
    midas_data_root, monkeypatch
):
    """Regression: until 2026-10-05 `_step` summed native closes into the book
    (world held AXFO.ST at 3 x 249.3 SEK, counted as EUR 747.9). Now each
    close is converted at the rate of the row's date, before sizing and
    before summing. Every number below is hand-computed."""
    import engine.fx as fx

    _currencies({"E.PA": "EUR", "S.ST": "SEK", "G.L": "GBP", "U": "USD"})
    d1, d2 = "2026-01-01", "2026-01-02"
    for t, c1, c2 in (("E.PA", 50.0, 55.0), ("S.ST", 100.0, 110.0), ("G.L", 20.0, 21.0), ("U", 30.0, 33.0)):
        _store(t, [(d1, c1), (d2, c2)])
    # EUR per unit, per day. engine.fx has no SEK route, so the rates are
    # injected; the real store's SEK gap is pinned by the next test.
    rates = {("SEK", d1): 0.09, ("GBP", d1): 1.15, ("USD", d1): 0.9,
             ("SEK", d2): 0.10, ("GBP", d2): 1.20, ("USD", d2): 0.8}

    def get_rate(frm, to, on=None):
        assert to == "EUR"
        return 1.0 if frm == to else rates.get((frm, on.isoformat()))

    monkeypatch.setattr(fx, "get_rate", get_rate)
    assert get_config().initial_capital == 10_000.0
    _advance(date(2026, 1, 2), tickers=["E.PA", "S.ST", "G.L", "U"], max_positions=4, currency="EUR")
    rows = _rows()
    # Day 1: a EUR 2,500 sleeve each. E.PA 50 x 50.00 = 2,500.00; S.ST
    # floor(2500 / 9.00) = 277 -> 2,493.00; G.L floor(2500 / 23.00) = 108 ->
    # 2,484.00; U floor(2500 / 27.00) = 92 -> 2,484.00. Cash 39.00.
    assert rows[0]["portfolio_value"] == pytest.approx(10_000.0)
    assert rows[0]["cash"] == pytest.approx(39.0)
    # Day 2: 39 + 50x55 + 277x110x0.10 + 108x21x1.20 + 92x33x0.80
    #      = 39 + 2,750 + 3,047 + 2,721.6 + 2,428.8 = 10,986.40
    assert rows[1]["portfolio_value"] == pytest.approx(10_986.40)
    assert all(r["currency"] == "EUR" for r in rows)
    held = load_coin_flip_state(coin_flip_state_path(_series_path())).holdings
    assert {t: h.currency for t, h in held.items()} == {
        "E.PA": "EUR", "S.ST": "SEK", "G.L": "GBP", "U": "USD"
    }
    assert held["S.ST"].mark_rate == pytest.approx(0.10)


def test_a_close_older_than_the_row_converts_at_the_rate_of_the_row(midas_data_root):
    """Regression: round-3 review, 2026-10-06. The books convert at the
    valuation date (`engine.valuation.value_position`), the coin flip
    converted each close at the close's own date, so the two disagreed
    whenever the store held no close for the row's date. U has no 01-02
    close: the 01-02 row values 10 x 30 USD at 01-02's rate (1/1.6), exactly
    as `value_position` values the same position, not at 01-01's (1/1.25)."""
    from engine.valuation import value_position

    _store("U", [("2026-01-01", 30.0), ("2026-01-03", 33.0)])
    _store("EURUSD=X", [("2026-01-01", 1.25), ("2026-01-02", 1.6), ("2026-01-03", 2.0)])
    _seed_state(
        CoinFlipState(
            date="2026-01-01",
            portfolio_value=100.0 + 10 * 30.0 * 0.8,
            cash=100.0,
            holdings={"U": CoinFlipHolding(10, "2026-01-01", 30.0, "USD", 0.8)},
        )
    )
    result = _advance(date(2026, 1, 2), tickers=["U"], max_positions=1, currency="EUR")
    assert result.concerns == []
    books = value_position("U", 10, "EUR", date(2026, 1, 2))
    assert books.ok and books.value == pytest.approx(187.5)
    assert _rows()[-1]["portfolio_value"] == pytest.approx(100.0 + books.value)
    # The repick sizes at the same converted price: floor(287.5 / 18.75) = 15.
    held = load_coin_flip_state(coin_flip_state_path(_series_path())).holdings["U"]
    assert held == CoinFlipHolding(15, "2026-01-01", 30.0, "USD", 1 / 1.6)


def test_a_schema_2_state_is_refused_with_its_reason(midas_data_root):
    """Schema 2 stored the rate at the close's date under the same field
    name; reading it as a valuation-date rate would be silently wrong."""
    _seed_store()
    _advance(_START + timedelta(days=3))
    path = coin_flip_state_path(_series_path())
    doc = json.loads(path.read_text())
    doc["schema"] = 2
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="schema 2 .* valuation date"):
        load_coin_flip_state(path)
    result = _advance(_START + timedelta(days=6))
    assert result.appended == 0 and "schema 2" in result.concerns[0]
    _assert_names_the_reinit_remedy(result.concerns[0])


def test_a_candidate_with_no_resolvable_currency_is_never_drawn(midas_data_root):
    """`ZZZ.XX` has no override, no registry entry and an unknown suffix, so
    `ticker_currency` answers None: it is never a candidate, whatever the seed."""
    from engine.quotes import ticker_currency

    assert ticker_currency("ZZZ.XX") is None
    for t in ("AAA", "ZZZ.XX"):
        _store(t, [(d, 10.0) for d in _days(_START, 10)])
    for offset in range(10):
        state = init_coin_flip_state(
            _AGENT, ["AAA", "ZZZ.XX"], 2, _START + timedelta(days=offset), 1e4, "USD"
        )
        assert set(state.holdings) == {"AAA"}


def test_a_candidate_with_no_rate_into_the_book_is_never_drawn(midas_data_root):
    """The real `engine.fx` has no SEK route: a SEK name is not drawn into a
    EUR book (it would otherwise be summed unconverted, the defect)."""
    from engine.fx import get_rate

    _currencies({"S.ST": "SEK", "E.PA": "EUR"})
    assert get_rate("SEK", "EUR", _START) is None
    for t in ("S.ST", "E.PA"):
        _store(t, [(d, 10.0) for d in _days(_START, 10)])
    for offset in range(10):
        state = init_coin_flip_state(
            _AGENT, ["S.ST", "E.PA"], 2, _START + timedelta(days=offset), 1e4, "EUR"
        )
        assert set(state.holdings) == {"E.PA"}


def test_a_held_name_that_loses_its_rate_is_frozen_and_raises_a_concern(
    midas_data_root, capsys
):
    """A held USD name in a EUR book whose rate file is gone: NO_FX_RATE, held
    at its recorded mark in the book currency, kept, and one concern."""
    _store("U", [(d, 30.0 + i) for i, d in enumerate(_days(_START, 4))])
    _currencies({"E.PA": "EUR"})  # a priceable name, so the book is advanced
    _store("E.PA", [(d, 10.0) for d in _days(_START, 4)])
    state = CoinFlipState(
        date="2026-01-01",
        portfolio_value=100.0 + 10 * 30.0 * 0.9,
        cash=100.0,
        holdings={"U": CoinFlipHolding(10, "2026-01-01", 30.0, "USD", 0.9)},
    )
    _seed_state(state)
    result = _advance(date(2026, 1, 3), tickers=["U", "E.PA"], max_positions=1, currency="EUR")
    warns = [l for l in capsys.readouterr().out.splitlines() if "[WARN]" in l]
    assert len(result.concerns) == 1 and len(warns) == 1
    assert "NO_FX_RATE" in warns[0] and _AGENT in warns[0] and "U" in warns[0]
    assert [r["portfolio_value"] for r in _rows()[1:]] == pytest.approx([370.0, 370.0])
    held = load_coin_flip_state(coin_flip_state_path(_series_path())).holdings
    assert held["U"] == CoinFlipHolding(10, "2026-01-01", 30.0, "USD", 0.9)


def test_a_name_bought_and_frozen_in_the_same_run_names_its_actual_mark(midas_data_root):
    """Review fix 5: a frozen ticker bought during the run is not in the run's
    starting state, and its concern used to print mark `?`. It names the mark
    it was bought at. The shape: bought 01-02, its rate is a zero row on 01-03
    (`get_rate` answers None), so it freezes on 01-03."""
    _store("U", [(d, 30.0) for d in _days(_START, 3)])
    _store("EURUSD=X", [("2026-01-01", 1.25), ("2026-01-02", 1.25), ("2026-01-03", 0.0)])
    # A EUR name keeps 01-03 drawable (a date with nothing drawable carries
    # the book instead), and at 1e9 it is never bought.
    _currencies({"E.PA": "EUR"})
    _store("E.PA", [(d, 1e9) for d in _days(_START, 3)])
    _seed_state(CoinFlipState(date="2026-01-01", portfolio_value=1_000.0, cash=1_000.0, holdings={}))
    result = _advance(date(2026, 1, 3), tickers=["U", "E.PA"], max_positions=2, currency="EUR")
    assert len(result.concerns) == 1
    concern = result.concerns[0]
    assert "NO_FX_RATE" in concern and "?" not in concern
    # Bought 01-02: floor(500 / (30 x 0.8)) = 20 shares, held at that mark.
    assert "2026-01-02" in concern and "20 x 30" in concern and "USD->EUR" in concern
    assert _rows()[-1]["portfolio_value"] == pytest.approx(1_000.0)


def test_a_fresh_path_reports_a_holding_it_froze(midas_data_root, capsys):
    """Regression: round-3 review, 2026-10-06. `_fresh_path` collected the
    holdings `_step` froze and dropped them, so a brand-new agent's first
    build reported no NO_FX_RATE freeze. Same fixture as the advance case
    above, with no state: the fresh path buys U on 01-01 and 01-02 and
    freezes it on 01-03."""
    _store("U", [(d, 30.0) for d in _days(_START, 3)])
    _store("EURUSD=X", [("2026-01-01", 1.25), ("2026-01-02", 1.25), ("2026-01-03", 0.0)])
    _currencies({"E.PA": "EUR"})
    _store("E.PA", [(d, 1e9) for d in _days(_START, 3)])
    result = _advance(date(2026, 1, 3), tickers=["U", "E.PA"], max_positions=2, currency="EUR")
    assert result.appended == 3 and len(result.concerns) == 1
    concern = result.concerns[0]
    assert _AGENT in concern and "U NO_FX_RATE" in concern and "2026-01-02" in concern
    assert f"  [WARN] {concern}" in capsys.readouterr().out.splitlines()
    rows = compute_coin_flip(_AGENT, ["U", "E.PA"], "EUR", 2, _START, date(2026, 1, 3))
    assert "NO_FX_RATE" in capsys.readouterr().out
    assert rows == _rows()


def _rate_gap_on_day_3_then_back(days: int = 4) -> None:
    """A held USD name in a EUR book whose rate is a zero row on 01-03 only;
    E.PA, too dear to buy, keeps every day drawable."""
    _store("U", [(d, 30.0) for d in _days(_START, days)])
    _store(
        "EURUSD=X",
        [(d, 0.0 if d == "2026-01-03" else 1.25) for d in _days(_START, days)],
    )
    _currencies({"E.PA": "EUR"})
    _store("E.PA", [(d, 1e9) for d in _days(_START, days)])
    _seed_state(
        CoinFlipState(
            date="2026-01-01",
            portfolio_value=100.0 + 10 * 30.0 * 0.8,
            cash=100.0,
            holdings={"U": CoinFlipHolding(10, "2026-01-01", 30.0, "USD", 0.8)},
        )
    )


def test_a_holding_frozen_and_valued_again_inside_the_window_is_no_concern(
    midas_data_root, capsys
):
    """Regression: round-4 review, 2026-10-06. A multi-day advance reported
    every holding that froze on any of its days: U froze on 01-03 (no rate)
    and was valued again and sold on 01-04, yet the run raised a NO_FX_RATE
    concern about a holding no longer held at its mark. It is an [INFO]
    note now; a holding still frozen at the end stays a concern."""
    _rate_gap_on_day_3_then_back()
    result = _advance(date(2026, 1, 4), tickers=["U", "E.PA"], max_positions=2, currency="EUR")
    assert result.appended == 3 and result.concerns == []
    out = capsys.readouterr().out
    assert "[WARN]" not in out
    assert any(
        "[INFO]" in l and "U was held at its mark (NO_FX_RATE)" in l and "2026-01-04" in l
        for l in out.splitlines()
    )


def test_a_holding_still_frozen_at_the_end_of_the_window_is_a_concern(midas_data_root):
    """The control: the same gap, the window ending on it."""
    _rate_gap_on_day_3_then_back()
    result = _advance(date(2026, 1, 3), tickers=["U", "E.PA"], max_positions=2, currency="EUR")
    assert len(result.concerns) == 1 and "U NO_FX_RATE" in result.concerns[0]


def test_a_held_name_whose_currency_no_longer_resolves_is_frozen(midas_data_root, capsys):
    """Its override removed, `ZZZ.XX` resolves to nothing: CURRENCY_UNRESOLVED."""
    _store("ZZZ.XX", [(d, 10.0) for d in _days(_START, 4)])
    _store("AAA", [(d, 10.0) for d in _days(_START, 4)])  # priceable, too dear for 5.0
    _seed_state(_held_state("ZZZ.XX", 10, "2026-01-01", 10.0, cash=5.0))
    result = _advance(date(2026, 1, 2), tickers=["ZZZ.XX", "AAA"], max_positions=1)
    assert len(result.concerns) == 1 and "CURRENCY_UNRESOLVED" in result.concerns[0]
    assert _rows()[-1]["portfolio_value"] == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# Integrity: the first new row chains from the state date
# ---------------------------------------------------------------------------


def test_a_series_without_a_state_is_not_advanced(midas_data_root, capsys):
    _seed_store()
    _advance(_START + timedelta(days=3))
    coin_flip_state_path(_series_path()).unlink()
    before = _series_path().read_text()
    result = _advance(_START + timedelta(days=6))
    assert result.appended == 0 and len(result.concerns) == 1
    assert "[WARN]" in capsys.readouterr().out
    assert _series_path().read_text() == before
    _assert_names_the_reinit_remedy(result.concerns[0])


def _assert_names_the_reinit_remedy(concern: str) -> None:
    """Review I2: the refusal says how to recover, and what recovering costs."""
    assert "scripts/init_coinflip_state.py --force" in concern
    assert "chore(data):" in concern and "its own" in concern
    assert "new seam" in concern and "METHODOLOGY" in concern


def test_a_state_behind_the_series_is_not_advanced(midas_data_root, capsys, monkeypatch):
    """A run that wrote the series but died before the state: the series is
    written first, so the next run sees the mismatch and recomputes nothing."""
    import engine.baselines as baselines

    _seed_store()
    _advance(_START + timedelta(days=3))

    def boom(*a, **k):
        raise RuntimeError("killed between the two writes")

    monkeypatch.setattr(baselines, "write_coin_flip_state", boom)
    with pytest.raises(RuntimeError):
        _advance(_START + timedelta(days=5))
    monkeypatch.undo()
    before = _series_path().read_text()
    result = _advance(_START + timedelta(days=7))
    assert result.appended == 0 and len(result.concerns) == 1
    assert _series_path().read_text() == before


def test_a_state_whose_value_differs_from_its_row_is_not_advanced(midas_data_root):
    _seed_store()
    _advance(_START + timedelta(days=3))
    path = coin_flip_state_path(_series_path())
    doc = json.loads(path.read_text())
    doc["portfolio_value"] += 1.0
    path.write_text(json.dumps(doc))
    result = _advance(_START + timedelta(days=6))
    assert result.appended == 0 and len(result.concerns) == 1
    _assert_names_the_reinit_remedy(result.concerns[0])


def test_a_state_without_a_series_is_not_advanced(midas_data_root):
    _seed_store()
    _advance(_START + timedelta(days=3))
    _series_path().unlink()
    result = _advance(_START + timedelta(days=6))
    assert result.appended == 0 and len(result.concerns) == 1
    assert not _series_path().exists()


def test_an_unreadable_state_is_not_advanced(midas_data_root):
    _seed_store()
    _advance(_START + timedelta(days=3))
    coin_flip_state_path(_series_path()).write_text("{not json")
    result = _advance(_START + timedelta(days=6))
    assert result.appended == 0 and len(result.concerns) == 1


_NAN, _INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    ("where", "field", "bad"),
    [
        ("holding", "shares", -1),
        ("holding", "shares", 1.5),
        ("holding", "shares", True),
        ("holding", "mark_close", 0),
        ("holding", "mark_close", -2.0),
        ("holding", "mark_close", _NAN),
        ("holding", "mark_close", _INF),
        ("holding", "mark_rate", 0),
        ("holding", "mark_rate", -0.9),
        ("holding", "mark_rate", _NAN),
        ("holding", "mark_rate", _INF),
        ("state", "cash", _NAN),
        ("state", "cash", _INF),
        ("state", "cash", -_INF),
        ("state", "portfolio_value", _NAN),
        ("state", "portfolio_value", _INF),
    ],
)
def test_a_state_with_an_invalid_number_is_refused(midas_data_root, where, field, bad):
    """Regression: round-3 review, 2026-10-06. The loader coerced with
    `int()`/`float()` only, so 0, negative, NaN and inf (which `json` reads)
    loaded, and `int(1.5)` truncated. Each is now a `ValueError`, and the
    advance refuses with the re-init remedy."""
    _seed_store()
    _advance(_START + timedelta(days=3))
    path = coin_flip_state_path(_series_path())
    doc = json.loads(path.read_text())
    assert doc["holdings"], "the fixture must hold something"
    target = doc if where == "state" else next(iter(doc["holdings"].values()))
    target[field] = bad
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=field):
        load_coin_flip_state(path)
    result = _advance(_START + timedelta(days=6))
    assert result.appended == 0 and len(result.concerns) == 1
    _assert_names_the_reinit_remedy(result.concerns[0])


def test_a_holding_of_zero_shares_still_loads(midas_data_root):
    _seed_store()
    _advance(_START + timedelta(days=3))
    path = coin_flip_state_path(_series_path())
    doc = json.loads(path.read_text())
    next(iter(doc["holdings"].values()))["shares"] = 0
    path.write_text(json.dumps(doc))
    load_coin_flip_state(path)


@pytest.mark.parametrize("universe", [["growth-stocks"], [], ["ZZZ.XX"]])
def test_a_universe_with_nothing_priceable_carries_the_book(
    midas_data_root, capsys, universe
):
    """Review fix 2, then regression: round-4 review, 2026-10-06.
    `resolve_agent_universe` returns a universe's bare name when its file is
    missing. With nothing priceable in it, the first step sold the whole
    established book to cash and drew nothing; the fix after it stopped the
    advance there, and since the universe stays missing every later run
    stopped at the same date. Now the book is carried untraded and revalued,
    the advance goes on, and one concern names the agent, the dates and the
    remedy for the cause."""
    _store("AAA", [(d, 10.0 + i) for i, d in enumerate(_days(_START, 5))])
    _store("ZZZ.XX", [(d, 10.0) for d in _days(_START, 5)])  # no currency
    _seed_state(_held_state("AAA", 100, "2026-01-01", 10.0, cash=50.0))
    result = _advance(date(2026, 1, 4), tickers=universe, max_positions=1)
    assert result.appended == 3 and len(result.concerns) == 1
    concern = result.concerns[0]
    assert _AGENT in concern and "2026-01-02..2026-01-04 (3 dates)" in concern
    assert "not sold to cash" in concern
    # The remedy follows the cause: the universe-file hint for a universe
    # with no close at all, the currency map for an unresolvable name.
    assert ("universe file" in concern) == (universe != ["ZZZ.XX"])
    assert ("ticker_currencies.json" in concern) == (universe == ["ZZZ.XX"])
    assert f"  [WARN] {concern}" in capsys.readouterr().out.splitlines()
    assert [r["portfolio_value"] for r in _rows()[1:]] == pytest.approx(
        [50.0 + 100 * 11.0, 50.0 + 100 * 12.0, 50.0 + 100 * 13.0]
    )
    state = load_coin_flip_state(coin_flip_state_path(_series_path()))
    assert state.date == "2026-01-04" and state.cash == 50.0
    assert state.holdings == {"AAA": CoinFlipHolding(100, "2026-01-04", 13.0, "USD", 1.0)}


def test_a_universe_whose_only_close_is_not_positive_carries_the_book(midas_data_root):
    """Regression: round-4 review, 2026-10-06. A close of 0 had a date, so the
    old pre-pass counted the day drawable; the step then sold the book to
    cash and skipped the 0 close at sizing, leaving all cash. A close that is
    not a positive number is no price: the day has no candidate and the
    book (BBB, which left the universe) is carried."""
    _store("AAA", [(d, 0.0) for d in _days(_START, 3)])
    _store("BBB", [(d, 20.0 + i) for i, d in enumerate(_days(_START, 3))])
    _seed_state(_held_state("BBB", 10, "2026-01-01", 20.0, cash=5.0))
    result = _advance(date(2026, 1, 3), tickers=["AAA"], max_positions=1)
    assert result.appended == 2 and len(result.concerns) == 1
    assert "NO_PRICE_DATA x1" in result.concerns[0]
    state = load_coin_flip_state(coin_flip_state_path(_series_path()))
    assert set(state.holdings) == {"BBB"} and state.cash == 5.0
    assert _rows()[-1]["portfolio_value"] == pytest.approx(5.0 + 10 * 22.0)


def _rate_withdrawn_on_day_3() -> None:
    """Two USD names in a EUR book; EURUSD=X is a zero row on 01-03 only, so
    01-03 is the one date with nothing drawable."""
    for t, c in (("U", 30.0), ("V", 40.0)):
        _store(t, [(d, c) for d in _days(_START, 4)])
    _store(
        "EURUSD=X",
        [("2026-01-01", 1.25), ("2026-01-02", 1.25), ("2026-01-03", 0.0), ("2026-01-04", 1.25)],
    )


def _cash_state(on: str, cash: float = 1_000.0) -> CoinFlipState:
    return CoinFlipState(date=on, portfolio_value=cash, cash=cash, holdings={})


def test_an_all_cash_book_keeps_advancing_through_an_undrawable_date(
    midas_data_root, capsys
):
    """Regression: round-4 review, 2026-10-06. The advance stopped before a
    date with nothing drawable even when the book was all cash, where a step
    loses nothing. It steps it in cash, says so as an [INFO] line, and buys
    again the next day."""
    _rate_withdrawn_on_day_3()
    _seed_state(_cash_state("2026-01-02"))
    result = _advance(date(2026, 1, 4), tickers=["U", "V"], max_positions=1, currency="EUR")
    assert result.appended == 2 and result.concerns == []
    out = capsys.readouterr().out
    assert "[WARN]" not in out
    assert any("[INFO]" in l and "2026-01-03" in l and _AGENT in l for l in out.splitlines())
    rows = _rows()
    assert [r["date"] for r in rows] == ["2026-01-02", "2026-01-03", "2026-01-04"]
    assert rows[1]["cash"] == 1_000.0
    state = load_coin_flip_state(coin_flip_state_path(_series_path()))
    assert state.date == "2026-01-04" and state.holdings, "it buys again on 01-04"


def test_a_book_with_liquid_holdings_is_carried_through_an_fx_gap(midas_data_root, capsys):
    """Regression: round-3 review, then round-4, 2026-10-06. The empty-draw
    guard first looked at the first new date only, and a rate withdrawn on a
    later date made the step sell the book to cash; the fix stopped the
    advance there for good. Now the book is carried on 01-03 (E.PA, a euro
    name that left the universe, is not sold), the advance goes on, and the
    concern names the rate ticker, the date and that the coin flip does not
    trade on a date that reads that row until it is revised; no universe-file
    hint, which is the wrong remedy for a rate."""
    _rate_withdrawn_on_day_3()
    _currencies({"E.PA": "EUR"})
    _store("E.PA", [(d, 50.0 + i) for i, d in enumerate(_days(_START, 4))])
    _seed_state(
        CoinFlipState(
            date="2026-01-02",
            portfolio_value=100.0 + 10 * 51.0,
            cash=100.0,
            holdings={"E.PA": CoinFlipHolding(10, "2026-01-02", 51.0, "EUR", 1.0)},
        )
    )
    result = _advance(date(2026, 1, 4), tickers=["U", "V"], max_positions=1, currency="EUR")
    assert result.appended == 2 and len(result.concerns) == 1
    concern = result.concerns[0]
    assert _AGENT in concern and "on 2026-01-03 (NO_FX_RATE x2)" in concern
    assert "EURUSD=X" in concern and "until the store's row is revised" in concern
    assert "universe file" not in concern
    assert [l for l in capsys.readouterr().out.splitlines() if "[WARN]" in l] == [
        f"  [WARN] {concern}"
    ]
    rows = _rows()
    assert rows[1]["date"] == "2026-01-03"
    assert rows[1]["portfolio_value"] == pytest.approx(100.0 + 10 * 52.0)
    assert rows[1]["cash"] == pytest.approx(100.0), "carried, not sold to cash"
    state = load_coin_flip_state(coin_flip_state_path(_series_path()))
    assert state.date == "2026-01-04" and "E.PA" not in state.holdings, "sold on 01-04"


def test_a_fresh_path_advances_through_a_later_date_with_nothing_drawable(midas_data_root):
    _rate_withdrawn_on_day_3()
    result = _advance(date(2026, 1, 4), tickers=["U", "V"], max_positions=1, currency="EUR")
    assert result.appended == 4
    assert load_coin_flip_state(coin_flip_state_path(_series_path())).date == "2026-01-04"


def test_an_unpriceable_universe_is_a_concern_of_the_build(midas_data_root, capsys):
    cfg = get_config()
    days = _days(_START, 10)
    _seed_benchmarks(cfg, days)
    _seed_store(n_days=10)
    universes = _desk_universes(cfg, ["AAA", "BBB", "CCC"])
    build_all_baselines(universes, _START, _START + timedelta(days=5))
    agent = next(iter(universes))
    capsys.readouterr()
    totals = build_all_baselines(
        {**universes, agent: ["missing-universe"]}, _START, _START + timedelta(days=9)
    )
    assert totals.concern == 1
    assert "[WARN] baselines: 1 concern(s)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Restart invariance (Hypothesis, `midas` profile)
# ---------------------------------------------------------------------------

_TICKERS = ["P", "Q", "R", "S"]


@given(
    paths=st.lists(
        st.lists(
            st.floats(min_value=0.5, max_value=500.0, allow_nan=False, allow_infinity=False),
            min_size=9,
            max_size=9,
        ),
        min_size=len(_TICKERS),
        max_size=len(_TICKERS),
    ),
    gaps=st.lists(st.booleans(), min_size=9, max_size=9),
    n=st.integers(min_value=1, max_value=7),
    max_positions=st.integers(min_value=0, max_value=5),
    book=st.sampled_from(["USD", "EUR"]),
)
def test_n_days_in_one_call_equal_n_one_day_calls(
    midas_data_root, paths, gaps, n, max_positions, book
):
    """Advancing N days in one call equals N one-day calls from the persisted
    state, in a native book and in a converted one (EUR book, USD names)."""
    import shutil

    days = _days(_START, 9)
    for ticker, closes in zip(_TICKERS, paths):
        # A gap (no bar) on a day exercises the last-close-on-or-before rule.
        _store(ticker, [(d, c) for d, c, g in zip(days, closes, gaps) if not g or d == days[0]])
    _store("EURUSD=X", [(d, 1.1 + 0.03 * ((i * 5) % 7)) for i, d in enumerate(days)])
    base = get_config().baselines_dir
    shutil.rmtree(base, ignore_errors=True)

    def run(to: date):
        return _advance(to, tickers=_TICKERS, max_positions=max_positions, currency=book)

    run(_START)
    run(_START + timedelta(days=n))
    one_call = (_series_path().read_text(), coin_flip_state_path(_series_path()).read_text())

    shutil.rmtree(base)
    run(_START)
    for k in range(1, n + 1):
        run(_START + timedelta(days=k))
    many_calls = (_series_path().read_text(), coin_flip_state_path(_series_path()).read_text())
    assert one_call == many_calls


# ---------------------------------------------------------------------------
# The build: both writers go through it, restatement is refused
# ---------------------------------------------------------------------------


def _desk_universes(cfg, tickers) -> dict[str, list[str]]:
    return {
        aid: tickers for aid in cfg.trading_roster if cfg.roster[aid].benchmark is not None
    }


def _seed_benchmarks(cfg, days: list[str]) -> None:
    for aid in cfg.trading_roster:
        spec = cfg.roster[aid].benchmark
        if spec is not None and not spec.is_cash_flat:
            _store(spec.ticker, [(d, 100.0 + i) for i, d in enumerate(days)])
    _store(cfg.global_reference.ticker, [(d, 100.0 + i) for i, d in enumerate(days)])
    # The desk's EUR books hold the bare (USD) test tickers: a flat 1.0 rate
    # keeps these build tests about the build, not about conversion.
    _store("EURUSD=X", [(d, 1.0) for d in days])


def test_a_coinflip_concern_is_counted_in_the_build(midas_data_root, capsys):
    """Review M2: a coin flip that cannot be advanced is a concern of the
    build, in its totals and its aggregate line, not only a stray [WARN]."""
    cfg = get_config()
    days = _days(_START, 10)
    _seed_benchmarks(cfg, days)
    _seed_store(n_days=10)
    universes = _desk_universes(cfg, ["AAA", "BBB", "CCC"])
    build_all_baselines(universes, _START, _START + timedelta(days=5))
    agent = next(iter(universes))
    coin_flip_state_path(cfg.baselines_dir / agent / "coinflip.json").unlink()
    capsys.readouterr()

    totals = build_all_baselines(universes, _START, _START + timedelta(days=9))

    out = capsys.readouterr().out
    assert totals.concern == 1
    assert "[WARN] baselines: 1 concern(s)" in out
    assert "coin flip" in out.split("[WARN] baselines:")[1]


@pytest.mark.parametrize("scope", [{"coinflip"}, {"goldfinger/coinflip"}])
def test_a_coinflip_restatement_scope_is_refused(midas_data_root, scope):
    with pytest.raises(ValueError, match="coin flip"):
        build_all_baselines({}, _START, _START, restate_series=scope, changelog_entry="x")


def test_a_replay_over_a_universe_refresh_gives_no_coinflip_concern(midas_data_root, capsys):
    """The 2026-09-26 shape: `3EUS.L` left a universe for `3EUS.MI` and every
    later coin-flip path moved. Now the refresh changes candidates from the
    next new date on and nothing published moves."""
    cfg = get_config()
    days = _days(_START, 10)
    _seed_benchmarks(cfg, days)
    _seed_store(n_days=10)
    _store("3EUS.MI", [(d, 4.0 + i / 10) for i, d in enumerate(days[3:])])
    build_all_baselines(_desk_universes(cfg, ["AAA", "BBB", "CCC"]), _START, _START + timedelta(days=5))
    published = {
        aid: (cfg.baselines_dir / aid / "coinflip.json").read_text()
        for aid in _desk_universes(cfg, [])
    }
    capsys.readouterr()
    build_all_baselines(
        _desk_universes(cfg, ["AAA", "BBB", "3EUS.MI", "DDD"]), _START, _START + timedelta(days=9)
    )
    out = capsys.readouterr().out
    assert "[WARN]" not in out
    for aid, before in published.items():
        rows = json.loads((cfg.baselines_dir / aid / "coinflip.json").read_text())
        assert rows[:6] == json.loads(before)
        assert len(rows) == 10


# ---------------------------------------------------------------------------
# Round-5 review, 2026-10-06
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, float("nan"), -3.0])
def test_a_holding_whose_newest_close_is_not_a_price_is_frozen_and_the_state_reloads(
    midas_data_root, bad
):
    """Regression: round-5 review, 2026-10-06. A held name whose newest close
    was 0 or NaN was revalued at it and carried with that ``mark_close``;
    ``load_coin_flip_state`` refuses it, so the next advance refused the state
    and the series stalled for good. The holding is frozen at its last valid
    mark (NO_PRICE_DATA), the state written round-trips, and the next advance
    goes on."""
    _store("AAA", [("2026-01-01", 10.0), ("2026-01-02", bad), ("2026-01-03", 11.0)])
    _store("BBB", [(d, 1e9) for d in _days(_START, 3)])  # drawable, too dear to buy
    _seed_state(_held_state("AAA", 100, "2026-01-01", 10.0, cash=5.0))
    result = _advance(date(2026, 1, 2), tickers=["AAA", "BBB"], max_positions=1)
    assert result.appended == 1
    assert len(result.concerns) == 1 and "AAA NO_PRICE_DATA" in result.concerns[0]
    assert "not a positive finite number" in result.concerns[0]
    state = load_coin_flip_state(coin_flip_state_path(_series_path()))
    assert state.holdings["AAA"] == CoinFlipHolding(100, "2026-01-01", 10.0, "USD", 1.0)
    assert _rows()[-1]["portfolio_value"] == pytest.approx(1_005.0)
    nxt = _advance(date(2026, 1, 3), tickers=["AAA", "BBB"], max_positions=1)
    assert nxt.appended == 1 and nxt.concerns == []
    assert _rows()[-1]["portfolio_value"] == pytest.approx(5.0 + 100 * 11.0)


def test_a_frozen_concern_is_worded_true_for_a_ratio_overflow_too(midas_data_root):
    """Regression: round-6 review, 2026-10-06. A holding frozen because the
    valuation ratio overflowed (every close finite and positive) was reported
    as "the newest close ... is not a positive finite number", which is
    false of it. The wording now covers the ratio."""
    _store("AAA", [("2026-01-01", 1.0), ("2026-01-02", 10.0)])
    _store("BBB", [(d, 1e9) for d in _days(_START, 2)])
    _seed_state(_held_state("AAA", 1, "2026-01-01", 1e308))
    result = _advance(date(2026, 1, 2), tickers=["AAA", "BBB"], max_positions=1)
    assert len(result.concerns) == 1 and "AAA NO_PRICE_DATA" in result.concerns[0]
    assert "valuation ratio" in result.concerns[0]
    assert "not finite" in result.concerns[0]


class _StubCloses:
    """A ``_Closes`` whose currency and rate answers are set per test."""

    book_currency = "EUR"

    def __init__(self, ccy: str | None, rate: float | None, reason: str | None) -> None:
        self._ccy, self._rate, self._reason = ccy, rate, reason

    def at(self, ticker: str, iso: str):
        return ("2026-01-01", 10.0) if ticker == "U" else None

    def currency(self, ticker: str):
        return self._ccy

    def rate(self, ccy, iso):
        return (self._rate, self._reason)


def test_a_frozen_holding_records_the_reason_current_at_the_end():
    """Regression: round-5 review, 2026-10-06. ``frozen.setdefault`` kept the
    first reason a holding froze for, so a holding that lost its rate and then
    its currency was still reported NO_FX_RATE at the end of the window, a
    condition that had cleared. The latest reason is recorded, with the
    holding's original mark."""
    from engine.baselines import CURRENCY_UNRESOLVED, NO_FX_RATE, _step

    held = CoinFlipHolding(10, "2026-01-01", 10.0, "USD", 0.8)
    frozen: dict = {}
    kw = dict(universe=[], excluded=set(), max_positions=1, frozen=frozen)
    s1 = _step(_AGENT, {"U": held}, 0.0, "2026-01-02", closes=_StubCloses("USD", None, NO_FX_RATE), **kw)
    assert frozen["U"] == (NO_FX_RATE, held)
    _step(_AGENT, s1.holdings, s1.cash, "2026-01-03", closes=_StubCloses(None, None, None), **kw)
    assert frozen["U"] == (CURRENCY_UNRESOLVED, held)


def test_an_empty_draw_over_a_book_of_carried_positions_is_a_concern(midas_data_root, capsys):
    """Regression: round-5 review, 2026-10-06. With nothing drawable and a
    book holding only a frozen position, the run printed "[INFO] ... nothing
    to sell; stepped in cash" of a book that was not in cash. It is a [WARN]
    concern naming the agent, the dates and the carried tickers."""
    _seed_state(_held_state("GONE", 100, "2026-01-01", 10.0, cash=0.0))
    result = _advance(date(2026, 1, 3), tickers=["NOFILE"], max_positions=1)
    assert result.appended == 2
    empty = [c for c in result.concerns if "no candidate" in c]
    assert len(empty) == 1
    assert _AGENT in empty[0] and "2026-01-02..2026-01-03 (2 dates)" in empty[0]
    assert "GONE" in empty[0] and "not in cash" in empty[0]
    out = capsys.readouterr().out
    assert "stepped in cash" not in out
    assert f"  [WARN] {empty[0]}" in out.splitlines()


def test_baselines_imports_fx_once_at_module_level():
    """Round-5 review, 2026-10-06: a function-local ``from engine.fx import``
    duplicated the module-level ``_fx`` import."""
    import ast

    import engine.baselines as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    local = [
        node.lineno
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.ImportFrom) and node.module == "engine.fx"
    ]
    assert local == []
