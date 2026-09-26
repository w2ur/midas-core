"""Store-level gap detection (`engine.store_gaps`).

Regression: 634222b53 — follow-up money review r6, I1: a hole was detected
only from the rows the vendor served inside the night's own window. On the
next night the one-day revision window never asked for that date again, the
run exited 0, and `failure-issue` closed the alert as "Recovered" while the
date stayed missing from the store for good. These tests pin the detector that
reads the STORED series instead.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from engine.store_gaps import (
    GAP_LOOKBACK_TRADING_DAYS,
    Verdict,
    bucket_traded,
    lookback_start,
    parse_ledger,
    render_ledger,
    scan_store,
    verdict,
)

# A fixed fortnight: 2026-09-14 (Mon) .. 2026-09-25 (Fri).
WEEK1 = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"]
WEEK2 = ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"]
DAYS = WEEK1 + WEEK2
START, END = DAYS[0], DAYS[-1]


def _bucket(symbol: str) -> str:
    head, dot, tail = symbol.rpartition(".")
    if dot and head:
        return f".{tail}"
    if symbol.endswith("-USD"):
        return "crypto"
    return ""


def _scan(store: dict[str, set[str]], scope: set[str] | None = None, start: str = START):
    return scan_store(
        {s: frozenset(d) for s, d in store.items()},
        bucket_of=_bucket,
        scope=set(store) if scope is None else scope,
        start=start,
        end=END,
    )


def _all_but(missing: str) -> set[str]:
    return {d for d in DAYS if d != missing}


class TestLookback:
    def test_counts_weekdays_back_from_end(self) -> None:
        assert lookback_start(date(2026, 9, 25), 5) == date(2026, 9, 21)
        assert lookback_start(date(2026, 9, 25), 6) == date(2026, 9, 18)

    def test_a_weekend_end_counts_from_the_friday_before(self) -> None:
        assert lookback_start(date(2026, 9, 27), 1) == date(2026, 9, 25)

    def test_the_default_spans_six_calendar_weeks(self) -> None:
        # 30 weekdays back from a Friday is the Monday six weeks earlier.
        assert GAP_LOOKBACK_TRADING_DAYS == 30
        assert lookback_start(date(2026, 9, 25)) == date(2026, 8, 17)

    def test_refuses_a_non_positive_window(self) -> None:
        with pytest.raises(ValueError, match="at least one trading day"):
            lookback_start(date(2026, 9, 25), 0)


class TestMemberGaps:
    def test_a_date_most_of_the_bucket_holds_is_a_gap_for_the_member_lacking_it(self) -> None:
        # The SAP.DE shape: 09-16, then 09-18, while its exchange traded 09-17.
        store = {f"P{i}.DE": set(DAYS) for i in range(4)}
        store["SAP.DE"] = _all_but("2026-09-17")
        scan = _scan(store)
        assert scan.member_gaps == {"SAP.DE": frozenset({"2026-09-17"})}
        assert scan.bucket_candidates == {}

    def test_exactly_half_is_not_a_majority(self) -> None:
        # Two of four hold it: the bucket's own calendar is undecided, so it is
        # not a member gap. It is a weekday, so the vendor probe decides.
        store = {"A.DE": set(DAYS), "B.DE": set(DAYS)}
        store["C.DE"] = _all_but("2026-09-17")
        store["D.DE"] = _all_but("2026-09-17")
        assert _scan(store).member_gaps == {}
        assert set(_scan(store).bucket_candidates) == {(".DE", "2026-09-17")}
        assert set(_scan(store).bucket_candidates[(".DE", "2026-09-17")]) == {"C.DE", "D.DE"}

    def test_dates_before_a_symbols_first_row_are_never_counted(self) -> None:
        # A first ingest has no history before its first date; that is not a gap.
        store = {f"P{i}.DE": set(DAYS) for i in range(4)}
        store["NEW.DE"] = set(WEEK2)
        assert _scan(store).member_gaps == {}

    def test_dates_after_a_symbols_last_row_are_not_interior_gaps(self) -> None:
        # A symbol that stopped (a day-late UCITS fund, a dead listing) is a
        # freshness question the nightly window owns, not a hole in its series.
        store = {f"P{i}.DE": set(DAYS) for i in range(4)}
        store["LATE.DE"] = set(DAYS[:-1])
        assert _scan(store).member_gaps == {}

    def test_dates_before_the_lookback_are_not_scanned(self) -> None:
        store = {f"P{i}.DE": set(DAYS) for i in range(4)}
        store["SAP.DE"] = _all_but("2026-09-15")
        assert _scan(store, start="2026-09-16").member_gaps == {}
        assert _scan(store).member_gaps == {"SAP.DE": frozenset({"2026-09-15"})}

    def test_an_out_of_scope_symbol_votes_but_is_not_reported(self) -> None:
        store = {f"P{i}.DE": set(DAYS) for i in range(4)}
        store["SAP.DE"] = _all_but("2026-09-17")
        store["BMW.DE"] = _all_but("2026-09-17")
        scan = _scan(store, scope={"BMW.DE"})
        assert scan.member_gaps == {"BMW.DE": frozenset({"2026-09-17"})}


class TestBucketCandidates:
    def test_a_us_date_spy_holds_and_most_of_the_bucket_lacks_is_a_candidate(self) -> None:
        # 2026-09-22: 94 of 576 US files held it, SPY among them.
        store = {"SPY": set(DAYS), "QQQ": set(DAYS)}
        for i in range(5):
            store[f"US{i}"] = _all_but("2026-09-22")
        scan = _scan(store)
        assert scan.member_gaps == {}
        assert set(scan.bucket_candidates) == {("", "2026-09-22")}
        assert set(scan.bucket_candidates[("", "2026-09-22")]) == {f"US{i}" for i in range(5)}

    def test_a_us_date_spy_also_lacks_is_undecided_not_a_holiday(self) -> None:
        # Regression: 82adee88d — follow-up review r7, I-1. A US holiday (Labor Day) and a
        # vendor hole that took SPY with it (09-22, had SPY not come back on
        # its own) are the same input here. SPY lacking the date used to mean
        # "closed", before any vendor probe; now it is a candidate, and the
        # probe decides.
        store = {"SPY": _all_but("2026-09-21"), "^VIX": set(DAYS)}
        for i in range(5):
            store[f"US{i}"] = _all_but("2026-09-21")
        scan = _scan(store)
        assert set(scan.bucket_candidates) == {("", "2026-09-21")}
        assert scan.member_gaps == {}

    def test_a_date_another_exchange_traded_is_a_candidate_not_a_verdict(self) -> None:
        # Both a wholesale vendor hole (.CO on 09-17) and a bank holiday (.L on
        # 08-31) look like this in the store. The candidate is what the vendor
        # is then asked about; the scan does not decide.
        store = {f"P{i}.PA": set(DAYS) for i in range(4)}
        for i in range(3):
            store[f"L{i}.L"] = _all_but("2026-09-17")
        scan = _scan(store)
        assert set(scan.bucket_candidates) == {(".L", "2026-09-17")}

    def test_a_date_no_other_equity_bucket_traded_is_still_undecided(self) -> None:
        # Regression: 82adee88d — follow-up review r7, I-1. A hole across every equity
        # bucket, or a European hole on a US holiday that also hits `.L`, left
        # no bucket holding the date, and it read as a holiday. A weekday a
        # non-calendar bucket lacks wholesale is undecided: both buckets go to
        # the vendor probe.
        store = {f"P{i}.PA": _all_but("2026-09-17") for i in range(4)}
        for i in range(3):
            store[f"L{i}.L"] = _all_but("2026-09-17")
        store["BTC-USD"] = set(DAYS) | {"2026-09-19", "2026-09-20"}
        assert set(_scan(store).bucket_candidates) == {
            (".PA", "2026-09-17"),
            (".L", "2026-09-17"),
        }

    def test_a_weekend_is_never_an_equity_candidate(self) -> None:
        store = {f"P{i}.PA": set(DAYS) for i in range(4)}
        store["BTC-USD"] = set(DAYS) | {"2026-09-19", "2026-09-20"}
        assert _scan(store).bucket_candidates == {}

    def test_a_crypto_day_is_always_a_trading_day(self) -> None:
        # 2026-08-05: every crypto pair missing the same day.
        store = {f"C{i}-USD": set(DAYS) | {"2026-09-20"} for i in range(3)}
        scan = _scan(store)
        assert set(scan.bucket_candidates) == {("crypto", "2026-09-19")}

    def test_probe_order_puts_the_best_covered_member_first(self) -> None:
        store = {"SPY": set(DAYS), "QQQ": set(DAYS)}
        store["THIN"] = {"2026-09-14", "2026-09-25"}
        for i in range(3):
            store[f"US{i}"] = _all_but("2026-09-22")
        order = _scan(store).bucket_candidates[("", "2026-09-22")]
        assert order[-1] == "THIN"


class TestVerdict:
    def test_a_served_close_fills(self) -> None:
        assert verdict({"2026-09-16": True, "2026-09-17": True}, "2026-09-17") is Verdict.FILLED

    def test_a_served_row_without_a_close_is_a_real_hole(self) -> None:
        assert verdict({"2026-09-17": False}, "2026-09-17") is Verdict.NO_CLOSE

    def test_a_date_the_vendors_own_series_skips_was_not_traded(self) -> None:
        served = {"2026-09-16": True, "2026-09-18": True}
        assert verdict(served, "2026-09-17") is Verdict.NOT_TRADED

    def test_no_frame_is_unfetched(self) -> None:
        assert verdict(None, "2026-09-17") is Verdict.UNFETCHED
        assert verdict({}, "2026-09-17") is Verdict.UNFETCHED

    def test_a_series_that_does_not_bracket_the_date_says_nothing_about_it(self) -> None:
        # 3EUS.L and SGLN.MI since 2026-09: the vendor serves one row, today's
        # quote, for any window. "No row for 09-17" there is not "did not trade".
        assert verdict({"2026-09-25": True}, "2026-09-17") is Verdict.UNFETCHED


class TestBucketTraded:
    def test_any_probe_served_the_date_means_it_traded(self) -> None:
        assert bucket_traded([Verdict.NOT_TRADED, Verdict.NO_CLOSE]) is True
        assert bucket_traded([Verdict.UNFETCHED, Verdict.FILLED]) is True

    def test_probes_whose_series_skip_it_mean_closed(self) -> None:
        assert bucket_traded([Verdict.NOT_TRADED, Verdict.UNFETCHED]) is False

    def test_no_evidence_is_unknown_not_closed(self) -> None:
        assert bucket_traded([Verdict.UNFETCHED, Verdict.UNFETCHED]) is None
        assert bucket_traded([]) is None


class TestLedger:
    def test_round_trip_is_stable(self) -> None:
        entries = {"SAP.DE": {"2026-09-17": "no-close"}, "AI.PA": {"2026-07-31": "unfetched"}}
        text = render_ledger(entries)
        assert parse_ledger(text) == entries
        assert render_ledger(parse_ledger(text)) == text
        assert text.endswith("\n")

    def test_an_empty_ledger_is_an_empty_object(self) -> None:
        assert render_ledger({}) == "{}\n"
        assert parse_ledger("{}\n") == {}

    @pytest.mark.parametrize(
        "text",
        [
            "[]",
            '{"SAP.DE": ["2026-09-17"]}',
            '{"SAP.DE": {"17 Sept": "no-close"}}',
            '{"SAP.DE": {"2026-09-17": "lost"}}',
            "{",
        ],
    )
    def test_a_malformed_ledger_raises(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_ledger(text)


class TestAcceptedEntries:
    """Regression: 579af1cd4 — an acceptance must carry its reason."""

    ACCEPTED = {"status": "accepted", "reason": "split-basis rebase", "accepted_on": "2026-09-26"}

    def test_an_accepted_entry_round_trips(self) -> None:
        entries = {"BYND": {"2026-08-13": dict(self.ACCEPTED)}, "AI.PA": {"2026-07-31": "unfetched"}}
        text = render_ledger(entries)
        assert parse_ledger(text) == entries
        assert render_ledger(parse_ledger(text)) == text

    @pytest.mark.parametrize(
        "entry, message",
        [
            ({"status": "accepted", "accepted_on": "2026-09-26"}, "reason"),
            ({"status": "accepted", "reason": "", "accepted_on": "2026-09-26"}, "reason"),
            ({"status": "accepted", "reason": "  ", "accepted_on": "2026-09-26"}, "reason"),
            ({"status": "accepted", "reason": "x"}, "accepted_on"),
            ({"status": "accepted", "reason": "x", "accepted_on": "26/09"}, "accepted_on"),
            ({"status": "waived", "reason": "x", "accepted_on": "2026-09-26"}, "status"),
        ],
    )
    def test_an_incomplete_acceptance_raises(self, entry: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_ledger(json.dumps({"BYND": {"2026-08-13": entry}}))


class TestANullRowIsUndecided:
    """Follow-up review r8 (r4 I1): a null row (no close, no volume) is how the
    vendor serves a real trading day it has no price for, and sometimes a
    holiday. Only "no row at all" is `NOT_TRADED`."""

    def test_a_null_row_is_undecided(self) -> None:
        served = {"2026-03-20": True, "2026-03-23": None, "2026-03-24": True}
        assert verdict(served, "2026-03-23") is Verdict.UNDECIDED

    def test_undecided_never_reaches_the_ledger(self) -> None:
        from engine.store_gaps import OPEN_VERDICTS

        assert Verdict.UNDECIDED not in OPEN_VERDICTS
        with pytest.raises(ValueError):
            parse_ledger('{"X.CO": {"2026-03-23": "undecided"}}')

    def test_positive_hourly_volume_is_traded(self) -> None:
        from engine.store_gaps import intraday_traded

        vols = {"2026-03-20": 248_960.0, "2026-03-23": 144_824.0, "2026-03-24": 1.0}
        assert intraday_traded(vols, "2026-03-23") is True

    def test_no_hourly_volume_inside_the_series_is_closed(self) -> None:
        from engine.store_gaps import intraday_traded

        assert intraday_traded({"2026-08-28": 5.0, "2026-09-01": 5.0}, "2026-08-31") is False
        assert intraday_traded({"2026-08-28": 5.0, "2026-08-31": 0.0, "2026-09-01": 5.0}, "2026-08-31") is False

    def test_no_hourly_evidence_is_unknown(self) -> None:
        from engine.store_gaps import intraday_traded

        assert intraday_traded(None, "2026-03-23") is None
        assert intraday_traded({}, "2026-03-23") is None
        assert intraday_traded({"2026-03-24": 5.0}, "2026-03-23") is None
