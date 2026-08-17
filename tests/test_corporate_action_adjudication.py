"""Adjudicating a quarantined row against the vendor's own action calendar.

The ingest tripwire can refuse a row but nothing could ever accept one back.
A real corporate action produces the same shape as the units flip the tripwire
hunts, so a split froze the symbol's store indefinitely: `fetch-ohlcv` exited 2
on every full-universe run, and the stale price stayed tradable because the
broker's BUY band compares against the *prior stored close* from the same
frozen file (ratio 1.0 by construction).

`detect_split` could not close this. It infers the ratio from vendor
restatement drift, and the vendor does not always restate — measured
2026-08-17, in one sitting:

  * JMAT.L — Yahoo restated all 2 601 stored rows; detect_split fired at 0.75.
  * MNST   — Yahoo had NOT restated six days after the split. Its series still
             ran 90.36 (08-07) -> 45.53 (08-11), adj_close equal to close on
             both sides. A full resweep revised 50 rows (all volume
             settlement) and detected nothing.

So detection is dark precisely when it is needed. The calendar is
authoritative, free, and arrives in the same vendor response.

Fixtures are the three REAL actions of August 2026, transcribed from the
committed quarantine sidecars and Yahoo's split calendar. Embedded as literals
— never `git show` or network at test time, matching the discipline in
tests/test_corporate_actions.py (CI checks out at fetch-depth: 1).
"""

from __future__ import annotations

import pytest

from engine.corporate_actions import CorporateAction, explain_quarantine
from engine.ohlcv_ingest import QuarantinedRow

# --------------------------------------------------------------------------
# Real fixtures. Ratios are as the tripwire recorded them: incoming / stored.
# --------------------------------------------------------------------------

#: MNST 2:1 effective 2026-08-11. Yahoo did NOT restate history, so every
#: refused row is a NEW ROW on the post-split basis against a pre-split store.
MNST_ACTION = CorporateAction("MNST", "2026-08-11", shares_ratio=2.0)
MNST_ROWS = [
    QuarantinedRow("MNST", "2026-08-11", "new-row", 91.43, 45.529998, 0.497976),
    QuarantinedRow("MNST", "2026-08-12", "new-row", 91.43, 45.979999, 0.502898),
    QuarantinedRow("MNST", "2026-08-13", "new-row", 91.43, 46.680000, 0.510554),
    QuarantinedRow("MNST", "2026-08-14", "new-row", 91.43, 46.819999, 0.512085),
]

#: JMAT.L 3:4 effective 2026-08-17. Yahoo DID restate history, so the refused
#: row is a REVISION of a date BEFORE the action took effect — the opposite
#: date relationship from MNST, which is why one rule cannot cover both.
JMAT_ACTION = CorporateAction("JMAT.L", "2026-08-17", shares_ratio=0.75)
JMAT_ROWS = [
    QuarantinedRow("JMAT.L", "2026-08-12", "revision", 22.12, 29.493332, 1.333333),
]

#: BYND 1:30 reverse effective 2026-08-14.
BYND_ACTION = CorporateAction("BYND", "2026-08-14", shares_ratio=1 / 30)
BYND_ROWS = [
    QuarantinedRow("BYND", "2026-08-12", "revision", 0.4141, 12.423000, 30.000001),
    QuarantinedRow("BYND", "2026-08-13", "new-row", 0.4141, 12.465000, 30.101426),
]


class TestTheRatioArithmetic:
    """A stored close is on the old basis; an incoming one is on the new."""

    def test_price_ratio_is_the_inverse_of_the_shares_ratio(self):
        # 2:1 doubles your shares and halves the price.
        assert MNST_ACTION.shares_ratio == 2.0
        assert MNST_ACTION.price_ratio == pytest.approx(0.5)

    def test_a_consolidation_moves_both_the_other_way(self):
        # 3:4 leaves you 0.75 of your shares at 4/3 of the price.
        assert JMAT_ACTION.price_ratio == pytest.approx(4 / 3)

    def test_the_shares_ratio_is_what_apply_split_consumes(self):
        """`apply_split` multiplies shares by its ratio — so it takes THIS one.

        Feeding it `price_ratio` would scale a holding by the reciprocal:
        a 2:1 split would halve the share count instead of doubling it, on a
        real book, silently. The two differ by exactly the inversion that is
        easiest to get backwards, which is why they are named rather than
        being one bare float.
        """
        assert MNST_ACTION.shares_ratio != MNST_ACTION.price_ratio


class TestTheThreeRealActions:
    def test_mnst_is_explained_although_the_vendor_never_restated(self):
        """The case detect_split structurally cannot see."""
        assert explain_quarantine(MNST_ROWS, [MNST_ACTION]) == MNST_ACTION

    def test_jmat_is_explained_by_an_action_dated_after_the_refused_row(self):
        """A revision restates history, so the action POSTDATES the row.

        The first draft of the design required the action to fall on or before
        every refused row's date, which is right for MNST and backwards here —
        it would have failed to explain the very case the rule was derived
        from.
        """
        assert explain_quarantine(JMAT_ROWS, [JMAT_ACTION]) == JMAT_ACTION

    def test_bynd_is_explained_across_both_row_kinds_at_once(self):
        assert explain_quarantine(BYND_ROWS, [BYND_ACTION]) == BYND_ACTION


class TestWhatMustNeverBeAdjudicated:
    """The falsifiable half. Each of these must stay quarantined.

    A tripwire that can be talked into accepting anything is not a tripwire,
    and this is the money path: an accepted row becomes a fill price.
    """

    def test_a_units_flip_with_no_calendar_entry_stays_quarantined(self):
        """The GBp/GBP class — 100x, no corporate action anywhere."""
        rows = [QuarantinedRow("LLOY.L", "2026-08-12", "revision", 1.166, 116.6, 100.0)]
        assert explain_quarantine(rows, []) is None

    def test_a_calendar_entry_whose_ratio_disagrees_stays_quarantined(self):
        """A 2:1 on the calendar cannot license a 100x jump.

        Both a calendar entry AND a ratio match are required, so a wrong or
        stale calendar alone can never admit a bad row.
        """
        rows = [QuarantinedRow("MNST", "2026-08-11", "new-row", 91.43, 0.9143, 0.01)]
        assert explain_quarantine(rows, [MNST_ACTION]) is None

    def test_a_partially_explained_symbol_is_refused_whole(self):
        """One unexplained row poisons the batch, deliberately.

        Adjudication triggers a full-history rewrite plus a share-count
        mutation. Doing that while one refused row remains unaccounted for is
        how a real defect rides in behind a real split.
        """
        rows = MNST_ROWS + [
            QuarantinedRow("MNST", "2026-08-14", "revision", 91.43, 9143.0, 100.0)
        ]
        assert explain_quarantine(rows, [MNST_ACTION]) is None

    def test_an_ancient_action_cannot_explain_todays_jump(self):
        """Recency is the only sound date constraint. See the class below."""
        ancient = CorporateAction("MNST", "2019-01-02", shares_ratio=2.0)
        assert explain_quarantine(MNST_ROWS, [ancient]) is None

    def test_two_actions_splitting_the_rows_between_them_are_refused(self):
        """A stacked split: no single ratio can correct it.

        `apply_split` scales every share by one factor while the right factor
        depends on when each position was opened — the same shape
        `detect_split` already refuses outright.
        """
        second = CorporateAction("MNST", "2026-08-13", shares_ratio=3.0)
        rows = MNST_ROWS + [
            QuarantinedRow("MNST", "2026-08-15", "new-row", 91.43, 15.24, 1 / 6)
        ]
        assert explain_quarantine(rows, [MNST_ACTION, second]) is None

    def test_no_rows_is_not_an_adjudication(self):
        """Vacuously explaining nothing must not report a licence to rewrite."""
        assert explain_quarantine([], [MNST_ACTION]) is None


class TestWhyThereIsNoPerRowDateRule:
    """The date check is RECENCY only, and that is a finding, not laziness.

    Two drafts of this rule were wrong, each refuted by a real fixture:

      draft 1  "the action falls on or before every refused row"
               — refuted by JMAT.L, whose action (08-17) postdates the
                 revision it explains (08-12), because a revision IS the
                 vendor restating history.

      draft 2  "new-row => action on or before it; revision => action after"
               — refuted by BYND, whose 08-13 row is a NEW ROW dated BEFORE
                 its 08-14 action, because Yahoo had already restated the
                 08-13 bar onto the post-split basis and our store had no
                 08-13 row to revise.

    The reason no per-row ordering works: whether a given bar arrives as a new
    row or as a revision depends on whether our store happened to hold that
    date already, and which basis it carries depends on whether the vendor has
    restated yet — which MNST proves is not guaranteed even six days on. Those
    two independent facts make all four combinations reachable and legitimate.

    What survives is the constraint that actually discriminates: every refused
    row for the symbol agrees on ONE ratio matching the action, and the action
    belongs to this fetch window rather than to history.
    """

    RECENT = CorporateAction("MNST", "2026-08-11", shares_ratio=2.0)

    def test_a_new_row_before_its_action_is_fine(self):
        """BYND's real shape: vendor restated the bar before the action date."""
        rows = [
            QuarantinedRow("BYND", "2026-08-13", "new-row", 0.4141, 12.465, 30.101426)
        ]
        assert explain_quarantine(rows, [BYND_ACTION]) == BYND_ACTION

    def test_a_revision_before_its_action_is_fine(self):
        """JMAT.L's real shape."""
        assert explain_quarantine(JMAT_ROWS, [JMAT_ACTION]) == JMAT_ACTION

    def test_recency_is_bounded_on_both_sides(self):
        far_future = CorporateAction("MNST", "2027-01-04", shares_ratio=2.0)
        assert explain_quarantine(MNST_ROWS, [far_future]) is None


class TestTolerance:
    """10%, because the price genuinely moves on the day of the action.

    Measured on the real rows: MNST's 08-11 came in 0.4% off its implied 0.5
    and BYND's 08-13 1.7% off its implied 30. The bound still separates a 2x
    split from a 100x units flip by more than an order of magnitude, and the
    calendar match is the real gate. A too-tight bound fails SAFE — the row
    stays quarantined and a human adjudicates, which is today's behaviour.
    """

    def test_a_real_same_day_move_on_top_of_the_split_is_tolerated(self):
        # 2:1 with the stock also down 6% that day.
        rows = [QuarantinedRow("MNST", "2026-08-11", "new-row", 91.43, 42.97, 0.47)]
        assert explain_quarantine(rows, [MNST_ACTION]) == MNST_ACTION

    def test_a_move_beyond_the_bound_is_refused(self):
        rows = [QuarantinedRow("MNST", "2026-08-11", "new-row", 91.43, 36.57, 0.40)]
        assert explain_quarantine(rows, [MNST_ACTION]) is None

    def test_the_bound_is_relative_not_absolute(self):
        """A 1:30 reverse has a ratio of 30, where 10% is 3.0 wide.

        An absolute tolerance would be simultaneously absurd here and useless
        on a 0.5 ratio.
        """
        rows = [
            QuarantinedRow("BYND", "2026-08-13", "new-row", 0.4141, 12.84, 31.0)
        ]
        assert explain_quarantine(rows, [BYND_ACTION]) == BYND_ACTION
