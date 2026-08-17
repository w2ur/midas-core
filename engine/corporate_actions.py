"""Corporate-action detection and position adjustment — stock splits.

Built after commit 4b6b8556 corrected 29,348 rows in the committed OHLCV
store, ~10k of them 11 tickers whose pre-split history had never been
restated. Nothing in the engine handled corporate actions before this
module: a position held through a split kept its pre-split share count
while the store's price became post-split, silently mis-valuing the book
by the split ratio (shares too low, price too low, and the two errors
don't cancel — they multiply in the same direction).

Two pure primitives, no I/O, matching ``engine.restatement``'s style:

- ``detect_split`` compares a symbol's already-stored history against a
  freshly fetched series over their overlapping dates and returns the split
  ratio if — and only if — the disagreement has a real split's *structure*:
  a transition date exists (rows after it already agree), the drifted rows
  sit before it, and the drifted rows nearest that transition all share one
  ratio within a tight tolerance. Getting this wrong in the false-positive
  direction is worse than the bug it fixes: a spurious detection would
  silently multiply a real position by a bogus ratio, unattended (the
  weekly ``--resweep-held`` workflow runs with no human in the loop).

- ``apply_split`` scales a held position's ``shares``/``avg_cost`` by the
  detected ratio, preserving cost basis (``shares × avg_cost``) exactly.

Calibration is against **all 11 real splits** the store carries (the
tickers corrected by commit ``4b6b8556``, replayed as the pre-sweep series
from ``314035e34`` against today's values) plus the three real Class-D
drift symbols (``SIE.DE``, ``ALV.DE``, ``BMW.DE``) that must never fire.
Every threshold below cites the real measurement that set it, and
``tests/test_corporate_actions.py`` pins all 14 as embedded fixtures.

Two ways a *real* split departs from a textbook single-constant-ratio
signature, both measured, both of which the first version of this module
mistook for "not a split":

- an isolated **bad tick** inside the drifted run (``FCIT.L`` carries one:
  ``2026-05-08`` stored at 5275.02 against a ~330 range, a ratio of 16.0
  where every one of its 2 531 other drifted rows is exactly 4.0);
- an **older, second body of drift** deeper in the same history (``WLN.PA``
  has 2 524 rows at ratio 0.0968 sitting before the 67 rows at 0.025 that
  are its actual 40-for-1 reverse split).

Both defeat any statistic taken over *all* drifted rows at once — a
max-minus-min spread reads 300% for ``FCIT.L`` and 75% for ``WLN.PA`` — so
the cluster is instead anchored on the drifted rows **nearest the
transition**, which is also the only ratio that can matter to a currently
held position.

The two are then treated differently, and the asymmetry is the point. A
lone bad tick is stepped over. A *second body* of drift is refused
outright, because it is bit-for-bit the shape of a **stacked split** (rows
before split1 at ``split1 × split2``, rows between the two at ``split2``,
then a clean tail) and no single ratio can correct it — ``apply_split``
multiplies every share of the ticker by one factor, while the right factor
depends on when each position was opened. ``WLN.PA``'s older body is, per
Yahoo's own calendar, *not* a second split; it is nonetheless
indistinguishable from one in the data this module receives, so ``WLN.PA``
is a **documented known miss** on a full-history overlap (it is still
detected under the 90-day window the weekly job actually uses, where the
older body falls outside the range). 10 of 11 detected, with the eleventh
refused for a stated reason, beats 11 of 11 with a silent wrong-ratio path.
See ``detect_split``'s docstring for the full rule and
``tests/test_corporate_actions.py`` for the measurement behind the miss.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from datetime import date, timedelta
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    # Type-only: importing it at runtime would make this module depend on the
    # ingest layer, and `engine.ohlcv_ingest` is the layer that will call back
    # into the adjudication helpers below.
    from engine.ohlcv_ingest import QuarantinedRow

# A row counts as "drifted" once its ratio departs from 1.0 by more than
# this. Real unchanged rows in this store are bit-identical (measured 0.0%
# deviation on CRWD/TIT.MI's post-split dates when diffing 4b6b8556^ against
# 4b6b8556) — this is a generous floor to absorb ordinary float noise, set
# far below the smallest real split's deviation from 1.0 (HON, 4.65%).
_DRIFT_TOLERANCE = 0.003

# A split candidate needs at least this many drifted rows before its ratios
# are even clustered. Real splits carry hundreds (CRWD 552, DD 546, TIT.MI
# 2570); the store's single-row "Day-1" errors (Class B in the sweep
# measurement) are exactly 1 row, so this alone rules those out without
# needing to look at the ratio at all.
_MIN_DRIFTED_ROWS = 10

# At least this many overlapping dates must exist between `stored` and
# `fetched` before concluding anything — "a handful", per the design brief.
_MIN_OVERLAP_ROWS = 10

# Max relative distance from the cluster's anchor ratio for a drifted row to
# count as part of that cluster — i.e. how far apart "the same ratio" may
# be. Measured against the 11 real splits: nine clusters are exact (0.0%
# internal spread — CRWD, DD, HON, SPGI, KLAC, CVNA, FDX, and AI.PA/TIT.MI
# at 0.26%/0.51%); the loosest is WLN.PA at 2.23% (2-decimal store rounding
# on a very low post-split price, ~EUR 0.50). Measured against real ordinary
# drift by fetching live data for ALV.DE (4.20% spread across 24 drifted
# rows) and BMW.DE (6.08% across 32): both comfortably clear this bar. 3%
# leaves ~0.8 percentage points of margin below the tightest real drift and
# ~0.8 above the loosest real split — WLN.PA is the closest real split ever
# measured to this boundary.
_CLUSTER_TOLERANCE = 0.03

# The cluster's anchor ratio must differ from 1.0 by at least this much to
# be worth calling a split, rather than some benign near-1 systematic
# shift. HON (0.9535, 4.65% from 1.0) is the closest real split to this
# floor — ~1.65 percentage points of margin. A threshold tuned around a
# "round" ratio like 2:1 or 4:1 would miss both HON (0.9535) and SPGI
# (1.057); this floor is set by the closest real case, not a round number.
# Real Class-D drift sits on the other side of it by a wide margin: SIE.DE
# 0.9930, ALV.DE 0.9940, BMW.DE 0.9926 — all inside 0.8% of 1.0.
_MATERIALITY_FLOOR = 0.03

# How many of the most recent drifted rows are used to anchor the cluster.
# The anchor is their MEDIAN, so it survives up to (n-1)//2 bad ticks among
# them: FCIT.L's real 16.0 tick is the newest drifted row of its ten and
# does not move the anchor off 4.0. Held at _MIN_DRIFTED_ROWS — a cluster
# has to be at least this big to be believed anyway, so anchoring on fewer
# rows than that would be anchoring on evidence too thin to act on.
_ANCHOR_ROWS = 10

# Non-conforming drifted rows tolerated *inside* the cluster before the walk
# concludes it has reached a genuinely different cluster. Set from the one
# real bad-tick class the store carries: FCIT.L has exactly 1 such row in
# 2 533 drifted rows. WLN.PA's older 0.0968 block — 2 524 consecutive
# non-conforming rows — ends the walk immediately, which is the whole point:
# the walk must stop at the boundary of the split's own cluster rather than
# average across two.
_MAX_CLUSTER_STRAYS = 2

# Drifted rows tolerated *after* the transition date. Measured requirement
# across all 11 real splits: zero — every one of them has a perfectly clean
# post-transition suffix (24 to 62 rows). The allowance exists because the
# production window is short (`--resweep-held --history-days 90`) and a bad
# tick of the FCIT.L class can land on the post-split side of the
# transition, where demanding zero exceptions would veto a real split on one
# stray row. 2 sits an order of magnitude below the smallest real negative
# (ALV.DE puts 24 drifted rows after its first undrifted one; SIE.DE 28,
# BMW.DE 32), so it cannot open the door to Class-D drift.
_MAX_POST_TRANSITION_STRAYS = 2


def _date_key(ts: object) -> str:
    return ts.date().isoformat() if hasattr(ts, "date") else str(ts)


def detect_split(stored: list[dict], fetched: pd.DataFrame) -> float | None:
    """Return the split ratio if `stored` and `fetched` disagree by one constant factor.

    Compares the ``close`` field of `stored` (rows as read from a
    ``{SYMBOL}.jsonl`` store file — the same shape ``engine.market_data``
    and ``engine.ohlcv_ingest`` use) against the ``Close`` column of
    `fetched` (a yfinance-shaped DataFrame, ``auto_adjust=False``) over
    their overlapping dates. Deliberately reads raw ``close``, not
    ``adj_close``: Yahoo restates raw ``close`` itself for a split (which is
    exactly the corruption this function detects) but re-bases
    ``adj_close`` after every dividend too, which would otherwise look
    identical to a split under a naive comparison — see the pre-flight
    finding in this plan's ledger.

    A split's signature, established empirically against all 11 tickers
    corrected in commit 4b6b8556: every date on or before the split shows
    the SAME ratio (``stored_close / fetched_close``), and every date after
    it is unchanged (ratio 1.0) — a drifted PREFIX ending at one transition
    date. Ordinary drift (real ``SIE.DE``/``ALV.DE``/``BMW.DE``) looks
    nothing like this: a couple of dozen rows scattered THROUGHOUT the
    history, interleaved with unchanged rows on both sides, at ratios both
    spread wider than any real split's cluster (4.2-6.1% vs. the loosest
    real split's 2.23%) and far nearer 1.0 than any real split (within
    0.8%).

    Four conditions, in order, all required:

    1. **A transition exists.** At least one overlapping row must already
       agree. This fails closed on a wholly-drifted overlap, where there is
       no observable transition at all — and that case is what a units
       mismatch looks like (a London ticker stored in GBp against a fetch
       in GBP is a clean, tight, material 100.0 across every row), which no
       statistic over the ratios can tell apart from a split. It costs
       nothing on real data: the store is appended to nightly, so any
       window reaching back past a split date necessarily also contains the
       post-split rows the cron has since appended at correct prices — all
       11 real splits carry 24 to 62 such rows. The one shape it refuses is
       a resweep run on the split date itself, before any post-split row
       exists; that is deliberately traded away, since the next weekly run
       catches it and a corrupted position cannot be un-corrupted.
    2. **The drift sits before the transition** — at most
       ``_MAX_POST_TRANSITION_STRAYS`` drifted rows after it, for isolated
       bad ticks. This is the check real scattered drift cannot survive.
    3. **The cluster is anchored on the transition.** The anchor ratio is
       the median of the last ``_ANCHOR_ROWS`` drifted rows, and must clear
       ``_MATERIALITY_FLOOR``. Anchoring at the transition rather than
       taking a statistic over every drifted row is what survives an
       isolated bad tick (``FCIT.L``); the recent ratio is in any case the
       only one that can apply to a position held today.
    4. **The cluster is tight and big enough.** Walking back from the
       newest drifted row, rows within ``_CLUSTER_TOLERANCE`` of the anchor
       join the cluster; more than ``_MAX_CLUSTER_STRAYS`` non-conforming
       rows end the walk. The cluster must still reach
       ``_MIN_DRIFTED_ROWS``.
    5. **Nothing of substance is left behind the cluster.** Fewer than
       ``_MIN_DRIFTED_ROWS`` drifted rows may sit older than the cluster.
       A second body there is the signature of a stacked split, which no
       single ratio can correct; refusing costs ``WLN.PA`` on a
       full-history overlap and is why it is a documented known miss.

    Returns ``None`` on anything that doesn't match. A missed detection
    leaves the pre-existing valuation bug in place; a false one would
    silently multiply a real position, unattended.

    Parameters
    ----------
    stored:
        Rows as read from the committed store, each at least
        ``{"date": "YYYY-MM-DD", "close": float}``.
    fetched:
        A DataFrame indexed by date-like values with a ``Close`` column
        (yfinance's raw, non-adjusted shape).

    Returns
    -------
    float | None
        The detected ratio (``stored_close / fetched_close``, e.g. ``4.0``
        for a 4-for-1 forward split, ``0.025`` for WLN.PA's 40-for-1
        reverse split), or ``None`` if no split is detected.
    """
    stored_close: dict[str, float] = {
        r["date"]: r["close"] for r in stored if r.get("close") is not None
    }

    dated_ratios: list[tuple[str, float]] = []
    for ts, row in fetched.iterrows():
        d = _date_key(ts)
        old = stored_close.get(d)
        if old is None or old == 0:
            continue
        new = row["Close"]
        if new is None:
            continue
        try:
            new = float(new)
        except (TypeError, ValueError):
            continue
        if new == 0 or math.isnan(new):
            continue
        dated_ratios.append((d, old / new))

    dated_ratios.sort(key=lambda pair: pair[0])  # chronological — required below

    if len(dated_ratios) < _MIN_OVERLAP_ROWS:
        return None

    is_drifted = [abs(r - 1.0) > _DRIFT_TOLERANCE for _, r in dated_ratios]
    drifted_positions = [i for i, d in enumerate(is_drifted) if d]
    if len(drifted_positions) < _MIN_DRIFTED_ROWS:
        return None

    # (1) A transition must exist. No undrifted row anywhere in the overlap
    # means no observable "and then it agreed again" — the shape a units
    # mismatch (GBp vs GBP, a clean 100.0 on every row) also has. Fail
    # closed: see condition 1 in the docstring for why this costs nothing
    # against the 11 real splits.
    first_undrifted = next((i for i, d in enumerate(is_drifted) if not d), None)
    if first_undrifted is None:
        return None

    # (2) The drift must sit before that transition. A calibrated allowance
    # rather than zero exceptions: one bad tick on the post-split side must
    # not veto a real split, but real scattered drift puts 24-32 rows here.
    strays_after = sum(1 for i in drifted_positions if i >= first_undrifted)
    if strays_after > _MAX_POST_TRANSITION_STRAYS:
        return None

    # (3) Anchor the cluster on the drifted rows NEAREST the transition —
    # the only ratio that can apply to a position held today. Median, so
    # isolated bad ticks among the anchor rows (FCIT.L) do not move it.
    anchor = statistics.median(
        dated_ratios[i][1] for i in drifted_positions[-_ANCHOR_ROWS:]
    )
    if anchor <= 0 or abs(anchor - 1.0) < _MATERIALITY_FLOOR:
        return None

    # (4) Walk back from the newest drifted row collecting rows that share
    # the anchor ratio. Isolated non-conforming rows are bad ticks and are
    # stepped over; more than _MAX_CLUSTER_STRAYS of them means the walk has
    # left this split's cluster and entered a different one, so it stops
    # there rather than averaging the two together.
    #
    # `strays_inside` deliberately shares no budget with
    # _MAX_POST_TRANSITION_STRAYS but the two ARE consumed by the same rows
    # when a post-transition stray also fails to conform: such a row is
    # counted once in (2) and again here. That double-count can only end the
    # walk earlier, i.e. shrink the cluster and push toward None — it can
    # never widen a cluster or move the returned ratio. Left as-is.
    cluster: list[float] = []
    oldest_in_cluster = drifted_positions[-1]
    strays_inside = 0
    for i in reversed(drifted_positions):
        ratio = dated_ratios[i][1]
        if abs(ratio - anchor) / anchor <= _CLUSTER_TOLERANCE:
            cluster.append(ratio)
            oldest_in_cluster = i
            continue
        strays_inside += 1
        if strays_inside > _MAX_CLUSTER_STRAYS:
            break
    if len(cluster) < _MIN_DRIFTED_ROWS:
        return None

    # (5) Nothing of substance may be left behind the cluster. A second body
    # of drift older than the cluster is the signature of a STACKED split —
    # rows before split1 at ratio split1*split2, rows between the two at
    # split2, then the undrifted tail. Returning the cluster's ratio there
    # would be a confident, clean-looking detection that silently
    # under-corrects every position opened before split1 by exactly split1.
    #
    # Refusing is right even when the older body is NOT a second split,
    # because a single ratio cannot express the correction either way:
    # `apply_split` multiplies every share count of the ticker by one
    # factor, while in a stacked history the correct factor depends on when
    # each position was opened. Measured cost: WLN.PA, whose older 2 524-row
    # cluster at 0.0968 is (per Yahoo's own calendar) NOT a split — but is
    # indistinguishable from one in the data this function receives. See
    # tests/test_corporate_actions.py::
    # test_detect_split_refuses_wln_pa_because_it_cannot_be_told_from_a_stacked_split
    # for the measurement and for what would have to change to detect it.
    older_drift = sum(1 for i in drifted_positions if i < oldest_in_cluster)
    if older_drift >= _MIN_DRIFTED_ROWS:
        return None

    return statistics.median(cluster)


def apply_split(positions: list[dict], ticker: str, ratio: float) -> list[dict]:
    """Scale a held position's shares/avg_cost for a detected stock split.

    Multiplies ``shares`` by ``ratio`` and divides ``avg_cost`` by it, so
    the cost basis (``shares × avg_cost``) is unchanged. Positions for
    every other ticker pass through untouched. Pure: returns a new list,
    never mutates `positions` or its dicts in place.

    Parameters
    ----------
    positions:
        Position dicts as stored in ``portfolio.json`` (at least
        ``ticker``, ``shares``, ``avg_cost`` — any other keys, e.g.
        ``date_opened``/``grid_level``, are preserved unchanged).
    ticker:
        The split-affected ticker.
    ratio:
        ``stored_close / fetched_close`` for the overlapping pre-split
        rows, as returned by ``detect_split`` — e.g. ``4.0`` for a 4-for-1
        forward split, ``0.025`` for a 40-for-1 reverse split.

    Returns
    -------
    list[dict]
        A new list with the matching ticker's position adjusted.

    Raises
    ------
    ValueError
        If ``ratio`` is not strictly positive.
    """
    if ratio <= 0:
        raise ValueError(f"split ratio must be positive, got {ratio}")

    adjusted: list[dict] = []
    for position in positions:
        if position["ticker"] != ticker:
            adjusted.append(position)
            continue
        new_position = dict(position)
        new_position["shares"] = position["shares"] * ratio
        new_position["avg_cost"] = position["avg_cost"] / ratio
        adjusted.append(new_position)
    return adjusted


# ---------------------------------------------------------------------------
# Calendar-based adjudication
# ---------------------------------------------------------------------------
#
# `detect_split` above INFERS a ratio from the disagreement between a stored
# series and a re-fetched one. That works only when the vendor restates history,
# and the vendor does not always restate: measured 2026-08-17, JMAT.L's 3:4
# moved all 2 601 stored rows while MNST's 2:1 had moved nothing six days on
# (its series still ran 90.36 -> 45.53 with adj_close equal to close on both
# sides). So inference is dark precisely when it matters, and what the
# unrestated split produces instead is a jump the ingest tripwire quarantines —
# freezing that symbol's store until a human intervenes.
#
# The authoritative answer costs nothing extra: the vendor publishes the action
# in its own calendar. These helpers are pure — the caller fetches the calendar
# and passes it in, so the decision is unit-testable without network and
# `engine.ohlcv_ingest` stays I/O-free.

#: A refused row's observed ratio may differ from the action's implied ratio by
#: this fraction and still be explained by it. The price genuinely moves on the
#: day of the action: measured, MNST's 08-11 row came in 0.4% off its implied
#: 0.5 and BYND's 08-13 row 1.7% off its implied 30. 10% clears those while
#: still separating a 2x split from a 100x units flip by an order of magnitude.
#: Erring tight fails SAFE — the row stays quarantined and a human adjudicates,
#: which is the behaviour this whole mechanism replaces.
RATIO_TOLERANCE = 0.10

#: How far an action's effective date may sit from the refused rows it explains.
#: Recency is the ONLY sound date constraint here — see `explain_quarantine`.
#: 7 days covers the widest real gap measured (JMAT.L: a revision of 08-12
#: explained by an action effective 08-17) with margin, while refusing an
#: ancient split whose ratio happens to match.
RECENCY_DAYS = 7


class CorporateAction(NamedTuple):
    """One action from the vendor's calendar.

    ``shares_ratio`` is the vendor's own convention and the multiplier on a
    HOLDING: 2.0 for a 2:1 split (twice the shares), 0.75 for a 3:4
    consolidation, 1/30 for a 1:30 reverse. It is what `apply_split` consumes.

    ``price_ratio`` is its inverse — the factor carrying a close from the old
    basis to the new one, which is what a quarantined row's ratio records.
    They are named rather than left as one bare float precisely because the
    inversion is easy to get backwards, and getting it backwards halves a real
    book's share count instead of doubling it.
    """

    symbol: str
    effective: str  # ISO date
    shares_ratio: float

    @property
    def price_ratio(self) -> float:
        return 1.0 / self.shares_ratio


def _within(observed: float, expected: float, tolerance: float) -> bool:
    """Relative comparison — an absolute one is meaningless across ratios.

    A 1:30 reverse implies a ratio of 30 where 10% is 3.0 wide; a 2:1 implies
    0.5 where the same 10% is 0.05.
    """
    if expected == 0 or observed <= 0:
        return False
    return abs(observed / expected - 1.0) <= tolerance


def ratios_agree(
    observed: float, expected: float, tolerance: float = RATIO_TOLERANCE
) -> bool:
    """Public form of the relative ratio comparison used by adjudication.

    Exported so the drift-inferred path (`detect_split`, which returns a bare
    ratio) can be matched to a calendar action on the SAME tolerance the
    calendar path uses. Two different tolerances for one question is how the
    two paths would start disagreeing about whether an action had already been
    applied — and the ledger key depends on that answer.
    """
    return _within(observed, expected, tolerance)


def _is_recent(action: CorporateAction, dates: list[str], days: int) -> bool:
    try:
        effective = date.fromisoformat(action.effective)
        first = date.fromisoformat(min(dates))
        last = date.fromisoformat(max(dates))
    except ValueError:
        return False
    return (first - timedelta(days=days)) <= effective <= (last + timedelta(days=days))


def explain_quarantine(
    rows: Sequence["QuarantinedRow"],
    actions: Sequence[CorporateAction],
    *,
    tolerance: float = RATIO_TOLERANCE,
    recency_days: int = RECENCY_DAYS,
) -> CorporateAction | None:
    """Return the single action explaining EVERY refused row, else ``None``.

    Adjudication is deliberately all-or-nothing per symbol. It triggers a
    full-history rewrite and a share-count mutation, and doing that while one
    refused row remains unaccounted for is how a real defect rides in behind a
    real corporate action.

    There is no per-row rule about which side of the action's effective date a
    row must fall on, and that is a finding rather than an omission. Two such
    rules were written and each was refuted by a real August 2026 fixture:
    JMAT.L's action POSTDATES the revision it explains (a revision is the
    vendor restating history), and BYND's 08-13 NEW ROW PREDATES its 08-14
    action (the vendor had already restated that bar and our store held no row
    to revise). Whether a bar arrives as a new row or a revision depends on
    what our store happens to hold; which basis it carries depends on whether
    the vendor has restated yet. Those are independent, so all four
    combinations are reachable and legitimate.

    What discriminates is that every refused row agrees on ONE ratio matching
    the action, and that the action belongs to this fetch window rather than to
    history. Both a calendar entry and a ratio match are required, so a wrong
    or stale calendar can never admit a bad row on its own.
    """
    if not rows:
        # Vacuous truth is not a licence to rewrite a store.
        return None

    dates = [r.date for r in rows]
    candidates = [
        action
        for action in actions
        if _is_recent(action, dates, recency_days)
        and all(_within(r.ratio, action.price_ratio, tolerance) for r in rows)
    ]
    # Exactly one, or nothing: two actions that each explain everything is an
    # ambiguity no single ratio can resolve, which is the stacked-split shape
    # `detect_split` already refuses.
    if len(candidates) != 1:
        return None
    return candidates[0]
