"""Store-level gap detection: trading days a symbol's stored series skips.

Follow-up money review r6, I1. Until 2026-09-26 a hole was seen only in the
rows the vendor served inside the night's own window (`MergeResult.holes`). On
night N the vendor served date D with no close and the run went red; on night
N+1 the symbol's last stored date was still D-1, the one-day revision window
asked for D-1..D+1, the vendor answered that short window with NO row for D at
all, D+1 was appended, the run exited 0, and `failure-issue` closed the alert
as "Recovered". D then stayed missing for good — 434 of the 1,205 equity
files whose span covers 2026-09-17 lacked it and 483 of 1,200 lacked 2026-09-22,
with non-holiday European gaps on 07-31 and 09-07 too — and every consumer
that counts rows (N-row lookbacks, momentum windows, daily returns, backtests,
the coin flip) read a series with holes in it.

So this module reads the STORED series, not tonight's window:

- A bucket (the same buckets the per-exchange hole detector uses) trades on the
  dates a MAJORITY of its members hold, counting only members whose stored span
  covers the date: a symbol has no history before its first row (a first
  ingest) and no interior gap after its last (a stale or dead symbol is a
  freshness question the nightly window owns). A member lacking such a date has
  a **member gap**.
- A date the bucket lacks wholesale is either a holiday or a vendor hole, and
  the store cannot tell them apart — `.L` on the 2026-08-31 bank holiday
  (0 of 157) looks exactly like `.CO` on 2026-09-17 (0 of 23). Every such
  weekday (every day, for crypto) is a **bucket candidate**, and no
  store-derived reference may call it closed: a reference can sit in the same
  hole (follow-up review r7, I-1 — SPY missing with the US, or no equity
  bucket holding the date). The vendor decides: asked over a wide window, it serves a
  closed day with no row inside an otherwise continuous series (measured
  2026-09-26: BP.L 08-31 and SPY 09-07 absent, SAP.DE 09-07/09-17, AI.PA 07-31
  and SPY 09-22 all present with a close).

Pure and I/O-free, like `engine.ohlcv_ingest.merge_rows`; the vendor calls and
the store writes live in `scripts/fetch_ohlcv.py`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

#: How far back each nightly run scans the stored series for a missing
#: trading day, in weekdays (~6 calendar weeks). It bounds DETECTION only:
#: once found, a gap is remembered by the ledger until the store holds it, so
#: it cannot age out of this window into a false "Recovered". What the window
#: has to cover is therefore the longest stretch with no full-universe run
#: plus the day a gap takes to become interior (D is a gap only once D+1 is
#: stored). The longest measured stretch is the 2026-08-25..09-04 outage (a
#: required check refused every bot push: nine trading days); 30 is three times
#: that. It also spans the 20-trading-day (one-month) return window the
#: momentum selectors rank on. Wider buys nothing the ledger does not already
#: hold, and reaches back into pre-day-one history this detector was not
#: calibrated on.
GAP_LOOKBACK_TRADING_DAYS = 30

#: Share of a bucket's in-span members that must hold a date for it to be the
#: bucket's own trading day. Strictly more than half: a date only half the
#: bucket holds is undecided, and is left to the reference and the vendor.
BUCKET_MAJORITY = 0.5

#: Members asked about a bucket candidate before it is called a holiday. More
#: than one, because a thin name can skip a day the exchange traded; the best
#: covered members go first. Three per bucket and date keeps a healthy night's
#: extra vendor requests to a handful.
PROBE_SIZE = 3

#: How far before the oldest date it asks about a refetch opens, in calendar
#: days. Wide, for the midas-market-data skill's request-shape rule: the vendor
#: served 2026-08-05 crypto over a long window and not over a short one, and the
#: I1 miss was the one-day window; 90 is the window `resweep-held-tickers` asks
#: for every week. Before the date rather than on it, because `verdict` calls a
#: day "not traded" only when the vendor's series runs across it.
HEAL_WINDOW_DAYS = 90

#: Buckets that trade every calendar day, weekends included.
EVERY_DAY_BUCKETS = frozenset({"crypto"})



def lookback_start(end: date, trading_days: int = GAP_LOOKBACK_TRADING_DAYS) -> date:
    """The earliest weekday of a window of ``trading_days`` weekdays ending at ``end``."""
    if trading_days < 1:
        raise ValueError("a lookback needs at least one trading day")
    d = end
    counted = 0
    while True:
        if d.weekday() < 5:
            counted += 1
            if counted == trading_days:
                return d
        d -= timedelta(days=1)


@dataclass(frozen=True)
class GapScan:
    #: Symbol -> dates its bucket traded (by majority) that it lacks.
    member_gaps: dict[str, frozenset[str]]
    #: (bucket, date) -> the in-scope symbols lacking a date their bucket does
    #: not hold by majority, on a day it could have traded. Ordered
    #: best-covered first: the vendor is probed in that order.
    bucket_candidates: dict[tuple[str, str], tuple[str, ...]]


def _days(start: str, end: str) -> list[str]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]


def _weekday(d: str) -> bool:
    return date.fromisoformat(d).weekday() < 5


def scan_store(
    dates_by_symbol: Mapping[str, frozenset[str]],
    *,
    bucket_of: Callable[[str], str],
    scope: Iterable[str],
    start: str,
    end: str,
) -> GapScan:
    """Every trading day in ``[start, end]`` an in-scope symbol's series skips.

    ``dates_by_symbol`` is the whole store: every member votes on its bucket's
    calendar, in scope or not, so a targeted run reads the same calendar as a
    full one. Only symbols in ``scope`` are reported.
    """
    in_scope = set(scope)
    spans = {s: (min(ds), max(ds)) for s, ds in dates_by_symbol.items() if ds}
    members: dict[str, list[str]] = {}
    for symbol in sorted(spans):
        members.setdefault(bucket_of(symbol), []).append(symbol)
    days = _days(start, end)

    majority: dict[str, set[str]] = {}
    lacking_by: dict[tuple[str, str], list[str]] = {}
    for bucket, symbols in members.items():
        majority[bucket] = set()
        for d in days:
            in_span = [s for s in symbols if spans[s][0] <= d <= spans[s][1]]
            if not in_span:
                continue
            lacking = [s for s in in_span if d not in dates_by_symbol[s]]
            if 1 - len(lacking) / len(in_span) > BUCKET_MAJORITY:
                majority[bucket].add(d)
            if lacking:
                lacking_by[(bucket, d)] = lacking

    def could_trade(bucket: str, d: str) -> bool:
        # No store-derived reference may call a weekday closed: a reference
        # can sit in the same hole (follow-up review r7, I-1 — SPY missing
        # with the US bucket, or no equity bucket holding the date at all),
        # and then the hole read as a holiday and the run exited 0. A weekday
        # a bucket lacks wholesale is undecided; the vendor probe
        # (`bucket_traded`) tells a closed day from a hole.
        return bucket in EVERY_DAY_BUCKETS or _weekday(d)

    def coverage(symbol: str) -> int:
        return sum(1 for d in dates_by_symbol[symbol] if start <= d <= end)

    member_gaps: dict[str, set[str]] = {}
    bucket_candidates: dict[tuple[str, str], tuple[str, ...]] = {}
    for (bucket, d), lacking in sorted(lacking_by.items()):
        reported = [s for s in lacking if s in in_scope]
        if not reported:
            continue
        if d in majority[bucket]:
            for s in reported:
                member_gaps.setdefault(s, set()).add(d)
        elif could_trade(bucket, d):
            bucket_candidates[(bucket, d)] = tuple(
                sorted(reported, key=lambda s: (-coverage(s), s))
            )
    return GapScan(
        {s: frozenset(ds) for s, ds in sorted(member_gaps.items())},
        bucket_candidates,
    )


class Verdict(str, Enum):
    """What the vendor's wide-window answer says about one missing date."""

    #: Served with a close: insert it.
    FILLED = "filled"
    #: Served with no close: the vendor knows the day and has no price. Held.
    NO_CLOSE = "no-close"
    #: The vendor's own series runs across the date with no row for it: the
    #: instrument did not trade (a holiday, a thin name's quiet day).
    NOT_TRADED = "not-traded"
    #: Nothing to judge by — no frame, or one that does not reach across the
    #: date (a vendor serving only today's quote). Unknown, never healthy.
    UNFETCHED = "unfetched"


#: The verdicts that leave a gap open, and so the reasons a ledger may record.
OPEN_VERDICTS = frozenset({Verdict.NO_CLOSE, Verdict.UNFETCHED})


def verdict(served: Mapping[str, bool] | None, d: str) -> Verdict:
    """Classify date ``d`` against what the vendor served: date -> has a close."""
    if not served:
        return Verdict.UNFETCHED
    if d in served:
        return Verdict.FILLED if served[d] else Verdict.NO_CLOSE
    if min(served) < d < max(served):
        return Verdict.NOT_TRADED
    return Verdict.UNFETCHED


def bucket_traded(probes: Iterable[Verdict]) -> bool | None:
    """Whether a bucket candidate's date traded, from its probes' verdicts.

    ``True`` when any probe was served the date, with or without a close — the
    vendor knows it as a trading day. ``False`` when none was, and at least one
    probe's series ran across it without a row: closed. ``None`` when no probe
    had evidence either way, which is not a holiday.
    """
    seen = list(probes)
    if any(v in (Verdict.FILLED, Verdict.NO_CLOSE) for v in seen):
        return True
    if any(v is Verdict.NOT_TRADED for v in seen):
        return False
    return None


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_LEDGER_REASONS = frozenset(v.value for v in OPEN_VERDICTS) | {"quarantined"}

#: The status of a gap a human has accepted as unfillable. Such an entry is an
#: object, ``{"status": "accepted", "reason": ..., "accepted_on": ...}``, not
#: an open reason string. It is green, and the scan never re-opens its date.
#: The reason is required: an acceptance nobody can explain is not a decision,
#: and a ledger carrying one is unreadable, which is never green. BYND
#: 2026-08-13 was the first: the vendor serves it on the post-split basis
#: against a pre-split 08-12, so filling it needs a basis rebase of the
#: history, and that is not insert-only.
ACCEPTED = "accepted"


def is_accepted(entry: object) -> bool:
    """Whether a parsed ledger entry is an accepted gap rather than an open one."""
    return isinstance(entry, dict) and entry.get("status") == ACCEPTED


def accepted_entry(reason: str, accepted_on: date) -> dict[str, str]:
    """The ledger entry that accepts a gap. Refuses an empty reason."""
    if not reason.strip():
        raise ValueError("an accepted gap needs a non-empty reason")
    return {"status": ACCEPTED, "reason": reason.strip(), "accepted_on": accepted_on.isoformat()}


def _check_accepted(symbol: str, d: str, entry: dict) -> None:
    if entry.get("status") != ACCEPTED:
        raise ValueError(f"{symbol} {d}: unknown status {entry.get('status')!r}")
    reason = entry.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{symbol} {d}: an accepted gap needs a non-empty reason")
    if not _ISO_DATE.fullmatch(str(entry.get("accepted_on", ""))):
        raise ValueError(f"{symbol} {d}: an accepted gap needs an ISO accepted_on date")
    extra = set(entry) - {"status", "reason", "accepted_on"}
    if extra:
        raise ValueError(f"{symbol} {d}: unexpected keys {sorted(extra)}")


def parse_ledger(text: str) -> dict[str, dict[str, str | dict[str, str]]]:
    """Read ``data/market/store_gaps.json``: symbol -> {date: entry}.

    An entry is an open reason (``no-close``, ``unfetched``, ``quarantined``)
    or an accepted gap (`ACCEPTED`). Raises ``ValueError`` on anything else,
    including an acceptance without a reason. The caller must not treat an
    unreadable ledger as an empty one: that would forget every held gap and
    report the store healthy.
    """
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("the ledger must be an object keyed by symbol")
    entries: dict[str, dict[str, str | dict[str, str]]] = {}
    for symbol, gaps in raw.items():
        if not isinstance(gaps, dict):
            raise ValueError(f"{symbol}: expected {{date: entry}}")
        for d, entry in gaps.items():
            if not _ISO_DATE.fullmatch(str(d)):
                raise ValueError(f"{symbol}: {d!r} is not an ISO date")
            if isinstance(entry, dict):
                _check_accepted(symbol, d, entry)
            elif not isinstance(entry, str) or entry not in _LEDGER_REASONS:
                raise ValueError(f"{symbol} {d}: unknown reason {entry!r}")
        entries[symbol] = dict(gaps)
    return entries


def render_ledger(entries: Mapping[str, Mapping[str, object]]) -> str:
    """Serialise the ledger. Stable, so an unchanged night writes no diff."""
    return json.dumps(entries, indent=2, sort_keys=True) + "\n"
