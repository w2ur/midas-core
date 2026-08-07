"""Reconcile fills whose notional was converted at the wrong quote currency.

Companion to ``scripts/restate_valuations.py``. That script re-prices
*valuations* and treats recorded cash as authoritative. This one addresses
the defect underneath: for a subset of fills the broker converted the trade
notional into the book's base currency using a currency it had **guessed
from the ticker suffix**, and guessed wrong. The resulting cash movement is
therefore wrong in the ledger itself — not merely mis-valued afterwards.

Three distinct failure modes, all fixed in ``engine/quotes.py`` (2026-08-07):

1. **Unit, not currency.** The LSE quotes in pence. ``LLOY.L`` at 116.60 is
   GBP 1.166, not GBP 116.60, so the fill debited 100x the true cost.
2. **Wrong currency outright.** ``.L`` is not uniformly sterling —
   ``PHAG.L`` and ``CRUD.L`` quote in USD but were converted as GBP.
3. **Currency pairs quote in their second leg.** ``EURJPY=X`` quotes in JPY,
   ``USDCAD=X`` in CAD, ``USDCHF=X`` in CHF, ``EURGBP=X`` in GBP — all four
   were converted as USD.

**The correction is multiplicative, never a recomputation from scratch,**
and that distinction is load-bearing. The FX rate a fill actually executed
at is not exactly reproducible today: re-deriving every recorded ``total``
from ``shares x price x convert(...)`` fails to reproduce **25 of the 203
unaffected trades** as well, by a median of 0.14% (worst 13.4%), because
the rate the broker used at execution time differs from the daily rate the
store now serves for that date — and because the FX series was itself swept
on this branch. Recomputing would therefore fold a silent FX-rate
restatement of every touched row into what must be a currency-only
correction, and would leave the corrected rows anchored to a different rate
source than the untouched ones.

So each affected fill's recorded ``total`` is **scaled** by

    factor = (native_price / recorded_price) * (rate_new / rate_old)

where ``rate_new``/``rate_old`` are same-date, same-source conversions of
one unit of the corrected and the guessed currency into the book's base
currency. The broker's own executed rate stays the anchor; only the
currency identification changes. For a pure unit error (``LLOY.L``, where
both readings are GBP) the factor collapses to exactly 1/100, which is the
right answer by inspection.

**Fees are recomputed only where a fee was actually charged.**
``engine/fees.py`` applies from its deploy date forward and explicitly does
not backfill; a fill recorded with ``fees == 0.0`` predates it and keeps
0.0 rather than acquiring a retroactive commission.

**Cash is re-derived, not patched.** After the totals are corrected, each
book's cash is recomputed as ``initial_capital + replay_holdings(...)`` over
its own corrected ledger — the same identity ``restate_valuations.py`` uses
as its book-level gate, so the two scripts agree by construction.

Every mutated record keeps its prior values under ``originally_recorded``
with a ``reconciliation_note``, following the precedent set by the
2026-08-02 ``sharp-shooter-eur`` reconciliation: the record shows what the
broker actually did as well as what it should have done.

Dry-run is the default — a tool that rewrites the published ledger does not
do so by accident. Pass ``--apply`` to write.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.config import get_config
from engine.fees import fee_for
from engine.fx import convert
from engine.quotes import quote_currency, ticker_currency
from engine.restatement import replay_holdings

NOTE = (
    "Notional converted at a currency guessed from the ticker suffix; "
    "corrected 2026-08-07 via engine/quotes.py. See METHODOLOGY.md, "
    "2026-08-07 entry."
)

# The pre-fix resolver, reproduced verbatim from paper_broker._ticker_currency
# as it stood at 33063100e^. Kept here rather than imported because the point
# is to reconstruct what the broker *did*, which the current code no longer
# does — and because a future edit to the live resolver must not silently
# change this script's idea of the defect.
_OVERRIDE_SENTINEL = object()


def _old_currency(ticker: str, overrides: dict[str, str]) -> str:
    if ticker in overrides:
        return overrides[ticker]
    if ticker.endswith("-EUR"):
        return "EUR"
    if ticker.endswith("-USD"):
        return "USD"
    if ticker.endswith((".PA", ".DE", ".AS", ".MI")):
        return "EUR"
    if ticker.endswith(".L"):
        return "GBP"
    if ticker.endswith(".SW"):
        return "CHF"
    if ticker.endswith(".T"):
        return "JPY"
    return "USD"


def _correction_factor(
    ticker: str, trade_date: date, base_currency: str, overrides: dict[str, str]
) -> tuple[float, str, str]:
    """Multiplicative correction for one ticker's recorded notional.

    Returns ``(factor, old_currency, new_currency)``. A factor of exactly
    1.0 means the fill was already correct and must not be touched.
    """
    old_ccy = _old_currency(ticker, overrides)
    new_ccy = ticker_currency(ticker)
    unit_scale = 0.01 if quote_currency(ticker) == "GBp" else 1.0

    if old_ccy == new_ccy and unit_scale == 1.0:
        return 1.0, old_ccy, new_ccy

    rate_old = convert(1.0, old_ccy, base_currency, trade_date)
    rate_new = convert(1.0, new_ccy, base_currency, trade_date)
    return unit_scale * (rate_new / rate_old), old_ccy, new_ccy


def _initial_capital(agent_id: str) -> float:
    """Inception cash for a book, back-solved the way restate_valuations does.

    Imported lazily from that module so the two cannot disagree.
    """
    import scripts.restate_valuations as rv

    root = Path(get_config().data_dir) / "data" / "portfolios" / agent_id
    trades = json.loads((root / "trades.json").read_text())
    snapshots = json.loads((root / "snapshots.json").read_text())
    return rv._initial_capital(agent_id, trades, snapshots)


def reconcile(apply: bool) -> int:
    from engine.quotes import _load_ticker_currency_overrides

    overrides = _load_ticker_currency_overrides()
    root = Path(get_config().data_dir) / "data"
    portfolios = root / "portfolios"

    agents = sorted(p.name for p in portfolios.iterdir() if p.is_dir())
    touched_ids: dict[str, tuple[float, str, str]] = {}
    cash_moves: dict[str, tuple[float, float]] = {}
    total_fills = 0

    for agent_id in agents:
        pf_path = portfolios / agent_id / "portfolio.json"
        tr_path = portfolios / agent_id / "trades.json"
        if not pf_path.exists() or not tr_path.exists():
            continue

        portfolio = json.loads(pf_path.read_text())
        trades = json.loads(tr_path.read_text())
        base = portfolio.get("currency", "EUR")

        changed = []
        for trade in trades:
            # Idempotence is not optional here: the correction is multiplicative,
            # so a second pass over an already-reconciled trade would square the
            # factor silently. A record carrying this script's own note is done.
            if trade.get("reconciliation_note") == NOTE:
                continue
            trade_date = datetime.fromisoformat(trade["timestamp"]).date()
            factor, old_ccy, new_ccy = _correction_factor(
                trade["ticker"], trade_date, base, overrides
            )
            if factor == 1.0:
                continue

            unit_scale = 0.01 if quote_currency(trade["ticker"]) == "GBp" else 1.0
            new_total = trade["total"] * factor
            new_price = trade["price"] * unit_scale
            # Pre-fee-model fills carry 0.0 and must not acquire a fee.
            new_fees = (
                0.0
                if trade.get("fees", 0.0) == 0.0
                else fee_for(trade["ticker"], new_total)
            )

            changed.append(
                {
                    "id": trade["id"],
                    "ticker": trade["ticker"],
                    "date": trade_date.isoformat(),
                    "action": trade["action"],
                    "old_total": trade["total"],
                    "new_total": new_total,
                    "old_price": trade["price"],
                    "new_price": new_price,
                    "old_fees": trade.get("fees", 0.0),
                    "new_fees": new_fees,
                    "old_ccy": old_ccy,
                    "new_ccy": new_ccy,
                }
            )
            touched_ids[trade["id"]] = (factor, old_ccy, new_ccy)

            if apply:
                trade.setdefault("originally_recorded", {}).update(
                    {
                        "price": trade["price"],
                        "total": trade["total"],
                        "fees": trade.get("fees", 0.0),
                        "assumed_currency": old_ccy,
                    }
                )
                trade["reconciliation_note"] = NOTE
                trade["price"] = new_price
                trade["total"] = new_total
                trade["fees"] = new_fees

        if not changed:
            continue

        total_fills += len(changed)
        print(f"=== {agent_id} ({base}) ===")
        for c in changed:
            print(
                f"  {c['date']}  {c['action']:<4} {c['ticker']:<10} "
                f"{c['old_ccy']}->{c['new_ccy']}  "
                f"total {c['old_total']:>10,.2f} -> {c['new_total']:>10,.2f}"
                + (
                    f"   price {c['old_price']:.4f} -> {c['new_price']:.4f}"
                    if c["old_price"] != c["new_price"]
                    else ""
                )
            )

        capital = _initial_capital(agent_id)
        old_cash = portfolio["cash"]
        corrected = json.loads(tr_path.read_text()) if not apply else trades
        if not apply:
            # Model the corrected ledger without writing it.
            by_id = {c["id"]: c for c in changed}
            corrected = [
                {
                    **t,
                    "total": by_id[t["id"]]["new_total"],
                    "fees": by_id[t["id"]]["new_fees"],
                }
                if t["id"] in by_id
                else t
                for t in corrected
            ]
        new_cash = capital + replay_holdings(corrected, date(2100, 1, 1))[1]
        cash_moves[agent_id] = (old_cash, new_cash)
        print(
            f"  cash {old_cash:,.2f} -> {new_cash:,.2f}  ({new_cash - old_cash:+,.2f})\n"
        )

        if apply:
            portfolio["cash"] = new_cash
            for pos in portfolio.get("positions", []):
                if (
                    quote_currency(pos["ticker"]) == "GBp"
                    and pos.get("reconciliation_note") != NOTE
                ):
                    pos.setdefault("originally_recorded", {}).setdefault(
                        "avg_cost", pos["avg_cost"]
                    )
                    pos["reconciliation_note"] = NOTE
                    pos["avg_cost"] = pos["avg_cost"] * 0.01
            tr_path.write_text(json.dumps(trades, indent=2) + "\n")
            pf_path.write_text(json.dumps(portfolio, indent=2) + "\n")

    # --- inbox fill rows -----------------------------------------------------
    inbox_changed = 0
    for channel in ("inbox", "manager-inbox"):
        directory = root / "orders" / channel
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            out, dirty = [], False
            for line in lines:
                row = json.loads(line)
                hit = touched_ids.get(row.get("order_id", ""))
                if (
                    hit is None
                    or row.get("status") != "filled"
                    or row.get("reconciliation_note") == NOTE
                ):
                    out.append(line)
                    continue
                factor, old_ccy, new_ccy = hit
                unit_scale = 0.01 if old_ccy == new_ccy else 1.0
                row.setdefault("originally_recorded", {}).update(
                    {
                        "fill_price": row.get("fill_price"),
                        "fill_currency": row.get("fill_currency"),
                        "notional_base": row.get("notional_base"),
                        "fees": row.get("fees"),
                    }
                )
                row["reconciliation_note"] = NOTE
                if row.get("fill_price") is not None:
                    row["fill_price"] = row["fill_price"] * unit_scale
                row["fill_currency"] = new_ccy
                if row.get("notional_base") is not None:
                    row["notional_base"] = row["notional_base"] * factor
                    if row.get("fees"):
                        row["fees"] = fee_for(
                            row.get("ticker", ""), row["notional_base"]
                        )
                out.append(json.dumps(row))
                dirty = True
                inbox_changed += 1
            if dirty and apply:
                path.write_text("\n".join(out) + "\n")

    print("=== Summary ===")
    print(f"  fills reconciled:   {total_fills}")
    print(f"  inbox rows updated: {inbox_changed}")
    print("  cash movement per book:")
    for agent_id, (old, new) in sorted(
        cash_moves.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0])
    ):
        print(f"    {agent_id:<18} {old:>12,.2f} -> {new:>12,.2f}  ({new - old:+,.2f})")
    print(f"  mode: {'APPLY' if apply else 'DRY RUN'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--apply", action="store_true", help="Write the reconciled ledger."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print without writing (default behavior).",
    )
    args = parser.parse_args()
    return reconcile(apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
