"""After-tax shadow ledger — reporting only.

Computes the French PFU (Prélèvement Forfaitaire Unique) tax drag on each
agent's realized P&L, derived purely from their executed trade log.  This
module is REPORTING ONLY — it never loads portfolio state and never writes
anywhere except ``data/tax_shadow/`` (via the script wrapper).

French tax regimes modeled
--------------------------
Two regimes are siloed under French law; crypto losses can NEVER offset
securities gains or vice-versa:

1. **Securities** (equity + FX tickers): per-ticker weighted-average cost
   basis (PRU — prix de revient unitaire).

   - BUY → (total + fees) added to the ticker's cost pool; shares increase.
   - SELL → gain = proceeds_net - (avg_cost_per_share * shares_sold),
     where proceeds_net = total - fees.  The remaining basis is reduced
     proportionally.

2. **Crypto** (crypto tickers as classified by ``engine.fees.classify_ticker``):
   global-portfolio weighted-average cost method (PVCT — plus-value sur
   cession de valeurs mobilières numériques, Art. 150 VH bis CGI).

   Under PVCT, every disposal's gain is:
       gain = proceeds - (total_acquisition_cost_net * proceeds / portfolio_FV_at_disposal)

   **v1 approximation**: The statutory PVCT formula requires marking the
   entire crypto portfolio to fair market value (FMV) at the time of each
   disposal.  This shadow ledger has no intraday live valuations.  As an
   approximation, we track per-coin cost pools (like the securities PRU)
   and use the per-coin weighted-average cost as the allocated basis for
   each disposal.  This is equivalent to the PVCT formula when portfolio
   FMV equals cost basis (i.e., at break-even), and is equivalent to the
   statutory formula when the agent holds a SINGLE crypto asset.  For
   multi-coin portfolios it understates allocated cost on losing positions
   and overstates it on winning positions.  The real tax filing must use
   live portfolio FMV per disposal.  This approximation is labelled
   ``method: "per-coin-PRU-approx"`` in the output JSON.

   DEBT: before real-money go-live, replace per-coin pooling with the true
   global crypto PVCT pool (live FMV per disposal) — today's €0 tax delta
   vs statutory is an artifact of every crypto book being in a loss year,
   not a structural guarantee.

   Note: under French *sursis d'imposition*, crypto-to-crypto swaps are
   tax-deferred.  The current agent universe trades crypto only against EUR
   fiat (e.g. BTC-EUR), so every crypto disposal here is a taxable fiat-out.
   If crypto-to-crypto swaps are introduced, this module must be updated.

Tax rate
--------
PFU rate is read from ``globals.jurisdiction.tax_rate_pct`` in roster.yaml
(e.g. 30.0 for France → 0.30 fraction) via ``_pfu_rate()``.  When no
jurisdiction block is present the rate defaults to 0.0 (no tax drag).
PFU applies as a flat rate on net positive annual realized gain per regime.
If the net is <= 0, PFU = 0 and the loss magnitude is recorded in
``realized_loss_by_year`` for transparency (French law has no PFU
loss carry-forward, but the amount is noted for future Manager reporting).

Usage
-----
::

    from engine.tax_shadow import compute_tax_shadow

    result = compute_tax_shadow(trades_list, agent="satoshi")
    # result["securities"]["pfu_due_by_year"]
    # result["crypto"]["lifetime_pfu"]
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from engine.fees import classify_ticker


def _pfu_rate() -> float:
    """Return the PFU rate as a fraction (e.g. 0.30 for FR).

    Reads ``globals.jurisdiction.tax_rate_pct`` from roster.yaml via
    ``get_config()``.  Defaults to 0.0 when no jurisdiction block is present
    (neutral / midas-core config).
    """
    from engine.config import get_config

    return get_config().jurisdiction.tax_rate_pct / 100.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_tax_shadow(trades: list[dict], agent: str = "") -> dict[str, Any]:
    """Compute the French PFU shadow ledger for an agent's trade history.

    Parameters
    ----------
    trades:
        List of trade dicts from ``data/portfolios/{agent}/trades.json``.
        Each dict must have: action ("BUY"|"SELL"), ticker, total (float,
        base-currency notional post-FX), fees (float), timestamp (ISO-8601),
        shares (float).  ``fees`` defaults to 0.0 if absent.
    agent:
        Agent identifier stored in the output for provenance.

    Returns
    -------
    dict
        Shadow ledger with keys: agent, generated_at, securities, crypto,
        notes.  Monetary values in the output are rounded to 2 decimal places;
        internal computation retains full float precision.
    """
    # Sort by timestamp so trades are processed in chronological order.
    sorted_trades = sorted(trades, key=lambda t: t["timestamp"])

    # Per-ticker cost pools for BOTH securities and crypto.
    # For securities this is the exact PRU; for crypto it is used as the v1
    # approximation of the PVCT allocated-cost denominator.
    # pool[ticker] = {"shares": float, "basis": float}
    # where basis = sum of (total + fees) for all buys minus disposed portions.
    pool: dict[str, dict[str, float]] = defaultdict(
        lambda: {"shares": 0.0, "basis": 0.0}
    )

    # Realized gain/loss per year, per regime (full precision — rounded only on output).
    sec_by_year: dict[str, float] = defaultdict(float)
    crypto_by_year: dict[str, float] = defaultdict(float)

    for trade in sorted_trades:
        action = trade["action"].upper()
        ticker = trade["ticker"]
        shares = float(trade["shares"])
        total = float(trade["total"])
        fees = float(trade.get("fees", 0.0))
        year = _extract_year(trade["timestamp"])
        is_crypto = classify_ticker(ticker) == "crypto"

        if action == "BUY":
            p = pool[ticker]
            p["shares"] += shares
            p["basis"] += total + fees

        elif action == "SELL":
            p = pool[ticker]
            proceeds_net = total - fees
            if p["shares"] > 0:
                avg_cost_per_share = p["basis"] / p["shares"]
                # Guard against oversell: cap fraction at 1.0 so basis/shares
                # never go negative (e.g. if SELL qty exceeds recorded holdings).
                # Use actual shares disposed (capped) for the allocated-cost calc.
                fraction_sold = min(1.0, shares / p["shares"])
                actual_shares_sold = min(shares, p["shares"])
                allocated_cost = avg_cost_per_share * actual_shares_sold
                gain = proceeds_net - allocated_cost
                p["basis"] -= p["basis"] * fraction_sold
                p["shares"] = max(0.0, p["shares"] - shares)
            else:
                # SELL with no prior BUY cost recorded — treat cost as zero.
                gain = proceeds_net

            if is_crypto:
                crypto_by_year[year] += gain
            else:
                sec_by_year[year] += gain

    # -- Aggregate by year into gain / loss / PFU dicts --
    sec = _aggregate_regime(sec_by_year)
    crypto = _aggregate_regime(crypto_by_year)
    crypto["method"] = "per-coin-PRU-approx"

    now_iso = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    return {
        "agent": agent,
        "generated_at": now_iso,
        "securities": sec,
        "crypto": crypto,
        "notes": [
            "Reporting only — no portfolio state mutated.",
            "Securities: per-ticker weighted-average PRU cost basis.",
            (
                "Crypto: per-coin weighted-average PRU used as v1 approximation of PVCT "
                "(per-coin-PRU-approx) — statutory formula requires live FMV at disposal; "
                "see module docstring for full caveat."
            ),
            "FX tickers (=X suffix) classified as securities regime.",
            "Crypto-to-crypto swaps are tax-deferred (sursis); none present in current universe.",
        ],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_regime(by_year: dict[str, float]) -> dict[str, Any]:
    """Convert a raw per-year gain/loss dict into the output regime structure.

    Positive net → recorded in realized_gain_by_year + pfu_due_by_year.
    Negative/zero net → recorded in realized_loss_by_year (absolute value); PFU = 0.
    All monetary values rounded to 2dp in the output dict.
    """
    gain_by_year: dict[str, float] = {}
    loss_by_year: dict[str, float] = {}
    pfu_by_year: dict[str, float] = {}
    lifetime_realized: float = 0.0
    lifetime_pfu: float = 0.0

    for year, net in sorted(by_year.items()):
        lifetime_realized += net
        if net > 0:
            pfu = _pfu_rate() * net
            gain_by_year[year] = round(net, 2)
            pfu_by_year[year] = round(pfu, 2)
            lifetime_pfu += pfu
        else:
            loss_by_year[year] = round(abs(net), 2)

    return {
        "realized_gain_by_year": gain_by_year,
        "realized_loss_by_year": loss_by_year,
        "pfu_due_by_year": pfu_by_year,
        "lifetime_realized": round(lifetime_realized, 2),
        "lifetime_pfu": round(lifetime_pfu, 2),
    }


def _extract_year(timestamp: str) -> str:
    """Extract the calendar year string from an ISO-8601 timestamp."""
    return timestamp[:4]
