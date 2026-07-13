"""engine.universes — aggregate re-exports + universe-name dispatch."""

from __future__ import annotations

from engine.universes.index import (
    get_sp500_tickers,
    get_dow30_tickers,
    get_nasdaq100_tickers,
    get_cac40_tickers,
    get_dax_tickers,
    get_ftse100_tickers,
    get_stoxx600_tickers,
)
from engine.universes.assets import (
    get_crypto_tickers,
    get_forex_tickers,
    get_metals_tickers,
    get_voo_only,
    get_classic_60_40,
    get_bearish_etf_tickers,
    get_crypto_eur_tickers,
    get_commodities_eur_tickers,
    get_bearish_etf_ucits_tickers,
    get_etf_sectors,
    get_etf_broad,
)
from engine.universes.alternative import (
    get_congressional_tickers,
    get_insider_tickers,
    get_high_short_tickers,
)


# Declared-but-unimplemented universes: present in VALID_UNIVERSES (so specs
# validate) but with no data source yet. They resolve to [] and are therefore
# rejected by the empty-resolution guard in resolve_universe below — never
# silently returned as an empty allowlist (which would disable the broker's
# TICKER_NOT_IN_UNIVERSE rail, i.e. fail open to allow-all).
_UNIMPLEMENTED: list[str] = ["dividend-aristocrats", "13f-whales"]


_RESOLVERS = {
    "sp500": get_sp500_tickers,
    "dow30": get_dow30_tickers,
    "nasdaq100": get_nasdaq100_tickers,
    "cac40": get_cac40_tickers,
    "dax": get_dax_tickers,
    "ftse100": get_ftse100_tickers,
    "stoxx-600": get_stoxx600_tickers,
    "crypto-top20": get_crypto_tickers,
    "crypto-top20-eur": get_crypto_eur_tickers,
    "forex-majors": get_forex_tickers,
    "metals-commodities": get_metals_tickers,
    "commodities-eur": get_commodities_eur_tickers,
    "single-voo": get_voo_only,
    "classic-60-40": get_classic_60_40,
    "bearish-etfs": get_bearish_etf_tickers,
    "bearish-etfs-ucits": get_bearish_etf_ucits_tickers,
    "congress": get_congressional_tickers,
    "insiders": get_insider_tickers,
    "high-short": get_high_short_tickers,
    "etf-sectors": get_etf_sectors,
    "etf-broad": get_etf_broad,
    **{name: (lambda: []) for name in _UNIMPLEMENTED},
}


def resolve_universe(name: str) -> list[str]:
    """Return the tickers for a named universe — the single universe registry.

    This is the one source of truth: the CLI backtesters and the backtester
    service all delegate here. Raises KeyError for unknown names AND for
    declared-but-unimplemented placeholders that resolve to an empty list.

    Refusing to return [] is deliberate: an empty allowlist silently disables
    the paper broker's TICKER_NOT_IN_UNIVERSE rail (paper_broker treats an
    empty allowed set as allow-everything). A placeholder must fail loudly, not
    fail open.
    """
    if name not in _RESOLVERS:
        raise KeyError(f"Unknown universe: {name!r}")
    tickers = list(_RESOLVERS[name]())
    if not tickers:
        raise KeyError(
            f"Universe {name!r} resolved to an empty ticker list — it is a "
            f"declared-but-unimplemented placeholder. Refusing to return [] "
            f"(an empty allowlist would fail open and disable the universe rail)."
        )
    return tickers


__all__ = ["resolve_universe"]
