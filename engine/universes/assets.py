"""Asset class universe resolvers — crypto, forex, metals/commodities.

All lists are static (no API calls). Tickers use yfinance format.
"""

from __future__ import annotations


def get_crypto_tickers() -> list[str]:
    """Return the top 20 crypto tickers in yfinance format (XXX-USD)."""
    return [
        "BTC-USD",
        "ETH-USD",
        "BNB-USD",
        "SOL-USD",
        "XRP-USD",
        "DOGE-USD",
        "ADA-USD",
        "AVAX-USD",
        "SHIB-USD",
        "DOT-USD",
        "LINK-USD",
        "LTC-USD",
        "BCH-USD",
        "UNI-USD",
        "MATIC-USD",
        "XLM-USD",
        "ATOM-USD",
        "FIL-USD",
        "HBAR-USD",
        "ICP-USD",
    ]


def get_forex_tickers() -> list[str]:
    """Return major forex pairs in yfinance format (XXXYYY=X)."""
    return [
        "EURUSD=X",
        "GBPUSD=X",
        "USDJPY=X",
        "AUDUSD=X",
        "USDCAD=X",
        "USDCHF=X",
        "NZDUSD=X",
        "EURGBP=X",
        "EURJPY=X",
        "GBPJPY=X",
    ]


def get_metals_tickers() -> list[str]:
    """Return metals and commodities tickers (futures + ETFs) in yfinance format."""
    return [
        "GC=F",  # Gold futures
        "SI=F",  # Silver futures
        "PL=F",  # Platinum futures
        "CL=F",  # Crude Oil WTI futures
        "HG=F",  # Copper futures
        "GLD",  # SPDR Gold ETF
        "SLV",  # iShares Silver ETF
        "USO",  # United States Oil Fund ETF
    ]


def get_etf_sectors() -> list[str]:
    """Return the 11 SPDR sector ETFs (US equity sector rotation universe)."""
    return [
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLI",
        "XLC",
        "XLY",
        "XLP",
        "XLU",
        "XLRE",
        "XLB",
    ]


def get_etf_broad() -> list[str]:
    """Return a broad cross-asset ETF universe (equity, bond, gold, intl)."""
    return ["VOO", "QQQ", "VEA", "VWO", "GLD", "BND", "TLT", "IWM", "DIA", "HYG"]


def get_voo_only() -> list[str]:
    """Return VOO as a single-ticker universe for buy-and-hold baseline."""
    return ["VOO"]


def get_classic_60_40() -> list[str]:
    """Return VOO + BND for the classic 60/40 portfolio baseline."""
    return ["VOO", "BND"]


def get_bearish_etf_tickers() -> list[str]:
    """Return US-domiciled inverse ETFs that express bearish views without shorting.

    These are regular long positions whose value rises when an index falls.
    Note: designed for daily returns — multi-day holds suffer volatility decay.

    **EU retail caveat**: most of these are blocked for EU retail investors
    because their issuers (ProShares, Direxion) do not publish PRIIPs KID
    documents. Kept in this universe for backtest continuity; EU real-money
    execution should use `get_bearish_etf_ucits_tickers()` instead.
    """
    return [
        "SH",  # ProShares Short S&P 500 (-1x)
        "PSQ",  # ProShares Short QQQ (-1x)
        "DOG",  # ProShares Short Dow 30 (-1x)
        "RWM",  # ProShares Short Russell 2000 (-1x)
        "SDS",  # ProShares UltraShort S&P 500 (-2x)
        "SPXS",  # Direxion Daily S&P 500 Bear 3x
        "SPXU",  # ProShares UltraPro Short S&P 500 (-3x)
        "SQQQ",  # ProShares UltraPro Short QQQ (-3x)
    ]


def get_crypto_eur_tickers() -> list[str]:
    """Return the top crypto tickers quoted in EUR (Kraken spot format on yfinance).

    Agents assigned to this universe trade BTC/EUR, ETH/EUR, etc. directly on
    Kraken without an EUR→USD conversion. Validated against yfinance
    2026-04-17; tickers without EUR pair coverage on Yahoo are omitted.
    """
    return [
        "BTC-EUR",
        "ETH-EUR",
        "SOL-EUR",
        "XRP-EUR",
        "ADA-EUR",
        "DOGE-EUR",
        "DOT-EUR",
        "LINK-EUR",
        "LTC-EUR",
        "BCH-EUR",
        "AVAX-EUR",
        "ATOM-EUR",
        "XLM-EUR",
        "FIL-EUR",
    ]


def get_commodities_eur_tickers() -> list[str]:
    """Return EUR-quoted and EUR-hedged commodity ETFs for goldfinger's EUR-only form.

    Gold, silver, and broad commodities via UCITS ETFs tradable from IBIE.
    Some are GBP-denominated on LSE but still UCITS-compliant; their EUR
    equivalents on Euronext may be thinner. Candidates validated against
    yfinance on 2026-04-17 (see fetch_ohlcv smoke runs).
    """
    return [
        "PHAU.L",  # WisdomTree Physical Gold (USD, LSE) — widely traded
        "PHAG.L",  # WisdomTree Physical Silver (USD, LSE)
        "SGLN.L",  # iShares Physical Gold ETC (USD, LSE)
        "SGLN.MI",  # iShares Physical Gold ETC (EUR, Milan listing)
        "4GLD.DE",  # Xetra-Gold (EUR, Xetra)
        "PPFB.DE",  # WisdomTree Physical Gold EUR (Xetra)
        "CRUD.L",  # WisdomTree Brent Crude Oil ETC (USD, LSE)
    ]


def get_bearish_etf_ucits_tickers() -> list[str]:
    """UCITS inverse and leveraged ETFs tradable by EU retail on LSE/Euronext/Xetra.

    All tickers validated against yfinance on 2026-04-17 (returned ≥10d of price
    history). PRIIPs KID published by the issuers; tradable from IBKR Ireland
    once the **Complex/Leveraged Products (CLP)** permission is activated and
    each instrument's KID is acknowledged in Client Portal.

    Covers: S&P 500, Nasdaq 100, FTSE 100, Euro Stoxx 50, CAC 40, DAX, IBEX 35.
    Mix of inverse and leveraged-long products so agents can express both
    directions within their risk mandate.
    """
    return [
        # --- US-equity-proxy (the most important replacements for US blocks) ---
        "3USS.L",  # WisdomTree S&P 500 3x Daily Short (LSE, USD) — replaces SPXU/SPXS
        "3USL.L",  # WisdomTree S&P 500 3x Daily Leveraged (LSE, USD) — replaces UPRO
        "QQQS.L",  # WisdomTree Nasdaq 100 3x Daily Short (LSE, USD) — replaces SQQQ
        "QQQ3.L",  # WisdomTree Nasdaq 100 3x Daily Leveraged (LSE, USD) — replaces TQQQ
        "DSP5.PA",  # Amundi S&P 500 Daily (-1x) Inverse (Euronext, EUR) — replaces SH
        # --- European-index exposure ---
        "3UKS.L",  # WisdomTree FTSE 100 3x Daily Short (LSE)
        "3EUS.L",  # WisdomTree Euro Stoxx 50 3x Daily Short (LSE)
        "BX4.PA",  # Amundi CAC 40 Daily (-2x) Inverse (Euronext Paris)
        "CL2.PA",  # Amundi CAC 40 Daily 2x Leveraged (Euronext Paris)
        "XDEB.DE",  # Xtrackers ShortDAX Daily x1 Swap (Xetra)
        "DXSN.DE",  # Xtrackers ShortDAX Daily x2 Swap (Xetra)
        "IBEXA.MC",  # Amundi IBEX 35 Daily 2x (Madrid) — the "IBKR example" ticker
    ]
