"""Tests for engine.tickers — registry I/O, idempotent merge, name + currency resolution."""

from pathlib import Path

from engine.tickers import (
    load_registry,
    save_registry,
    merge,
    resolve_name,
)


def test_load_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    assert load_registry(path=tmp_path / "nope.json") == {}


def test_round_trip_preserves_data(tmp_path: Path) -> None:
    reg = {
        "AAPL": {"name": "Apple Inc.", "type": "equity"},
        "VOO": {"name": "Vanguard S&P 500 ETF", "type": "etf"},
    }
    path = tmp_path / "tickers.json"
    save_registry(reg, path=path)
    assert load_registry(path=path) == reg


def test_merge_adds_new_entry() -> None:
    existing = {"AAPL": {"name": "Apple Inc.", "type": "equity"}}
    fresh = {"MSFT": {"name": "Microsoft Corporation", "type": "equity"}}
    out = merge(existing, fresh)
    assert out["AAPL"] == {"name": "Apple Inc.", "type": "equity"}
    assert out["MSFT"] == {"name": "Microsoft Corporation", "type": "equity"}


def test_merge_keeps_existing_when_fresh_name_is_null() -> None:
    existing = {"AAPL": {"name": "Apple Inc.", "type": "equity"}}
    fresh = {"AAPL": {"name": None, "type": "unknown"}}
    out = merge(existing, fresh)
    assert out["AAPL"] == {"name": "Apple Inc.", "type": "equity"}


def test_merge_replaces_existing_when_fresh_name_is_non_null() -> None:
    existing = {"AAPL": {"name": None, "type": "unknown"}}
    fresh = {"AAPL": {"name": "Apple Inc.", "type": "equity"}}
    out = merge(existing, fresh)
    assert out["AAPL"] == {"name": "Apple Inc.", "type": "equity"}


def test_merge_overwrites_existing_when_fresh_name_changes() -> None:
    existing = {"X": {"name": "Old Name", "type": "equity"}}
    fresh = {"X": {"name": "New Name", "type": "equity"}}
    out = merge(existing, fresh)
    assert out["X"] == {"name": "New Name", "type": "equity"}


def test_merge_preserves_keys_only_in_existing() -> None:
    existing = {"AAPL": {"name": "Apple Inc.", "type": "equity"}}
    fresh = {"MSFT": {"name": "Microsoft Corporation", "type": "equity"}}
    out = merge(existing, fresh)
    assert "AAPL" in out


def test_resolve_uses_long_name_when_present() -> None:
    info = {"longName": "Apple Inc.", "shortName": "Apple", "quoteType": "EQUITY"}
    assert resolve_name("AAPL", info) == {
        "name": "Apple Inc.",
        "type": "equity",
        "currency": None,
    }


def test_resolve_falls_back_to_short_name_when_long_empty() -> None:
    info = {"longName": "", "shortName": "Microsoft", "quoteType": "EQUITY"}
    assert resolve_name("MSFT", info) == {
        "name": "Microsoft",
        "type": "equity",
        "currency": None,
    }


def test_resolve_treats_etf_quote_type() -> None:
    info = {"longName": "Vanguard S&P 500 ETF", "quoteType": "ETF"}
    assert resolve_name("VOO", info) == {
        "name": "Vanguard S&P 500 ETF",
        "type": "etf",
        "currency": None,
    }


def test_resolve_crypto_usd_from_static_map_when_info_missing() -> None:
    assert resolve_name("BTC-USD", None) == {
        "name": "Bitcoin",
        "type": "crypto",
        "currency": None,
    }


def test_resolve_crypto_eur_from_static_map_when_info_missing() -> None:
    assert resolve_name("ETH-EUR", None) == {
        "name": "Ethereum",
        "type": "crypto",
        "currency": None,
    }


def test_resolve_crypto_unknown_base_returns_unknown_name() -> None:
    # WIF-USD: real coin, not in the static map. We must not invent a name.
    assert resolve_name("WIF-USD", None) == {
        "name": None,
        "type": "crypto",
        "currency": None,
    }


def test_resolve_forex_pattern() -> None:
    assert resolve_name("EURUSD=X", None) == {
        "name": "EUR/USD",
        "type": "forex",
        "currency": None,
    }


def test_resolve_unknown_symbol_with_no_info() -> None:
    assert resolve_name("MYSTERY", None) == {
        "name": None,
        "type": "unknown",
        "currency": None,
    }


def test_resolve_prefers_yfinance_name_over_static_map() -> None:
    # If yfinance has a richer name for a crypto, use it.
    info = {"longName": "Bitcoin USD", "quoteType": "CRYPTOCURRENCY"}
    assert resolve_name("BTC-USD", info) == {
        "name": "Bitcoin USD",
        "type": "crypto",
        "currency": None,
    }


def test_resolve_currency_quote_type_maps_to_forex() -> None:
    info = {"longName": "EUR/USD", "quoteType": "CURRENCY"}
    assert resolve_name("EURUSD=X", info) == {
        "name": "EUR/USD",
        "type": "forex",
        "currency": None,
    }


def test_resolve_ignores_empty_string_long_name() -> None:
    info = {"longName": "   ", "shortName": "BTC", "quoteType": "CRYPTOCURRENCY"}
    assert resolve_name("BTC-USD", info) == {
        "name": "BTC",
        "type": "crypto",
        "currency": None,
    }


# ---------------------------------------------------------------------------
# Currency capture (2026-08-07)
#
# The registry gained a `currency` field so `engine.quotes` could stop
# guessing a ticker's quote unit from its suffix — a heuristic that valued
# every `.L` position 100x high (LSE quotes pence) and answered USD for 126
# European tickers in `world`'s universe.
# ---------------------------------------------------------------------------


def test_resolve_captures_the_vendor_currency() -> None:
    info = {"longName": "Airbus SE", "quoteType": "EQUITY", "currency": "EUR"}
    assert resolve_name("AIR.PA", info)["currency"] == "EUR"


def test_resolve_keeps_a_sub_unit_quote_verbatim() -> None:
    """`GBp` must survive into the registry — it is the whole point.

    `financialCurrency` is GBP for an LSE listing while the quote is in
    pence; preferring it (or upper-casing) would erase the 100:1 distinction
    the field exists to record.
    """
    info = {
        "longName": "Lloyds Banking Group plc",
        "quoteType": "EQUITY",
        "currency": "GBp",
        "financialCurrency": "GBP",
    }
    assert resolve_name("LLOY.L", info)["currency"] == "GBp"


def test_resolve_currency_is_none_when_the_vendor_is_silent() -> None:
    assert resolve_name("MYSTERY", {"quoteType": "EQUITY"})["currency"] is None


def test_merge_keeps_a_known_currency_when_the_fresh_entry_has_none() -> None:
    """A yfinance hiccup must not drop a ticker back onto the heuristic."""
    existing = {"LLOY.L": {"name": "Lloyds", "type": "equity", "currency": "GBp"}}
    fresh = {"LLOY.L": {"name": "Lloyds Banking Group plc", "type": "equity", "currency": None}}
    out = merge(existing, fresh)
    assert out["LLOY.L"]["currency"] == "GBp"
    assert out["LLOY.L"]["name"] == "Lloyds Banking Group plc"


def test_merge_replaces_a_currency_when_the_fresh_entry_has_one() -> None:
    existing = {"X.L": {"name": "X", "type": "equity", "currency": "GBp"}}
    fresh = {"X.L": {"name": "X", "type": "equity", "currency": "USD"}}
    assert merge(existing, fresh)["X.L"]["currency"] == "USD"


def test_resolve_rejects_a_currency_that_is_not_a_currency_code() -> None:
    """The vendor sometimes answers with a number.

    A full-universe sweep on 2026-08-07 returned `"3.3"` for ENX.AS and
    `"9.2"` for HMB.ST. Writing those into the registry would resolve to a
    currency `engine.fx` cannot convert, which takes the whole holding
    book's valuation to None — worse than the heuristic they replace.
    """
    for junk in ("3.3", "9.2", "", "   ", "EURO", "US"):
        assert resolve_name("X", {"currency": junk})["currency"] is None
