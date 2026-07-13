"""Tests for engine.tickers — registry I/O, idempotent merge, name resolution."""

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
    assert resolve_name("AAPL", info) == {"name": "Apple Inc.", "type": "equity"}


def test_resolve_falls_back_to_short_name_when_long_empty() -> None:
    info = {"longName": "", "shortName": "Microsoft", "quoteType": "EQUITY"}
    assert resolve_name("MSFT", info) == {"name": "Microsoft", "type": "equity"}


def test_resolve_treats_etf_quote_type() -> None:
    info = {"longName": "Vanguard S&P 500 ETF", "quoteType": "ETF"}
    assert resolve_name("VOO", info) == {"name": "Vanguard S&P 500 ETF", "type": "etf"}


def test_resolve_crypto_usd_from_static_map_when_info_missing() -> None:
    assert resolve_name("BTC-USD", None) == {"name": "Bitcoin", "type": "crypto"}


def test_resolve_crypto_eur_from_static_map_when_info_missing() -> None:
    assert resolve_name("ETH-EUR", None) == {"name": "Ethereum", "type": "crypto"}


def test_resolve_crypto_unknown_base_returns_unknown_name() -> None:
    # WIF-USD: real coin, not in the static map. We must not invent a name.
    assert resolve_name("WIF-USD", None) == {"name": None, "type": "crypto"}


def test_resolve_forex_pattern() -> None:
    assert resolve_name("EURUSD=X", None) == {"name": "EUR/USD", "type": "forex"}


def test_resolve_unknown_symbol_with_no_info() -> None:
    assert resolve_name("MYSTERY", None) == {"name": None, "type": "unknown"}


def test_resolve_prefers_yfinance_name_over_static_map() -> None:
    # If yfinance has a richer name for a crypto, use it.
    info = {"longName": "Bitcoin USD", "quoteType": "CRYPTOCURRENCY"}
    assert resolve_name("BTC-USD", info) == {"name": "Bitcoin USD", "type": "crypto"}


def test_resolve_currency_quote_type_maps_to_forex() -> None:
    info = {"longName": "EUR/USD", "quoteType": "CURRENCY"}
    assert resolve_name("EURUSD=X", info) == {"name": "EUR/USD", "type": "forex"}


def test_resolve_ignores_empty_string_long_name() -> None:
    info = {"longName": "   ", "shortName": "BTC", "quoteType": "CRYPTOCURRENCY"}
    assert resolve_name("BTC-USD", info) == {"name": "BTC", "type": "crypto"}
