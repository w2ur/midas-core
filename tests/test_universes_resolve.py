"""Tests for engine.universes.resolve_universe."""

import pytest

from engine.types import VALID_UNIVERSES
from engine.universes import resolve_universe
from engine.universes import _RESOLVERS


class TestResolveUniverse:
    def test_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="Unknown universe"):
            resolve_universe("not-a-universe")

    def test_static_universe_returns_nonempty(self) -> None:
        # single-voo is a static 1-ticker universe, safe to call without network
        tickers = resolve_universe("single-voo")
        assert isinstance(tickers, list)
        assert len(tickers) >= 1

    def test_classic_60_40_returns_static_list(self) -> None:
        tickers = resolve_universe("classic-60-40")
        assert isinstance(tickers, list)
        assert len(tickers) >= 2  # at least a stock + bond ticker

    def test_bearish_etfs_ucits_returns_list(self) -> None:
        tickers = resolve_universe("bearish-etfs-ucits")
        assert isinstance(tickers, list)

    def test_every_valid_universe_has_a_resolver(self) -> None:
        # Every VALID_UNIVERSES entry must be resolvable (even if empty placeholder).
        for name in VALID_UNIVERSES:
            assert name in _RESOLVERS, f"Missing resolver for {name}"

    def test_etf_broad_now_resolves_to_real_tickers(self) -> None:
        # Previously a lambda: [] placeholder; the static list now lives in engine.
        tickers = resolve_universe("etf-broad")
        assert "VOO" in tickers and "BND" in tickers
        assert len(tickers) >= 5

    def test_etf_sectors_now_resolves_to_real_tickers(self) -> None:
        tickers = resolve_universe("etf-sectors")
        assert "XLK" in tickers
        assert len(tickers) == 11

    def test_unimplemented_placeholder_raises_not_empty(self) -> None:
        # A declared-but-unimplemented universe must RAISE, never return [] —
        # an empty allowlist would fail open at the broker's universe rail.
        with pytest.raises(KeyError, match="empty ticker list"):
            resolve_universe("dividend-aristocrats")
        with pytest.raises(KeyError, match="empty ticker list"):
            resolve_universe("13f-whales")
