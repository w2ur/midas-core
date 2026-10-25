"""engine.indicators.wilder_rsi — pinned to the pandas-ta output it replaced.

The rsi-oversold selector feeds the live desk, so swapping pandas-ta's
`ta.rsi(close, length=14)` for an inline implementation must not move a single
signal. The expected values below were produced by pandas-ta 0.4.71b0 (pandas
3.0.3, numpy 2.2.6, no TA-Lib) BEFORE the dependency was dropped. They are
committed data, not computed here: pandas-ta is no longer installed, so a test
that called it would have nothing to compare against.

None stands for NaN in the literals. A `None` expected value for a whole case
means pandas-ta returned None (input shorter than length + 1).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from engine.adapter import SELECTOR_REGISTRY
from engine.indicators import wilder_rsi
from engine.types import FundingConfig, StrategyRules, StrategySpec

LENGTH = 14

CASES = {
    "synthetic": (
        [
            99.15, 99.62, 101.0, 102.6, 106.17, 107.18, 108.93, 110.58, 111.9, 112.32,
            109.3, 109.62, 108.5, 115.23, 116.78, 120.48, 117.11, 116.66, 115.06,
            116.98, 113.11, 112.29, 115.68, 115.67, 118.04, 113.23, 111.26, 112.41,
            110.1, 107.87, 108.03, 109.0, 107.85, 110.64, 106.82, 107.05, 106.56,
            107.82, 110.66, 112.68, 114.06, 117.84, 118.44, 116.68, 115.18, 118.33,
            114.35, 117.85, 117.54, 118.89, 119.78, 123.1, 122.31, 120.87, 125.46,
            128.86, 128.83, 128.36, 135.02, 137.96, 136.16, 132.95, 131.85, 133.83,
            134.95, 134.52, 135.19, 135.91, 136.96, 137.8, 140.06, 138.58, 142.25,
            138.59, 135.48, 135.0, 136.85, 137.7, 136.78, 132.49, 130.36, 130.89,
            130.03, 126.54, 126.7, 121.22, 126.53, 123.3, 120.71, 120.21, 113.81,
            112.67, 111.18, 109.91, 107.1, 102.28, 101.13, 103.57, 101.12, 98.8, 96.37,
            93.05, 90.58, 90.82, 89.99, 88.4, 88.01, 88.97, 88.84, 89.68, 91.85, 92.29,
            94.4, 94.72, 93.28, 89.65, 87.64, 85.1, 87.24, 87.57, 86.48, 87.45, 88.54,
            87.4, 88.46, 91.38, 91.77, 94.45, 93.8, 95.25, 97.55, 97.72, 99.91, 98.78,
            102.49, 103.17, 102.62, 102.5, 97.61, 98.82, 98.71, 100.14, 99.24, 100.41,
            104.77, 104.84, 103.59, 104.04, 107.15, 111.49, 108.35, 107.68, 110.88,
            112.58, 110.22, 107.75, 108.91, 109.39, 112.81, 116.52,
        ],
        [
            None, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
            80.19487789631283, 80.63257840319548, 74.4322670313838, 82.92758497002723,
            84.22742894390169, 86.80922178786338, 74.79955937179399, 73.34048967266736,
            68.2433177331959, 70.86049742453223, 60.107668841965655, 58.09601862958651,
            63.53011684401863, 63.50395562608111, 66.97495656759095, 55.44884424379075,
            51.536884297885365, 53.59505985198499, 49.085604886333655,
            45.13728531805319, 45.47616408188438, 47.58974916005288, 45.34546432556148,
            51.34088684371509, 44.19281236410397, 44.69211278002552, 43.79317192181393,
            46.75873876637932, 52.803274962732864, 56.5792545320006,
            58.992982672558824, 64.76997083764076, 65.5984199184219, 61.06242104485101,
            57.41829708159954, 62.481944130497965, 53.77996361527379,
            59.165845179604815, 58.51545283744704, 60.54928888457433,
            61.87627339822673, 66.41458671878877, 64.44857483551354, 60.90920704883627,
            67.10956629196225, 70.80376414022872, 70.7282774032727, 69.47851877548452,
            75.96063847056139, 78.1651825635342, 73.70838784200278, 66.43369133765049,
            64.09904105114079, 66.3887254308305, 67.64573511831995, 66.61566888127074,
            67.44743453520373, 68.35974091520289, 69.69366288916656, 70.75585717894288,
            73.45184932618194, 68.96786129276362, 73.31770733512022, 63.72441382887986,
            56.910239273880414, 55.91640902754925, 58.89577233688612, 60.2258567829553,
            58.03683526845944, 49.07875885805818, 45.337039648809935,
            46.43142103432934, 44.86192326486797, 39.0876197123746, 39.4722605714937,
            32.01542765928376, 43.210547359293244, 39.003174402883154,
            35.97806573795753, 35.40713071797769, 29.052059167363616,
            28.085084695490863, 26.828199207813377, 25.76960416159344,
            23.55494120389607, 20.327814706749688, 19.636564985182176,
            25.430605925293893, 23.591371061066873, 21.97092212038182,
            20.39104055986461, 18.440051359314516, 17.12711621248414,
            17.739986961494832, 17.26447509950714, 16.359805670233634,
            16.136450143807117, 19.06560932049977, 18.968988503139116,
            21.72918800997514, 28.504553499043784, 29.830898275467575,
            35.965759892906235, 36.86725982833603, 34.5125992874037,
            29.412804517989557, 27.030974074680085, 24.347757569538167,
            30.598447374573574, 31.537828141096398, 30.089125395674692,
            33.03702156610922, 36.28810002961034, 34.406627707690774,
            37.64403475268414, 45.60807501204987, 46.58925906733496, 52.87967292735227,
            51.30153758973825, 54.55944292077505, 59.21977964005018, 59.54999554854948,
            63.63516370101684, 60.25377839073511, 66.54016931479782, 67.55314230452457,
            65.81752436896606, 65.42257678681614, 51.78557205802453, 54.32275864708656,
            54.044327699285425, 57.121179346973285, 54.641578231082455,
            57.24022981874789, 65.23367823723032, 65.34568708794885, 61.53327132477047,
            62.384099714097324, 67.70120854145588, 73.36030784130388, 64.5483789473994,
            62.814574552189946, 67.3284023859168, 69.45001853752485,
            63.304238722025396, 57.56285479944401, 59.42407577598231,
            60.201901025721156, 65.30516460348089, 69.82538758813944,
        ],
    ),
    "flat": (
        [
            50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
            50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        ],
        [
            None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
        ],
    ),
    "monotonic_up": (
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        ],
        [
            None, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        ],
    ),
    "monotonic_down": (
        [
            20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0,
            7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ],
        [
            None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    ),
    "flat_then_up": (
        [
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 12.0,
            13.0, 12.5, 14.0, 15.0,
        ],
        [
            None, None, None, None, None, None, None, None, None, None, 100.0, 100.0,
            100.0, 83.82647648237652, 89.3771999309125, 91.47720526964456,
        ],
    ),
    "interior_gaps": (
        [
            1.0, 2.0, None, 3.0, None, None, 2.0, 1.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0,
            2.0, 1.0, 2.0, 3.0,
        ],
        [
            None, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 89.97463572997445,
            91.85439153060422, 92.36807855119676, 92.853431115754, 86.90178862611073,
            81.29049094218534, 76.00527186532818, 71.03178838235921, 66.35572846056785,
            68.58301698421081, 70.67378589747815,
        ],
    ),
    "leading_gaps": (
        [
            None, None, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 20.0, 19.0, 21.0,
        ],
        [
            None, None, None, None, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
            92.85714285714286, 86.22448979591837, 88.06122448979592,
        ],
    ),
    "exactly_length_plus_one": (
        [
            1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0, 7.0, 9.0, 8.0,
        ],
        [
            None, 100.0, 96.2962962962963, 96.56992084432719, 92.87526839742337,
            93.41767586388669, 89.73898746036839, 90.5412429451628, 86.88348956944792,
            87.933434055963, 84.2998956649582, 85.58300901931413, 81.97554914723125,
            83.47575311941341, 79.89509328315425,
        ],
    ),
    "too_short": (
        [
            1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0,
        ],
        None,
    ),
}


def _series(values: list[float | None]) -> pd.Series:
    return pd.Series([math.nan if v is None else v for v in values], dtype=float)


# Regression: pandas-ta removal, 2026-10-25 — wilder_rsi must reproduce
# ta.rsi(close, length=14) exactly; the live rsi-oversold signal depends on it.
@pytest.mark.parametrize("name", sorted(CASES))
def test_wilder_rsi_matches_committed_pandas_ta_output(name: str) -> None:
    closes, expected = CASES[name]
    got = wilder_rsi(_series(closes), length=LENGTH)

    assert len(got) == len(closes)
    if expected is None:
        # pandas-ta returned None; ours is all-NaN on the same index.
        assert got.isna().all()
        return
    want = _series(expected).to_numpy()
    have = got.to_numpy()
    assert np.array_equal(np.isnan(have), np.isnan(want)), "NaN positions differ"
    mask = ~np.isnan(want)
    assert np.max(np.abs(have[mask] - want[mask]), initial=0.0) <= 1e-9


def test_synthetic_case_crosses_the_oversold_line() -> None:
    """The fixture exercises the threshold the selector actually reads."""
    _, expected = CASES["synthetic"]
    values = [v for v in expected if v is not None]
    assert min(values) < 30 < max(values)


def _rsi_spec() -> StrategySpec:
    return StrategySpec(
        id="test-rsi-oversold",
        name="test",
        universe="sp500",
        selector="rsi-oversold",
        manager="equal-weight",
        funding=FundingConfig(initial=10000),
        dividends="cash",
        rules=StrategyRules(max_positions=3, max_position_pct=30, min_hold_days=3),
    )


def test_selector_signal_matches_pandas_ta_threshold() -> None:
    closes, expected = CASES["synthetic"]
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    prices = pd.DataFrame({"AAA": closes}, index=idx)
    [algo] = SELECTOR_REGISTRY["rsi-oversold"](_rsi_spec(), prices)
    want = pd.Series([v is not None and v < 30 for v in expected], index=idx)
    assert algo.signal["AAA"].tolist() == want.tolist()


def test_selector_on_too_short_history_selects_nothing() -> None:
    """pandas-ta returned None here; its column compared False everywhere."""
    closes, _ = CASES["too_short"]
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    prices = pd.DataFrame({"AAA": closes}, index=idx)
    [algo] = SELECTOR_REGISTRY["rsi-oversold"](_rsi_spec(), prices)
    assert not algo.signal["AAA"].any()


def test_non_positive_length_is_rejected() -> None:
    with pytest.raises(ValueError):
        wilder_rsi(pd.Series([1.0, 2.0]), length=0)


# --- properties (example count from the `midas` profile in conftest.py) -----

prices = st.floats(
    min_value=1e-2,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
price_paths = st.lists(prices, min_size=LENGTH + 1, max_size=200)


@given(price_paths)
def test_rsi_is_bounded_to_0_100(closes: list[float]) -> None:
    """Within 1e-6: 100 * g / g can round one ulp above 100 (hypothesis found
    100.00000000000001). pandas-ta does the same arithmetic and does not clip,
    so neither do we — clipping would break the exact match above."""
    out = wilder_rsi(pd.Series(closes), length=LENGTH).dropna()
    assert ((out >= -1e-6) & (out <= 100.0 + 1e-6)).all()


@given(
    flat=st.integers(min_value=0, max_value=40),
    start=prices,
    rest=st.lists(prices, min_size=LENGTH + 1, max_size=120),
)
def test_nan_prefix_is_first_row_plus_leading_flat_run(
    flat: int, start: float, rest: list[float]
) -> None:
    """No warm-up window: only row 0 and a flat opening stretch (0/0) are NaN."""
    if rest[0] == start:
        rest = [start * 1.5] + rest[1:]
    closes = [start] * (flat + 1) + rest
    out = wilder_rsi(pd.Series(closes), length=LENGTH)
    nan_rows = np.flatnonzero(out.isna().to_numpy())
    assert nan_rows.tolist() == list(range(flat + 1))


@given(price=prices, n=st.integers(min_value=0, max_value=200))
def test_constant_series_is_all_nan(price: float, n: int) -> None:
    out = wilder_rsi(pd.Series([price] * n), length=LENGTH)
    assert len(out) == n
    assert out.isna().all()


@given(price_paths, st.floats(min_value=1e-3, max_value=1e3))
def test_rsi_is_scale_invariant(closes: list[float], k: float) -> None:
    a = wilder_rsi(pd.Series(closes), length=LENGTH)
    b = wilder_rsi(pd.Series(closes) * k, length=LENGTH)
    assert np.array_equal(a.isna().to_numpy(), b.isna().to_numpy())
    assert np.allclose(a.dropna(), b.dropna(), rtol=0.0, atol=1e-6)
