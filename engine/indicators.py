"""Technical indicators computed in plain pandas.

`wilder_rsi` replaces the pandas-ta dependency (`ta.rsi(close, length=14)`).
It reproduces pandas-ta 0.4.71b0's non-TA-Lib path operation for operation, so
the live desk's rsi-oversold signal is unchanged by the swap; the committed
fixture in tests/test_indicators.py pins that equivalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder (RMA) smoothing, scaled 0-100.

    Same arithmetic as pandas-ta's `rsi` without TA-Lib:

    - gains/losses are the one-step diff split by sign, the leading diff NaN
      kept as NaN;
    - each is smoothed with `ewm(alpha=1/length, adjust=False).mean()` and NO
      min_periods — so there is no warm-up window: only the first row (and any
      leading NaN closes) come out NaN, and a gap carries the last average;
    - RSI = 100 * avg_gain / (avg_gain + |avg_loss|). No losses gives 100; a
      flat stretch from the start (0 / 0) gives NaN.

    One deliberate shape difference: for fewer than `length + 1` closes
    pandas-ta returns None; this returns an all-NaN Series on the same index.
    Every comparison against either is False, so a selector built on it emits
    the same signal.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if close.size < length + 1:
        return pd.Series(np.nan, index=close.index, dtype=float, name=f"RSI_{length}")

    negative = close.diff(1)
    positive = negative.copy()
    positive[positive < 0] = 0
    negative[negative > 0] = 0

    alpha = 1.0 / length
    positive_avg = positive.ewm(alpha=alpha, adjust=False).mean()
    negative_avg = negative.ewm(alpha=alpha, adjust=False).mean()

    rsi = 100.0 * positive_avg / (positive_avg + negative_avg.abs())
    rsi.name = f"RSI_{length}"
    return rsi
