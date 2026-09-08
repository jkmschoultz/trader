"""Vectorised technical primitives, every one backward-looking.

The rule these all obey: row ``i`` of the output is a function of input rows
``0..i`` only. That means ``.rolling``, ``.ewm``, ``.expanding``, ``.diff`` and
``.shift(k)`` with ``k >= 0`` -- never ``center=True`` and never a negative
shift. Break that and a model trains on its own answer.

Inputs and outputs are pandas Series/DataFrames aligned to the caller's index;
nothing here reindexes or sorts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "atr",
    "log_return",
    "macd",
    "range_pct",
    "rolling_vol",
    "rsi",
    "volume_zscore",
]


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over ``periods`` bars: ``ln(close_t / close_{t-periods})``."""
    return np.log(close / close.shift(periods))


def rolling_vol(close: pd.Series, window: int) -> pd.Series:
    """Sample standard deviation of 1-bar log returns over ``window`` bars."""
    return log_return(close, 1).rolling(window).std()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI in ``[0, 100]`` using an EWMA of gains and losses."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, its signal line, and the histogram (line minus signal)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return line, signal_line, line - signal_line


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range: an EWMA of the true range over ``window`` bars."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def range_pct(frame: pd.DataFrame) -> pd.Series:
    """Bar range as a fraction of its close: ``(high - low) / close``."""
    return (frame["high"] - frame["low"]) / frame["close"]


def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    """Volume standardised against its own trailing mean and stdev."""
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std
