"""Price indicators on pandas OHLC frames (columns: open, high, low, close, volume)."""

from __future__ import annotations

import pandas as pd

from fno_research.models import Candle


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    frame = pd.DataFrame([c.model_dump() for c in candles])
    return frame.set_index("timestamp").sort_index()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).fillna(100.0)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift()
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(frame: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Returns +1 while in an uptrend, -1 while in a downtrend."""
    hl2 = (frame["high"] + frame["low"]) / 2
    band = multiplier * atr(frame, period)
    basic_upper = (hl2 + band).to_numpy()
    basic_lower = (hl2 - band).to_numpy()
    close = frame["close"].to_numpy()
    upper, lower = basic_upper.copy(), basic_lower.copy()
    trend = [1] * len(frame)
    for i in range(1, len(frame)):
        if basic_upper[i] > upper[i - 1] and close[i - 1] <= upper[i - 1]:
            upper[i] = upper[i - 1]
        if basic_lower[i] < lower[i - 1] and close[i - 1] >= lower[i - 1]:
            lower[i] = lower[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1 if close[i] < lower[i] else 1
        else:
            trend[i] = 1 if close[i] > upper[i] else -1
    return pd.Series(trend, index=frame.index)
