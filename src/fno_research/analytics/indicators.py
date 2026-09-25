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


def adx(frame: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder's ADX. Returns columns plus_di, minus_di, adx."""
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = atr(frame, period)
    alpha = 1 / period
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return pd.DataFrame(
        {
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx": dx.fillna(0).ewm(alpha=alpha, adjust=False).mean(),
        }
    )


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": line - signal_line})


def bollinger(close: pd.Series, period: int = 20, width: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper, lower = mid + width * std, mid - width * std
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "pct_b": (close - lower) / (upper - lower),
            "bandwidth": (upper - lower) / mid,
        }
    )


def obv(frame: pd.DataFrame) -> pd.Series:
    direction = frame["close"].diff().apply(lambda d: 1 if d > 0 else (-1 if d < 0 else 0))
    return (direction * frame["volume"]).cumsum()


def rolling_vwap(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    pv = (typical * frame["volume"]).rolling(period).sum()
    return pv / frame["volume"].rolling(period).sum().replace(0, float("nan"))
