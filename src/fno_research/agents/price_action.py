"""Rule-based read of trend and momentum on daily and intraday candles."""

from __future__ import annotations

from fno_research.agents.base import make_signal
from fno_research.analytics.indicators import atr, candles_to_frame, ema, rsi, supertrend
from fno_research.data.base import MarketDataProvider
from fno_research.models import AgentSignal


class PriceActionAgent:
    name = "price_action"

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def analyse(self, underlying: str) -> AgentSignal:
        daily = candles_to_frame(self.provider.candles(underlying, "day", 200))
        intraday = candles_to_frame(self.provider.candles(underlying, "15minute", 5))
        close = daily["close"]
        last = float(close.iloc[-1])

        ema20 = float(ema(close, 20).iloc[-1])
        ema50 = float(ema(close, 50).iloc[-1])
        daily_rsi = float(rsi(close).iloc[-1])
        daily_atr = float(atr(daily).iloc[-1])
        st_daily = int(supertrend(daily).iloc[-1])
        st_intraday = int(supertrend(intraday).iloc[-1])
        intraday_ema = float(ema(intraday["close"], 20).iloc[-1])
        intraday_last = float(intraday["close"].iloc[-1])

        # Each component votes in [-1, 1]; weights sum to 1.
        votes = {
            "trend_ema": (1.0 if ema20 > ema50 else -1.0, 0.25),
            "price_vs_ema20": (_scaled((last - ema20) / daily_atr, 1.5), 0.20),
            "supertrend_daily": (float(st_daily), 0.20),
            "momentum_rsi": (_scaled((daily_rsi - 50) / 50, 0.4), 0.15),
            "supertrend_15m": (float(st_intraday), 0.10),
            "intraday_vs_ema": (1.0 if intraday_last > intraday_ema else -1.0, 0.10),
        }
        score = sum(v * w for v, w in votes.values())
        # Confidence falls when the components disagree.
        same_sign = sum(w for v, w in votes.values() if v * score > 0)
        confidence = 0.3 + 0.6 * same_sign * min(abs(score) / 0.5, 1.0)
        if daily_rsi > 75 or daily_rsi < 25:
            confidence *= 0.8  # stretched; mean-reversion risk

        trend = "up" if ema20 > ema50 else "down"
        rationale = (
            f"Daily trend {trend} (EMA20 {ema20:,.0f} vs EMA50 {ema50:,.0f}); "
            f"spot {last:,.0f} is {(last - ema20) / daily_atr:+.1f} ATR from EMA20; "
            f"RSI {daily_rsi:.0f}; Supertrend daily {'+' if st_daily > 0 else '-'}, "
            f"15m {'+' if st_intraday > 0 else '-'}."
        )
        features = {k: round(v, 3) for k, (v, _) in votes.items()}
        features.update(
            ema20=round(ema20, 2), ema50=round(ema50, 2), rsi=round(daily_rsi, 2),
            atr=round(daily_atr, 2),
        )
        return make_signal(self.name, score, confidence, rationale, features)


def _scaled(x: float, full_scale: float) -> float:
    return max(-1.0, min(1.0, x / full_scale))
