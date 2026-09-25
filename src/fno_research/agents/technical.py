"""Technical agents. Each reads a daily OHLCV frame and nothing else, so each can be
backtested on its own (see fno_research.backtest)."""

from __future__ import annotations

import pandas as pd

from fno_research.agents.base import clamp, make_signal, vote_confidence
from fno_research.analytics.indicators import (
    adx,
    atr,
    bollinger,
    ema,
    macd,
    obv,
    rolling_vwap,
    rsi,
    supertrend,
)
from fno_research.models import AgentSignal

MIN_BARS = 60


class TechnicalAgent:
    name = ""
    group = "technical"

    def analyse_frame(self, frame: pd.DataFrame) -> AgentSignal:
        raise NotImplementedError

    def _not_enough(self, frame: pd.DataFrame) -> AgentSignal | None:
        if len(frame) < MIN_BARS:
            return make_signal(self.name, 0, 0, f"Only {len(frame)} bars; need {MIN_BARS}.",
                               {}, self.group)
        return None


class TrendAgent(TechnicalAgent):
    """MA cross for direction, ADX for whether there is a trend worth following."""

    name = "trend"

    def analyse_frame(self, frame: pd.DataFrame) -> AgentSignal:
        if short := self._not_enough(frame):
            return short
        close = frame["close"]
        ema20, ema50 = ema(close, 20).iloc[-1], ema(close, 50).iloc[-1]
        dmi = adx(frame).iloc[-1]
        st = int(supertrend(frame).iloc[-1])
        votes = {
            "ma_cross": (1.0 if ema20 > ema50 else -1.0, 0.45),
            "di_spread": (clamp((dmi.plus_di - dmi.minus_di) / 20), 0.30),
            "supertrend": (float(st), 0.25),
        }
        score = sum(v * w for v, w in votes.values())
        # ADX < 15: no trend, the cross is noise. ADX >= 30: strong trend.
        strength = clamp((dmi.adx - 15) / 15, 0.0, 1.0)
        confidence = vote_confidence(votes, score) * (0.4 + 0.6 * strength)
        rationale = (
            f"EMA20 {ema20:,.0f} {'>' if ema20 > ema50 else '<'} EMA50 {ema50:,.0f}; "
            f"ADX {dmi.adx:.0f} ({'trending' if dmi.adx >= 25 else 'weak trend'}), "
            f"+DI {dmi.plus_di:.0f} / -DI {dmi.minus_di:.0f}; "
            f"Supertrend {'up' if st > 0 else 'down'}."
        )
        return make_signal(self.name, score, confidence, rationale, {
            **{k: round(v, 3) for k, (v, _) in votes.items()},
            "adx": round(float(dmi.adx), 2),
        }, self.group)


class MomentumAgent(TechnicalAgent):
    """RSI and MACD histogram."""

    name = "momentum"

    def analyse_frame(self, frame: pd.DataFrame) -> AgentSignal:
        if short := self._not_enough(frame):
            return short
        close = frame["close"]
        r = float(rsi(close).iloc[-1])
        m = macd(close)
        hist, prev_hist = float(m["hist"].iloc[-1]), float(m["hist"].iloc[-2])
        a = float(atr(frame).iloc[-1])
        votes = {
            "rsi": (clamp((r - 50) / 20), 0.5),
            "macd_hist": (clamp(hist / (0.25 * a)) if a else 0.0, 0.35),
            "macd_turn": (1.0 if hist > prev_hist else -1.0, 0.15),
        }
        score = sum(v * w for v, w in votes.values())
        confidence = vote_confidence(votes, score)
        if r > 75 or r < 25:
            confidence *= 0.7  # stretched: momentum is real but reversal risk is high
        rationale = (
            f"RSI {r:.0f}; MACD histogram {hist:+.1f} "
            f"({'rising' if hist > prev_hist else 'falling'})."
        )
        return make_signal(self.name, score, confidence, rationale, {
            **{k: round(v, 3) for k, (v, _) in votes.items()}, "rsi_value": round(r, 2),
        }, self.group)


class VolatilityAgent(TechnicalAgent):
    """ATR and Bollinger Bands. Directional only on a band breakout, especially after a squeeze."""

    name = "volatility"

    def analyse_frame(self, frame: pd.DataFrame) -> AgentSignal:
        if short := self._not_enough(frame):
            return short
        close = frame["close"]
        bb = bollinger(close)
        last = bb.iloc[-1]
        bandwidth = bb["bandwidth"].dropna()
        lookback = bandwidth.iloc[-120:]
        squeeze_pct = float((lookback <= lookback.iloc[-6]).mean())  # 5 bars ago
        was_squeezed = squeeze_pct <= 0.2
        expanding = bandwidth.iloc[-1] > bandwidth.iloc[-6]
        a = atr(frame)
        atr_pct = float(a.iloc[-1] / close.iloc[-1] * 100)
        atr_rising = a.iloc[-1] > a.iloc[-6]

        if last.pct_b > 1 and expanding:
            score, confidence, state = 0.8, 0.75 if was_squeezed else 0.55, "upside breakout"
        elif last.pct_b < 0 and expanding:
            score, confidence, state = -0.8, 0.75 if was_squeezed else 0.55, "downside breakout"
        else:
            # Inside the bands: only a faint lean from where price sits in them.
            score, confidence = clamp((last.pct_b - 0.5) * 0.6), 0.15
            state = "squeeze" if squeeze_pct <= 0.2 else "inside bands"
        rationale = (
            f"{state.capitalize()}: %B {last.pct_b:.2f}, bandwidth "
            f"{last.bandwidth * 100:.1f}% ({'expanding' if expanding else 'contracting'}); "
            f"ATR {atr_pct:.2f}% of price and {'rising' if atr_rising else 'falling'}."
        )
        return make_signal(self.name, score, confidence, rationale, {
            "pct_b": round(float(last.pct_b), 3),
            "bandwidth": round(float(last.bandwidth), 4),
            "atr_pct": round(atr_pct, 3),
            "state": state,
        }, self.group)


class VolumeAgent(TechnicalAgent):
    """OBV trend and price versus 20-day VWAP. Needs real volume."""

    name = "volume"

    def analyse_frame(self, frame: pd.DataFrame) -> AgentSignal:
        if short := self._not_enough(frame):
            return short
        if frame["volume"].iloc[-20:].sum() <= 0:
            return make_signal(self.name, 0, 0, "No volume in this data feed.", {}, self.group)
        close = frame["close"]
        on_balance = obv(frame)
        obv_fast, obv_slow = ema(on_balance, 10).iloc[-1], ema(on_balance, 30).iloc[-1]
        avg_vol = frame["volume"].iloc[-20:].mean()
        vwap = float(rolling_vwap(frame).iloc[-1])
        a = float(atr(frame).iloc[-1])
        votes = {
            "obv_trend": (clamp((obv_fast - obv_slow) / (3 * avg_vol)), 0.5),
            "price_vs_vwap": (clamp((close.iloc[-1] - vwap) / a) if a else 0.0, 0.5),
        }
        score = sum(v * w for v, w in votes.values())
        confidence = vote_confidence(votes, score, floor=0.2)
        rationale = (
            f"OBV {'rising' if obv_fast > obv_slow else 'falling'} "
            f"(10/30 EMA); close {close.iloc[-1]:,.0f} vs 20-day VWAP {vwap:,.0f}."
        )
        return make_signal(self.name, score, confidence, rationale, {
            **{k: round(v, 3) for k, (v, _) in votes.items()}, "vwap20": round(vwap, 2),
        }, self.group)


TECHNICAL_AGENTS: list[type[TechnicalAgent]] = [
    TrendAgent, MomentumAgent, VolatilityAgent, VolumeAgent,
]
