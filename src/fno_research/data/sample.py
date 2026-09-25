"""Deterministic synthetic data for tests and for exploring the dashboard offline.

Nothing here is real market data. The dashboard labels runs that use it.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from fno_research.analytics.options import bs_price
from fno_research.models import Candle, NewsItem, OptionChain, OptionQuote

_PROFILES = {
    "NIFTY": {"spot": 25_000.0, "step": 50.0, "lot": 65, "vol": 0.13},
    "BANKNIFTY": {"spot": 55_000.0, "step": 100.0, "lot": 30, "vol": 0.16},
    "FINNIFTY": {"spot": 26_500.0, "step": 50.0, "lot": 60, "vol": 0.15},
    "MIDCPNIFTY": {"spot": 13_000.0, "step": 25.0, "lot": 120, "vol": 0.18},
}


class SampleDataProvider:
    name = "sample"

    def __init__(self, seed: int = 7, drift: float = 0.0006, as_of: datetime | None = None):
        self.seed = seed
        self.drift = drift  # daily drift; positive makes the series trend up
        self.as_of = as_of or datetime(2026, 9, 22, 11, 0)

    def _profile(self, underlying: str) -> dict:
        try:
            return _PROFILES[underlying.upper()]
        except KeyError as exc:
            raise ValueError(f"Unsupported underlying {underlying!r}") from exc

    def _closes(self, underlying: str, interval: str, n: int, per_bar_vol: float) -> np.ndarray:
        rng = np.random.default_rng(self.seed + sum(map(ord, underlying + interval)))
        returns = rng.normal(self.drift, per_bar_vol, n)
        path = np.exp(np.cumsum(returns))
        return self._profile(underlying)["spot"] * path / path[-1]

    def spot(self, underlying: str) -> float:
        return self._profile(underlying)["spot"]

    def candles(self, underlying: str, interval: str, lookback_days: int) -> list[Candle]:
        p = self._profile(underlying)
        if interval == "day":
            n, step, bar_vol = lookback_days, timedelta(days=1), p["vol"] / np.sqrt(252)
        else:
            minutes = int(interval.removesuffix("minute") or 1)
            n = lookback_days * (375 // minutes)
            step = timedelta(minutes=minutes)
            bar_vol = p["vol"] / np.sqrt(252 * 375 / minutes)
        closes = self._closes(underlying, interval, n, bar_vol)
        rng = np.random.default_rng(self.seed)
        candles = []
        for i, close in enumerate(closes):
            open_ = closes[i - 1] if i else close
            wiggle = abs(rng.normal(0, bar_vol)) * close
            candles.append(
                Candle(
                    timestamp=self.as_of - step * (n - 1 - i),
                    open=open_,
                    high=max(open_, close) + wiggle,
                    low=min(open_, close) - wiggle,
                    close=close,
                    volume=float(rng.integers(1e5, 5e5)),
                )
            )
        return candles

    def option_chain(self, underlying: str, strikes_each_side: int = 15) -> OptionChain:
        p = self._profile(underlying)
        rng = np.random.default_rng(self.seed + 1)
        spot, step = p["spot"], p["step"]
        expiry = _next_tuesday(self.as_of.date())
        t = max((expiry - self.as_of.date()).days, 0.5) / 365
        atm = round(spot / step) * step
        quotes = []
        for k in range(-strikes_each_side, strikes_each_side + 1):
            strike = atm + k * step
            moneyness = (strike - spot) / spot
            for opt in ("CE", "PE"):
                vol = p["vol"] * (1 + 1.5 * moneyness**2 * 100) + (0.01 if opt == "PE" else 0)
                price = max(bs_price(spot, strike, t, vol, opt), 0.05)
                # OI peaks at round strikes a little OTM on each side.
                otm = (strike - spot) if opt == "CE" else (spot - strike)
                oi = 2e6 * np.exp(-(((otm - 3 * step) / (6 * step)) ** 2)) if otm > 0 else 3e5
                oi *= 1.6 if strike % (step * 10) == 0 else 1.0
                oi *= 1.15 if opt == "PE" else 1.0
                spread = max(0.05, price * 0.004)
                quotes.append(
                    OptionQuote(
                        strike=strike,
                        option_type=opt,
                        tradingsymbol=f"{underlying}{expiry:%y%b}{int(strike)}{opt}".upper(),
                        last_price=round(price, 2),
                        oi=round(oi),
                        oi_change=round(oi * rng.normal(0.08 if opt == "PE" else 0.03, 0.05)),
                        volume=round(oi * rng.uniform(2, 6)),
                        bid=round(price - spread / 2, 2),
                        ask=round(price + spread / 2, 2),
                    )
                )
        return OptionChain(
            underlying=underlying.upper(), spot=spot, expiry=expiry,
            lot_size=p["lot"], as_of=self.as_of, quotes=quotes,
        )


def _next_tuesday(d: date) -> date:
    # NSE moved NIFTY weekly expiry to Tuesday in Sept 2025.
    days = (1 - d.weekday()) % 7 or 7
    return d + timedelta(days=days)


class StaticNewsProvider:
    def __init__(self, items: list[NewsItem] | None = None):
        self.items = items if items is not None else SAMPLE_NEWS

    def headlines(self, underlying: str, limit: int = 25) -> list[NewsItem]:
        return self.items[:limit]


SAMPLE_NEWS = [
    NewsItem(title="FIIs turn net buyers in cash market for third straight session",
             source="sample"),
    NewsItem(title="RBI keeps repo rate unchanged, retains neutral stance", source="sample"),
    NewsItem(title="Crude oil slips below $70 on demand concerns", source="sample"),
    NewsItem(title="Banking stocks under pressure after weak asset-quality commentary",
             source="sample"),
]
