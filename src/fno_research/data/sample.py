"""Deterministic synthetic data for tests and for exploring the dashboard offline.

Nothing here is real market data. The dashboard labels runs that use it.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np

from fno_research.analytics.options import bs_price
from fno_research.config import lot_size
from fno_research.models import Candle, FlowSnapshot, NewsItem, OptionChain, OptionQuote

_PROFILES = {
    "NIFTY": {"spot": 25_000.0, "step": 50.0, "vol": 0.13},
    "BANKNIFTY": {"spot": 55_000.0, "step": 100.0, "vol": 0.16},
    "FINNIFTY": {"spot": 26_500.0, "step": 50.0, "vol": 0.15},
    "MIDCPNIFTY": {"spot": 13_000.0, "step": 25.0, "vol": 0.18},
}


class SampleDataProvider:
    name = "sample"

    def __init__(self, seed: int = 7, drift: float = 0.0006, as_of: datetime | None = None,
                 spot: float | None = None, vol: float | None = None):
        self.seed = seed
        self.drift = drift  # daily drift; positive makes the series trend up
        self.as_of = as_of or datetime(2026, 9, 22, 11, 0)
        self.spot_override = spot  # move the market for scenario tests
        self.vol_override = vol

    def _profile(self, underlying: str) -> dict:
        try:
            profile = dict(_PROFILES[underlying.upper()])
        except KeyError as exc:
            raise ValueError(f"Unsupported underlying {underlying!r}") from exc
        if self.spot_override:
            profile["spot"] = self.spot_override
        if self.vol_override:
            profile["vol"] = self.vol_override
        return profile

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
        return synthetic_chain(underlying, p["spot"], p["step"], p["vol"],
                               lot_size(underlying), self.as_of,
                               _next_tuesday(self.as_of.date()), strikes_each_side, self.seed)

    def vix(self) -> float | None:
        return self.vix_history(5)[-1]

    def vix_history(self, lookback_days: int = 365) -> list[float]:
        rng = np.random.default_rng(self.seed + 99)
        # Mean-reverting around 14 with occasional spikes.
        values, v = [], 14.0
        for _ in range(lookback_days):
            v += 0.1 * (14 - v) + rng.normal(0, 0.6)
            values.append(max(v, 9.0))
        return values

    def ban_list(self) -> set[str]:
        return {"SAMPLEBANNEDSTOCK"}

    def fii_dii(self) -> FlowSnapshot | None:
        return FlowSnapshot(date=self.as_of.date(), fii_net=1_250.0, dii_net=-400.0)


def synthetic_chain(underlying: str, spot: float, step: float, base_vol: float, lot: int,
                    as_of: datetime, expiry: date, strikes_each_side: int = 15,
                    seed: int = 7) -> OptionChain:
    """Black-Scholes-priced chain with OI humps a few strikes out of the money."""
    rng = np.random.default_rng(seed + 1)
    t = max((expiry - as_of.date()).days, 0.5) / 365
    atm = round(spot / step) * step
    quotes = []
    for k in range(-strikes_each_side, strikes_each_side + 1):
        strike = atm + k * step
        moneyness = (strike - spot) / spot
        for opt in ("CE", "PE"):
            vol = base_vol * (1 + 1.5 * moneyness**2 * 100) + (0.01 if opt == "PE" else 0)
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
                    tradingsymbol=f"{underlying}{expiry:%y%b}{strike:g}{opt}".upper(),
                    last_price=round(price, 2),
                    oi=round(oi),
                    oi_change=round(oi * rng.normal(0.08 if opt == "PE" else 0.03, 0.05)),
                    # Puts cheapening while OI rises (put writing), calls firming.
                    price_change=round(price * rng.normal(-0.08 if opt == "PE" else 0.04,
                                                          0.03), 2),
                    volume=round(oi * rng.uniform(2, 6)),
                    bid=round(price - spread / 2, 2),
                    ask=round(price + spread / 2, 2),
                )
            )
    return OptionChain(
        underlying=underlying.upper(), spot=spot, expiry=expiry,
        lot_size=lot, as_of=as_of, quotes=quotes,
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
