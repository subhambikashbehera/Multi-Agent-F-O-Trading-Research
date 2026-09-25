"""Provider interfaces. Agents only ever see these, never a broker SDK or a scraper."""

from __future__ import annotations

from typing import Protocol

from fno_research.models import Candle, FlowSnapshot, NewsItem, OptionChain


class MarketDataProvider(Protocol):
    name: str

    def spot(self, underlying: str) -> float: ...

    def candles(self, underlying: str, interval: str, lookback_days: int) -> list[Candle]:
        """interval: "day", "15minute" or "5minute"."""
        ...

    def option_chain(self, underlying: str, strikes_each_side: int = 15) -> OptionChain:
        """Chain for the nearest expiry, `strikes_each_side` strikes around ATM."""
        ...

    def vix(self) -> float | None: ...

    def vix_history(self, lookback_days: int = 365) -> list[float]: ...


class FlowsProvider(Protocol):
    def fii_dii(self) -> FlowSnapshot | None:
        """Latest provisional FII/DII cash-market flows."""
        ...


class NewsProvider(Protocol):
    def headlines(self, underlying: str, limit: int = 25) -> list[NewsItem]: ...
