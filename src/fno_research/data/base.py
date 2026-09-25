"""Provider interfaces. Agents only ever see these, never a broker SDK directly."""

from __future__ import annotations

from typing import Protocol

from fno_research.models import Candle, NewsItem, OptionChain


class MarketDataProvider(Protocol):
    name: str

    def spot(self, underlying: str) -> float: ...

    def candles(self, underlying: str, interval: str, lookback_days: int) -> list[Candle]:
        """interval is a Kite interval string: "day", "15minute", "5minute", ..."""
        ...

    def option_chain(self, underlying: str, strikes_each_side: int = 15) -> OptionChain:
        """Chain for the nearest expiry, `strikes_each_side` strikes around ATM."""
        ...


class NewsProvider(Protocol):
    def headlines(self, underlying: str, limit: int = 25) -> list[NewsItem]: ...
