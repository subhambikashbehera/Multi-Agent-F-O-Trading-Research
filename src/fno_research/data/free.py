"""Free market data: NSE website for spot, option chain, VIX and flows; Yahoo for history."""

from __future__ import annotations

import logging

from fno_research.data.nse import NSEClient
from fno_research.data.yahoo import YAHOO_TICKERS, YahooClient
from fno_research.models import Candle, FlowSnapshot, OptionChain

log = logging.getLogger(__name__)


class FreeWebProvider:
    name = "nse"

    def __init__(self, nse: NSEClient | None = None, yahoo: YahooClient | None = None):
        self.nse = nse or NSEClient()
        self.yahoo = yahoo or YahooClient()

    def spot(self, underlying: str) -> float:
        try:
            return self.nse.spot(underlying)
        except Exception as exc:
            log.info("NSE spot failed (%s); using Yahoo", exc)
            return self.candles(underlying, "day", 5)[-1].close

    def candles(self, underlying: str, interval: str, lookback_days: int) -> list[Candle]:
        return self.yahoo.candles(YAHOO_TICKERS[underlying.upper()], interval, lookback_days)

    def option_chain(self, underlying: str, strikes_each_side: int = 15) -> OptionChain:
        return self.nse.option_chain(underlying, strikes_each_side)

    def vix(self) -> float | None:
        try:
            return self.nse.vix()
        except Exception as exc:
            log.info("NSE VIX failed (%s); using Yahoo", exc)
            history = self.vix_history(10)
            return history[-1] if history else None

    def vix_history(self, lookback_days: int = 365) -> list[float]:
        return [c.close for c in self.yahoo.candles(YAHOO_TICKERS["INDIAVIX"], "day",
                                                     lookback_days)]

    def fii_dii(self) -> FlowSnapshot | None:
        return self.nse.fii_dii()
