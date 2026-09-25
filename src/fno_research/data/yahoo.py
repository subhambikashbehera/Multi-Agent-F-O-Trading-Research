"""Free OHLCV history from Yahoo Finance's chart endpoint (NSE indices and India VIX)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fno_research.data.web import WebClient
from fno_research.models import Candle

IST = ZoneInfo("Asia/Kolkata")
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

YAHOO_TICKERS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
    "INDIAVIX": "^INDIAVIX",
}
INTERVALS = {"day": "1d", "15minute": "15m", "5minute": "5m", "60minute": "60m"}
# Yahoo only serves this much intraday history.
MAX_INTRADAY_DAYS = {"15m": 59, "5m": 59, "60m": 729}


def parse_chart(payload: dict) -> list[Candle]:
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        error = (payload.get("chart") or {}).get("error")
        raise ValueError(f"Yahoo chart returned no data: {error}")
    r = result[0]
    quote = r["indicators"]["quote"][0]
    candles = []
    for i, ts in enumerate(r.get("timestamp") or []):
        o, h, lo, c = (quote[k][i] for k in ("open", "high", "low", "close"))
        if None in (o, h, lo, c):
            continue  # Yahoo leaves holes for halted or partial bars
        candles.append(
            Candle(
                timestamp=datetime.fromtimestamp(ts, IST).replace(tzinfo=None),
                open=o, high=h, low=lo, close=c,
                volume=(quote.get("volume") or [0] * (i + 1))[i] or 0,
            )
        )
    return candles


class YahooClient:
    def __init__(self, web: WebClient | None = None):
        self.web = web or WebClient(cache_ttl=300, min_interval=0.5)

    def candles(self, ticker: str, interval: str, lookback_days: int) -> list[Candle]:
        yf_interval = INTERVALS[interval]
        lookback_days = min(lookback_days, MAX_INTRADAY_DAYS.get(yf_interval, lookback_days))
        end = datetime.now(IST)
        start = end - timedelta(days=lookback_days)
        payload = self.web.get_json(
            CHART_URL.format(ticker=ticker),
            {
                "period1": int(start.timestamp()),
                "period2": int(end.timestamp()),
                "interval": yf_interval,
                "includePrePost": "false",
            },
        )
        return parse_chart(payload)
