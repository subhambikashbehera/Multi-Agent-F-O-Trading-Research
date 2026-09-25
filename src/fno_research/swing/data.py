"""Daily bars for individual stocks: Yahoo Finance (SYMBOL.NS) with a local cache, or a
synthetic universe for tests and demos."""

from __future__ import annotations

import io
import sqlite3
import zlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from fno_research.analytics.indicators import candles_to_frame
from fno_research.data.nse import NSEClient
from fno_research.data.sample import synthetic_chain
from fno_research.data.yahoo import YAHOO_TICKERS, YahooClient
from fno_research.models import OptionChain
from fno_research.swing.models import Stock

BENCHMARK = "NIFTY"


class SwingData(Protocol):
    name: str

    def daily(self, symbol: str, lookback_days: int = 420) -> pd.DataFrame: ...

    def benchmark(self, lookback_days: int = 420) -> pd.DataFrame: ...

    def lot_sizes(self) -> dict[str, int]:
        """F&O stocks and their lot sizes; stocks missing here can't be traded bearish."""
        ...

    def option_chain(self, symbol: str, lot: int) -> OptionChain: ...


def yahoo_symbol(symbol: str) -> str:
    # Yahoo uses the NSE symbol with .NS; '&' (e.g. M&M) is URL-encoded by requests.
    return f"{symbol}.NS"


class YahooSwingData:
    """Fetches each symbol at most once per day; bars are cached in SQLite."""

    name = "nse"

    def __init__(self, db_path: Path, yahoo: YahooClient | None = None,
                 nse: NSEClient | None = None, option_min_days: int = 7):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS bar_cache "
                          "(ticker TEXT PRIMARY KEY, fetched_on TEXT, payload TEXT)")
        self.yahoo = yahoo or YahooClient()
        self.nse = nse or NSEClient()
        self.option_min_days = option_min_days
        self._lots: dict[str, int] | None = None

    def _frame(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        today = date.today().isoformat()
        row = self.conn.execute("SELECT fetched_on, payload FROM bar_cache WHERE ticker = ?",
                                (ticker,)).fetchone()
        if row and row[0] == today:
            frame = pd.read_json(io.StringIO(row[1]), orient="split")
            frame.index = pd.to_datetime(frame.index)
            return frame
        frame = candles_to_frame(self.yahoo.candles(ticker, "day", lookback_days))
        self.conn.execute("INSERT OR REPLACE INTO bar_cache VALUES (?, ?, ?)",
                          (ticker, today, frame.to_json(orient="split", date_format="iso")))
        self.conn.commit()
        return frame

    def daily(self, symbol: str, lookback_days: int = 420) -> pd.DataFrame:
        return self._frame(yahoo_symbol(symbol), lookback_days)

    def benchmark(self, lookback_days: int = 420) -> pd.DataFrame:
        return self._frame(YAHOO_TICKERS[BENCHMARK], lookback_days)

    def lot_sizes(self) -> dict[str, int]:
        if self._lots is None:
            self._lots = self.nse.lot_sizes()
        return self._lots

    def option_chain(self, symbol: str, lot: int) -> OptionChain:
        return self.nse.stock_option_chain(symbol, lot, min_days=self.option_min_days)


# -- synthetic ---------------------------------------------------------------------------

SAMPLE_INDUSTRIES = ["Financial Services", "Information Technology", "Automobile",
                     "Capital Goods", "Healthcare", "FMCG", "Metals & Mining", "Chemicals"]


def sample_universe(n: int = 40) -> list[Stock]:
    """Clearly fake names: DEMO01..DEMOnn, spread over real NSE industry labels."""
    return [Stock(symbol=f"DEMO{i:02d}", name=f"Demo Company {i:02d}",
                  industry=SAMPLE_INDUSTRIES[i % len(SAMPLE_INDUSTRIES)])
            for i in range(1, n + 1)]


class SampleSwingData:
    name = "sample"

    def __init__(self, as_of: datetime | None = None, seed: int = 11,
                 overrides: dict[str, pd.DataFrame] | None = None):
        self.as_of = as_of or datetime(2026, 9, 22, 15, 30)
        self.seed = seed
        self.overrides = overrides or {}

    def _series(self, key: str, drift: float, vol: float, base: float, days: int,
                volume: float) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + zlib.crc32(key.encode()))
        n = int(days * 250 / 365)
        closes = base * np.exp(np.cumsum(rng.normal(drift, vol, n)))
        idx = pd.bdate_range(end=self.as_of.date(), periods=n)
        opens = np.r_[closes[0], closes[:-1]] * (1 + rng.normal(0, vol / 4, n))
        wiggle = np.abs(rng.normal(0, vol, n)) * closes
        return pd.DataFrame({
            "open": opens, "high": np.maximum(opens, closes) + wiggle,
            "low": np.minimum(opens, closes) - wiggle, "close": closes,
            "volume": rng.integers(int(volume * 0.5), int(volume * 1.5), n).astype(float),
        }, index=idx)

    def daily(self, symbol: str, lookback_days: int = 420) -> pd.DataFrame:
        if symbol in self.overrides:
            return self.overrides[symbol]
        rng = np.random.default_rng(zlib.crc32(symbol.encode()))
        drift = rng.normal(0.0004, 0.0012)  # some trend up, some down
        base = float(rng.choice([150, 400, 900, 1800, 3500]))
        return self._series(symbol, drift, rng.uniform(0.012, 0.025), base, lookback_days,
                            volume=float(rng.choice([2e5, 1e6, 4e6])))

    def benchmark(self, lookback_days: int = 420) -> pd.DataFrame:
        return self._series("NIFTY", 0.0004, 0.009, 18_000, lookback_days, 3e8)

    def lot_sizes(self) -> dict[str, int]:
        # Odd-numbered demo stocks are "in F&O"; lots sized to ~₹15 lakh contract value.
        out = {}
        for i in range(1, 100, 2):
            sym = f"DEMO{i:02d}"
            close = float(self.daily(sym)["close"].iloc[-1])
            out[sym] = max(1, round(1_500_000 / close / 25) * 25)
        return out

    def option_chain(self, symbol: str, lot: int) -> OptionChain:
        frame = self.daily(symbol)
        spot = float(frame["close"].iloc[-1])
        step = next(s for s in (1, 2.5, 5, 10, 20, 50, 100) if spot / s <= 120)
        expiry = (self.as_of + timedelta(days=30)).date()
        return synthetic_chain(symbol, spot, step, 0.28, lot, self.as_of, expiry, 10, self.seed)
