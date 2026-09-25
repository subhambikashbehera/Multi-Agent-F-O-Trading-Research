"""Zerodha Kite Connect market data.

Kite's quote API does not return the previous session's open interest, so OI change
is computed against the last snapshot this provider saved on an earlier trading day
(stored as JSON under data/cache/). On the first day of use oi_change is zero.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from kiteconnect import KiteConnect

from fno_research.config import INDEX_SPOT_SYMBOLS, Settings
from fno_research.models import Candle, OptionChain, OptionQuote, TradeIdea

IST = ZoneInfo("Asia/Kolkata")
VIX_SYMBOL = "NSE:INDIA VIX"
QUOTE_BATCH = 500  # Kite's per-request instrument limit for quote()


class KiteDataProvider:
    name = "kite"

    def __init__(self, settings: Settings, cache_dir: Path = Path("data/cache")):
        if not settings.has_kite:
            raise RuntimeError(
                "KITE_API_KEY and KITE_ACCESS_TOKEN must be set. "
                "Run `fno-research kite-login` to get today's access token."
            )
        self.kite = KiteConnect(api_key=settings.kite_api_key)
        self.kite.set_access_token(settings.kite_access_token)
        self.cache_dir = cache_dir
        self._nfo: list[dict] | None = None

    # -- helpers ---------------------------------------------------------

    def _spot_symbol(self, underlying: str) -> str:
        try:
            return INDEX_SPOT_SYMBOLS[underlying.upper()]
        except KeyError as exc:
            raise ValueError(f"Unsupported underlying {underlying!r}") from exc

    def _nfo_instruments(self) -> list[dict]:
        if self._nfo is None:
            self._nfo = self.kite.instruments("NFO")
        return self._nfo

    def _quotes(self, symbols: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for i in range(0, len(symbols), QUOTE_BATCH):
            out.update(self.kite.quote(symbols[i : i + QUOTE_BATCH]))
        return out

    def _load_prev_oi(self, underlying: str, expiry: date, today: date) -> dict[str, float]:
        path = self.cache_dir / f"oi_{underlying}_{expiry.isoformat()}.json"
        if not path.exists():
            return {}
        snapshots = json.loads(path.read_text())
        earlier = sorted(d for d in snapshots if d < today.isoformat())
        return snapshots[earlier[-1]] if earlier else {}

    def _save_oi(self, underlying: str, expiry: date, today: date, oi: dict[str, float]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"oi_{underlying}_{expiry.isoformat()}.json"
        snapshots = json.loads(path.read_text()) if path.exists() else {}
        snapshots[today.isoformat()] = oi
        path.write_text(json.dumps(snapshots))

    # -- MarketDataProvider ---------------------------------------------

    def spot(self, underlying: str) -> float:
        symbol = self._spot_symbol(underlying)
        return float(self.kite.ltp([symbol])[symbol]["last_price"])

    def candles(self, underlying: str, interval: str, lookback_days: int) -> list[Candle]:
        symbol = self._spot_symbol(underlying)
        token = self.kite.ltp([symbol])[symbol]["instrument_token"]
        to_dt = datetime.now(IST)
        rows = self.kite.historical_data(
            token, to_dt - timedelta(days=lookback_days), to_dt, interval
        )
        return [
            Candle(
                timestamp=r["date"], open=r["open"], high=r["high"],
                low=r["low"], close=r["close"], volume=r.get("volume", 0),
            )
            for r in rows
        ]

    def option_chain(self, underlying: str, strikes_each_side: int = 15) -> OptionChain:
        underlying = underlying.upper()
        now = datetime.now(IST)
        today = now.date()
        options = [
            i for i in self._nfo_instruments()
            if i["name"] == underlying and i["segment"] == "NFO-OPT" and i["expiry"] >= today
        ]
        if not options:
            raise RuntimeError(f"No NFO options found for {underlying}")
        expiry = min(i["expiry"] for i in options)
        options = [i for i in options if i["expiry"] == expiry]

        spot = self.spot(underlying)
        strikes = sorted({i["strike"] for i in options})
        atm_index = min(range(len(strikes)), key=lambda k: abs(strikes[k] - spot))
        window = set(
            strikes[max(atm_index - strikes_each_side, 0) : atm_index + strikes_each_side + 1]
        )
        options = [i for i in options if i["strike"] in window]

        quotes = self._quotes([f"NFO:{i['tradingsymbol']}" for i in options])
        prev_oi = self._load_prev_oi(underlying, expiry, today)
        chain_quotes: list[OptionQuote] = []
        for inst in options:
            q = quotes.get(f"NFO:{inst['tradingsymbol']}")
            if not q:
                continue
            depth = q.get("depth") or {}
            bids = depth.get("buy") or [{}]
            asks = depth.get("sell") or [{}]
            oi = float(q.get("oi", 0))
            prev_close = float((q.get("ohlc") or {}).get("close", 0) or 0)
            last = float(q.get("last_price", 0))
            chain_quotes.append(
                OptionQuote(
                    strike=inst["strike"],
                    option_type=inst["instrument_type"],
                    tradingsymbol=inst["tradingsymbol"],
                    last_price=last,
                    oi=oi,
                    oi_change=oi - prev_oi.get(inst["tradingsymbol"], oi),
                    price_change=last - prev_close if prev_close else 0.0,
                    volume=float(q.get("volume", 0)),
                    bid=float(bids[0].get("price", 0) or 0),
                    ask=float(asks[0].get("price", 0) or 0),
                )
            )
        self._save_oi(underlying, expiry, today, {q.tradingsymbol: q.oi for q in chain_quotes})
        return OptionChain(
            underlying=underlying,
            spot=spot,
            expiry=expiry,
            lot_size=int(options[0]["lot_size"]),
            as_of=now.replace(tzinfo=None),
            quotes=chain_quotes,
        )


    def vix(self) -> float | None:
        return float(self.kite.ltp([VIX_SYMBOL])[VIX_SYMBOL]["last_price"])

    def vix_history(self, lookback_days: int = 365) -> list[float]:
        token = self.kite.ltp([VIX_SYMBOL])[VIX_SYMBOL]["instrument_token"]
        to_dt = datetime.now(IST)
        rows = self.kite.historical_data(token, to_dt - timedelta(days=lookback_days), to_dt,
                                         "day")
        return [float(r["close"]) for r in rows]

    def margin_required(self, idea: TradeIdea) -> float:
        """Broker-computed margin for the whole basket, with hedge benefit."""
        orders = [
            {
                "exchange": "NFO",
                "tradingsymbol": leg.tradingsymbol,
                "transaction_type": leg.action,
                "variety": "regular",
                "product": "NRML",
                "order_type": "MARKET",
                "quantity": leg.lots * idea.lot_size,
                "price": 0,
            }
            for leg in idea.legs
        ]
        result = self.kite.basket_order_margins(orders, consider_positions=True, mode="compact")
        return float(result["final"]["total"])

    def available_margin(self) -> float:
        return float(self.kite.margins("equity")["net"])


def login_url(settings: Settings) -> str:
    return KiteConnect(api_key=settings.kite_api_key).login_url()


def exchange_request_token(settings: Settings, request_token: str) -> str:
    """Swap the request_token from the login redirect for today's access token."""
    kite = KiteConnect(api_key=settings.kite_api_key)
    session = kite.generate_session(request_token, api_secret=settings.kite_api_secret)
    return session["access_token"]
