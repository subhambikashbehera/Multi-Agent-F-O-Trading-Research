"""Free data from nseindia.com's public JSON endpoints (the ones its own web pages call).

These endpoints are undocumented and change without notice; the parsers below accept the
field-name variants seen so far. Keep polling gentle (the client caches and throttles).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from fno_research.config import NSE_INDEX_NAMES, lot_size
from fno_research.data.web import WebClient
from fno_research.models import FlowSnapshot, OptionChain, OptionQuote

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
BASE = "https://www.nseindia.com"

# NSE's option chain reports open interest in contracts; we store units (contracts x lot
# size) so thresholds mean the same thing as with Kite, which reports units.
NSE_OI_IN_CONTRACTS = True


def _num(value) -> float:
    if value in (None, "", "-"):
        return 0.0
    if isinstance(value, str):
        value = value.replace(",", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first(d: dict, *keys: str) -> float:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return _num(d[k])
    return 0.0


def parse_expiry(text: str) -> date:
    return datetime.strptime(text, "%d-%b-%Y").date()


def parse_option_chain(payload: dict, underlying: str, strikes_each_side: int = 15,
                       expiry: date | None = None) -> OptionChain:
    """Parse NSE option-chain JSON (legacy `option-chain-indices` or `option-chain-v3`)."""
    records = payload.get("records") or payload
    rows = records.get("data") or []
    if not rows:
        raise ValueError("Option chain response has no data")
    lot = lot_size(underlying)
    oi_scale = lot if NSE_OI_IN_CONTRACTS else 1

    def row_expiry(row: dict) -> date | None:
        text = row.get("expiryDate") or row.get("expiryDates") or \
            (row.get("CE") or row.get("PE") or {}).get("expiryDate")
        return parse_expiry(text) if text else None

    if expiry is None:
        expiries = sorted({e for e in (row_expiry(r) for r in rows) if e})
        if not expiries and records.get("expiryDates"):
            expiries = sorted(parse_expiry(e) for e in records["expiryDates"])
        expiry = expiries[0]
    rows = [r for r in rows if row_expiry(r) in (expiry, None)]

    spot = _num(records.get("underlyingValue"))
    if not spot:
        for r in rows:
            spot = _first(r.get("CE") or r.get("PE") or {}, "underlyingValue")
            if spot:
                break

    strikes = sorted({_num(r.get("strikePrice")) for r in rows})
    atm = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    window = set(strikes[max(atm - strikes_each_side, 0) : atm + strikes_each_side + 1])

    quotes: list[OptionQuote] = []
    for r in rows:
        strike = _num(r.get("strikePrice"))
        if strike not in window:
            continue
        for opt in ("CE", "PE"):
            leg = r.get(opt)
            if not leg:
                continue
            iv = _first(leg, "impliedVolatility")
            quotes.append(
                OptionQuote(
                    strike=strike,
                    option_type=opt,
                    tradingsymbol=f"{underlying} {expiry:%d%b%y} {strike:g} {opt}".upper(),
                    last_price=_first(leg, "lastPrice"),
                    oi=_first(leg, "openInterest") * oi_scale,
                    oi_change=_first(leg, "changeinOpenInterest") * oi_scale,
                    volume=_first(leg, "totalTradedVolume"),
                    bid=_first(leg, "bidprice", "bidPrice", "buyPrice1"),
                    ask=_first(leg, "askPrice", "askprice", "sellPrice1"),
                    iv=iv / 100 if iv > 0 else None,
                )
            )

    stamp = records.get("timestamp")
    try:
        as_of = datetime.strptime(stamp, "%d-%b-%Y %H:%M:%S") if stamp else None
    except ValueError:
        as_of = None
    return OptionChain(
        underlying=underlying, spot=spot, expiry=expiry, lot_size=lot,
        as_of=as_of or datetime.now(IST).replace(tzinfo=None), quotes=quotes,
    )


def parse_all_indices(payload: dict) -> dict[str, float]:
    """{index name: last price} from /api/allIndices."""
    out = {}
    for row in payload.get("data", []):
        name = row.get("index") or row.get("indexSymbol")
        if name:
            out[name.upper()] = _num(row.get("last"))
    return out


def parse_fii_dii(payload: list) -> FlowSnapshot | None:
    fii = dii = None
    day = None
    for row in payload or []:
        category = (row.get("category") or "").upper()
        if "FII" in category or "FPI" in category:
            fii = _num(row.get("netValue"))
        elif "DII" in category:
            dii = _num(row.get("netValue"))
        if row.get("date"):
            day = datetime.strptime(row["date"], "%d-%b-%Y").date()
    if fii is None or dii is None or day is None:
        return None
    return FlowSnapshot(date=day, fii_net=fii, dii_net=dii)


class NSEClient:
    def __init__(self, web: WebClient | None = None):
        self.web = web or WebClient(
            warmup_url=f"{BASE}/option-chain", referer=f"{BASE}/option-chain"
        )

    def indices(self) -> dict[str, float]:
        return parse_all_indices(self.web.get_json(f"{BASE}/api/allIndices"))

    def spot(self, underlying: str) -> float:
        name = NSE_INDEX_NAMES[underlying.upper()]
        value = self.indices().get(name)
        if not value:
            raise RuntimeError(f"{name} not found in NSE allIndices")
        return value

    def vix(self) -> float | None:
        return self.indices().get("INDIA VIX") or None

    def option_chain(self, underlying: str, strikes_each_side: int = 15) -> OptionChain:
        underlying = underlying.upper()
        try:
            info = self.web.get_json(f"{BASE}/api/option-chain-contract-info",
                                     {"symbol": underlying})
            expiries = sorted(parse_expiry(e) for e in info["expiryDates"])
            today = datetime.now(IST).date()
            expiry = next(e for e in expiries if e >= today)
            payload = self.web.get_json(
                f"{BASE}/api/option-chain-v3",
                {"type": "Indices", "symbol": underlying, "expiry": f"{expiry:%d-%b-%Y}"},
            )
            return parse_option_chain(payload, underlying, strikes_each_side, expiry)
        except Exception as exc:  # fall back to the older endpoint
            log.info("option-chain-v3 failed (%s); trying option-chain-indices", exc)
            payload = self.web.get_json(f"{BASE}/api/option-chain-indices",
                                        {"symbol": underlying})
            return parse_option_chain(payload, underlying, strikes_each_side)

    def fii_dii(self) -> FlowSnapshot | None:
        return parse_fii_dii(self.web.get_json(f"{BASE}/api/fiidiiTradeReact"))
