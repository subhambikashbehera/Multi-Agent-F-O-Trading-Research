"""Option chain analytics: Black-Scholes, implied volatility, PCR, max pain, OI walls."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fno_research.models import OptionChain

RISK_FREE_RATE = 0.065  # approx. Indian 91-day T-bill yield


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(spot: float, strike: float, t: float, vol: float, option_type: str,
             r: float = RISK_FREE_RATE) -> float:
    if t <= 0 or vol <= 0:
        intrinsic = spot - strike if option_type == "CE" else strike - spot
        return max(intrinsic, 0.0)
    d1 = (math.log(spot / strike) + (r + vol**2 / 2) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, t: float, vol: float, option_type: str,
             r: float = RISK_FREE_RATE) -> float:
    if t <= 0 or vol <= 0:
        itm = spot > strike if option_type == "CE" else spot < strike
        return (1.0 if itm else 0.0) * (1 if option_type == "CE" else -1)
    d1 = (math.log(spot / strike) + (r + vol**2 / 2) * t) / (vol * math.sqrt(t))
    return _norm_cdf(d1) if option_type == "CE" else _norm_cdf(d1) - 1


def implied_vol(price: float, spot: float, strike: float, t: float, option_type: str,
                r: float = RISK_FREE_RATE) -> float | None:
    """Bisection IV solve; returns None when the price is outside no-arbitrage bounds."""
    if price <= 0 or t <= 0:
        return None
    lo, hi = 1e-4, 5.0
    if not bs_price(spot, strike, t, lo, option_type, r) <= price <= bs_price(
        spot, strike, t, hi, option_type, r
    ):
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        if bs_price(spot, strike, t, mid, option_type, r) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2


def years_to_expiry(chain: OptionChain) -> float:
    # NSE index options expire at 15:30 IST; count to that moment.
    expiry_close = chain.as_of.replace(
        year=chain.expiry.year, month=chain.expiry.month, day=chain.expiry.day,
        hour=15, minute=30, second=0, microsecond=0,
    )
    seconds = max((expiry_close - chain.as_of).total_seconds(), 0.0)
    return seconds / (365 * 24 * 3600)


def fill_implied_vols(chain: OptionChain) -> OptionChain:
    t = years_to_expiry(chain)
    for q in chain.quotes:
        if q.iv is None:
            q.iv = implied_vol(q.last_price, chain.spot, q.strike, t, q.option_type)
    return chain


def pcr(chain: OptionChain) -> float:
    put_oi = sum(q.oi for q in chain.quotes if q.option_type == "PE")
    call_oi = sum(q.oi for q in chain.quotes if q.option_type == "CE")
    return put_oi / call_oi if call_oi else float("nan")


def max_pain(chain: OptionChain) -> float:
    """Strike at which option writers pay out the least at expiry."""
    strikes = chain.strikes()

    def payout(settle: float) -> float:
        total = 0.0
        for q in chain.quotes:
            if q.option_type == "CE":
                total += max(settle - q.strike, 0) * q.oi
            else:
                total += max(q.strike - settle, 0) * q.oi
        return total

    return min(strikes, key=payout)


@dataclass
class OIWalls:
    call_wall: float  # strike with the most call OI (resistance)
    put_wall: float  # strike with the most put OI (support)
    call_oi_added: float  # OI change in calls within the chain window
    put_oi_added: float


def oi_walls(chain: OptionChain) -> OIWalls:
    calls = [q for q in chain.quotes if q.option_type == "CE"]
    puts = [q for q in chain.quotes if q.option_type == "PE"]
    return OIWalls(
        call_wall=max(calls, key=lambda q: q.oi).strike,
        put_wall=max(puts, key=lambda q: q.oi).strike,
        call_oi_added=sum(q.oi_change for q in calls),
        put_oi_added=sum(q.oi_change for q in puts),
    )


def atm_iv(chain: OptionChain) -> float | None:
    atm = chain.atm_strike()
    ivs = [q.iv for q in chain.quotes if q.strike == atm and q.iv]
    return sum(ivs) / len(ivs) if ivs else None


def iv_skew(chain: OptionChain, width: int = 3) -> float | None:
    """OTM put IV minus OTM call IV, `width` strikes away from ATM. Positive = put skew."""
    strikes = chain.strikes()
    i = strikes.index(chain.atm_strike())
    if i - width < 0 or i + width >= len(strikes):
        return None
    put = chain.get(strikes[i - width], "PE")
    call = chain.get(strikes[i + width], "CE")
    if not put or not call or put.iv is None or call.iv is None:
        return None
    return put.iv - call.iv
