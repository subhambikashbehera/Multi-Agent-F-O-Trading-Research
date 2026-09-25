"""OI build-up: read price change and open-interest change together.

    price up,   OI up   -> long build-up   (fresh buyers)
    price down, OI up   -> short build-up  (fresh writers)
    price up,   OI down -> short covering  (writers exiting)
    price down, OI down -> long unwinding  (buyers exiting)

For options the meaning flips with the side: call writing (call short build-up) caps the
upside and is bearish, while put writing (put short build-up) builds a floor and is bullish.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fno_research.models import OptionChain

LONG_BUILDUP = "long build-up"
SHORT_BUILDUP = "short build-up"
SHORT_COVERING = "short covering"
LONG_UNWINDING = "long unwinding"
NO_SIGNAL = "no change"

# Directional read for the underlying, per option side and state. Unwinding is weaker
# than fresh positions, so it gets half weight.
BIAS = {
    ("CE", LONG_BUILDUP): 1.0,
    ("CE", SHORT_BUILDUP): -1.0,
    ("CE", SHORT_COVERING): 1.0,
    ("CE", LONG_UNWINDING): -0.5,
    ("PE", LONG_BUILDUP): -1.0,
    ("PE", SHORT_BUILDUP): 1.0,
    ("PE", SHORT_COVERING): -1.0,
    ("PE", LONG_UNWINDING): 0.5,
}


def classify(price_change: float, oi_change: float) -> str:
    if price_change == 0 or oi_change == 0:
        return NO_SIGNAL
    if oi_change > 0:
        return LONG_BUILDUP if price_change > 0 else SHORT_BUILDUP
    return SHORT_COVERING if price_change > 0 else LONG_UNWINDING


@dataclass
class BuildUp:
    score: float  # -1..1, weighted by |OI change| near the money
    call_state: str  # dominant state on the call side
    put_state: str
    rows: list[dict] = field(default_factory=list)  # per strike and side, for display


def oi_buildup(chain: OptionChain, strikes_each_side: int = 5) -> BuildUp:
    strikes = chain.strikes()
    i = strikes.index(chain.atm_strike())
    near = set(strikes[max(i - strikes_each_side, 0) : i + strikes_each_side + 1])

    rows, weighted, total = [], 0.0, 0.0
    side_weight: dict[str, dict[str, float]] = {"CE": {}, "PE": {}}
    for q in chain.quotes:
        if q.strike not in near:
            continue
        state = classify(q.price_change, q.oi_change)
        weight = abs(q.oi_change)
        rows.append({"strike": q.strike, "side": q.option_type, "price_change": q.price_change,
                     "oi_change": q.oi_change, "state": state})
        if state == NO_SIGNAL:
            continue
        weighted += BIAS[(q.option_type, state)] * weight
        total += weight
        side = side_weight[q.option_type]
        side[state] = side.get(state, 0.0) + weight

    def dominant(side: str) -> str:
        states = side_weight[side]
        return max(states, key=states.get) if states else NO_SIGNAL

    return BuildUp(
        score=weighted / total if total else 0.0,
        call_state=dominant("CE"),
        put_state=dominant("PE"),
        rows=sorted(rows, key=lambda r: (r["strike"], r["side"])),
    )
