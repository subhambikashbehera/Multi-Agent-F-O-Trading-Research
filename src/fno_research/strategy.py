"""Turns a directional view into a defined-risk option structure.

Builds one lot; the risk engine sizes it.
"""

from __future__ import annotations

from fno_research.models import (
    AgentSignal,
    AggregateSignal,
    Direction,
    OptionChain,
    OptionLeg,
    OptionQuote,
    TradeIdea,
)


def _fill_price(q: OptionQuote, action: str) -> float:
    # Assume we pay the ask and receive the bid; fall back to LTP when depth is empty.
    price = q.ask if action == "BUY" else q.bid
    return price if price > 0 else q.last_price


def _leg(q: OptionQuote, action: str) -> OptionLeg:
    return OptionLeg(
        action=action, option_type=q.option_type, strike=q.strike,
        tradingsymbol=q.tradingsymbol, price=round(_fill_price(q, action), 2),
    )


def build_idea(chain: OptionChain, view: AggregateSignal,
               options_signal: AgentSignal | None = None,
               min_width_steps: int = 2, max_width_steps: int = 8) -> TradeIdea | None:
    if view.direction == Direction.NEUTRAL:
        return None

    strikes = chain.strikes()
    atm = chain.atm_strike()
    i = strikes.index(atm)
    bullish = view.direction == Direction.BULLISH
    opt = "CE" if bullish else "PE"
    sign = 1 if bullish else -1

    # Target the OI wall in the direction of the trade when it's a sensible distance away.
    wall_key = "call_wall" if bullish else "put_wall"
    wall = (options_signal.features.get(wall_key) if options_signal else None)
    width, at_wall = 4, False
    if isinstance(wall, (int, float)) and wall in strikes:
        steps = (strikes.index(wall) - i) * sign
        if min_width_steps <= steps <= max_width_steps:
            width, at_wall = steps, True
    j = i + sign * width
    if not 0 <= j < len(strikes):
        return None

    long_q = chain.get(atm, opt)
    short_q = chain.get(strikes[j], opt)
    if not long_q or not short_q:
        return None
    long_leg, short_leg = _leg(long_q, "BUY"), _leg(short_q, "SELL")

    debit = long_leg.price - short_leg.price
    spread_width = abs(short_leg.strike - long_leg.strike)
    if debit <= 0 or debit >= spread_width:
        return None
    name = "Bull call spread" if bullish else "Bear put spread"
    breakeven = long_leg.strike + debit if bullish else long_leg.strike - debit

    return TradeIdea(
        underlying=chain.underlying,
        expiry=chain.expiry,
        strategy=name,
        direction=view.direction,
        legs=[long_leg, short_leg],
        lot_size=chain.lot_size,
        max_loss=round(debit * chain.lot_size, 2),
        max_profit=round((spread_width - debit) * chain.lot_size, 2),
        breakeven=round(breakeven, 2),
        rationale=(
            f"{name}: buy {long_leg.strike:,.0f}{opt} / sell {short_leg.strike:,.0f}{opt} "
            f"for a debit of {debit:.2f}; short strike at "
            f"{'the OI wall' if at_wall else f'{width} strikes out'}. "
            f"Aggregate score {view.score:+.2f}, confidence {view.confidence:.2f}."
        ),
    )
