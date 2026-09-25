"""Rule-based read of where option writers are positioned."""

from __future__ import annotations

import math

from fno_research.agents.base import make_signal
from fno_research.analytics.options import (
    atm_iv,
    fill_implied_vols,
    iv_skew,
    max_pain,
    oi_walls,
    pcr,
)
from fno_research.data.base import MarketDataProvider
from fno_research.models import AgentSignal, OptionChain


class OptionsPositioningAgent:
    name = "options_positioning"

    def __init__(self, provider: MarketDataProvider, strikes_each_side: int = 15):
        self.provider = provider
        self.strikes_each_side = strikes_each_side
        self.last_chain: OptionChain | None = None

    def analyse(self, underlying: str) -> AgentSignal:
        chain = fill_implied_vols(self.provider.option_chain(underlying, self.strikes_each_side))
        self.last_chain = chain
        return self.analyse_chain(chain)

    def analyse_chain(self, chain: OptionChain) -> AgentSignal:
        spot = chain.spot
        ratio = pcr(chain)
        pain = max_pain(chain)
        walls = oi_walls(chain)
        iv = atm_iv(chain)
        skew = iv_skew(chain)

        # PCR: heavy put writing (>1) is read as support, i.e. bullish; very high is crowded.
        if math.isnan(ratio):
            pcr_vote = 0.0
        elif ratio > 1.5:
            pcr_vote = 0.3
        else:
            pcr_vote = max(-1.0, min(1.0, (ratio - 1.0) / 0.35))

        # Fresh OI: puts being added faster than calls = writers defending downside.
        added = walls.put_oi_added + walls.call_oi_added
        flow_vote = (
            (walls.put_oi_added - walls.call_oi_added) / abs(added) if added else 0.0
        )
        flow_vote = max(-1.0, min(1.0, flow_vote))

        # Where spot sits between the put wall (support) and call wall (resistance).
        span = walls.call_wall - walls.put_wall
        if span > 0:
            position = (spot - walls.put_wall) / span  # 0 at support, 1 at resistance
            range_vote = max(-1.0, min(1.0, 1 - 2 * position))
        else:
            range_vote = 0.0

        # Max pain acts as a weak magnet into expiry.
        pain_vote = max(-1.0, min(1.0, (pain - spot) / (0.01 * spot)))

        votes = {
            "pcr": (pcr_vote, 0.30),
            "oi_flow": (flow_vote, 0.30),
            "range_position": (range_vote, 0.25),
            "max_pain_pull": (pain_vote, 0.15),
        }
        score = sum(v * w for v, w in votes.values())
        agree = sum(w for v, w in votes.values() if v * score > 0)
        confidence = 0.25 + 0.6 * agree * min(abs(score) / 0.4, 1.0)
        if span <= 0:
            confidence *= 0.7  # walls inverted: positioning is muddled

        rationale = (
            f"PCR {ratio:.2f}; put wall {walls.put_wall:,.0f} / call wall "
            f"{walls.call_wall:,.0f} with spot {spot:,.0f}; fresh OI puts "
            f"{walls.put_oi_added:+,.0f} vs calls {walls.call_oi_added:+,.0f}; "
            f"max pain {pain:,.0f}"
            + (f"; ATM IV {iv * 100:.1f}%" if iv else "")
            + (f", skew {skew * 100:+.1f} pts" if skew is not None else "")
            + "."
        )
        features = {k: round(v, 3) for k, (v, _) in votes.items()}
        features.update(
            pcr_value=round(ratio, 3), max_pain=pain, put_wall=walls.put_wall,
            call_wall=walls.call_wall, atm_iv=round(iv, 4) if iv else None,
            iv_skew=round(skew, 4) if skew is not None else None,
        )
        return make_signal(self.name, score, confidence, rationale, features)
