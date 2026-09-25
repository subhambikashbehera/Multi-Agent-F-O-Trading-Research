"""Risk engine: sizes a trade idea and runs every check. Any failed check blocks the idea."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from fno_research.config import RiskLimits
from fno_research.models import (
    AggregateSignal,
    MarketContext,
    OptionChain,
    RiskCheck,
    RiskDecision,
    TradeIdea,
)


@dataclass
class MarginInfo:
    required: float
    available: float
    source: str  # "broker" when the broker computed it, "estimate" otherwise


def estimate_margin(idea: TradeIdea, capital: float) -> MarginInfo:
    """Without a broker, a debit spread blocks roughly its net debit (short leg is hedged)."""
    return MarginInfo(required=max(idea.net_debit, idea.max_loss), available=capital,
                      source="estimate")


class RiskEngine:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    @property
    def risk_budget(self) -> float:
        return self.limits.capital * self.limits.max_risk_pct / 100

    def size(self, idea: TradeIdea, multiplier: float = 1.0) -> TradeIdea:
        """Lots = what the risk budget allows, capped at max_lots, scaled by the aggregator's
        allocation multiplier; never below 1 (the budget check then blocks it if needed)."""
        per_lot_loss = idea.max_loss / idea.legs[0].lots
        cap = min(self.limits.max_lots, math.floor(self.risk_budget / per_lot_loss))
        lots = max(1, math.floor(cap * multiplier))
        scale = lots / idea.legs[0].lots
        return idea.model_copy(
            update={
                "legs": [leg.model_copy(update={"lots": lots}) for leg in idea.legs],
                "max_loss": round(idea.max_loss * scale, 2),
                "max_profit": (
                    round(idea.max_profit * scale, 2) if idea.max_profit is not None else None
                ),
            }
        )

    def evaluate(self, idea: TradeIdea, view: AggregateSignal, chain: OptionChain,
                 context: MarketContext | None = None, margin: MarginInfo | None = None,
                 kill_switch: dict | None = None, today: date | None = None,
                 open_positions: int = 0) -> RiskDecision:
        today = today or chain.as_of.date()
        lim = self.limits
        margin = margin or estimate_margin(idea, lim.capital)
        margin_cap = margin.available * lim.max_margin_utilisation
        kill = kill_switch or {"active": False}
        regime = context.vol_regime if context else "unknown"

        checks = [
            RiskCheck(
                name="Kill switch",
                passed=not kill.get("active"),
                detail=kill.get("reason", "Tripped") if kill.get("active") else "Not tripped",
            ),
            RiskCheck(
                name="Existing exposure",
                passed=open_positions < lim.max_open_per_underlying,
                detail=f"{open_positions} open {idea.underlying} position(s) vs cap "
                f"{lim.max_open_per_underlying}",
            ),
            RiskCheck(
                name="No conflict veto",
                passed=not view.veto,
                detail=view.veto_reason or "Technical and context groups do not conflict",
            ),
            RiskCheck(
                name="Allocation multiplier",
                passed=view.allocation_multiplier > 0,
                detail=f"{view.allocation_multiplier:.2f}",
            ),
            RiskCheck(
                name="Signal confidence",
                passed=view.confidence >= lim.min_confidence,
                detail=f"{view.confidence:.2f} vs minimum {lim.min_confidence:.2f}",
            ),
            RiskCheck(
                name="Agent agreement",
                passed=view.agreement >= lim.min_agreement,
                detail=f"{view.agreement:.2f} vs minimum {lim.min_agreement:.2f}",
            ),
            RiskCheck(
                name="Defined risk",
                passed=idea.max_profit is not None or not lim.require_defined_risk,
                detail="Every short leg is covered" if idea.max_profit is not None
                else "Structure has unbounded risk",
            ),
            RiskCheck(
                name="Risk budget",
                passed=idea.max_loss <= self.risk_budget,
                detail=f"Max loss ₹{idea.max_loss:,.0f} vs budget ₹{self.risk_budget:,.0f} "
                f"({lim.max_risk_pct:.1f}% of ₹{lim.capital:,.0f})",
            ),
            RiskCheck(
                name="Margin",
                passed=margin.required <= margin_cap,
                detail=f"₹{margin.required:,.0f} ({margin.source}) vs ₹{margin_cap:,.0f} "
                f"({lim.max_margin_utilisation:.0%} of ₹{margin.available:,.0f})",
            ),
            RiskCheck(
                name="Position size",
                passed=all(leg.lots <= lim.max_lots for leg in idea.legs),
                detail=f"{idea.legs[0].lots} lot(s) vs cap {lim.max_lots}",
            ),
            RiskCheck(
                name="Days to expiry",
                passed=(idea.expiry - today).days >= lim.min_days_to_expiry,
                detail=f"{(idea.expiry - today).days} day(s) vs minimum {lim.min_days_to_expiry}",
            ),
            RiskCheck(
                name="Volatility regime",
                passed=not (lim.block_extreme_vol and regime == "extreme"),
                detail=f"{regime}" + (f" (VIX {context.vix:.1f})" if context and context.vix
                                      else ""),
            ),
        ]
        for leg in idea.legs:
            q = chain.get(leg.strike, leg.option_type)
            if q is None:
                checks.append(RiskCheck(name=f"Liquidity {leg.tradingsymbol}", passed=False,
                                        detail="Not in chain"))
                continue
            mid = (q.bid + q.ask) / 2
            spread_pct = (q.ask - q.bid) / mid * 100 if mid > 0 and q.bid > 0 else float("inf")
            checks.append(
                RiskCheck(
                    name=f"Liquidity {leg.tradingsymbol}",
                    passed=q.oi >= lim.min_oi and spread_pct <= lim.max_spread_pct,
                    detail=f"OI {q.oi:,.0f} (min {lim.min_oi:,.0f}), spread "
                    f"{spread_pct:.1f}% (max {lim.max_spread_pct:.1f}%)",
                )
            )
        return RiskDecision(approved=all(c.passed for c in checks), checks=checks)
