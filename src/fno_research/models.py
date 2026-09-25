"""Core data types shared by providers, agents, the aggregator and the risk engine."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

    @classmethod
    def from_score(cls, score: float, threshold: float) -> Direction:
        if score >= threshold:
            return cls.BULLISH
        if score <= -threshold:
            return cls.BEARISH
        return cls.NEUTRAL


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class OptionQuote(BaseModel):
    """One strike of the chain. `oi_change` is change versus the previous session."""

    strike: float
    option_type: str  # "CE" or "PE"
    tradingsymbol: str
    last_price: float
    oi: float = 0.0
    oi_change: float = 0.0
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    iv: float | None = None  # annualised, as a fraction (0.14 = 14%)


class OptionChain(BaseModel):
    underlying: str
    spot: float
    expiry: date
    lot_size: int
    as_of: datetime
    quotes: list[OptionQuote]

    def strikes(self) -> list[float]:
        return sorted({q.strike for q in self.quotes})

    def get(self, strike: float, option_type: str) -> OptionQuote | None:
        for q in self.quotes:
            if q.strike == strike and q.option_type == option_type:
                return q
        return None

    def atm_strike(self) -> float:
        return min(self.strikes(), key=lambda k: abs(k - self.spot))


class NewsItem(BaseModel):
    title: str
    summary: str = ""
    source: str = ""
    published: datetime | None = None
    url: str = ""


class AgentSignal(BaseModel):
    """What every analysis agent returns.

    score is in [-1, 1] (bearish to bullish), confidence in [0, 1].
    """

    agent: str
    group: str = ""  # "technical" or "context"
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    direction: Direction
    rationale: str
    features: dict[str, float | str | None] = Field(default_factory=dict)


class GroupScore(BaseModel):
    score: float
    confidence: float
    direction: Direction


class AggregateSignal(BaseModel):
    score: float
    confidence: float
    direction: Direction
    agreement: float  # 1.0 when every confident agent points the same way
    contributions: dict[str, float]
    groups: dict[str, GroupScore] = Field(default_factory=dict)
    veto: bool = False
    veto_reason: str = ""
    # 0 = no position, 1 = full size allowed by the risk budget.
    allocation_multiplier: float = 0.0


class FlowSnapshot(BaseModel):
    """FII/DII provisional cash-market flows for one day, in ₹ crore."""

    date: date
    fii_net: float
    dii_net: float


class MarketContext(BaseModel):
    data_source: str
    vix: float | None = None
    vix_percentile: float | None = None  # 0-100, versus the past year
    vol_regime: str = "unknown"  # low / normal / elevated / extreme
    days_to_expiry: int | None = None
    expiry_tag: str = "unknown"  # expiry_day / near_expiry / mid_cycle
    flows: list[FlowSnapshot] = Field(default_factory=list)


class OptionLeg(BaseModel):
    action: str  # "BUY" or "SELL"
    option_type: str
    strike: float
    tradingsymbol: str
    price: float
    lots: int = 1


class TradeIdea(BaseModel):
    underlying: str
    expiry: date
    strategy: str
    direction: Direction
    legs: list[OptionLeg]
    lot_size: int
    max_loss: float  # rupees for the whole position, positive number
    max_profit: float | None  # None when unbounded
    breakeven: float | None
    rationale: str

    @property
    def net_debit(self) -> float:
        per_unit = sum(
            (leg.price if leg.action == "BUY" else -leg.price) * leg.lots for leg in self.legs
        )
        return per_unit * self.lot_size


class RiskCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class RiskDecision(BaseModel):
    approved: bool
    checks: list[RiskCheck]

    @property
    def failures(self) -> list[RiskCheck]:
        return [c for c in self.checks if not c.passed]


class ResearchReport(BaseModel):
    id: str
    created_at: datetime
    underlying: str
    spot: float | None  # None when no data source could be reached
    context: MarketContext | None = None
    signals: list[AgentSignal]
    aggregate: AggregateSignal
    idea: TradeIdea | None
    risk: RiskDecision | None
    summary: str
