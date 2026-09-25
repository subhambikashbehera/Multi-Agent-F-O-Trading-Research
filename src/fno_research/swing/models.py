from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from fno_research.models import RiskDecision, TradeIdea


class Stock(BaseModel):
    symbol: str
    name: str = ""
    industry: str = ""


LONG, SHORT = "long", "short"


class StockScore(BaseModel):
    symbol: str
    name: str = ""
    industry: str = ""
    close: float
    score: float
    confidence: float
    agents: dict[str, dict[str, float]] = Field(default_factory=dict)  # agent -> score/conf
    rs_20: float  # stock minus benchmark return, 20 sessions
    rs_60: float
    above_200: bool
    avg_value_cr: float
    fno: bool = False  # has stock options/futures on NSE (needed for bearish trades)
    side: str | None = None  # "long" / "short" when it qualifies
    qualified: bool
    reason: str  # why it did or didn't qualify
    rationale: str = ""


class TradePlan(BaseModel):
    symbol: str
    entry: float
    stop: float
    target: float
    qty: int
    atr: float
    time_stop_sessions: int

    @property
    def risk_per_share(self) -> float:
        return self.entry - self.stop

    @property
    def risk_amount(self) -> float:
        return self.risk_per_share * self.qty

    @property
    def position_value(self) -> float:
        return self.entry * self.qty

    @property
    def reward_risk(self) -> float:
        return (self.target - self.entry) / self.risk_per_share if self.risk_per_share else 0.0

    @property
    def stop_pct(self) -> float:
        return self.risk_per_share / self.entry * 100


class SwingIdea(BaseModel):
    """Long ideas are delivery buys (plan). Short ideas are bear put spreads on the stock's
    options (option_idea), since cash shorts can't be held overnight in India."""

    id: str
    created_at: datetime
    as_of: date  # date of the last daily bar the plan was built on
    side: str = LONG
    stock: StockScore
    plan: TradePlan | None = None
    option_idea: TradeIdea | None = None
    risk: RiskDecision

    @property
    def symbol(self) -> str:
        return self.stock.symbol

    @property
    def max_loss(self) -> float:
        if self.plan is not None:
            return self.plan.risk_amount
        return self.option_idea.max_loss if self.option_idea else 0.0


class ExitAlert(BaseModel):
    symbol: str
    side: str
    position_id: int
    score: float
    reason: str


class ScanReport(BaseModel):
    id: str
    created_at: datetime
    universe: str
    source: str
    benchmark_ret_60: float
    scored: list[StockScore]
    ideas: list[SwingIdea]
    exit_alerts: list[ExitAlert] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)  # symbol -> error
