"""Runs every agent, aggregates, builds a trade idea and passes it through the risk engine.

Side effects per run (when a store / paper book is attached): open paper positions are
marked to market, the kill switch is re-checked, and the run's features, context and
option-chain snapshot are written to the feature store.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fno_research.agents import (
    TECHNICAL_AGENTS,
    FlowsAgent,
    NewsAgent,
    OptionsPositioningAgent,
)
from fno_research.agents.base import make_signal
from fno_research.aggregator import aggregate
from fno_research.analytics.indicators import candles_to_frame
from fno_research.analytics.regime import expiry_tag, vol_regime
from fno_research.config import NSE_INDEX_NAMES, Settings
from fno_research.data.base import FlowsProvider, MarketDataProvider, NewsProvider
from fno_research.models import (
    AgentSignal,
    FlowSnapshot,
    MarketContext,
    OptionChain,
    ResearchReport,
)
from fno_research.paper import PaperBook
from fno_research.risk import MarginInfo, RiskEngine
from fno_research.store import FeatureStore
from fno_research.strategy import build_idea

log = logging.getLogger(__name__)
DAILY_LOOKBACK_DAYS = 400  # calendar days; ~270 sessions, enough for every indicator


class ResearchPipeline:
    def __init__(self, settings: Settings, market: MarketDataProvider, news: NewsProvider,
                 flows: FlowsProvider | None = None, news_agent: NewsAgent | None = None,
                 store: FeatureStore | None = None, paper: PaperBook | None = None):
        self.settings = settings
        self.market = market
        self.flows_provider = flows
        self.technical_agents = [cls() for cls in TECHNICAL_AGENTS]
        self.options_agent = OptionsPositioningAgent(market, settings.strikes_each_side)
        self.flows_agent = FlowsAgent()
        self.news_agent = news_agent or NewsAgent(news, model=settings.anthropic_model)
        self.risk = RiskEngine(settings.risk)
        self.store = store
        self.paper = paper
        self.last_chain: OptionChain | None = None

    @staticmethod
    def _failed(agent, exc: Exception) -> AgentSignal:
        log.exception("%s failed", agent.name)
        return make_signal(agent.name, 0.0, 0.0, f"Agent failed: {exc}", {}, agent.group)

    def _technical(self, underlying: str) -> list[AgentSignal]:
        try:
            daily = candles_to_frame(
                self.market.candles(underlying, "day", DAILY_LOOKBACK_DAYS)
            )
        except Exception as exc:
            return [self._failed(a, exc) for a in self.technical_agents]
        signals = []
        for agent in self.technical_agents:
            try:
                signals.append(agent.analyse_frame(daily))
            except Exception as exc:
                signals.append(self._failed(agent, exc))
        return signals

    def _flows(self) -> list[FlowSnapshot]:
        latest = None
        if self.flows_provider is not None:
            try:
                latest = self.flows_provider.fii_dii()
            except Exception as exc:
                log.info("FII/DII fetch failed: %s", exc)
        if self.store is not None:
            if latest:
                self.store.save_flows(latest)
            return self.store.flows(5)
        return [latest] if latest else []

    def _context(self, chain: OptionChain | None) -> MarketContext:
        ctx = MarketContext(data_source=self.market.name)
        try:
            ctx.vix = self.market.vix()
            history = self.market.vix_history(365)
            ctx.vol_regime, ctx.vix_percentile = vol_regime(ctx.vix, history)
        except Exception as exc:
            log.info("VIX unavailable: %s", exc)
            ctx.vol_regime, _ = vol_regime(ctx.vix)
        if chain is not None:
            ctx.days_to_expiry, ctx.expiry_tag = expiry_tag(chain.expiry, chain.as_of.date())
        return ctx

    def _margin(self, idea) -> MarginInfo | None:
        required = getattr(self.market, "margin_required", None)
        available = getattr(self.market, "available_margin", None)
        if required is None or available is None:
            return None
        try:
            return MarginInfo(required=required(idea), available=available(), source="broker")
        except Exception as exc:
            log.info("Broker margin check failed, using estimate: %s", exc)
            return None

    def _ban_list(self, underlying: str) -> set[str] | None:
        if underlying in NSE_INDEX_NAMES:
            return None  # indices are never banned; skip the fetch
        fetch = getattr(self.market, "ban_list", None) or getattr(self.flows_provider,
                                                                   "ban_list", None)
        if fetch is None:
            return None
        try:
            return fetch()
        except Exception as exc:
            log.info("Ban list unavailable: %s", exc)
            return None

    def run(self, underlying: str) -> ResearchReport:
        underlying = underlying.upper()
        signals = self._technical(underlying)

        try:
            signals.append(self.options_agent.analyse(underlying))
        except Exception as exc:
            signals.append(self._failed(self.options_agent, exc))
        chain = self.options_agent.last_chain
        self.last_chain = chain

        flows = self._flows()
        signals.append(self.flows_agent.analyse_flows(flows))

        if self.settings.has_anthropic or self.news_agent.has_client:
            try:
                signals.append(self.news_agent.analyse(underlying))
            except Exception as exc:
                signals.append(self._failed(self.news_agent, exc))
            voting = signals
        else:
            # Not configured is different from failed: leave it out of the vote entirely.
            voting = list(signals)
            signals.append(make_signal("news", 0.0, 0.0,
                                       "ANTHROPIC_API_KEY not set; news skipped.", {},
                                       "context"))

        context = self._context(chain)
        context.flows = flows
        view = aggregate(voting, self.settings.weights, self.settings.aggregator)

        kill = {"active": False}
        if self.paper is not None:
            if chain is not None:
                self.paper.mark(chain)
            today = chain.as_of.date() if chain else datetime.now().date()
            kill = self.paper.check_kill_switch(self.settings.risk.daily_loss_limit, today)

        idea = risk = None
        if chain is not None:
            options_signal = next(s for s in signals if s.agent == "options_positioning")
            idea = build_idea(chain, view, options_signal)
            if idea is not None:
                idea = self.risk.size(idea, view.allocation_multiplier)
                open_here = (
                    sum(1 for p in self.paper.positions("open") if p["underlying"] == underlying)
                    if self.paper is not None else 0
                )
                risk = self.risk.evaluate(idea, view, chain, context, self._margin(idea), kill,
                                          open_positions=open_here,
                                          banned=self._ban_list(underlying))

        try:
            spot = chain.spot if chain else self.market.spot(underlying)
        except Exception:
            spot = None
        report = ResearchReport(
            id=uuid.uuid4().hex[:12],
            created_at=datetime.now(),
            underlying=underlying,
            spot=spot,
            context=context,
            signals=signals,
            aggregate=view,
            idea=idea,
            risk=risk,
            summary=_summary(underlying, view, idea, risk),
        )
        if self.store is not None:
            self.store.record(report, chain)
        return report


def _summary(underlying, view, idea, risk) -> str:
    head = (f"{underlying}: {view.direction.value} (score {view.score:+.2f}, "
            f"confidence {view.confidence:.2f}, agreement {view.agreement:.2f}, "
            f"allocation {view.allocation_multiplier:.2f}).")
    if view.veto:
        return head + f" Vetoed: {view.veto_reason}"
    if idea is None:
        return head + " No trade: the view is not directional enough or no structure fits."
    if risk and risk.approved:
        return head + f" {idea.strategy} passed all risk checks and awaits review."
    failed = ", ".join(c.name for c in risk.failures) if risk else "not evaluated"
    return head + f" {idea.strategy} blocked by risk: {failed}."

