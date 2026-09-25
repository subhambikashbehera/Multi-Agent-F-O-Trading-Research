"""Runs every agent, aggregates, builds a trade idea and passes it through the risk engine."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fno_research.agents import NewsAgent, OptionsPositioningAgent, PriceActionAgent
from fno_research.agents.base import make_signal
from fno_research.aggregator import aggregate
from fno_research.config import Settings
from fno_research.data.base import MarketDataProvider, NewsProvider
from fno_research.models import AgentSignal, OptionChain, ResearchReport
from fno_research.risk import RiskEngine
from fno_research.strategy import build_idea

log = logging.getLogger(__name__)


class ResearchPipeline:
    def __init__(self, settings: Settings, market: MarketDataProvider, news: NewsProvider,
                 news_agent: NewsAgent | None = None):
        self.settings = settings
        self.market = market
        self.price_agent = PriceActionAgent(market)
        self.options_agent = OptionsPositioningAgent(market, settings.strikes_each_side)
        self.news_agent = news_agent or NewsAgent(news, model=settings.anthropic_model)
        self.risk = RiskEngine(settings.risk)
        self.last_chain: OptionChain | None = None

    def _run_agent(self, agent, underlying: str) -> AgentSignal:
        try:
            return agent.analyse(underlying)
        except Exception as exc:  # an agent failing should lower confidence, not crash the run
            log.exception("%s failed", agent.name)
            return make_signal(agent.name, 0.0, 0.0, f"Agent failed: {exc}", {})

    def run(self, underlying: str) -> ResearchReport:
        underlying = underlying.upper()
        signals = [
            self._run_agent(self.price_agent, underlying),
            self._run_agent(self.options_agent, underlying),
        ]
        voting = list(signals)
        if self.settings.has_anthropic or self.news_agent.has_client:
            signals.append(self._run_agent(self.news_agent, underlying))
            voting.append(signals[-1])
        else:
            # Not configured is different from failed: leave it out of the vote entirely
            # rather than dragging every run's confidence down.
            signals.append(make_signal("news", 0.0, 0.0,
                                       "ANTHROPIC_API_KEY not set; news skipped.", {}))

        view = aggregate(voting, self.settings.weights.as_dict(),
                         self.settings.direction_threshold)
        chain = self.options_agent.last_chain
        self.last_chain = chain

        idea = risk = None
        if chain is not None:
            options_signal = next(s for s in signals if s.agent == "options_positioning")
            idea = build_idea(chain, view, options_signal)
            if idea is not None:
                idea = self.risk.size(idea)
                risk = self.risk.evaluate(idea, view, chain)

        spot = chain.spot if chain else self.market.spot(underlying)
        return ResearchReport(
            id=uuid.uuid4().hex[:12],
            created_at=datetime.now(),
            underlying=underlying,
            spot=spot,
            signals=signals,
            aggregate=view,
            idea=idea,
            risk=risk,
            summary=_summary(underlying, view, idea, risk),
        )


def _summary(underlying, view, idea, risk) -> str:
    head = (f"{underlying}: {view.direction.value} (score {view.score:+.2f}, "
            f"confidence {view.confidence:.2f}, agreement {view.agreement:.2f}).")
    if idea is None:
        return head + " No trade: the view is not directional enough or no structure fits."
    if risk and risk.approved:
        return head + f" {idea.strategy} passed all risk checks and awaits review."
    failed = ", ".join(c.name for c in risk.failures) if risk else "not evaluated"
    return head + f" {idea.strategy} blocked by risk: {failed}."
