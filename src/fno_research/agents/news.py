"""News sentiment agent. Claude reads the headlines and returns a structured view."""

from __future__ import annotations

import logging

import anthropic
from pydantic import BaseModel

from fno_research.agents.base import make_signal
from fno_research.data.base import NewsProvider
from fno_research.models import AgentSignal, NewsItem

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the news analyst on an Indian index derivatives research desk. You receive recent \
market headlines and judge their likely effect on the named index over the next one to \
five trading sessions.

Weigh what actually moves Indian indices: FII/DII flows, RBI policy and liquidity, \
inflation and GDP prints, crude oil and the rupee, US yields and Fed signals, global risk \
sentiment, heavyweight earnings (banks for BANKNIFTY), and scheduled event risk. Ignore \
stock-specific stories unless the company is an index heavyweight.

score runs from -1 (strongly bearish) to 1 (strongly bullish). confidence runs from 0 to 1 \
and should be low when the headlines are stale, mixed, or not index-relevant. Say so \
plainly when the news carries no real signal; a neutral, low-confidence view is a valid \
answer."""


class NewsView(BaseModel):
    score: float
    confidence: float
    key_drivers: list[str]
    event_risk: list[str]
    rationale: str


class NewsAgent:
    name = "news"

    def __init__(self, news: NewsProvider, model: str = "claude-opus-5",
                 client: anthropic.Anthropic | None = None, effort: str = "medium"):
        self.news = news
        self.model = model
        self.effort = effort
        self._client = client

    @property
    def has_client(self) -> bool:
        return self._client is not None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def analyse(self, underlying: str) -> AgentSignal:
        items = self.news.headlines(underlying)
        if not items:
            return self._neutral("No headlines available.")
        try:
            view = self._ask_claude(underlying, items)
        except anthropic.APIConnectionError as exc:
            log.warning("News agent could not reach the Claude API: %s", exc)
            return self._neutral("Claude API unreachable; news not assessed.")
        except anthropic.AuthenticationError:
            return self._neutral("No valid Anthropic credentials; news not assessed.")
        except anthropic.APIStatusError as exc:
            log.warning("News agent API error %s (request %s)", exc.status_code,
                        exc.request_id if hasattr(exc, "request_id") else "?")
            return self._neutral(f"Claude API error {exc.status_code}; news not assessed.")
        if view is None:
            return self._neutral("Model declined to assess these headlines.")

        return make_signal(
            self.name,
            view.score,
            view.confidence,
            view.rationale,
            {
                "headlines": len(items),
                "key_drivers": "; ".join(view.key_drivers),
                "event_risk": "; ".join(view.event_risk),
            },
        )

    def _ask_claude(self, underlying: str, items: list[NewsItem]) -> NewsView | None:
        lines = []
        for i, item in enumerate(items, 1):
            when = item.published.strftime("%d %b %H:%M") if item.published else "undated"
            line = f"{i}. [{when}, {item.source}] {item.title}"
            if item.summary:
                line += f" — {item.summary}"
            lines.append(line)
        prompt = f"Index: {underlying}\n\nHeadlines:\n" + "\n".join(lines)

        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            # On a safety decline, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_format=NewsView,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return None
        return response.parsed_output

    def _neutral(self, reason: str) -> AgentSignal:
        return make_signal(self.name, 0.0, 0.0, reason, {})
