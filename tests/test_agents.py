from types import SimpleNamespace

import anthropic
import httpx2

from fno_research.agents import NewsAgent, OptionsPositioningAgent, PriceActionAgent
from fno_research.agents.news import NewsView
from fno_research.data.sample import SampleDataProvider, StaticNewsProvider
from fno_research.models import Direction


def test_price_action_reads_trend():
    bull = PriceActionAgent(SampleDataProvider(drift=0.01)).analyse("NIFTY")
    bear = PriceActionAgent(SampleDataProvider(drift=-0.01)).analyse("NIFTY")
    assert bull.direction == Direction.BULLISH and bull.score > 0.5
    assert bear.direction == Direction.BEARISH and bear.score < -0.5


def test_options_positioning_signal_is_bounded(provider):
    sig = OptionsPositioningAgent(provider).analyse("NIFTY")
    assert -1 <= sig.score <= 1 and 0 <= sig.confidence <= 1
    assert sig.features["put_wall"] < sig.features["call_wall"]


class FakeMessages:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def fake_client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def test_news_agent_uses_structured_output():
    view = NewsView(score=0.6, confidence=0.7, key_drivers=["FII buying"],
                    event_risk=["US CPI"], rationale="Flows supportive.")
    messages = FakeMessages(SimpleNamespace(stop_reason="end_turn", parsed_output=view))
    sig = NewsAgent(StaticNewsProvider(), client=fake_client(messages)).analyse("NIFTY")
    assert sig.direction == Direction.BULLISH
    assert sig.confidence == 0.7
    call = messages.calls[0]
    assert call["output_format"] is NewsView
    assert "FIIs turn net buyers" in call["messages"][0]["content"]


def test_news_agent_clamps_out_of_range_model_output():
    view = NewsView(score=3, confidence=2, key_drivers=[], event_risk=[], rationale="x")
    messages = FakeMessages(SimpleNamespace(stop_reason="end_turn", parsed_output=view))
    sig = NewsAgent(StaticNewsProvider(), client=fake_client(messages)).analyse("NIFTY")
    assert sig.score == 1.0 and sig.confidence == 1.0


def test_news_agent_is_neutral_on_refusal():
    messages = FakeMessages(SimpleNamespace(stop_reason="refusal", parsed_output=None))
    sig = NewsAgent(StaticNewsProvider(), client=fake_client(messages)).analyse("NIFTY")
    assert sig.score == 0 and sig.confidence == 0


def test_news_agent_is_neutral_on_connection_error():
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    sig = NewsAgent(StaticNewsProvider(), client=fake_client(FakeMessages(error=error))) \
        .analyse("NIFTY")
    assert sig.confidence == 0 and "unreachable" in sig.rationale


def test_news_agent_without_headlines():
    sig = NewsAgent(StaticNewsProvider([]), client=fake_client(FakeMessages())).analyse("NIFTY")
    assert sig.confidence == 0
