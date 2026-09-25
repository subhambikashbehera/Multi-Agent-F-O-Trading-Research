from datetime import date
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from fno_research.agents import (
    FlowsAgent,
    MomentumAgent,
    NewsAgent,
    OptionsPositioningAgent,
    TrendAgent,
    VolatilityAgent,
    VolumeAgent,
)
from fno_research.agents.news import NewsView
from fno_research.analytics.indicators import candles_to_frame
from fno_research.data.sample import SampleDataProvider, StaticNewsProvider
from fno_research.models import Direction, FlowSnapshot


def daily(drift, n=250):
    return candles_to_frame(SampleDataProvider(drift=drift).candles("NIFTY", "day", n))


@pytest.mark.parametrize("agent_cls", [TrendAgent, MomentumAgent, VolumeAgent])
def test_directional_technical_agents_read_trend(agent_cls):
    bull = agent_cls().analyse_frame(daily(0.004))
    bear = agent_cls().analyse_frame(daily(-0.004))
    assert bull.group == "technical"
    assert bull.direction == Direction.BULLISH
    assert bear.direction == Direction.BEARISH


def test_trend_confidence_needs_adx():
    strong = TrendAgent().analyse_frame(daily(0.01))
    flat = TrendAgent().analyse_frame(daily(0.0))
    assert strong.features["adx"] > flat.features["adx"]
    assert strong.confidence > flat.confidence


def test_volatility_agent_flags_breakout():
    frame = daily(0.0)
    # Quiet market, then a sharp two-day jump through the upper band.
    frame.loc[:, ["open", "high", "low", "close"]] = 25_000.0
    frame.iloc[:-2, frame.columns.get_loc("high")] = 25_020.0
    frame.iloc[:-2, frame.columns.get_loc("low")] = 24_980.0
    for i, px in ((-2, 25_300.0), (-1, 25_600.0)):
        frame.iloc[i, [frame.columns.get_loc(c) for c in ("open", "high", "low", "close")]] = \
            [px - 100, px + 20, px - 120, px]
    sig = VolatilityAgent().analyse_frame(frame)
    assert sig.features["state"] == "upside breakout"
    assert sig.direction == Direction.BULLISH and sig.confidence >= 0.55


def test_volume_agent_without_volume_abstains():
    frame = daily(0.01)
    frame["volume"] = 0.0
    sig = VolumeAgent().analyse_frame(frame)
    assert sig.confidence == 0 and "No volume" in sig.rationale


def test_technical_agents_need_enough_bars():
    sig = TrendAgent().analyse_frame(daily(0.01, n=30))
    assert sig.confidence == 0


def test_flows_agent():
    days = [FlowSnapshot(date=date(2026, 9, d), fii_net=3_000, dii_net=500) for d in
            range(15, 20)]
    sig = FlowsAgent().analyse_flows(days)
    assert sig.group == "context" and sig.score == pytest.approx(1.0)
    assert sig.confidence == pytest.approx(0.4)
    one = FlowsAgent().analyse_flows(days[:1])
    assert one.confidence < sig.confidence
    assert FlowsAgent().analyse_flows([]).confidence == 0


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
