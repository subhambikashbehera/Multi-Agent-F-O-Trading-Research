import pytest

from fno_research.agents.base import make_signal
from fno_research.aggregator import aggregate
from fno_research.models import Direction

W = {"price_action": 0.4, "options_positioning": 0.4, "news": 0.2}


def test_agreeing_agents():
    signals = [make_signal("price_action", 0.8, 0.8, "", {}),
               make_signal("options_positioning", 0.6, 0.8, "", {}),
               make_signal("news", 0.4, 0.5, "", {})]
    view = aggregate(signals, W)
    assert view.direction == Direction.BULLISH
    assert view.agreement == pytest.approx(1.0)
    assert 0.6 < view.score < 0.8


def test_disagreeing_agents_cancel():
    signals = [make_signal("price_action", 0.8, 0.8, "", {}),
               make_signal("options_positioning", -0.8, 0.8, "", {})]
    view = aggregate(signals, W)
    assert view.direction == Direction.NEUTRAL
    assert view.agreement == pytest.approx(0.0)
    assert view.confidence == pytest.approx(0.4)


def test_zero_confidence_agent_has_no_say():
    signals = [make_signal("price_action", 0.5, 0.6, "", {}),
               make_signal("news", -1.0, 0.0, "", {})]
    view = aggregate(signals, W)
    assert view.score == pytest.approx(0.5)
    assert view.contributions["news"] == 0


def test_all_zero_confidence_is_neutral():
    view = aggregate([make_signal("news", 1.0, 0.0, "", {})], W)
    assert view.direction == Direction.NEUTRAL and view.confidence == 0
