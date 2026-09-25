import pytest

from fno_research.agents.base import make_signal
from fno_research.aggregator import aggregate
from fno_research.config import AgentWeights, AggregatorSettings
from fno_research.models import Direction


def tech(score, conf=0.8, agent="trend"):
    return make_signal(agent, score, conf, "", {}, "technical")


def ctx(score, conf=0.8, agent="options_positioning"):
    return make_signal(agent, score, conf, "", {}, "context")


def test_confluence_gets_full_size():
    view = aggregate([tech(0.7), tech(0.6, agent="momentum"), ctx(0.6), ctx(0.5, agent="flows")])
    assert view.direction == Direction.BULLISH
    assert view.groups["technical"].direction == view.groups["context"].direction
    assert view.agreement == pytest.approx(1.0)
    assert view.allocation_multiplier == pytest.approx(1.0)
    assert not view.veto


def test_conflict_veto():
    view = aggregate([tech(0.8), ctx(-0.6)])
    assert view.veto and view.direction == Direction.NEUTRAL
    assert view.allocation_multiplier == 0
    assert "conflicts" in view.veto_reason


def test_weak_conflict_is_not_vetoed():
    # Context leans the other way, but below the conflict threshold: no veto, half size.
    view = aggregate([tech(0.8), ctx(-0.1)])
    assert not view.veto
    assert view.direction == Direction.BULLISH
    assert 0 < view.allocation_multiplier <= 0.5


def test_single_group_is_capped_at_half():
    view = aggregate([tech(0.9, conf=0.9)])
    assert view.direction == Direction.BULLISH
    assert view.allocation_multiplier == pytest.approx(0.5)


def test_multiplier_scales_with_confidence():
    low = aggregate([tech(0.6, conf=0.3), ctx(0.6, conf=0.3)])
    high = aggregate([tech(0.6, conf=0.9), ctx(0.6, conf=0.9)])
    assert low.allocation_multiplier < high.allocation_multiplier == 1.0


def test_zero_confidence_agent_has_no_say():
    view = aggregate([tech(0.5, conf=0.6), ctx(-1.0, conf=0.0)])
    assert view.score == pytest.approx(0.5)
    assert view.contributions["options_positioning"] == 0


def test_all_zero_confidence_is_neutral():
    view = aggregate([tech(1.0, conf=0.0)])
    assert view.direction == Direction.NEUTRAL and view.confidence == 0


def test_unknown_agents_and_zero_weights_are_ignored():
    weights = AgentWeights()
    weights.agent["momentum"] = 0
    view = aggregate([make_signal("mystery", 1, 1, "", {}, "technical"),
                      tech(-1, agent="momentum")], weights)
    assert view.direction == Direction.NEUTRAL


def test_conflict_threshold_is_configurable():
    cfg = AggregatorSettings(conflict_threshold=0.9)
    assert not aggregate([tech(0.8), ctx(-0.6)], cfg=cfg).veto
