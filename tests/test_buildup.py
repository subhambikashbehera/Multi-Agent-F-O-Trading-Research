import pytest

from fno_research.agents import OptionsPositioningAgent
from fno_research.analytics.buildup import (
    LONG_BUILDUP,
    LONG_UNWINDING,
    NO_SIGNAL,
    SHORT_BUILDUP,
    SHORT_COVERING,
    classify,
    oi_buildup,
)


@pytest.mark.parametrize("price, oi, state", [
    (5, 100, LONG_BUILDUP), (-5, 100, SHORT_BUILDUP),
    (5, -100, SHORT_COVERING), (-5, -100, LONG_UNWINDING), (0, 100, NO_SIGNAL),
])
def test_classify(price, oi, state):
    assert classify(price, oi) == state


def with_changes(chain, ce, pe):
    """Set every CE/PE quote's (price_change, oi_change)."""
    quotes = [q.model_copy(update=dict(zip(("price_change", "oi_change"),
                                           ce if q.option_type == "CE" else pe, strict=True)))
              for q in chain.quotes]
    return chain.model_copy(update={"quotes": quotes})


def test_put_writing_and_call_covering_is_bullish(chain):
    b = oi_buildup(with_changes(chain, ce=(5, -1000), pe=(-5, 1000)))
    assert b.call_state == SHORT_COVERING and b.put_state == SHORT_BUILDUP
    assert b.score == pytest.approx(1.0)


def test_call_writing_and_put_buying_is_bearish(chain):
    b = oi_buildup(with_changes(chain, ce=(-5, 1000), pe=(5, 1000)))
    assert b.call_state == SHORT_BUILDUP and b.put_state == LONG_BUILDUP
    assert b.score == pytest.approx(-1.0)


def test_only_near_the_money_counts(chain):
    b = oi_buildup(chain, strikes_each_side=2)
    assert len({r["strike"] for r in b.rows}) == 5


def test_positioning_agent_uses_buildup(chain):
    bear = OptionsPositioningAgent(None).analyse_chain(
        with_changes(chain, ce=(-5, 1000), pe=(5, 1000)))
    bull = OptionsPositioningAgent(None).analyse_chain(
        with_changes(chain, ce=(5, -1000), pe=(-5, 1000)))
    assert bear.features["oi_buildup"] == -1 and bull.features["oi_buildup"] == 1
    assert bull.score > bear.score
    assert "calls: short build-up" in bear.rationale
