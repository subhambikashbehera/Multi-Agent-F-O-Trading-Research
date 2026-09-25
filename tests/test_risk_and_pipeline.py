from datetime import date

import pytest

from fno_research.agents.base import make_signal
from fno_research.aggregator import aggregate
from fno_research.data.sample import SampleDataProvider, StaticNewsProvider
from fno_research.models import AggregateSignal, Direction
from fno_research.pipeline import ResearchPipeline
from fno_research.review import APPROVED, BLOCKED, PENDING, ReviewQueue
from fno_research.risk import RiskEngine
from fno_research.strategy import build_idea


def view(direction, confidence=0.6, agreement=0.9, score=0.5):
    return AggregateSignal(score=score if direction == Direction.BULLISH else -score,
                           confidence=confidence, direction=direction, agreement=agreement,
                           contributions={})


def test_bull_call_spread_shape(chain):
    idea = build_idea(chain, view(Direction.BULLISH))
    long, short = idea.legs
    assert (long.action, long.option_type, short.action) == ("BUY", "CE", "SELL")
    assert long.strike == chain.atm_strike() < short.strike
    width = (short.strike - long.strike) * chain.lot_size
    assert idea.max_loss + idea.max_profit == pytest.approx(width)
    assert idea.net_debit == pytest.approx(idea.max_loss)


def test_bear_put_spread_targets_put_wall(chain):
    opts = make_signal("options_positioning", 0, 0.5, "", {"put_wall": 24_850.0})
    idea = build_idea(chain, view(Direction.BEARISH), opts)
    assert idea.strategy == "Bear put spread"
    assert idea.legs[1].strike == 24_850


def test_neutral_view_gives_no_trade(chain):
    assert build_idea(chain, view(Direction.NEUTRAL)) is None


def test_sizing_respects_budget(settings, chain):
    settings.risk.capital = 2_000_000  # ₹20k budget fits a few lots
    engine = RiskEngine(settings.risk)
    one_lot = build_idea(chain, view(Direction.BULLISH))
    idea = engine.size(one_lot)
    lots = idea.legs[0].lots
    assert lots == min(settings.risk.max_lots, int(engine.risk_budget // one_lot.max_loss))
    assert lots > 1 and idea.max_loss <= engine.risk_budget
    assert idea.max_loss == pytest.approx(one_lot.max_loss * lots)


def test_risk_blocks_low_confidence_and_expiry_day(settings, chain):
    engine = RiskEngine(settings.risk)
    idea = engine.size(build_idea(chain, view(Direction.BULLISH)))
    decision = engine.evaluate(idea, view(Direction.BULLISH, confidence=0.1), chain,
                               today=chain.expiry)
    failed = {c.name for c in decision.failures}
    assert not decision.approved
    assert {"Signal confidence", "Days to expiry"} <= failed


def test_risk_blocks_when_one_lot_exceeds_budget(settings, chain):
    settings.risk.capital = 50_000
    engine = RiskEngine(settings.risk)
    idea = engine.size(build_idea(chain, view(Direction.BULLISH)))
    decision = engine.evaluate(idea, view(Direction.BULLISH), chain, today=date(2026, 9, 22))
    assert "Risk budget" in {c.name for c in decision.failures}


def test_pipeline_to_review_queue(settings):
    pipeline = ResearchPipeline(settings, SampleDataProvider(drift=-0.01), StaticNewsProvider())
    report = pipeline.run("nifty")
    assert report.underlying == "NIFTY"
    assert report.aggregate.direction == Direction.BEARISH
    assert report.idea is not None and report.risk is not None
    # Without an API key the news agent is shown but does not vote.
    news = next(s for s in report.signals if s.agent == "news")
    assert "skipped" in news.rationale and "news" not in report.aggregate.contributions

    queue = ReviewQueue(settings.db_path)
    status = queue.add(report)
    assert status in (PENDING, BLOCKED)
    if status == PENDING:
        queue.decide(report.id, approve=True, note="looks fine")
        assert queue.list(APPROVED)[0]["reviewer_note"] == "looks fine"
        with pytest.raises(ValueError):
            queue.decide(report.id, approve=False)


def test_failed_agent_does_not_crash_pipeline(settings):
    class Broken(SampleDataProvider):
        def candles(self, *a, **k):
            raise RuntimeError("feed down")

    report = ResearchPipeline(settings, Broken(), StaticNewsProvider()).run("NIFTY")
    price = next(s for s in report.signals if s.agent == "price_action")
    assert price.confidence == 0 and "feed down" in price.rationale


def test_aggregate_ignores_unknown_agents():
    view_ = aggregate([make_signal("mystery", 1, 1, "", {})], {"price_action": 1})
    assert view_.direction == Direction.NEUTRAL
