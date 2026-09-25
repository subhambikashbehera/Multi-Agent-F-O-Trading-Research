from datetime import date

import pytest

from fno_research.agents.base import make_signal
from fno_research.data.sample import SampleDataProvider, StaticNewsProvider
from fno_research.models import AggregateSignal, Direction, MarketContext
from fno_research.paper import PaperBook
from fno_research.pipeline import ResearchPipeline
from fno_research.review import APPROVED, PENDING, ReviewQueue, decision_log
from fno_research.risk import MarginInfo, RiskEngine
from fno_research.store import FeatureStore
from fno_research.strategy import build_idea


def view(direction, confidence=0.6, agreement=0.9, score=0.5, multiplier=1.0, veto=False):
    return AggregateSignal(score=score if direction == Direction.BULLISH else -score,
                           confidence=confidence, direction=direction, agreement=agreement,
                           contributions={}, allocation_multiplier=multiplier, veto=veto)


def failed(decision):
    return {c.name for c in decision.failures}


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


def test_sizing_uses_budget_cap_and_multiplier(settings, chain):
    settings.risk.capital = 2_000_000  # ₹20k budget fits a few lots
    settings.risk.max_lots = 10
    engine = RiskEngine(settings.risk)
    one_lot = build_idea(chain, view(Direction.BULLISH))
    budget_lots = int(engine.risk_budget // one_lot.max_loss)
    full = engine.size(one_lot, 1.0)
    half = engine.size(one_lot, 0.5)
    assert full.legs[0].lots == min(10, budget_lots)
    assert half.legs[0].lots == max(1, full.legs[0].lots // 2)
    assert full.max_loss == pytest.approx(one_lot.max_loss * full.legs[0].lots)


def test_testing_default_is_one_lot(settings, chain):
    assert settings.risk.max_lots == 1
    idea = RiskEngine(settings.risk).size(build_idea(chain, view(Direction.BULLISH)))
    assert idea.legs[0].lots == 1


def test_all_checks_pass_on_a_clean_idea(settings, chain):
    settings.risk.capital = 1_000_000  # one lot (~₹5.6k max loss) fits the 1% budget
    engine = RiskEngine(settings.risk)
    idea = engine.size(build_idea(chain, view(Direction.BULLISH)))
    decision = engine.evaluate(idea, view(Direction.BULLISH), chain,
                               MarketContext(data_source="sample", vol_regime="normal"))
    assert decision.approved, failed(decision)


def test_risk_blocks(settings, chain):
    engine = RiskEngine(settings.risk)
    idea = engine.size(build_idea(chain, view(Direction.BULLISH)))
    decision = engine.evaluate(
        idea, view(Direction.BULLISH, confidence=0.1, multiplier=0), chain,
        MarketContext(data_source="sample", vol_regime="extreme", vix=34.0),
        MarginInfo(required=400_000, available=500_000, source="broker"),
        kill_switch={"active": True, "reason": "test"},
        today=chain.expiry,
    )
    assert {"Signal confidence", "Days to expiry", "Kill switch", "Volatility regime",
            "Margin", "Allocation multiplier"} <= failed(decision)


def test_risk_blocks_when_one_lot_exceeds_budget(settings, chain):
    settings.risk.capital = 50_000
    engine = RiskEngine(settings.risk)
    idea = engine.size(build_idea(chain, view(Direction.BULLISH)))
    decision = engine.evaluate(idea, view(Direction.BULLISH), chain, today=date(2026, 9, 22))
    assert "Risk budget" in failed(decision)


def test_pipeline_end_to_end(settings):
    store, book = FeatureStore(settings.db_path), PaperBook(settings.db_path)
    settings.risk.capital = 1_000_000
    pipeline = ResearchPipeline(settings, SampleDataProvider(drift=0.004),
                                StaticNewsProvider(), flows=SampleDataProvider(),
                                store=store, paper=book)
    report = pipeline.run("nifty")
    assert report.underlying == "NIFTY"
    assert report.aggregate.groups["technical"].direction == Direction.BULLISH
    assert report.aggregate.groups["context"].direction == Direction.BULLISH
    assert report.context.vix and report.context.vol_regime != "unknown"
    assert report.context.expiry_tag == "mid_cycle"
    # Without an API key the news agent is shown but does not vote.
    news = next(s for s in report.signals if s.agent == "news")
    assert "skipped" in news.rationale and "news" not in report.aggregate.contributions
    # Feature store captured the run.
    assert not store.feature_history("NIFTY", "trend", "adx").empty
    assert store.chains("NIFTY") and store.flows()

    queue = ReviewQueue(settings.db_path)
    assert queue.add(report) == PENDING, report.summary
    approved = queue.decide(report.id, approve=True, note="looks fine")
    book.open_from_report(approved)
    assert queue.list(APPROVED)[0]["reviewer_note"] == "looks fine"
    assert len(book.positions("open")) == 1
    with pytest.raises(ValueError):
        queue.decide(report.id, approve=False)

    log = decision_log(queue, book)
    assert log[0]["paper"] == "open" and log[0]["pnl"] == pytest.approx(0)
    assert "F&O ban list" not in failed(report.risk)


def test_failed_data_does_not_crash_pipeline(settings):
    class Broken(SampleDataProvider):
        def candles(self, *a, **k):
            raise RuntimeError("feed down")

        def vix(self):
            raise RuntimeError("vix down")

    report = ResearchPipeline(settings, Broken(), StaticNewsProvider()).run("NIFTY")
    trend = next(s for s in report.signals if s.agent == "trend")
    assert trend.confidence == 0 and "feed down" in trend.rationale
    assert report.context.vol_regime == "unknown"


def test_tripped_kill_switch_blocks_new_ideas(settings):
    book = PaperBook(settings.db_path)
    book._set_state({"active": True, "reason": "test trip", "tripped_at": "x"})
    pipeline = ResearchPipeline(settings, SampleDataProvider(drift=-0.01),
                                StaticNewsProvider(), paper=book)
    report = pipeline.run("NIFTY")
    if report.risk:
        assert "Kill switch" in failed(report.risk)


def test_existing_exposure_blocks_stacking(settings, chain):
    settings.risk.capital = 1_000_000
    engine = RiskEngine(settings.risk)
    idea = engine.size(build_idea(chain, view(Direction.BULLISH)))
    ctx = MarketContext(data_source="sample", vol_regime="normal")
    assert engine.evaluate(idea, view(Direction.BULLISH), chain, ctx,
                           open_positions=0).approved
    decision = engine.evaluate(idea, view(Direction.BULLISH), chain, ctx, open_positions=1)
    assert failed(decision) == {"Existing exposure"}


def test_report_round_trips_when_all_data_is_down(settings):
    class Down(SampleDataProvider):
        def candles(self, *a, **k):
            raise ConnectionError("blocked")

        option_chain = spot = vix = candles

    report = ResearchPipeline(settings, Down(), StaticNewsProvider()).run("NIFTY")
    assert report.spot is None and report.idea is None
    queue = ReviewQueue(settings.db_path)
    queue.add(report)
    assert queue.list()[0]["report"].id == report.id


def test_ban_list_check(settings, chain):
    engine = RiskEngine(settings.risk)
    idea = engine.size(build_idea(chain, view(Direction.BULLISH)))
    stock = idea.model_copy(update={"underlying": "RBLBANK"})
    b = Direction.BULLISH
    assert "F&O ban list" in failed(engine.evaluate(stock, view(b), chain, banned={"RBLBANK"}))
    assert "F&O ban list" in failed(engine.evaluate(stock, view(b), chain, banned=None))
    assert "F&O ban list" not in failed(engine.evaluate(stock, view(b), chain, banned=set()))
    # Indices are never banned, even with no list.
    assert "F&O ban list" not in failed(engine.evaluate(idea, view(b), chain, banned=None))
