"""Scan a stock universe both ways.

Every stock is scored with the technical agents plus relative strength versus NIFTY (and,
for a shortlist, a Claude read of its news). Then:

- Long: above the 200-day EMA and score >= min_score -> a delivery buy with an ATR/swing-low
  stop, a 2R target and a time stop, sized so a stop-out costs `risk_per_trade_pct`.
- Short: below the 200-day EMA and score <= -min_score, and the stock trades in F&O ->
  a bear put spread on its options (cash shorts can't be held overnight in India; stock
  futures lots of ~₹15 lakh are too big for a small account's risk budget).
- Held positions whose score turns against them raise an exit alert.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from collections.abc import Callable
from datetime import datetime

import pandas as pd

from fno_research.agents.base import clamp
from fno_research.agents.news import NewsAgent
from fno_research.agents.technical import TECHNICAL_AGENTS
from fno_research.analytics.indicators import atr, ema
from fno_research.config import NSE_INDEX_NAMES, SwingSettings
from fno_research.models import (
    AggregateSignal,
    Direction,
    OptionChain,
    RiskCheck,
    RiskDecision,
    TradeIdea,
)
from fno_research.strategy import build_idea
from fno_research.swing.data import SwingData
from fno_research.swing.models import (
    LONG,
    SHORT,
    ExitAlert,
    ScanReport,
    Stock,
    StockScore,
    SwingIdea,
    TradePlan,
)

log = logging.getLogger(__name__)
MIN_BARS = 220  # 200-day EMA plus a little


def company_query(name: str) -> str:
    """'Tata Motors Ltd.' -> '"Tata Motors"' for a news search."""
    core = re.sub(r"\b(Ltd\.?|Limited|Ltd)\s*$", "", name or "", flags=re.I).strip(" .,")
    return f'"{core}"' if core else ""


def ret(close: pd.Series, n: int) -> float:
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1) if len(close) > n else 0.0


def combine(agents: dict[str, dict[str, float]], weights: dict[str, float]) -> tuple[float, float]:
    total_w = sum(weights.get(k, 0) for k in agents)
    eff = {k: weights.get(k, 0) * v["confidence"] for k, v in agents.items()}
    eff_total = sum(eff.values())
    if not total_w or not eff_total:
        return 0.0, 0.0
    return (sum(eff[k] * v["score"] for k, v in agents.items()) / eff_total,
            eff_total / total_w)


def qualify(s: StockScore, cfg: SwingSettings) -> StockScore:
    """Set side/qualified/reason from the current score and filters."""
    s.side, s.qualified = None, False
    if s.avg_value_cr < cfg.min_avg_value_cr:
        s.reason = f"Avg traded value ₹{s.avg_value_cr:.1f} cr below ₹{cfg.min_avg_value_cr:.0f} cr"
    elif s.score >= cfg.min_score and s.above_200:
        s.side, s.qualified, s.reason = LONG, True, "Long setup"
    elif s.score >= cfg.min_score:
        s.reason = "Bullish score but below its 200-day EMA"
    elif s.score <= -cfg.min_score and not s.above_200:
        if not cfg.allow_short:
            s.reason = "Bearish setup; shorts are switched off"
        elif not s.fno:
            s.reason = "Bearish setup, but not in F&O so it can't be shorted overnight"
        else:
            s.side, s.qualified, s.reason = SHORT, True, "Short setup (bear put spread)"
    elif s.score <= -cfg.min_score:
        s.reason = "Bearish score but above its 200-day EMA"
    else:
        s.reason = f"Score {s.score:+.2f} inside ±{cfg.min_score:.2f}"
    return s


def score_stock(stock: Stock, frame: pd.DataFrame, bench: pd.DataFrame,
                cfg: SwingSettings, fno: bool = False) -> StockScore:
    close = frame["close"]
    agents: dict[str, dict[str, float]] = {}
    notes = []
    for cls in TECHNICAL_AGENTS:
        sig = cls().analyse_frame(frame)
        agents[sig.agent] = {"score": sig.score, "confidence": sig.confidence}
        notes.append(sig.rationale)

    bclose = bench["close"]
    rs_20, rs_60 = ret(close, 20) - ret(bclose, 20), ret(close, 60) - ret(bclose, 60)
    rs_score = clamp(0.6 * clamp(rs_60 / 0.15) + 0.4 * clamp(rs_20 / 0.08))
    agents["relative_strength"] = {"score": round(rs_score, 4), "confidence": 0.6}
    score, confidence = combine(agents, cfg.weights)

    s = StockScore(
        symbol=stock.symbol, name=stock.name, industry=stock.industry,
        close=round(float(close.iloc[-1]), 2), score=round(score, 4),
        confidence=round(confidence, 4), agents=agents, rs_20=round(rs_20, 4),
        rs_60=round(rs_60, 4), above_200=bool(close.iloc[-1] > ema(close, 200).iloc[-1]),
        avg_value_cr=round(float((close * frame["volume"]).iloc[-20:].mean() / 1e7), 2),
        fno=fno, qualified=False, reason="", rationale=" ".join(notes),
    )
    return qualify(s, cfg)


def apply_news(s: StockScore, score: float, confidence: float, rationale: str,
               cfg: SwingSettings) -> StockScore:
    s.agents["news"] = {"score": score, "confidence": confidence}
    total, conf = combine(s.agents, cfg.weights)
    s.score, s.confidence = round(total, 4), round(conf, 4)
    s.rationale = f"News: {rationale} " + s.rationale
    return qualify(s, cfg)


def build_plan(symbol: str, frame: pd.DataFrame, cfg: SwingSettings) -> TradePlan:
    """Entry at the last close; stop at the tighter of a 2-ATR stop and just under the
    10-day swing low, but never closer than 1 ATR; target at `reward_risk` x risk."""
    close = float(frame["close"].iloc[-1])
    a = float(atr(frame).iloc[-1])
    atr_stop = close - cfg.atr_stop_mult * a
    swing_low = float(frame["low"].iloc[-10:].min()) - 0.25 * a
    stop = min(max(atr_stop, swing_low), close - a)
    risk_per_share = close - stop
    target = close + cfg.reward_risk * risk_per_share
    qty_by_risk = math.floor(cfg.capital * cfg.risk_per_trade_pct / 100 / risk_per_share)
    qty_by_size = math.floor(cfg.capital * cfg.max_position_pct / 100 / close)
    return TradePlan(symbol=symbol, entry=round(close, 2), stop=round(stop, 2),
                     target=round(target, 2), qty=max(0, min(qty_by_risk, qty_by_size)),
                     atr=round(a, 2), time_stop_sessions=cfg.time_stop_sessions)


def build_short(s: StockScore, chain: OptionChain, cfg: SwingSettings) -> TradeIdea | None:
    """Bear put spread from the ATM put. Tries widths of 4 strikes down to 1 and keeps the
    widest whose one-lot max loss fits the risk budget (else the narrowest, which the risk
    check then blocks), and sizes to as many lots as the budget allows."""
    view = AggregateSignal(score=s.score, confidence=s.confidence, direction=Direction.BEARISH,
                           agreement=1.0, contributions={}, allocation_multiplier=1.0)
    budget = cfg.capital * cfg.risk_per_trade_pct / 100
    idea = None
    for width in (4, 3, 2, 1):
        candidate = build_idea(chain, view, default_width=width)
        if candidate is not None:
            idea = candidate
            if candidate.max_loss <= budget:
                break
    if idea is None:
        return None
    lots = max(1, math.floor(budget / idea.max_loss))
    return idea.model_copy(update={
        "legs": [leg.model_copy(update={"lots": lots}) for leg in idea.legs],
        "max_loss": round(idea.max_loss * lots, 2),
        "max_profit": round(idea.max_profit * lots, 2) if idea.max_profit else None,
    })


def _book_checks(symbol: str, industry: str, cfg: SwingSettings, open_symbols: list[str],
                 open_industries: list[str], kill: dict) -> list[RiskCheck]:
    same_industry = sum(1 for i in open_industries if i and i == industry)
    return [
        RiskCheck(name="Kill switch", passed=not kill.get("active"),
                  detail=kill.get("reason", "Tripped") if kill.get("active") else "Not tripped"),
        RiskCheck(name="Not already held", passed=symbol not in open_symbols,
                  detail="Held" if symbol in open_symbols else "Not held"),
        RiskCheck(name="Open positions", passed=len(open_symbols) < cfg.max_open,
                  detail=f"{len(open_symbols)} open vs cap {cfg.max_open}"),
        RiskCheck(name="Industry concentration", passed=same_industry < cfg.max_per_industry,
                  detail=f"{same_industry} open in {industry or 'unknown'} vs cap "
                  f"{cfg.max_per_industry}"),
    ]


def check_plan(plan: TradePlan, score: StockScore, cfg: SwingSettings,
               open_symbols: list[str], open_industries: list[str],
               kill_switch: dict | None = None) -> RiskDecision:
    adv = score.avg_value_cr * 1e7
    budget = cfg.capital * cfg.risk_per_trade_pct / 100
    checks = _book_checks(plan.symbol, score.industry, cfg, open_symbols, open_industries,
                          kill_switch or {"active": False}) + [
        RiskCheck(name="Quantity", passed=plan.qty >= 1, detail=f"{plan.qty} shares"),
        RiskCheck(name="Stop distance",
                  passed=cfg.min_stop_pct <= plan.stop_pct <= cfg.max_stop_pct,
                  detail=f"{plan.stop_pct:.1f}% (allowed {cfg.min_stop_pct:.0f}–"
                  f"{cfg.max_stop_pct:.0f}%)"),
        RiskCheck(name="Reward to risk", passed=plan.reward_risk >= cfg.reward_risk - 0.02,
                  detail=f"{plan.reward_risk:.1f}R"),
        RiskCheck(name="Risk per trade", passed=plan.risk_amount <= budget + 1,
                  detail=f"₹{plan.risk_amount:,.0f} vs ₹{budget:,.0f}"),
        RiskCheck(name="Position size",
                  passed=plan.position_value <= cfg.capital * cfg.max_position_pct / 100 + 1,
                  detail=f"₹{plan.position_value:,.0f} "
                  f"({plan.position_value / cfg.capital:.0%} of capital)"),
        RiskCheck(name="Liquidity",
                  passed=adv > 0 and plan.position_value <= adv * cfg.max_participation_pct / 100,
                  detail=f"{plan.position_value / adv:.2%} of avg daily value" if adv
                  else "No volume data"),
    ]
    return RiskDecision(approved=all(c.passed for c in checks), checks=checks)


def check_short(idea: TradeIdea, chain: OptionChain, score: StockScore, cfg: SwingSettings,
                open_symbols: list[str], open_industries: list[str], banned: set[str] | None,
                kill_switch: dict | None = None) -> RiskDecision:
    budget = cfg.capital * cfg.risk_per_trade_pct / 100
    days = (idea.expiry - chain.as_of.date()).days
    checks = _book_checks(score.symbol, score.industry, cfg, open_symbols, open_industries,
                          kill_switch or {"active": False}) + [
        RiskCheck(name="F&O ban list",
                  passed=banned is not None and score.symbol not in banned,
                  detail="Ban list unavailable" if banned is None else
                  ("In today's ban period" if score.symbol in banned else "Not banned")),
        RiskCheck(name="Defined risk", passed=idea.max_profit is not None,
                  detail="Long put covers the short put"),
        RiskCheck(name="Risk per trade", passed=idea.max_loss <= budget + 1,
                  detail=f"Max loss ₹{idea.max_loss:,.0f} vs ₹{budget:,.0f} "
                  f"({idea.legs[0].lots} lot(s) of {idea.lot_size})"),
        RiskCheck(name="Days to expiry", passed=days >= cfg.option_min_days,
                  detail=f"{days} days vs minimum {cfg.option_min_days}"),
    ]
    for leg in idea.legs:
        q = chain.get(leg.strike, leg.option_type)
        mid = (q.bid + q.ask) / 2 if q else 0
        spread = (q.ask - q.bid) / mid * 100 if q and mid > 0 and q.bid > 0 else float("inf")
        checks.append(RiskCheck(
            name=f"Liquidity {leg.tradingsymbol}", passed=spread <= 5.0 and q is not None
            and q.oi > 0, detail=f"spread {spread:.1f}%, OI {q.oi:,.0f}" if q else "No quote"))
    return RiskDecision(approved=all(c.passed for c in checks), checks=checks)


class SwingScanner:
    def __init__(self, cfg: SwingSettings, data: SwingData, universe: list[Stock],
                 book=None, paper=None, news_agent: NewsAgent | None = None,
                 kill_switch: Callable[[], dict] | None = None,
                 ban_list: Callable[[], set[str]] | None = None):
        self.cfg = cfg
        self.data = data
        self.universe = universe
        self.book = book  # SwingBook: long cash positions
        self.paper = paper  # PaperBook: bearish option spreads
        self.news_agent = news_agent
        self.kill_switch = kill_switch
        self.ban_list = ban_list

    def _held_shorts(self) -> list[dict]:
        if self.paper is None:
            return []
        return [p for p in self.paper.positions("open") if p["underlying"] not in NSE_INDEX_NAMES]

    def run(self, progress: Callable[[int, int, str], None] | None = None) -> ScanReport:
        bench = self.data.benchmark()
        try:
            lots = self.data.lot_sizes()
        except Exception as exc:
            log.info("F&O lot sizes unavailable, bearish ideas off: %s", exc)
            lots = {}
        frames: dict[str, pd.DataFrame] = {}
        scored: list[StockScore] = []
        failed: dict[str, str] = {}
        for i, stock in enumerate(self.universe):
            if progress:
                progress(i, len(self.universe), stock.symbol)
            try:
                frame = self.data.daily(stock.symbol)
                if len(frame) < MIN_BARS:
                    failed[stock.symbol] = f"Only {len(frame)} daily bars"
                    continue
                frames[stock.symbol] = frame
                scored.append(score_stock(stock, frame, bench, self.cfg,
                                          fno=stock.symbol in lots))
            except Exception as exc:
                log.info("Scan of %s failed: %s", stock.symbol, exc)
                failed[stock.symbol] = str(exc)

        # Walk held long positions forward; re-mark held bearish spreads.
        held_longs = self.book.positions("open") if self.book is not None else []
        held_shorts = self._held_shorts()
        if self.book is not None:
            for sym in {p["symbol"] for p in held_longs} - set(frames):
                try:
                    frames[sym] = self.data.daily(sym)
                except Exception as exc:
                    failed.setdefault(sym, str(exc))
            self.book.update(frames)
            held_longs = self.book.positions("open")
        for pos in held_shorts:
            try:
                self.paper.mark(self.data.option_chain(pos["underlying"], pos["lot_size"]))
            except Exception as exc:
                failed.setdefault(pos["underlying"], f"Could not mark spread: {exc}")
        held_shorts = self._held_shorts()

        # News for a shortlist: the strongest candidates each way plus everything held.
        by_symbol = {s.symbol: s for s in scored}
        if self.news_agent is not None:
            ranked = sorted(scored, key=lambda s: s.score)
            shortlist = {s.symbol for s in ranked[-self.cfg.news_shortlist:]}
            shortlist |= {s.symbol for s in ranked[: self.cfg.news_shortlist]}
            shortlist |= {p["symbol"] for p in held_longs}
            shortlist |= {p["underlying"] for p in held_shorts}
            for sym in shortlist & set(by_symbol):
                s = by_symbol[sym]
                sig = self.news_agent.analyse(sym, query=company_query(s.name) or sym)
                if sig.confidence > 0:
                    apply_news(s, sig.score, sig.confidence, sig.rationale, self.cfg)

        alerts = []
        for pos in held_longs:
            s = by_symbol.get(pos["symbol"])
            if s is None:
                continue
            news = s.agents.get("news", {})
            bad_news = (news.get("score", 0) <= -self.cfg.bad_news_score
                        and news.get("confidence", 0) >= 0.5)
            if s.score <= -self.cfg.exit_score or bad_news:
                reason = ("Strongly negative news" if bad_news and s.score > -self.cfg.exit_score
                          else "Signals have turned bearish"
                          + (" (news negative)" if bad_news else ""))
                alerts.append(ExitAlert(symbol=s.symbol, side=LONG, position_id=pos["id"],
                                        score=s.score, reason=reason))
        for pos in held_shorts:
            s = by_symbol.get(pos["underlying"])
            if s is None:
                continue
            news = s.agents.get("news", {})
            good_news = (news.get("score", 0) >= self.cfg.bad_news_score
                         and news.get("confidence", 0) >= 0.5)
            if s.score >= self.cfg.exit_score or good_news:
                alerts.append(ExitAlert(symbol=s.symbol, side=SHORT, position_id=pos["id"],
                                        score=s.score,
                                        reason="Strongly positive news" if good_news
                                        and s.score < self.cfg.exit_score
                                        else "Signals have turned bullish"))

        scored.sort(key=lambda s: s.score, reverse=True)
        open_symbols = [p["symbol"] for p in held_longs] + [p["underlying"] for p in held_shorts]
        open_industries = [p["industry"] for p in held_longs] + [
            by_symbol[p["underlying"]].industry if p["underlying"] in by_symbol else ""
            for p in held_shorts
        ]
        kill = self.kill_switch() if self.kill_switch else None
        banned = None
        if self.ban_list is not None:
            try:
                banned = self.ban_list()
            except Exception as exc:
                log.info("Ban list unavailable: %s", exc)

        longs = [s for s in scored if s.side == LONG][: self.cfg.top_n]
        shorts = [s for s in reversed(scored) if s.side == SHORT][: self.cfg.top_n]
        ideas = []
        # Strongest setups claim the open-position slots first, whichever side they're on.
        for s in sorted(longs + shorts, key=lambda s: abs(s.score), reverse=True):
            frame = frames[s.symbol]
            as_of = frame.index[-1].date()
            if s.side == LONG:
                plan = build_plan(s.symbol, frame, self.cfg)
                risk = check_plan(plan, s, self.cfg, open_symbols, open_industries, kill)
                idea = SwingIdea(id=uuid.uuid4().hex[:12], created_at=datetime.now(),
                                 as_of=as_of, side=LONG, stock=s, plan=plan, risk=risk)
            else:
                try:
                    chain = self.data.option_chain(s.symbol, lots[s.symbol])
                except Exception as exc:
                    failed[s.symbol] = f"Option chain unavailable: {exc}"
                    continue
                spread = build_short(s, chain, self.cfg)
                if spread is None:
                    failed[s.symbol] = "No sensible put spread in the chain"
                    continue
                risk = check_short(spread, chain, s, self.cfg, open_symbols, open_industries,
                                   banned, kill)
                idea = SwingIdea(id=uuid.uuid4().hex[:12], created_at=datetime.now(),
                                 as_of=as_of, side=SHORT, stock=s, option_idea=spread,
                                 risk=risk)
            ideas.append(idea)
            if risk.approved:  # later ideas see this one as if it were taken
                open_symbols.append(s.symbol)
                open_industries.append(s.industry)

        return ScanReport(id=uuid.uuid4().hex[:12], created_at=datetime.now(),
                          universe=self.cfg.universe, source=self.data.name,
                          benchmark_ret_60=round(ret(bench["close"], 60), 4),
                          scored=scored, ideas=ideas, exit_alerts=alerts, failed=failed)
