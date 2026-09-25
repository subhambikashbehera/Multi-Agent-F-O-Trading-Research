"""Combines agent signals into one decision.

1. Within each group (technical, context) agents are averaged by weight x confidence.
2. The groups are combined the same way using the group weights.
3. Conflict veto: if the groups point opposite ways and both are strong and confident,
   there is no trade, whatever the net score says.
4. allocation_multiplier (0..1) scales position size for the risk engine: full size needs
   confluence (both groups agree) and confidence at `full_size_confidence`; a lone group
   gets at most half.

All thresholds live in AggregatorSettings and are placeholders until backtested.
"""

from __future__ import annotations

from fno_research.config import AgentWeights, AggregatorSettings
from fno_research.models import AgentSignal, AggregateSignal, Direction, GroupScore


def _weighted(signals: list[AgentSignal], weights: dict[str, float]) -> tuple[float, float]:
    """(score, confidence) of a set of signals; confidence is relative to the weight present."""
    total = sum(weights.get(s.agent, 0.0) for s in signals)
    eff = sum(weights.get(s.agent, 0.0) * s.confidence for s in signals)
    if total == 0 or eff == 0:
        return 0.0, 0.0
    score = sum(weights.get(s.agent, 0.0) * s.confidence * s.score for s in signals) / eff
    return score, eff / total


def aggregate(signals: list[AgentSignal], weights: AgentWeights | None = None,
              cfg: AggregatorSettings | None = None) -> AggregateSignal:
    weights = weights or AgentWeights()
    cfg = cfg or AggregatorSettings()
    threshold = cfg.direction_threshold

    by_group: dict[str, list[AgentSignal]] = {}
    for s in signals:
        if weights.agent.get(s.agent, 0) > 0 and weights.group.get(s.group, 0) > 0:
            by_group.setdefault(s.group, []).append(s)

    groups: dict[str, GroupScore] = {}
    for name, members in by_group.items():
        score, confidence = _weighted(members, weights.agent)
        groups[name] = GroupScore(score=round(score, 4), confidence=round(confidence, 4),
                                  direction=Direction.from_score(score, threshold))

    group_total = sum(weights.group[g] for g in groups)
    group_eff = sum(weights.group[g] * gs.confidence for g, gs in groups.items())
    if group_total == 0 or group_eff == 0:
        return AggregateSignal(
            score=0.0, confidence=0.0, direction=Direction.NEUTRAL, agreement=0.0,
            contributions={s.agent: 0.0 for s in signals}, groups=groups,
        )
    score = sum(weights.group[g] * gs.confidence * gs.score for g, gs in groups.items()) \
        / group_eff

    # Agreement across individual agents, each weighted by its share of the final vote.
    def share(s: AgentSignal) -> float:
        members = by_group.get(s.group, [])
        if s not in members:
            return 0.0
        in_group = sum(weights.agent[m.agent] for m in members)
        return weights.group[s.group] / group_total * weights.agent[s.agent] / in_group

    net = sum(share(s) * s.confidence * s.score for s in signals)
    gross = sum(share(s) * s.confidence * abs(s.score) for s in signals)
    agreement = abs(net) / gross if gross else 0.0
    confidence = group_eff / group_total * (0.5 + 0.5 * agreement)
    direction = Direction.from_score(score, threshold)

    veto, veto_reason = False, ""
    tech, ctx = groups.get("technical"), groups.get("context")
    if tech and ctx and tech.score * ctx.score < 0:
        strong = min(abs(tech.score), abs(ctx.score)) >= cfg.conflict_threshold
        sure = min(tech.confidence, ctx.confidence) >= cfg.conflict_min_confidence
        if strong and sure:
            veto = True
            veto_reason = (
                f"Technical {tech.direction.value} ({tech.score:+.2f}) conflicts with "
                f"context {ctx.direction.value} ({ctx.score:+.2f})."
            )
            direction = Direction.NEUTRAL

    multiplier = 0.0
    if direction != Direction.NEUTRAL:
        multiplier = min(1.0, confidence / cfg.full_size_confidence)
        directional = [g for g in groups.values() if g.direction != Direction.NEUTRAL]
        confluence = len(groups) > 1 and len(directional) == len(groups)
        if not confluence:
            multiplier *= 0.5

    return AggregateSignal(
        score=round(score, 4),
        confidence=round(confidence, 4),
        direction=direction,
        agreement=round(agreement, 4),
        contributions={s.agent: round(share(s) * s.confidence * s.score, 4) for s in signals},
        groups=groups,
        veto=veto,
        veto_reason=veto_reason,
        allocation_multiplier=round(multiplier, 3),
    )
