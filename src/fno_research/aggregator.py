"""Combines agent signals into one view using configured weights and each agent's confidence."""

from __future__ import annotations

from fno_research.models import AgentSignal, AggregateSignal, Direction


def aggregate(signals: list[AgentSignal], weights: dict[str, float],
              threshold: float = 0.15) -> AggregateSignal:
    total_weight = sum(weights.get(s.agent, 0.0) for s in signals)
    effective = {s.agent: weights.get(s.agent, 0.0) * s.confidence for s in signals}
    eff_total = sum(effective.values())

    if total_weight == 0 or eff_total == 0:
        return AggregateSignal(
            score=0.0, confidence=0.0, direction=Direction.NEUTRAL, agreement=0.0,
            contributions={s.agent: 0.0 for s in signals},
        )

    score = sum(effective[s.agent] * s.score for s in signals) / eff_total
    # 1.0 when all confident agents point the same way, 0 when they cancel out.
    gross = sum(effective[s.agent] * abs(s.score) for s in signals)
    agreement = abs(score * eff_total) / gross if gross else 0.0
    avg_confidence = eff_total / total_weight
    confidence = avg_confidence * (0.5 + 0.5 * agreement)

    return AggregateSignal(
        score=round(score, 4),
        confidence=round(confidence, 4),
        direction=Direction.from_score(score, threshold),
        agreement=round(agreement, 4),
        contributions={
            s.agent: round(effective[s.agent] * s.score / total_weight, 4) for s in signals
        },
    )
