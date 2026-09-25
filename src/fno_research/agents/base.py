from __future__ import annotations

from fno_research.models import AgentSignal, Direction


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def make_signal(agent: str, score: float, confidence: float, rationale: str,
                features: dict, threshold: float = 0.15) -> AgentSignal:
    score = clamp(score)
    return AgentSignal(
        agent=agent,
        score=round(score, 4),
        confidence=round(clamp(confidence, 0.0, 1.0), 4),
        direction=Direction.from_score(score, threshold),
        rationale=rationale,
        features=features,
    )
