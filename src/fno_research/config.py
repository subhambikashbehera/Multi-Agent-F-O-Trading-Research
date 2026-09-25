"""Runtime settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Index underlyings we support, mapped to the NSE quote symbol Kite uses for the spot.
INDEX_SPOT_SYMBOLS = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else float(default)


@dataclass
class AgentWeights:
    price_action: float = 0.40
    options_positioning: float = 0.40
    news: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {
            "price_action": self.price_action,
            "options_positioning": self.options_positioning,
            "news": self.news,
        }


@dataclass
class RiskLimits:
    capital: float = field(default_factory=lambda: _env_float("FNO_CAPITAL", 500_000))
    max_risk_pct: float = field(default_factory=lambda: _env_float("FNO_MAX_RISK_PCT", 1.0))
    min_confidence: float = 0.35
    min_agreement: float = 0.5
    max_lots: int = 5
    min_oi: float = 5_000  # contracts of open interest on every leg
    max_spread_pct: float = 5.0  # bid/ask spread as % of mid on every leg
    min_days_to_expiry: int = 1  # avoid opening on expiry day
    require_defined_risk: bool = True


@dataclass
class Settings:
    kite_api_key: str = field(default_factory=lambda: os.getenv("KITE_API_KEY", ""))
    kite_api_secret: str = field(default_factory=lambda: os.getenv("KITE_API_SECRET", ""))
    kite_access_token: str = field(default_factory=lambda: os.getenv("KITE_ACCESS_TOKEN", ""))
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("FNO_CLAUDE_MODEL", "claude-opus-5")
    )
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("FNO_DB_PATH", "data/fno_research.db"))
    )
    weights: AgentWeights = field(default_factory=AgentWeights)
    risk: RiskLimits = field(default_factory=RiskLimits)
    direction_threshold: float = 0.15  # |aggregate score| needed to call a direction
    strikes_each_side: int = 15  # chain width fetched around ATM

    @property
    def has_kite(self) -> bool:
        return bool(self.kite_api_key and self.kite_access_token)

    @property
    def has_anthropic(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
