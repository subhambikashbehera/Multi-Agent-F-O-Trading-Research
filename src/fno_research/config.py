"""Runtime settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SUPPORTED_UNDERLYINGS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

# NSE spot index names (as used by nseindia.com and Kite's NSE segment).
NSE_INDEX_NAMES = {
    "NIFTY": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NIFTY MID SELECT",
}
INDEX_SPOT_SYMBOLS = {k: f"NSE:{v}" for k, v in NSE_INDEX_NAMES.items()}

# F&O lot sizes. NSE revises these a few times a year; check the latest circular and
# override with FNO_LOT_<UNDERLYING> (e.g. FNO_LOT_NIFTY=65) when they change.
DEFAULT_LOT_SIZES = {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120}


def lot_size(underlying: str) -> int:
    underlying = underlying.upper()
    return int(os.getenv(f"FNO_LOT_{underlying}", DEFAULT_LOT_SIZES[underlying]))


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else float(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass
class AgentWeights:
    """Placeholder weights. Tune them from the backtest results, not by feel."""

    group: dict[str, float] = field(
        default_factory=lambda: {"technical": 0.5, "context": 0.5}
    )
    agent: dict[str, float] = field(
        default_factory=lambda: {
            # technical group
            "trend": 0.35,
            "momentum": 0.30,
            "volatility": 0.15,
            "volume": 0.20,
            # context group
            "options_positioning": 0.50,
            "news": 0.30,
            "flows": 0.20,
        }
    )


@dataclass
class AggregatorSettings:
    direction_threshold: float = 0.15  # |score| needed to call a direction
    # Veto when the technical and context groups point opposite ways and both are at
    # least this strong (score) with at least `conflict_min_confidence`.
    conflict_threshold: float = 0.25
    conflict_min_confidence: float = 0.3
    full_size_confidence: float = 0.6  # confidence at which the multiplier reaches 1


@dataclass
class RiskLimits:
    capital: float = field(default_factory=lambda: _env_float("FNO_CAPITAL", 500_000))
    max_risk_pct: float = field(default_factory=lambda: _env_float("FNO_MAX_RISK_PCT", 1.0))
    # Hard cap while testing. Raise only after paper trading proves the signals.
    max_lots: int = field(default_factory=lambda: _env_int("FNO_MAX_LOTS", 1))
    daily_loss_pct: float = field(default_factory=lambda: _env_float("FNO_DAILY_LOSS_PCT", 2.0))
    max_open_per_underlying: int = 1  # no stacking a second idea on an open position
    max_margin_utilisation: float = 0.5  # share of capital one idea may block as margin
    min_confidence: float = 0.35
    min_agreement: float = 0.5
    min_oi: float = 5_000  # open interest (units) on every leg
    max_spread_pct: float = 5.0  # bid/ask spread as % of mid on every leg
    min_days_to_expiry: int = 1  # avoid opening on expiry day
    require_defined_risk: bool = True
    block_extreme_vol: bool = True

    @property
    def daily_loss_limit(self) -> float:
        return self.capital * self.daily_loss_pct / 100


@dataclass
class SwingSettings:
    """Long-only cash swing trading on an NSE index universe. Values are starting points."""

    # Any niftyindices.com constituent list name: niftylargemidcap250, nifty200, nifty500...
    universe: str = field(
        default_factory=lambda: os.getenv("FNO_SWING_UNIVERSE", "niftylargemidcap250")
    )
    capital: float = field(default_factory=lambda: _env_float("FNO_SWING_CAPITAL", 500_000))
    risk_per_trade_pct: float = 1.0  # rupees lost if the stop is hit, as % of capital
    max_position_pct: float = 20.0  # cap on one stock's value, as % of capital
    max_open: int = 5
    max_per_industry: int = 2
    min_avg_value_cr: float = 10.0  # 20-day average traded value, ₹ crore
    max_participation_pct: float = 1.0  # position value vs average daily traded value
    min_score: float = 0.35
    atr_stop_mult: float = 2.0
    min_stop_pct: float = 1.0
    max_stop_pct: float = 10.0
    reward_risk: float = 2.0
    time_stop_sessions: int = 15
    top_n: int = 10  # per side
    allow_short: bool = True  # bearish ideas as bear put spreads on stock options
    option_min_days: int = 7  # skip stock-option expiries closer than this
    exit_score: float = 0.2  # alert when a held position's score turns this far against it
    bad_news_score: float = 0.5  # news this far against a holding alerts on its own
    news_shortlist: int = 10  # stocks per side (plus holdings) that get a news read
    weights: dict[str, float] = field(default_factory=lambda: {
        "trend": 0.30, "momentum": 0.20, "volatility": 0.10, "volume": 0.15,
        "relative_strength": 0.25, "news": 0.15,
    })


@dataclass
class Settings:
    # "nse" = free public data (NSE website + Yahoo Finance), "kite", or "sample".
    data_source: str = field(default_factory=lambda: os.getenv("FNO_DATA_SOURCE", "nse"))
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
    aggregator: AggregatorSettings = field(default_factory=AggregatorSettings)
    risk: RiskLimits = field(default_factory=RiskLimits)
    swing: SwingSettings = field(default_factory=SwingSettings)
    strikes_each_side: int = 15  # chain width fetched around ATM

    @property
    def has_kite(self) -> bool:
        return bool(self.kite_api_key and self.kite_access_token)

    @property
    def has_anthropic(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
