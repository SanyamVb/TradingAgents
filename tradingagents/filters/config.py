"""
Configuration for signal quality filters.

All thresholds are configurable via YAML or dict.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ConsensusFilterConfig:
    """Consensus Filter: require 60%+ agent agreement for OW/UW."""
    enabled: bool = True
    min_agreement_pct: float = 0.60     # 60% of agents must agree on direction
    apply_to_signals: list = field(default_factory=lambda: ["BUY", "SELL"])


@dataclass
class ConfidenceFilterConfig:
    """Confidence Threshold: require avg confidence > 70 for actionable signals."""
    enabled: bool = True
    min_avg_confidence: float = 0.70    # 0-1 scale
    # Per-agent minimum (agents below this drag down the average meaningfully)
    min_agent_confidence: float = 0.40


@dataclass
class RiskFilterConfig:
    """Risk Filters: volatility, liquidity, stop-loss distance."""
    enabled: bool = True
    # Volatility: skip if intraday range > 50% of open
    max_intraday_range_pct: float = 0.50
    # Liquidity: skip if volume < threshold * avg_volume
    min_volume_ratio: float = 0.30      # At least 30% of 20d average volume
    min_absolute_volume: float = 100_000  # Hard floor regardless of average
    # Stop-loss: reject if SL distance > 10% from entry
    max_stop_loss_distance_pct: float = 0.10


@dataclass
class HistoricalPerformanceConfig:
    """Historical Performance Filter: track win rate and adjust position size."""
    enabled: bool = True
    # Tickers below this win rate get reduced sizing
    poor_performer_threshold: float = 0.40  # < 40% win rate
    # Tickers above this get boosted sizing
    strong_performer_threshold: float = 0.60  # > 60% win rate
    # Position size adjustments
    poor_performer_size_mult: float = 0.50   # Half size
    strong_performer_size_mult: float = 1.25  # 25% boost (capped internally)
    max_position_size_mult: float = 1.50
    # Minimum trades before historical filter kicks in
    min_trades_required: int = 5
    # Seed data: known strong/weak performers (can be overridden in config)
    known_strong: list = field(default_factory=lambda: ["HINDUNILVR", "TCS", "HDFCBANK"])
    known_weak: list = field(default_factory=list)


@dataclass
class MarketRegimeConfig:
    """Market Regime Filter: detect bull/bear/sideways and adjust strategy."""
    enabled: bool = True
    # Nifty50 rolling return thresholds for regime classification
    bull_threshold_pct: float = 0.015    # +1.5% day = strong bull
    bear_threshold_pct: float = -0.015   # -1.5% day = strong bear
    # In strong bull regime, skip UW signals
    skip_uw_in_bull: bool = True
    # In strong bear regime, skip OW signals
    skip_ow_in_bear: bool = True
    # VIX above this = high volatility regime (more conservative)
    high_vix_threshold: float = 25.0


@dataclass
class DissentAnalysisConfig:
    """Dissent Analysis: log and optionally veto on high-confidence dissent."""
    enabled: bool = True
    # If a dissenting agent has confidence >= this, log prominently
    strong_dissent_confidence: float = 0.80
    # If dissenter confidence >= this, trigger veto
    veto_confidence_threshold: float = 0.90
    # Only veto for these agent roles
    veto_eligible_agents: list = field(default_factory=lambda: ["risk_manager", "bear"])


@dataclass
class FilterConfig:
    """Top-level configuration for the full filter pipeline."""
    consensus: ConsensusFilterConfig = field(default_factory=ConsensusFilterConfig)
    confidence: ConfidenceFilterConfig = field(default_factory=ConfidenceFilterConfig)
    risk: RiskFilterConfig = field(default_factory=RiskFilterConfig)
    historical: HistoricalPerformanceConfig = field(default_factory=HistoricalPerformanceConfig)
    regime: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    dissent: DissentAnalysisConfig = field(default_factory=DissentAnalysisConfig)

    # Global settings
    log_filtered_signals: bool = True
    track_filter_effectiveness: bool = True
    # Path to persist historical performance data
    performance_db_path: str = "~/.tradingagents/filter_performance.json"

    @classmethod
    def from_yaml(cls, path: str) -> "FilterConfig":
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml required: pip install pyyaml")
        with open(os.path.expanduser(path)) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "FilterConfig":
        def _update(obj, sub):
            if not sub:
                return obj
            for k, v in sub.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            return obj

        cfg = cls()
        _update(cfg.consensus, d.get("consensus", {}))
        _update(cfg.confidence, d.get("confidence", {}))
        _update(cfg.risk, d.get("risk", {}))
        _update(cfg.historical, d.get("historical", {}))
        _update(cfg.regime, d.get("regime", {}))
        _update(cfg.dissent, d.get("dissent", {}))
        for k in ("log_filtered_signals", "track_filter_effectiveness", "performance_db_path"):
            if k in d:
                setattr(cfg, k, d[k])
        return cfg

    def to_dict(self) -> dict:
        import dataclasses
        def _dc(obj):
            return {k: v for k, v in dataclasses.asdict(obj).items()}
        return {
            "consensus": _dc(self.consensus),
            "confidence": _dc(self.confidence),
            "risk": _dc(self.risk),
            "historical": _dc(self.historical),
            "regime": _dc(self.regime),
            "dissent": _dc(self.dissent),
            "log_filtered_signals": self.log_filtered_signals,
            "track_filter_effectiveness": self.track_filter_effectiveness,
            "performance_db_path": self.performance_db_path,
        }
