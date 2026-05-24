"""
Data models for the signal quality filter pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FilterDecision(str, Enum):
    PASS = "PASS"
    FILTERED = "FILTERED"
    HOLD_OVERRIDE = "HOLD_OVERRIDE"


class FilterStage(str, Enum):
    CONSENSUS = "consensus"
    CONFIDENCE = "confidence"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    STOP_LOSS = "stop_loss"
    HISTORICAL_PERFORMANCE = "historical_performance"
    MARKET_REGIME = "market_regime"
    DISSENT_ANALYSIS = "dissent_analysis"


@dataclass
class RawSignal:
    """Input signal from the swarm consensus engine."""
    ticker: str
    signal: str                          # BUY / SELL / HOLD / SKIP
    recommendation: str                  # OVERWEIGHT / UNDERWEIGHT / HOLD / SKIP
    confidence: float                    # 0.0 - 1.0 overall swarm confidence
    weighted_score: float                # Raw weighted vote score
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None  # type: ignore[assignment]
    take_profit: Optional[float] = None
    # Per-agent votes: list of {"agent": ..., "signal": ..., "confidence": ...}
    agent_votes: List[Dict] = field(default_factory=list)
    dissenting_agents: List[Dict] = field(default_factory=list)
    trade_date: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class MarketData:
    """Market context data for filter evaluation."""
    ticker: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    avg_volume_20d: float = 0.0
    market_cap: float = 0.0
    # Market regime context
    nifty50_change_pct: float = 0.0      # Benchmark index % change
    vix: float = 0.0                     # Implied volatility index
    trade_date: str = ""


@dataclass
class StageResult:
    """Result from a single filter stage."""
    stage: FilterStage
    decision: FilterDecision
    reason: str
    details: Dict = field(default_factory=dict)
    metric: Optional[float] = None       # The computed metric that drove this decision


@dataclass
class FilterResult:
    """Final result from the full filter pipeline."""
    ticker: str
    original_signal: str
    final_signal: str
    final_recommendation: str
    decision: FilterDecision
    # Individual stage outcomes
    stage_results: List[StageResult] = field(default_factory=list)
    # Filter summary
    stages_passed: int = 0
    stages_failed: int = 0
    first_failure: Optional[FilterStage] = None
    failure_reason: str = ""
    # Adjusted sizing (0.0 - 1.0 multiplier; 1.0 = full position)
    position_size_multiplier: float = 1.0
    # Logging
    trade_date: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.decision == FilterDecision.PASS

    def to_report(self) -> str:
        lines = [
            f"## Signal Filter Report — {self.ticker}",
            f"",
            f"Original Signal : {self.original_signal}",
            f"Final Signal    : {self.final_signal}",
            f"Decision        : {self.decision.value}",
            f"Position Size   : {self.position_size_multiplier:.0%}",
        ]
        if self.failure_reason:
            lines.append(f"Failure Reason  : {self.failure_reason}")
        lines += ["", "### Stage Results", ""]
        for sr in self.stage_results:
            icon = "✓" if sr.decision == FilterDecision.PASS else "✗"
            metric_str = f" ({sr.metric:.3f})" if sr.metric is not None else ""
            lines.append(f"  {icon} [{sr.stage.value}]{metric_str} — {sr.reason}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "original_signal": self.original_signal,
            "final_signal": self.final_signal,
            "final_recommendation": self.final_recommendation,
            "decision": self.decision.value,
            "stages_passed": self.stages_passed,
            "stages_failed": self.stages_failed,
            "first_failure": self.first_failure.value if self.first_failure else None,
            "failure_reason": self.failure_reason,
            "position_size_multiplier": round(self.position_size_multiplier, 4),
            "trade_date": self.trade_date,
            "stage_results": [
                {
                    "stage": sr.stage.value,
                    "decision": sr.decision.value,
                    "reason": sr.reason,
                    "metric": sr.metric,
                }
                for sr in self.stage_results
            ],
        }
