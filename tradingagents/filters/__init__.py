"""
Signal quality filters and confidence scoring for TradingAgents.

Filter pipeline that runs BEFORE signal emission to prevent bad trades.

Usage:
    from tradingagents.filters import SignalFilterPipeline, FilterConfig
    from tradingagents.filters.performance import FilterPerformanceTracker

    config = FilterConfig.from_yaml("filters_config.yaml")
    pipeline = SignalFilterPipeline(config)
    tracker = FilterPerformanceTracker()

    result = pipeline.run(signal, market_data, consensus_result)
    tracker.record(result)
"""

from .config import FilterConfig
from .models import (
    RawSignal,
    MarketData,
    FilterResult,
    FilterDecision,
    FilterStage,
)
from .pipeline import SignalFilterPipeline
from .performance import FilterPerformanceTracker

__all__ = [
    "FilterConfig",
    "RawSignal",
    "MarketData",
    "FilterResult",
    "FilterDecision",
    "FilterStage",
    "SignalFilterPipeline",
    "FilterPerformanceTracker",
]
