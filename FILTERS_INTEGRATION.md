# Signal Quality Filters — Integration Guide

## Overview

The `tradingagents.filters` module adds a multi-stage filter pipeline that runs
**before signal emission** to prevent bad trades. Every signal from the swarm
consensus engine passes through 6 filter stages, each independently configurable.

## Architecture

```
SwarmConsensusEngine
      │
      ▼
 RawSignal (BUY/SELL/HOLD/SKIP)
      │
      ▼
SignalFilterPipeline
  ├─ Stage 1: Dissent Analysis     (veto by risk_manager/bear agent)
  ├─ Stage 2: Consensus Filter     (60%+ agent agreement)
  ├─ Stage 3: Confidence Threshold (avg confidence > 70%)
  ├─ Stage 4: Volatility Check     (intraday range < 50%)
  ├─ Stage 5: Liquidity Check      (volume > floor)
  ├─ Stage 6: Stop-Loss Distance   (SL < 10% from entry)
  ├─ Stage 7: Market Regime        (skip conflicting signals)
  └─ Stage 8: Historical Perf.     (position size adjustment only)
      │
      ▼
FilterResult (PASS / FILTERED / HOLD_OVERRIDE + position_size_multiplier)
```

## Quick Start

```python
from tradingagents.filters import (
    FilterConfig, SignalFilterPipeline, FilterPerformanceTracker,
    RawSignal, MarketData,
)

# 1. Load config (from YAML or defaults)
config = FilterConfig.from_yaml("filters_config.yaml")
# or: config = FilterConfig()  # sensible defaults

# 2. Create pipeline
pipeline = SignalFilterPipeline(config)

# 3. Load historical win rates from the tracker
tracker = FilterPerformanceTracker(config.performance_db_path)
win_rates = tracker.get_win_rates()
trade_counts = tracker.get_trade_counts()

# 4. Build inputs from swarm consensus result
signal = RawSignal(
    ticker="RELIANCE",
    signal="BUY",
    recommendation="OVERWEIGHT",
    confidence=0.82,
    weighted_score=0.55,
    entry_price=2500.0,
    stop_loss=2350.0,
    agent_votes=[...],         # list of per-agent vote dicts
    dissenting_agents=[...],   # from ConsensusResult.dissenting
    trade_date="2025-03-01",
)

market = MarketData(
    ticker="RELIANCE",
    open_price=2490.0,
    high_price=2520.0,
    low_price=2480.0,
    close_price=2510.0,
    volume=800_000,
    avg_volume_20d=600_000,
    nifty50_change_pct=0.007,
    vix=16.5,
)

# 5. Run the filter
result = pipeline.run(signal, market, win_rates, trade_counts)

# 6. Use the result
if result.passed:
    position_size = base_size * result.position_size_multiplier
    print(f"TRADE: {result.ticker} {result.final_signal} size={position_size:.0%}")
else:
    print(f"FILTERED: {result.ticker} — {result.failure_reason}")

# 7. Record for tracking
tracker.record(result)

# 8. After trade resolves, record outcome
tracker.record_outcome("RELIANCE", "2025-03-01", outcome="win", pnl_pct=2.3)
```

## Connecting to the Swarm

```python
from tradingagents.swarm.consensus import ConsensusResult, AgentVote
from tradingagents.filters import RawSignal

def consensus_to_raw_signal(
    ticker: str,
    consensus: ConsensusResult,
    entry_price: float = None,
    stop_loss: float = None,
    trade_date: str = "",
) -> RawSignal:
    """Convert a ConsensusResult into a RawSignal for filtering."""
    return RawSignal(
        ticker=ticker,
        signal=consensus.signal,
        recommendation=consensus.recommendation,
        confidence=consensus.confidence,
        weighted_score=consensus.weighted_score,
        entry_price=entry_price,
        stop_loss=stop_loss,
        agent_votes=[
            {
                "agent": v.agent,
                "signal": v.signal,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
            }
            for v in consensus.votes
        ],
        dissenting_agents=[
            {
                "agent": v.agent,
                "signal": v.signal,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
                "key_points": v.key_points,
            }
            for v in consensus.dissenting
        ],
        trade_date=trade_date,
    )
```

## Configuration

Edit `filters_config.yaml` to tune thresholds without code changes:

```yaml
consensus:
  min_agreement_pct: 0.60    # raise to 0.80 for more conservative trading

confidence:
  min_avg_confidence: 0.70   # lower to 0.60 in low-volatility environments

risk:
  max_stop_loss_distance_pct: 0.10  # tighten to 0.07 for smaller risk

historical:
  known_strong: [HINDUNILVR, TCS, HDFCBANK]  # add/remove tickers
```

## Filter Performance Report

```python
tracker = FilterPerformanceTracker()
print(tracker.generate_report())
```

Sample output:
```
# Filter Performance Report

Total runs      : 150
Passed          : 89 (59%)
Filtered out    : 61 (41%)

## Filter Trigger Rates

  dissent_analysis               triggered    3/150  = 2%
  consensus                      triggered   18/150  = 12%
  confidence                     triggered   15/150  = 10%
  volatility                     triggered    8/150  = 5%
  liquidity                      triggered    5/150  = 3%
  stop_loss                      triggered   12/150  = 8%
  market_regime                  triggered    0/150  = 0%

## Ticker Performance

  HINDUNILVR       trades= 12  win_rate=75%  avg_pnl=+1.85%
  RELIANCE         trades= 20  win_rate=60%  avg_pnl=+0.92%
```

## Files

| File | Description |
|---|---|
| `tradingagents/filters/__init__.py` | Public API |
| `tradingagents/filters/models.py` | RawSignal, MarketData, FilterResult data classes |
| `tradingagents/filters/config.py` | FilterConfig and per-stage configs |
| `tradingagents/filters/filters.py` | Individual filter implementations |
| `tradingagents/filters/pipeline.py` | SignalFilterPipeline orchestrator |
| `tradingagents/filters/performance.py` | FilterPerformanceTracker persistence |
| `filters_config.yaml` | Default threshold configuration |
| `tests/test_signal_filters.py` | 45 unit tests |
