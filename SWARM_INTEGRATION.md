# MiroFish Swarm Integration — TradingAgents

## Overview

This integration adds a MiroFish-inspired **5-agent swarm debate** to the TradingAgents
pipeline. Rather than using MiroFish directly (a social simulation engine with 21-42 min
latency and wrong-domain plumbing), we implement its core insight — multi-agent consensus
through structured debate rounds — purpose-built for stock analysis.

## Architecture

```
Market Data (yfinance, NSE feeds)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              Existing Analyst Layer                      │
│  Market Analyst → Social Analyst → News → Fundamentals  │
└───────────────────────────┬─────────────────────────────┘
                            │  (reports injected into state)
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Swarm Analyst Node                      │
│                                                         │
│  Round 1 (parallel):                                    │
│    Bull Agent │ Bear Agent │ Technical Agent             │
│    Fundamental Agent │ Sentiment Agent                   │
│                    ↓                                    │
│  Round 2 (sequential): Each agent sees all R1 analyses  │
│    Can revise signal + confidence (must justify)        │
│                    ↓                                    │
│  Round 3 (parallel): Final locked votes                 │
│                    ↓                                    │
│  Consensus Engine: Weighted vote → OVERWEIGHT/HOLD/     │
│                    UNDERWEIGHT/SKIP + confidence score  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
              Bull/Bear Researcher Debate
                            │
                            ▼
              Research Manager → Trader → Risk → Portfolio Manager
```

## Agents

| # | Agent | Perspective | LLM |
|---|-------|------------|-----|
| 1 | Bull Agent | Bullish/growth catalysts | quick_llm |
| 2 | Bear Agent | Bearish/risk-focused | deep_llm |
| 3 | Technical Agent | Chart patterns, momentum | quick_llm |
| 4 | Fundamental Agent | Valuation, earnings | quick_llm |
| 5 | Sentiment Agent | News, social, flows | quick_llm |

Optional: Kronos Quant Agent can be injected as a 6th synthetic vote if the Kronos
analyst node ran upstream (set `kronos_weight > 0` in `create_swarm_node`).

## Consensus Rules

Default agent weights (from `system_design.md` calibration):
```
bull:        0.20
bear:        0.20  (also has veto power)
technical:   0.25
fundamental: 0.20
sentiment:   0.15
```

Signal computation:
```python
score = sum(weight[agent] * signal_value * confidence for each vote)
# BUY=+1, HOLD=0, SELL=-1

if score > 0.3:   -> OVERWEIGHT
if score < -0.3:  -> UNDERWEIGHT
else:             -> HOLD

# SKIP if score in dead zone (±0.1) AND confidence < 0.5
# SKIP if bear agent triggers veto (portfolio drawdown, correlation, liquidity)
# ABORT if < 3 agents respond
```

## Usage

### 1. Add Swarm as an Analyst Option

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Standard workflow + swarm debate after all analysts
graph = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    config=config,
)
# Swarm runs automatically after all analysts when setup_graph sees swarm_node

# Or: add "swarm" explicitly to the analyst list (runs in sequence)
graph = TradingAgentsGraph(
    selected_analysts=["market", "fundamentals", "swarm"],
    config=config,
)
```

### 2. Direct Swarm Node Usage

```python
from tradingagents.swarm import create_swarm_node, SwarmConfig

config = SwarmConfig(
    round1_parallel=True,
    round3_parallel=True,
    min_agents_required=3,
    buy_threshold=0.3,
    sell_threshold=-0.3,
)
node = create_swarm_node(
    quick_llm=my_claude_haiku,
    deep_llm=my_claude_sonnet,
    config=config,
    kronos_weight=0.30,  # If Kronos ran upstream
)

# Use in LangGraph
workflow.add_node("Swarm Analyst", node)
```

### 3. Consensus Engine Standalone

```python
from tradingagents.swarm.consensus import ConsensusEngine, AgentVote

engine = ConsensusEngine()
votes = [
    AgentVote("bull", "BUY", 0.85, "Strong earnings growth"),
    AgentVote("bear", "HOLD", 0.6, "High valuation risk"),
    AgentVote("technical", "BUY", 0.75, "Breakout on volume"),
    AgentVote("fundamental", "BUY", 0.70, "P/E below sector"),
    AgentVote("sentiment", "HOLD", 0.55, "Mixed news flow"),
]

result = engine.compute(votes)
print(result.recommendation)   # "OVERWEIGHT"
print(result.confidence)       # ~0.45
print(result.to_report())      # Full markdown report
```

## State Fields

After the swarm node runs, `AgentState` is populated with:

- `swarm_report` (str): Full markdown debate report with all agent votes and dissent
- `swarm_consensus` (dict): Structured result including:
  - `recommendation`: OVERWEIGHT / HOLD / UNDERWEIGHT / SKIP
  - `signal`: BUY / HOLD / SELL / SKIP
  - `confidence`: 0.0-1.0 (magnitude of weighted score)
  - `weighted_score`: Raw weighted vote sum
  - `agents_responded`: How many agents returned (out of 5)
  - `veto_triggered`: Boolean
  - `veto_reason`: String
  - `skip_reason`: String (if SKIP)
  - `votes`: List of all agent votes with agent, signal, confidence, reasoning
  - `dissenting`: Votes that differ from consensus

The `investment_plan` field is also updated with the swarm consensus summary, feeding
directly into the Bull/Bear Researcher debate and Research Manager.

## Error Handling

| Scenario | Behavior |
|----------|----------|
| 1-2 agents fail | Continue with remaining (fallback to majority) |
| < 3 agents respond | Return SKIP with reason |
| All agents fail | Return SKIP, never raise |
| Agent timeout (>5min) | ThreadPoolExecutor timeout → agent treated as failed |
| LLM returns bad JSON | _extract_json fallback patterns + default to HOLD |
| Kronos unavailable | Skipped (kronos_weight treated as 0) |

## File Structure

```
tradingagents/swarm/
├── __init__.py          # Package exports
├── agents.py            # 5 specialist agent classes (BaseSwarmAgent + subclasses)
├── consensus.py         # ConsensusEngine, AgentVote, ConsensusResult
├── prompts.py           # AGENT_PROMPTS: system/round1/round2/round3 per agent
└── swarm_node.py        # create_swarm_node() — LangGraph-compatible node

tests/
└── test_swarm_consensus.py   # 26 tests covering consensus, agents, integration
```

## Performance Notes

- Round 1 parallel: ~30-60s with 5 Claude agents (network bound)
- Round 2 sequential: ~30s (5 calls, needs all R1 results)
- Round 3 parallel: ~15-30s
- Total swarm time: ~75-150s for 5 agents (3 rounds)
- Add upstream analyst reports to context to reduce Round 1 LLM work

For production pre-market use (5:30 AM IST, 60-min window):
- Use Claude Haiku for Round 1/3 (quick agents)
- Use Claude Sonnet only for Bear agent (deep_llm)
- Batch tickers: process ~20-30 tickers in the window
