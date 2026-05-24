"""
MiroFish-inspired swarm consensus module for TradingAgents.

Provides a 5-agent swarm with:
- Parallel independent analysis (Round 1)
- Challenge round where agents see each other's analysis (Round 2)
- Final vote with weighted consensus (Round 3)
- Confidence scoring and dissent logging
- Timeout and fallback handling
"""

from .agents import (
    BullAgent,
    BearAgent,
    TechnicalAgent,
    FundamentalAgent,
    SentimentAgent,
)
from .consensus import ConsensusEngine, ConsensusResult, AgentVote
from .swarm_node import create_swarm_node, SwarmConfig
from .prompts import AGENT_PROMPTS

__all__ = [
    "BullAgent",
    "BearAgent",
    "TechnicalAgent",
    "FundamentalAgent",
    "SentimentAgent",
    "ConsensusEngine",
    "ConsensusResult",
    "AgentVote",
    "create_swarm_node",
    "SwarmConfig",
    "AGENT_PROMPTS",
]
