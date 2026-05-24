"""
Consensus engine for the 5-agent trading swarm.

Implements weighted majority vote with:
- Agent reliability weights (calibrated from design doc)
- Risk Manager veto power
- SKIP signal on low-confidence ambiguity
- Dissent tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Agent weights from architecture design (system_design.md)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "kronos_quant": 0.30,  # Highest: 93% better RankIC vs baseline
    "technical": 0.20,
    "fundamental": 0.20,
    "sentiment": 0.15,
    "risk_manager": 0.15,  # Has veto power despite lower weight
}

# Without Kronos, redistribute weight proportionally among 4 Claude agents
FALLBACK_WEIGHTS: Dict[str, float] = {
    "bull": 0.20,        # Maps to "technical" optimistic view
    "bear": 0.20,        # Maps to "risk_manager" cautious view
    "technical": 0.25,
    "fundamental": 0.20,
    "sentiment": 0.15,
}

SIGNAL_VALUES = {"BUY": 1, "HOLD": 0, "SELL": -1, "SKIP": 0}

RECOMMENDATION_MAP = {
    "BUY": "OVERWEIGHT",
    "SELL": "UNDERWEIGHT",
    "HOLD": "HOLD",
    "SKIP": "SKIP",
}


@dataclass
class AgentVote:
    agent: str
    signal: str           # BUY / HOLD / SELL / SKIP
    confidence: float     # 0.0 - 1.0
    reasoning: str
    key_points: List[str] = field(default_factory=list)
    round_revised: bool = False
    revision_reason: str = ""


@dataclass
class ConsensusResult:
    recommendation: str              # OVERWEIGHT / HOLD / UNDERWEIGHT / SKIP
    signal: str                      # BUY / HOLD / SELL / SKIP
    confidence: float                # 0.0 - 1.0
    weighted_score: float            # Raw weighted vote score
    votes: List[AgentVote] = field(default_factory=list)
    dissenting: List[AgentVote] = field(default_factory=list)
    veto_triggered: bool = False
    veto_reason: str = ""
    agents_responded: int = 0
    agents_required: int = 3         # Minimum required (abort if < 3)
    skip_reason: str = ""

    def to_report(self) -> str:
        """Format consensus result as human-readable markdown report."""
        lines = [
            f"## Swarm Consensus Report",
            f"",
            f"**Recommendation**: {self.recommendation}",
            f"**Signal**: {self.signal}",
            f"**Confidence**: {self.confidence:.1%}",
            f"**Weighted Score**: {self.weighted_score:+.3f}",
            f"**Agents**: {self.agents_responded}/5 responded",
            "",
        ]

        if self.veto_triggered:
            lines += [
                f"**VETO TRIGGERED**: {self.veto_reason}",
                "",
            ]

        if self.skip_reason:
            lines += [
                f"**SKIP Reason**: {self.skip_reason}",
                "",
            ]

        lines += ["### Agent Votes", ""]
        for vote in self.votes:
            revised_tag = " *(revised)*" if vote.round_revised else ""
            lines.append(
                f"- **{vote.agent.upper()}**: {vote.signal} "
                f"(conf={vote.confidence:.0%}){revised_tag} — {vote.reasoning[:120]}"
            )

        if self.dissenting:
            lines += ["", "### Dissenting Views", ""]
            for vote in self.dissenting:
                lines.append(
                    f"- **{vote.agent.upper()}** dissents: {vote.signal} — "
                    + "; ".join(vote.key_points[:2])
                )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "recommendation": self.recommendation,
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "weighted_score": round(self.weighted_score, 4),
            "agents_responded": self.agents_responded,
            "veto_triggered": self.veto_triggered,
            "veto_reason": self.veto_reason,
            "skip_reason": self.skip_reason,
            "votes": [
                {
                    "agent": v.agent,
                    "signal": v.signal,
                    "confidence": round(v.confidence, 4),
                    "reasoning": v.reasoning,
                    "key_points": v.key_points,
                    "round_revised": v.round_revised,
                }
                for v in self.votes
            ],
            "dissenting": [
                {
                    "agent": v.agent,
                    "signal": v.signal,
                    "confidence": round(v.confidence, 4),
                    "key_points": v.key_points,
                }
                for v in self.dissenting
            ],
        }


class ConsensusEngine:
    """
    Compute weighted majority vote from swarm agent votes.
    
    Uses design from system_design.md:
    - Weighted score: sum(weight * signal_value * confidence)
    - Score > 0.3 -> BUY, Score < -0.3 -> SELL, else HOLD
    - Overall confidence = |score| / sum(weights)
    - Risk Manager (bear agent) can veto on extreme risk flags
    - SKIP if confidence < threshold and score in dead zone
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
        skip_confidence_threshold: float = 0.5,
        dead_zone: float = 0.1,
        min_agents: int = 3,
    ):
        self.weights = weights or FALLBACK_WEIGHTS
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.skip_confidence_threshold = skip_confidence_threshold
        self.dead_zone = dead_zone
        self.min_agents = min_agents

    def compute(
        self,
        votes: List[AgentVote],
        risk_veto_flag: bool = False,
        veto_reason: str = "",
    ) -> ConsensusResult:
        """Compute consensus from agent votes.
        
        Args:
            votes: List of AgentVote from each agent
            risk_veto_flag: If True (set externally by risk checks), override to SKIP
            veto_reason: Human-readable reason for veto
        """
        n_responded = len(votes)

        # Abort if too few agents
        if n_responded < self.min_agents:
            logger.warning(f"Only {n_responded}/{self.min_agents} agents responded — aborting")
            return ConsensusResult(
                recommendation="SKIP",
                signal="SKIP",
                confidence=0.0,
                weighted_score=0.0,
                votes=votes,
                agents_responded=n_responded,
                agents_required=self.min_agents,
                skip_reason=f"Insufficient agents: {n_responded} < {self.min_agents} required",
            )

        # Compute weighted score
        total_weight = 0.0
        weighted_score = 0.0
        for vote in votes:
            w = self.weights.get(vote.agent, 1.0 / len(votes))
            sv = SIGNAL_VALUES.get(vote.signal, 0)
            weighted_score += w * sv * vote.confidence
            total_weight += w

        normalized_score = weighted_score / total_weight if total_weight > 0 else 0.0
        confidence = abs(normalized_score)

        # Map to signal
        if normalized_score > self.buy_threshold:
            signal = "BUY"
        elif normalized_score < self.sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        # SKIP on ambiguity: dead zone AND low confidence
        skip_reason = ""
        if abs(normalized_score) <= self.dead_zone and confidence < self.skip_confidence_threshold:
            signal = "SKIP"
            skip_reason = (
                f"Score {normalized_score:+.3f} in dead zone ±{self.dead_zone} "
                f"and confidence {confidence:.1%} < {self.skip_confidence_threshold:.0%} threshold"
            )

        # Risk Manager veto overrides everything
        if risk_veto_flag and veto_reason:
            signal = "SKIP"
            recommendation = "SKIP"
            skip_reason = veto_reason
        else:
            recommendation = RECOMMENDATION_MAP.get(signal, "HOLD")

        # Dissenting agents = those who didn't vote for the consensus signal
        dissenting = [v for v in votes if v.signal != signal and v.signal != "SKIP"]

        result = ConsensusResult(
            recommendation=recommendation,
            signal=signal,
            confidence=round(confidence, 4),
            weighted_score=round(normalized_score, 4),
            votes=votes,
            dissenting=dissenting,
            veto_triggered=risk_veto_flag and bool(veto_reason),
            veto_reason=veto_reason,
            agents_responded=n_responded,
            agents_required=self.min_agents,
            skip_reason=skip_reason,
        )

        logger.info(
            f"Consensus: {signal} (conf={confidence:.1%}, score={normalized_score:+.3f}, "
            f"{n_responded} agents, {len(dissenting)} dissenting)"
        )
        return result

    def count_signals(self, votes: List[AgentVote]) -> Dict[str, int]:
        """Simple plurality count (for logging/debugging)."""
        counts: Dict[str, int] = {"BUY": 0, "HOLD": 0, "SELL": 0, "SKIP": 0}
        for v in votes:
            counts[v.signal] = counts.get(v.signal, 0) + 1
        return counts
