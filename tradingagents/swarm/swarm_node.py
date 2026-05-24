"""
LangGraph-compatible swarm node for TradingAgents.

Replaces single-analyst analysis with a 5-agent swarm debate.
The node:
  1. Gathers market data (reusing existing TradingAgents data tools)
  2. Runs 3-round debate across 5 specialist agents
  3. Computes weighted consensus
  4. Returns swarm_report + swarm_consensus to AgentState
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage

from .agents import BullAgent, BearAgent, TechnicalAgent, FundamentalAgent, SentimentAgent
from .consensus import ConsensusEngine, AgentVote, ConsensusResult
from .prompts import AGENT_PROMPTS

logger = logging.getLogger(__name__)

AGENT_TIMEOUT_SECONDS = 300  # 5 minutes per agent max


@dataclass
class SwarmConfig:
    """Configuration for the swarm node."""
    max_workers: int = 5           # Parallel workers for Round 1 and 3
    round1_parallel: bool = True   # Run Round 1 in parallel
    round3_parallel: bool = True   # Run Round 3 in parallel
    agent_timeout: float = AGENT_TIMEOUT_SECONDS
    min_agents_required: int = 3   # Abort if < 3 agents respond
    buy_threshold: float = 0.3
    sell_threshold: float = -0.3
    skip_confidence: float = 0.5
    use_quick_llm_for_analysts: bool = True  # Use quick LLM for bull/bear/tech/fund/sent


def _build_market_data_summary(state: dict) -> str:
    """Aggregate available data from AgentState for swarm agents."""
    parts = []
    for key, label in [
        ("market_report", "TECHNICAL/MARKET DATA"),
        ("sentiment_report", "SENTIMENT/SOCIAL DATA"),
        ("news_report", "NEWS DATA"),
        ("fundamentals_report", "FUNDAMENTAL DATA"),
        ("market_report_kronos", "KRONOS QUANT SIGNAL"),
        ("past_context", "HISTORICAL CONTEXT"),
    ]:
        val = state.get(key, "")
        if val:
            parts.append(f"### {label}\n{val}\n")
    return "\n".join(parts) if parts else "No market data available."


def _run_round1(agent, ticker, trade_date, market_data):
    """Thread target for parallel Round 1."""
    try:
        return agent.agent_id, agent.analyze_round1(ticker, trade_date, market_data)
    except Exception as e:
        logger.error(f"Agent {agent.agent_id} Round 1 failed: {e}", exc_info=True)
        return agent.agent_id, None


def _run_round2(agent, ticker, other_analyses):
    """Thread target for sequential Round 2."""
    try:
        return agent.agent_id, agent.analyze_round2(ticker, other_analyses)
    except Exception as e:
        logger.error(f"Agent {agent.agent_id} Round 2 failed: {e}", exc_info=True)
        return agent.agent_id, None


def _run_round3(agent, ticker):
    """Thread target for parallel Round 3."""
    try:
        return agent.agent_id, agent.final_vote(ticker)
    except Exception as e:
        logger.error(f"Agent {agent.agent_id} Round 3 failed: {e}", exc_info=True)
        return agent.agent_id, None


def create_swarm_node(
    quick_llm: Any,
    deep_llm: Optional[Any] = None,
    config: Optional[SwarmConfig] = None,
    kronos_weight: float = 0.0,  # Set >0 only if Kronos analyst already ran
):
    """Create a LangGraph-compatible swarm node.

    Args:
        quick_llm: LLM for bull/bear/technical/fundamental/sentiment agents
        deep_llm: LLM for bear agent (risk-focused, deeper reasoning). Defaults to quick_llm
        config: SwarmConfig settings
        kronos_weight: If Kronos analyst ran upstream, its weight is included in consensus
    
    Returns:
        swarm_node function compatible with LangGraph StateGraph
    """
    cfg = config or SwarmConfig()
    deep = deep_llm or quick_llm
    engine = ConsensusEngine(
        buy_threshold=cfg.buy_threshold,
        sell_threshold=cfg.sell_threshold,
        skip_confidence_threshold=cfg.skip_confidence,
        min_agents=cfg.min_agents_required,
    )

    def swarm_node(state: dict) -> dict:
        ticker = state.get("company_of_interest", "UNKNOWN")
        trade_date = state.get("trade_date", "unknown")
        start_time = time.time()

        logger.info(f"[Swarm] Starting 3-round debate for {ticker} on {trade_date}")

        # Build market data summary from upstream analyst reports
        market_data = _build_market_data_summary(state)

        # Instantiate agents
        from .prompts import AGENT_PROMPTS as AP
        agents = {
            "bull": BullAgent(quick_llm, AP["bull"]),
            "bear": BearAgent(deep, AP["bear"]),          # Deep LLM for bear (risk-focused)
            "technical": TechnicalAgent(quick_llm, AP["technical"]),
            "fundamental": FundamentalAgent(quick_llm, AP["fundamental"]),
            "sentiment": SentimentAgent(quick_llm, AP["sentiment"]),
        }
        agent_list = list(agents.values())

        # ── ROUND 1: Parallel independent analysis ──────────────────────────
        logger.info(f"[Swarm] Round 1: parallel analysis ({len(agent_list)} agents)")
        round1_results: Dict[str, Any] = {}

        if cfg.round1_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
                futures = {
                    pool.submit(_run_round1, ag, ticker, trade_date, market_data): ag.agent_id
                    for ag in agent_list
                }
                for future in concurrent.futures.as_completed(
                    futures, timeout=cfg.agent_timeout
                ):
                    agent_id, result = future.result()
                    round1_results[agent_id] = result
        else:
            for ag in agent_list:
                _, result = _run_round1(ag, ticker, trade_date, market_data)
                round1_results[ag.agent_id] = result

        # ── ROUND 2: Sequential challenge round (sees all Round 1) ──────────
        logger.info("[Swarm] Round 2: challenge round (sequential)")
        other_analyses_text = "\n\n".join(
            f"**{aid.upper()} (Round 1)**:\n```json\n{json.dumps(res, indent=2)}\n```"
            for aid, res in round1_results.items()
            if res is not None
        )
        for ag in agent_list:
            if round1_results.get(ag.agent_id) is not None:
                try:
                    ag.analyze_round2(ticker, other_analyses_text)
                except Exception as e:
                    logger.error(f"Agent {ag.agent_id} Round 2 failed: {e}")

        # ── ROUND 3: Parallel final votes ─────────────────────────────────
        logger.info("[Swarm] Round 3: final locked votes")
        votes: list[AgentVote] = []
        failed_agents = []

        if cfg.round3_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
                futures = {
                    pool.submit(_run_round3, ag, ticker): ag.agent_id
                    for ag in agent_list
                    if round1_results.get(ag.agent_id) is not None
                }
                for future in concurrent.futures.as_completed(
                    futures, timeout=cfg.agent_timeout
                ):
                    agent_id, vote = future.result()
                    if vote is not None:
                        votes.append(vote)
                    else:
                        failed_agents.append(agent_id)
        else:
            for ag in agent_list:
                if round1_results.get(ag.agent_id) is not None:
                    _, vote = _run_round3(ag, ticker)
                    if vote is not None:
                        votes.append(vote)
                    else:
                        failed_agents.append(ag.agent_id)

        if failed_agents:
            logger.warning(f"[Swarm] Agents that failed final vote: {failed_agents}")

        # Optionally inject Kronos as a synthetic vote if it ran upstream
        if kronos_weight > 0:
            kronos_report = state.get("market_report_kronos", "")
            if kronos_report and "KRONOS ANALYSIS UNAVAILABLE" not in kronos_report:
                kronos_vote = _parse_kronos_vote(kronos_report, kronos_weight)
                if kronos_vote:
                    votes.append(kronos_vote)
                    logger.info(f"[Swarm] Kronos synthetic vote injected: {kronos_vote.signal}")

        # ── CONSENSUS ─────────────────────────────────────────────────────
        consensus = engine.compute(votes)
        elapsed = time.time() - start_time

        logger.info(
            f"[Swarm] Consensus for {ticker}: {consensus.recommendation} "
            f"(conf={consensus.confidence:.1%}, {elapsed:.1f}s)"
        )

        # Format swarm report
        report = consensus.to_report()
        report += f"\n\n---\n*Swarm elapsed: {elapsed:.1f}s | "
        report += f"Agents responded: {consensus.agents_responded}/5*\n"

        # Log dissent
        if consensus.dissenting:
            dissent_log = "\n".join(
                f"  - {v.agent}: {v.signal} (conf={v.confidence:.0%}) — {v.reasoning[:100]}"
                for v in consensus.dissenting
            )
            logger.info(f"[Swarm] Dissenting opinions for {ticker}:\n{dissent_log}")

        return {
            "messages": [AIMessage(content=report)],
            "swarm_report": report,
            "swarm_consensus": consensus.to_dict(),
            # Inject into investment_plan so downstream Research Manager uses it
            "investment_plan": (
                f"SWARM CONSENSUS: **{consensus.recommendation}** "
                f"(confidence: {consensus.confidence:.1%})\n\n"
                + report
            ),
        }

    return swarm_node


def _parse_kronos_vote(kronos_report: str, weight: float) -> Optional[AgentVote]:
    """Parse Kronos analyst report to synthesize a swarm vote."""
    try:
        signal = "HOLD"
        confidence = 0.5
        if "SIGNAL: BUY" in kronos_report or "Signal: BUY" in kronos_report:
            signal = "BUY"
        elif "SIGNAL: SELL" in kronos_report or "Signal: SELL" in kronos_report:
            signal = "SELL"
        elif "SIGNAL: SKIP" in kronos_report or "Signal: SKIP" in kronos_report:
            signal = "SKIP"

        import re
        conf_match = re.search(r"Confidence[:\s]+(\d+\.?\d*)%?", kronos_report, re.IGNORECASE)
        if conf_match:
            val = float(conf_match.group(1))
            confidence = val / 100.0 if val > 1 else val

        return AgentVote(
            agent="kronos_quant",
            signal=signal,
            confidence=confidence,
            reasoning=f"Kronos-base quantitative forecast (weight={weight})",
            key_points=["ML-native OHLCV prediction", f"Confidence: {confidence:.1%}"],
        )
    except Exception as e:
        logger.warning(f"Could not parse Kronos vote: {e}")
        return None
