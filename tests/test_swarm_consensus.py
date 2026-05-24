"""
Tests for the MiroFish-inspired swarm consensus module.

Tests:
- ConsensusEngine: weighted voting, thresholds, veto, SKIP conditions
- AgentVote: dataclass fields
- ConsensusResult: report generation, dict serialization
- Swarm agents: round parsing, JSON extraction
- Swarm node: graceful failure handling
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from tradingagents.swarm.consensus import (
    ConsensusEngine,
    AgentVote,
    ConsensusResult,
    FALLBACK_WEIGHTS,
    SIGNAL_VALUES,
)
from tradingagents.swarm.agents import _extract_json, _safe_signal, _safe_confidence


# ── ConsensusEngine tests ────────────────────────────────────────────────────

class TestConsensusEngine:

    def _make_vote(self, agent, signal, confidence=0.8):
        return AgentVote(agent=agent, signal=signal, confidence=confidence, reasoning="test")

    def test_majority_buy(self):
        engine = ConsensusEngine()
        votes = [
            self._make_vote("bull", "BUY", 0.9),
            self._make_vote("technical", "BUY", 0.8),
            self._make_vote("fundamental", "BUY", 0.7),
            self._make_vote("sentiment", "HOLD", 0.6),
            self._make_vote("bear", "SELL", 0.5),
        ]
        result = engine.compute(votes)
        assert result.signal == "BUY"
        assert result.recommendation == "OVERWEIGHT"
        assert result.confidence > 0

    def test_majority_sell(self):
        engine = ConsensusEngine()
        votes = [
            self._make_vote("bull", "SELL", 0.9),
            self._make_vote("technical", "SELL", 0.8),
            self._make_vote("fundamental", "SELL", 0.7),
            self._make_vote("sentiment", "SELL", 0.6),
            self._make_vote("bear", "HOLD", 0.5),
        ]
        result = engine.compute(votes)
        assert result.signal == "SELL"
        assert result.recommendation == "UNDERWEIGHT"

    def test_hold_on_split(self):
        engine = ConsensusEngine()
        votes = [
            self._make_vote("bull", "BUY", 0.5),
            self._make_vote("technical", "SELL", 0.5),
            self._make_vote("fundamental", "HOLD", 0.5),
            self._make_vote("sentiment", "BUY", 0.5),
            self._make_vote("bear", "SELL", 0.5),
        ]
        result = engine.compute(votes)
        # Low confidence split should land near 0 -> HOLD or SKIP
        assert result.signal in ("HOLD", "SKIP")

    def test_veto_overrides_buy(self):
        engine = ConsensusEngine()
        votes = [
            self._make_vote("bull", "BUY", 0.95),
            self._make_vote("technical", "BUY", 0.9),
            self._make_vote("fundamental", "BUY", 0.85),
            self._make_vote("sentiment", "BUY", 0.8),
            self._make_vote("bear", "BUY", 0.7),
        ]
        result = engine.compute(votes, risk_veto_flag=True, veto_reason="Drawdown > 5%")
        assert result.signal == "SKIP"
        assert result.veto_triggered is True
        assert "Drawdown" in result.veto_reason

    def test_insufficient_agents_returns_skip(self):
        engine = ConsensusEngine(min_agents=3)
        votes = [
            self._make_vote("bull", "BUY", 0.9),
            self._make_vote("technical", "BUY", 0.8),
        ]
        result = engine.compute(votes)
        assert result.signal == "SKIP"
        assert "Insufficient" in result.skip_reason
        assert result.confidence == 0.0

    def test_dissent_tracking(self):
        engine = ConsensusEngine()
        votes = [
            self._make_vote("bull", "BUY", 0.9),
            self._make_vote("technical", "BUY", 0.85),
            self._make_vote("fundamental", "BUY", 0.8),
            self._make_vote("sentiment", "HOLD", 0.7),
            self._make_vote("bear", "SELL", 0.6),
        ]
        result = engine.compute(votes)
        assert result.signal == "BUY"
        # sentiment (HOLD) and bear (SELL) should be in dissenting
        dissent_agents = {v.agent for v in result.dissenting}
        assert "bear" in dissent_agents

    def test_signal_values_coverage(self):
        for sig in ["BUY", "HOLD", "SELL", "SKIP"]:
            assert sig in SIGNAL_VALUES

    def test_report_generation(self):
        engine = ConsensusEngine()
        votes = [self._make_vote("bull", "BUY", 0.8) for _ in range(3)]
        result = engine.compute(votes)
        report = result.to_report()
        assert "Swarm Consensus Report" in report
        assert result.recommendation in report

    def test_to_dict_serializable(self):
        engine = ConsensusEngine()
        votes = [self._make_vote("bull", "BUY", 0.8) for _ in range(3)]
        result = engine.compute(votes)
        d = result.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["signal"] == result.signal
        assert "votes" in parsed

    def test_count_signals(self):
        engine = ConsensusEngine()
        votes = [
            self._make_vote("bull", "BUY"),
            self._make_vote("technical", "BUY"),
            self._make_vote("fundamental", "HOLD"),
            self._make_vote("sentiment", "SELL"),
            self._make_vote("bear", "SELL"),
        ]
        counts = engine.count_signals(votes)
        assert counts["BUY"] == 2
        assert counts["HOLD"] == 1
        assert counts["SELL"] == 2

    def test_skip_on_low_confidence_dead_zone(self):
        engine = ConsensusEngine(dead_zone=0.3, skip_confidence_threshold=0.5)
        # Weak balanced votes: score near 0, low confidence
        votes = [
            self._make_vote("bull", "BUY", 0.2),
            self._make_vote("technical", "SELL", 0.2),
            self._make_vote("fundamental", "HOLD", 0.2),
        ]
        result = engine.compute(votes)
        assert result.signal == "SKIP"


# ── Agent utility function tests ─────────────────────────────────────────────

class TestAgentUtils:

    def test_extract_json_plain(self):
        text = '{"signal": "BUY", "confidence": 0.8}'
        result = _extract_json(text)
        assert result["signal"] == "BUY"

    def test_extract_json_fenced(self):
        text = '```json\n{"signal": "SELL", "confidence": 0.6}\n```'
        result = _extract_json(text)
        assert result["signal"] == "SELL"

    def test_extract_json_fenced_no_lang(self):
        text = '```\n{"signal": "HOLD", "confidence": 0.5}\n```'
        result = _extract_json(text)
        assert result["signal"] == "HOLD"

    def test_extract_json_fallback_on_bad(self):
        text = "I recommend buying this stock because the outlook is good."
        result = _extract_json(text)
        assert isinstance(result, dict)

    def test_safe_signal_valid(self):
        for sig in ["BUY", "HOLD", "SELL", "SKIP"]:
            assert _safe_signal({"signal": sig}) == sig

    def test_safe_signal_invalid_defaults_hold(self):
        assert _safe_signal({"signal": "STRONG_BUY"}) == "HOLD"
        assert _safe_signal({}) == "HOLD"

    def test_safe_confidence_clamp(self):
        assert _safe_confidence({"confidence": 1.5}) == 1.0
        assert _safe_confidence({"confidence": -0.2}) == 0.0
        assert _safe_confidence({"confidence": 0.75}) == pytest.approx(0.75)

    def test_safe_confidence_missing(self):
        assert _safe_confidence({}) == 0.5


# ── AgentVote dataclass tests ─────────────────────────────────────────────────

class TestAgentVote:

    def test_defaults(self):
        vote = AgentVote(agent="bull", signal="BUY", confidence=0.8, reasoning="strong trend")
        assert vote.key_points == []
        assert vote.round_revised is False
        assert vote.revision_reason == ""

    def test_with_key_points(self):
        vote = AgentVote(
            agent="technical",
            signal="HOLD",
            confidence=0.5,
            reasoning="mixed signals",
            key_points=["RSI neutral", "MACD flat"],
        )
        assert len(vote.key_points) == 2


# ── Swarm node integration test (mocked LLM) ─────────────────────────────────

class TestSwarmNode:

    def _make_mock_llm(self, signal="BUY", confidence=0.8):
        """Return a mock LLM that always responds with a valid JSON vote."""
        mock_llm = MagicMock()
        response_content = json.dumps({
            "signal": signal,
            "confidence": confidence,
            "reasoning": "Mock analysis for testing",
            "key_points": ["point1", "point2"],
            "final_reasoning": "Final mock reasoning",
        })
        mock_response = MagicMock()
        mock_response.content = response_content
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    def test_swarm_node_returns_consensus(self):
        from tradingagents.swarm.swarm_node import create_swarm_node, SwarmConfig

        mock_llm = self._make_mock_llm("BUY", 0.85)
        cfg = SwarmConfig(round1_parallel=False, round3_parallel=False)
        node = create_swarm_node(quick_llm=mock_llm, config=cfg)

        state = {
            "company_of_interest": "RELIANCE.NS",
            "trade_date": "2024-01-15",
            "market_report": "Strong momentum observed.",
            "sentiment_report": "Positive news flow.",
            "news_report": "No negative news.",
            "fundamentals_report": "P/E within range.",
        }

        result = node(state)

        assert "swarm_report" in result
        assert "swarm_consensus" in result
        assert result["swarm_consensus"]["signal"] in ("BUY", "HOLD", "SELL", "SKIP")
        assert 0.0 <= result["swarm_consensus"]["confidence"] <= 1.0
        assert "investment_plan" in result

    def test_swarm_node_handles_agent_failure(self):
        from tradingagents.swarm.swarm_node import create_swarm_node, SwarmConfig

        # LLM that raises on some calls
        call_count = [0]
        mock_llm = MagicMock()
        def side_effect(msgs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise RuntimeError("Simulated LLM timeout")
            resp = MagicMock()
            resp.content = json.dumps({
                "signal": "HOLD", "confidence": 0.6,
                "reasoning": "ok", "key_points": [],
                "final_reasoning": "ok"
            })
            return resp
        mock_llm.invoke.side_effect = side_effect

        cfg = SwarmConfig(round1_parallel=False, round3_parallel=False, min_agents_required=2)
        node = create_swarm_node(quick_llm=mock_llm, config=cfg)

        state = {
            "company_of_interest": "TCS.NS",
            "trade_date": "2024-01-15",
        }
        # Should not raise even if some agents fail
        result = node(state)
        assert "swarm_consensus" in result

    def test_swarm_node_skip_when_too_few_agents(self):
        from tradingagents.swarm.swarm_node import create_swarm_node, SwarmConfig

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("All agents fail")

        cfg = SwarmConfig(round1_parallel=False, round3_parallel=False, min_agents_required=3)
        node = create_swarm_node(quick_llm=mock_llm, config=cfg)

        state = {
            "company_of_interest": "HDFC.NS",
            "trade_date": "2024-01-15",
        }
        result = node(state)
        assert result["swarm_consensus"]["signal"] == "SKIP"


# ── Consensus weights test ────────────────────────────────────────────────────

def test_fallback_weights_sum_to_one():
    total = sum(FALLBACK_WEIGHTS.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"


def test_consensus_with_custom_weights():
    engine = ConsensusEngine(weights={"a": 0.5, "b": 0.3, "c": 0.2})
    votes = [
        AgentVote("a", "BUY", 0.9, ""),
        AgentVote("b", "BUY", 0.8, ""),
        AgentVote("c", "SELL", 0.7, ""),
    ]
    result = engine.compute(votes)
    assert result.signal == "BUY"
