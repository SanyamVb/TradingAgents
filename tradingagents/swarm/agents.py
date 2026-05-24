"""
Individual swarm agent implementations.

Each agent wraps an LLM and implements the 3-round debate protocol:
  Round 1: Independent analysis
  Round 2: Challenge round (sees other agents' analyses)
  Round 3: Final locked vote
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from .consensus import AgentVote

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    # Try to find JSON block
    patterns = [
        r"```json\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
        r"(\{[\s\S]*\})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    # Last resort: try parsing the whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON from: {text[:200]}")
        return {}


def _safe_signal(d: dict) -> str:
    sig = str(d.get("signal", "HOLD")).upper().strip()
    return sig if sig in ("BUY", "HOLD", "SELL", "SKIP") else "HOLD"


def _safe_confidence(d: dict) -> float:
    try:
        c = float(d.get("confidence", 0.5))
        return max(0.0, min(1.0, c))
    except (TypeError, ValueError):
        return 0.5


class BaseSwarmAgent:
    """Base class for all swarm agents."""

    def __init__(self, agent_id: str, llm: Any, system_prompt: str, prompts: dict):
        self.agent_id = agent_id
        self.llm = llm
        self.system_prompt = system_prompt
        self.prompts = prompts
        self._round1_result: Optional[dict] = None
        self._round2_result: Optional[dict] = None

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a human message and return text."""
        from langchain_core.messages import HumanMessage, SystemMessage
        msgs = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(msgs)
        return response.content if hasattr(response, "content") else str(response)

    def analyze_round1(self, ticker: str, trade_date: str, market_data: str) -> dict:
        """Round 1: Independent analysis."""
        prompt = self.prompts["round1"].format(
            ticker=ticker,
            trade_date=trade_date,
            market_data=market_data[:4000],  # Limit context
        )
        text = self._call_llm(prompt)
        result = _extract_json(text)
        if not result:
            result = {"signal": "HOLD", "confidence": 0.3, "reasoning": text[:500], "key_points": []}
        self._round1_result = result
        return result

    def analyze_round2(self, ticker: str, other_analyses: str) -> dict:
        """Round 2: Challenge round - sees other agents' Round 1 analyses."""
        own = json.dumps(self._round1_result or {}, indent=2)
        prompt = self.prompts["round2"].format(
            ticker=ticker,
            own_analysis=own,
            other_analyses=other_analyses[:3000],
        )
        text = self._call_llm(prompt)
        result = _extract_json(text)
        if not result:
            result = self._round1_result or {"signal": "HOLD", "confidence": 0.3, "reasoning": text[:500]}
        self._round2_result = result
        return result

    def final_vote(self, ticker: str) -> AgentVote:
        """Round 3: Final locked vote."""
        r1 = json.dumps(self._round1_result or {}, indent=2)
        r2 = json.dumps(self._round2_result or {}, indent=2)
        all_rounds = f"Round 1:\n{r1}\n\nRound 2:\n{r2}"

        prompt = self.prompts["round3"].format(
            ticker=ticker,
            all_rounds=all_rounds,
        )
        text = self._call_llm(prompt)
        result = _extract_json(text)

        if not result:
            # Fall back to Round 2 or Round 1 result
            result = self._round2_result or self._round1_result or {}

        # Track if signal changed between rounds
        r1_signal = _safe_signal(self._round1_result or {})
        final_signal = _safe_signal(result)
        revised = final_signal != r1_signal
        revision_reason = result.get("revision_reason", "")

        return AgentVote(
            agent=self.agent_id,
            signal=final_signal,
            confidence=_safe_confidence(result),
            reasoning=result.get("final_reasoning") or result.get("reasoning", ""),
            key_points=result.get("key_points", []),
            round_revised=revised,
            revision_reason=revision_reason,
        )


class BullAgent(BaseSwarmAgent):
    def __init__(self, llm: Any, prompts: dict):
        from .prompts import AGENT_PROMPTS
        p = prompts or AGENT_PROMPTS["bull"]
        super().__init__("bull", llm, p["system"], p)


class BearAgent(BaseSwarmAgent):
    def __init__(self, llm: Any, prompts: dict):
        from .prompts import AGENT_PROMPTS
        p = prompts or AGENT_PROMPTS["bear"]
        super().__init__("bear", llm, p["system"], p)


class TechnicalAgent(BaseSwarmAgent):
    def __init__(self, llm: Any, prompts: dict):
        from .prompts import AGENT_PROMPTS
        p = prompts or AGENT_PROMPTS["technical"]
        super().__init__("technical", llm, p["system"], p)


class FundamentalAgent(BaseSwarmAgent):
    def __init__(self, llm: Any, prompts: dict):
        from .prompts import AGENT_PROMPTS
        p = prompts or AGENT_PROMPTS["fundamental"]
        super().__init__("fundamental", llm, p["system"], p)


class SentimentAgent(BaseSwarmAgent):
    def __init__(self, llm: Any, prompts: dict):
        from .prompts import AGENT_PROMPTS
        p = prompts or AGENT_PROMPTS["sentiment"]
        super().__init__("sentiment", llm, p["system"], p)
