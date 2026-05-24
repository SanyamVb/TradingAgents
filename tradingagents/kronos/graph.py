"""
Kronos-augmented TradingAgentsGraph.

Drop-in replacement for the standard TradingAgentsGraph that:
1. Runs Kronos quantitative analysis BEFORE the main LLM pipeline
2. Injects the Kronos forecast report as context into the initial state
3. Stores the structured signal in self.last_kronos_signal for downstream use

Usage::

    from tradingagents.kronos.graph import KronosTradingGraph

    ta = KronosTradingGraph(
        selected_analysts=["market", "news", "fundamentals"],
        kronos_enabled=True,
        config={
            "llm_provider": "anthropic",
            "deep_think_llm": "claude-sonnet-4-5",
            "quick_think_llm": "claude-haiku-4-5",
        }
    )
    state, decision = ta.propagate("RELIANCE", "2025-05-01")
    print(decision)
    print("Kronos signal:", ta.last_kronos_signal)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tradingagents.graph.trading_graph import TradingAgentsGraph
from .analyst import create_kronos_analyst

logger = logging.getLogger(__name__)


class KronosTradingGraph(TradingAgentsGraph):
    """TradingAgentsGraph extended with a Kronos quantitative pre-analysis step.

    Kronos runs BEFORE the standard LangGraph pipeline. Its forecast report
    is injected as the first message so the LLM analysts can build on it.
    """

    def __init__(
        self,
        selected_analysts: List[str] = None,
        debug: bool = False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
        kronos_enabled: bool = True,
        kronos_repo_path: Optional[str] = None,
        kronos_pred_len: int = 5,
        kronos_lookback: int = 200,
        kronos_sample_count: int = 3,
    ):
        self._kronos_enabled = kronos_enabled
        self._kronos_repo_path = kronos_repo_path
        self._kronos_pred_len = kronos_pred_len
        self._kronos_lookback = kronos_lookback
        self._kronos_sample_count = kronos_sample_count
        self.last_kronos_signal: Optional[dict] = None

        if selected_analysts is None:
            selected_analysts = ["market", "news", "fundamentals"]

        super().__init__(
            selected_analysts=selected_analysts,
            debug=debug,
            config=config,
            callbacks=callbacks,
        )

        if self._kronos_enabled:
            logger.info("KronosTradingGraph: Kronos pre-analysis enabled")

    def _run_kronos_preanalysis(self, company_name: str, trade_date) -> str:
        """Run Kronos and return the formatted report (empty string on failure)."""
        node = create_kronos_analyst(
            kronos_repo_path=self._kronos_repo_path,
            pred_len=self._kronos_pred_len,
            lookback=self._kronos_lookback,
            sample_count=self._kronos_sample_count,
        )
        stub = {"company_of_interest": company_name, "trade_date": str(trade_date), "messages": []}
        try:
            result = node(stub)
            self.last_kronos_signal = result.get("kronos_signal")
            return result.get("market_report_kronos", "")
        except Exception as e:
            logger.warning(f"Kronos pre-analysis failed for {company_name}: {e}")
            self.last_kronos_signal = None
            return ""

    def propagate(self, company_name, trade_date):
        """Run Kronos first, then the standard TradingAgents pipeline."""
        self.last_kronos_signal = None

        if self._kronos_enabled:
            kronos_report = self._run_kronos_preanalysis(company_name, trade_date)
            if kronos_report:
                # Store report for access by propagator / initial state injection
                self._pending_kronos_report = kronos_report
                logger.info(
                    "Kronos pre-analysis complete: signal=%s conf=%.0f%%",
                    self.last_kronos_signal.get("signal") if self.last_kronos_signal else "N/A",
                    (self.last_kronos_signal.get("confidence", 0) * 100) if self.last_kronos_signal else 0,
                )
            else:
                self._pending_kronos_report = ""
        else:
            self._pending_kronos_report = ""

        return super().propagate(company_name, trade_date)
