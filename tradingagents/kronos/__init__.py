"""
Kronos LLM Integration for TradingAgents
-----------------------------------------
Hybrid strategy: Kronos handles quantitative OHLCV forecasting;
Claude/OpenAI handles qualitative analysis (news, fundamentals, sentiment).

Key components:
  KronosWrapper       — thin wrapper around KronosPredictor with lazy loading
  KronosSignalParser  — converts raw Kronos predictions to structured trading signals
  create_kronos_analyst — builds a LangGraph-compatible node for the TradingAgents graph
"""

from .wrapper import KronosWrapper
from .signal_parser import KronosSignalParser
from .analyst import create_kronos_analyst

__all__ = ["KronosWrapper", "KronosSignalParser", "create_kronos_analyst"]
