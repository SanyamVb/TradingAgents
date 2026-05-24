"""
Kronos Analyst node for TradingAgents LangGraph.

This node is a drop-in addition to the analyst pipeline. It:
  1. Loads the Kronos model (singleton, loads once)
  2. Fetches real OHLCV data for the ticker
  3. Generates a 5-day price forecast
  4. Derives a BUY/SELL/HOLD/SKIP signal with confidence
  5. Returns a formatted text report as the market_report_kronos state entry
     and injects it into the LangGraph messages list

The node can be added to the TradingAgents graph alongside existing analysts.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import AIMessage

from .wrapper import KronosWrapper
from .signal_parser import KronosSignalParser

logger = logging.getLogger(__name__)


def create_kronos_analyst(
    kronos_repo_path: Optional[str] = None,
    pred_len: int = 5,
    lookback: int = 200,
    sample_count: int = 3,
    confidence_threshold: float = 0.30,
):
    """Create a LangGraph-compatible Kronos analyst node.

    Args:
        kronos_repo_path:   Path to Kronos git clone. Defaults to ~/trading_enhancement/kronos
        pred_len:           Days to forecast (default: 5)
        lookback:           Historical bars for model context (default: 200)
        sample_count:       Monte Carlo samples for uncertainty (default: 3)
        confidence_threshold: Below this → SKIP signal (default: 0.30)

    Returns:
        kronos_analyst_node function compatible with LangGraph StateGraph
    """
    # Lazy singleton wrapper
    wrapper = KronosWrapper.get_instance(
        kronos_repo_path=kronos_repo_path,
        sample_count=sample_count,
    )

    def kronos_analyst_node(state):
        ticker = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "unknown")

        logger.info(f"Kronos analyst running for {ticker} on {trade_date}")

        report_text = ""
        signal_data = None
        try:
            # Load model if not yet loaded
            if not wrapper.is_loaded():
                wrapper.load_model()

            prediction = wrapper.predict(ticker, pred_len=pred_len, lookback=lookback)
            signal_data = KronosSignalParser.parse(prediction)

            # Override SKIP if confidence is below threshold
            if signal_data["confidence"] < confidence_threshold:
                signal_data["signal"] = "SKIP"

            report_text = KronosSignalParser.format_report(signal_data)

        except Exception as e:
            logger.warning(f"Kronos analyst failed for {ticker}: {e}", exc_info=True)
            report_text = (
                f"KRONOS ANALYSIS UNAVAILABLE for {ticker}: {str(e)}\n"
                "Falling back to Claude-only analysis."
            )

        message = AIMessage(
            content=report_text,
            name="KronosQuantAnalyst",
        )

        return {
            "messages": [message],
            "market_report_kronos": report_text,
            "kronos_signal": signal_data,
        }

    return kronos_analyst_node
