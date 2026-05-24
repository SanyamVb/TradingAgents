"""
Signal Filter Pipeline — orchestrates all filters in sequence.

Usage:
    pipeline = SignalFilterPipeline(config)
    result = pipeline.run(signal, market_data, win_rates={}, trade_counts={})
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .config import FilterConfig
from .filters import (
    run_confidence_filter,
    run_consensus_filter,
    run_dissent_analysis,
    run_historical_performance_filter,
    run_liquidity_filter,
    run_market_regime_filter,
    run_stop_loss_filter,
    run_volatility_filter,
)
from .models import FilterDecision, FilterResult, MarketData, RawSignal, StageResult

logger = logging.getLogger(__name__)

RECOMMENDATION_MAP = {
    "BUY": "OVERWEIGHT",
    "SELL": "UNDERWEIGHT",
    "HOLD": "HOLD",
    "SKIP": "SKIP",
}


class SignalFilterPipeline:
    """
    Runs all filter stages in order, short-circuiting on the first FILTERED result.

    Stage order:
    1. Dissent Analysis  (high-confidence veto — early exit avoids wasted work)
    2. Consensus Filter
    3. Confidence Threshold
    4. Volatility Check
    5. Liquidity Check
    6. Stop-Loss Distance
    7. Market Regime
    8. Historical Performance (sizing only — never blocks signal, runs last)
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()

    def run(
        self,
        signal: RawSignal,
        market: Optional[MarketData] = None,
        win_rates: Optional[Dict[str, float]] = None,
        trade_counts: Optional[Dict[str, int]] = None,
    ) -> FilterResult:
        """
        Run the full filter pipeline.

        Args:
            signal: The raw signal from the swarm consensus engine.
            market: Market data for risk/regime filters. If None, those stages are skipped.
            win_rates: {ticker: win_rate} for historical performance filter.
            trade_counts: {ticker: n_trades} for historical performance filter.

        Returns:
            FilterResult with final signal, decision, and per-stage details.
        """
        win_rates = win_rates or {}
        trade_counts = trade_counts or {}

        cfg = self.config
        stage_results = []
        position_size_mult = 1.0

        # HOLD/SKIP signals pass through unfiltered (they're already conservative)
        if signal.signal in ("HOLD", "SKIP"):
            return FilterResult(
                ticker=signal.ticker,
                original_signal=signal.signal,
                final_signal=signal.signal,
                final_recommendation=RECOMMENDATION_MAP.get(signal.signal, "HOLD"),
                decision=FilterDecision.PASS,
                stage_results=[],
                stages_passed=0,
                stages_failed=0,
                position_size_multiplier=1.0,
                trade_date=signal.trade_date,
            )

        def _apply(result: StageResult) -> bool:
            """Return False if we should stop the pipeline."""
            stage_results.append(result)
            if result.decision == FilterDecision.FILTERED:
                return False
            if result.decision == FilterDecision.HOLD_OVERRIDE:
                return False
            return True

        # --- Stage 1: Dissent Analysis ---
        sr = run_dissent_analysis(signal, cfg.dissent)
        if not _apply(sr):
            return self._build_result(signal, stage_results, position_size_mult, sr)

        # --- Stage 2: Consensus Filter ---
        sr = run_consensus_filter(signal, cfg.consensus)
        if not _apply(sr):
            return self._build_result(signal, stage_results, position_size_mult, sr)

        # --- Stage 3: Confidence Threshold ---
        sr = run_confidence_filter(signal, cfg.confidence)
        if not _apply(sr):
            return self._build_result(signal, stage_results, position_size_mult, sr)

        # --- Stages 4-6: Risk Filters (require market data) ---
        if market is not None:
            sr = run_volatility_filter(market, cfg.risk)
            if not _apply(sr):
                return self._build_result(signal, stage_results, position_size_mult, sr)

            sr = run_liquidity_filter(market, cfg.risk)
            if not _apply(sr):
                return self._build_result(signal, stage_results, position_size_mult, sr)

            sr = run_stop_loss_filter(signal, cfg.risk)
            if not _apply(sr):
                return self._build_result(signal, stage_results, position_size_mult, sr)

            # --- Stage 7: Market Regime ---
            sr = run_market_regime_filter(signal, market, cfg.regime)
            if not _apply(sr):
                return self._build_result(signal, stage_results, position_size_mult, sr)

        # --- Stage 8: Historical Performance (sizing, non-blocking) ---
        sr = run_historical_performance_filter(
            signal, win_rates, trade_counts, cfg.historical
        )
        stage_results.append(sr)
        # Extract position multiplier from details
        if sr.details and "position_size_multiplier" in sr.details:
            position_size_mult = float(sr.details["position_size_multiplier"])

        # All stages passed
        n_passed = sum(1 for s in stage_results if s.decision == FilterDecision.PASS)
        result = FilterResult(
            ticker=signal.ticker,
            original_signal=signal.signal,
            final_signal=signal.signal,
            final_recommendation=RECOMMENDATION_MAP.get(signal.signal, "HOLD"),
            decision=FilterDecision.PASS,
            stage_results=stage_results,
            stages_passed=n_passed,
            stages_failed=0,
            position_size_multiplier=position_size_mult,
            trade_date=signal.trade_date,
        )

        if cfg.log_filtered_signals:
            logger.info(
                f"[FILTER PASS] {signal.ticker} {signal.signal} "
                f"size={position_size_mult:.0%} stages={n_passed}"
            )

        return result

    def _build_result(
        self,
        signal: RawSignal,
        stage_results,
        position_size_mult: float,
        failing_stage: StageResult,
    ) -> FilterResult:
        cfg = self.config
        n_passed = sum(1 for s in stage_results[:-1] if s.decision == FilterDecision.PASS)

        # Determine final signal
        if failing_stage.decision == FilterDecision.HOLD_OVERRIDE:
            final_signal = "HOLD"
            final_rec = "HOLD"
        else:
            final_signal = "SKIP"
            final_rec = "SKIP"

        result = FilterResult(
            ticker=signal.ticker,
            original_signal=signal.signal,
            final_signal=final_signal,
            final_recommendation=final_rec,
            decision=failing_stage.decision,
            stage_results=stage_results,
            stages_passed=n_passed,
            stages_failed=1,
            first_failure=failing_stage.stage,
            failure_reason=failing_stage.reason,
            position_size_multiplier=0.0,
            trade_date=signal.trade_date,
        )

        if cfg.log_filtered_signals:
            logger.warning(
                f"[FILTER OUT] {signal.ticker} {signal.signal} -> {final_signal} "
                f"at stage={failing_stage.stage.value}: {failing_stage.reason}"
            )

        return result
