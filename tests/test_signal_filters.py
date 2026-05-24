"""
Unit tests for signal quality filters and confidence scoring.

Run with: pytest tests/test_signal_filters.py -v
"""

from __future__ import annotations

import pytest

from tradingagents.filters import (
    FilterConfig,
    FilterDecision,
    FilterResult,
    FilterStage,
    MarketData,
    RawSignal,
    SignalFilterPipeline,
)
from tradingagents.filters.config import (
    ConfidenceFilterConfig,
    ConsensusFilterConfig,
    DissentAnalysisConfig,
    HistoricalPerformanceConfig,
    MarketRegimeConfig,
    RiskFilterConfig,
)
from tradingagents.filters.filters import (
    run_confidence_filter,
    run_consensus_filter,
    run_dissent_analysis,
    run_historical_performance_filter,
    run_liquidity_filter,
    run_market_regime_filter,
    run_stop_loss_filter,
    run_volatility_filter,
)
from tradingagents.filters.models import FilterDecision, FilterStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_signal(
    ticker="RELIANCE",
    signal="BUY",
    confidence=0.80,
    weighted_score=0.50,
    agent_votes=None,
    dissenting_agents=None,
    entry_price=2500.0,
    stop_loss=2300.0,
) -> RawSignal:
    return RawSignal(
        ticker=ticker,
        signal=signal,
        recommendation="OVERWEIGHT",
        confidence=confidence,
        weighted_score=weighted_score,
        entry_price=entry_price,
        stop_loss=stop_loss,
        agent_votes=agent_votes or [
            {"agent": "bull", "signal": "BUY", "confidence": 0.85},
            {"agent": "technical", "signal": "BUY", "confidence": 0.80},
            {"agent": "fundamental", "signal": "BUY", "confidence": 0.75},
            {"agent": "sentiment", "signal": "BUY", "confidence": 0.70},
            {"agent": "risk_manager", "signal": "HOLD", "confidence": 0.60},
        ],
        dissenting_agents=dissenting_agents or [],
        trade_date="2025-03-01",
    )


def make_market(
    ticker="RELIANCE",
    open_price=2500.0,
    high_price=2550.0,
    low_price=2480.0,
    close_price=2530.0,
    volume=500_000.0,
    avg_volume_20d=400_000.0,
    nifty50_change_pct=0.005,
    vix=18.0,
) -> MarketData:
    return MarketData(
        ticker=ticker,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
        volume=volume,
        avg_volume_20d=avg_volume_20d,
        nifty50_change_pct=nifty50_change_pct,
        vix=vix,
        trade_date="2025-03-01",
    )


# ---------------------------------------------------------------------------
# 1. Consensus Filter
# ---------------------------------------------------------------------------

class TestConsensusFilter:
    def test_passes_with_high_agreement(self):
        # 4/5 = 80% agree on BUY — should pass
        sig = make_signal()
        cfg = ConsensusFilterConfig(min_agreement_pct=0.60)
        result = run_consensus_filter(sig, cfg)
        assert result.decision == FilterDecision.PASS
        assert result.metric == pytest.approx(4 / 5)

    def test_fails_with_low_agreement(self):
        # Only 2/5 agree on BUY
        votes = [
            {"agent": "bull", "signal": "BUY", "confidence": 0.8},
            {"agent": "technical", "signal": "BUY", "confidence": 0.7},
            {"agent": "fundamental", "signal": "SELL", "confidence": 0.8},
            {"agent": "sentiment", "signal": "HOLD", "confidence": 0.6},
            {"agent": "risk_manager", "signal": "SELL", "confidence": 0.9},
        ]
        sig = make_signal(agent_votes=votes)
        cfg = ConsensusFilterConfig(min_agreement_pct=0.60)
        result = run_consensus_filter(sig, cfg)
        assert result.decision == FilterDecision.HOLD_OVERRIDE
        assert result.metric == pytest.approx(2 / 5)

    def test_hold_signal_not_subject_to_filter(self):
        sig = make_signal(signal="HOLD")
        cfg = ConsensusFilterConfig(min_agreement_pct=0.99)  # unreachable
        result = run_consensus_filter(sig, cfg)
        assert result.decision == FilterDecision.PASS

    def test_disabled_filter_always_passes(self):
        sig = make_signal(agent_votes=[{"agent": "bull", "signal": "BUY", "confidence": 0.8}])
        cfg = ConsensusFilterConfig(enabled=False, min_agreement_pct=1.0)
        result = run_consensus_filter(sig, cfg)
        assert result.decision == FilterDecision.PASS

    def test_no_votes_passes(self):
        sig = make_signal(agent_votes=[])
        cfg = ConsensusFilterConfig()
        result = run_consensus_filter(sig, cfg)
        assert result.decision == FilterDecision.PASS


# ---------------------------------------------------------------------------
# 2. Confidence Filter
# ---------------------------------------------------------------------------

class TestConfidenceFilter:
    def test_passes_above_threshold(self):
        sig = make_signal(confidence=0.85)
        cfg = ConfidenceFilterConfig(min_avg_confidence=0.70)
        result = run_confidence_filter(sig, cfg)
        assert result.decision == FilterDecision.PASS

    def test_fails_below_threshold(self):
        votes = [{"agent": a, "signal": "BUY", "confidence": 0.50} for a in ["bull", "technical"]]
        sig = make_signal(confidence=0.50, agent_votes=votes)
        cfg = ConfidenceFilterConfig(min_avg_confidence=0.70)
        result = run_confidence_filter(sig, cfg)
        assert result.decision == FilterDecision.FILTERED
        assert result.metric < 0.70

    def test_uses_agent_votes_for_avg_if_available(self):
        votes = [
            {"agent": "bull", "signal": "BUY", "confidence": 0.80},
            {"agent": "technical", "signal": "BUY", "confidence": 0.60},
        ]
        # Agent avg = 0.70, overall signal confidence doesn't matter
        sig = make_signal(confidence=0.90, agent_votes=votes)
        cfg = ConfidenceFilterConfig(min_avg_confidence=0.70)
        result = run_confidence_filter(sig, cfg)
        assert result.decision == FilterDecision.PASS
        assert result.metric == pytest.approx(0.70)

    def test_disabled_always_passes(self):
        sig = make_signal(confidence=0.10)
        cfg = ConfidenceFilterConfig(enabled=False)
        result = run_confidence_filter(sig, cfg)
        assert result.decision == FilterDecision.PASS


# ---------------------------------------------------------------------------
# 3. Risk Filters
# ---------------------------------------------------------------------------

class TestVolatilityFilter:
    def test_passes_normal_range(self):
        market = make_market(open_price=2500, high_price=2550, low_price=2480)
        # range = 70 / 2500 = 2.8%
        cfg = RiskFilterConfig(max_intraday_range_pct=0.50)
        result = run_volatility_filter(market, cfg)
        assert result.decision == FilterDecision.PASS

    def test_fails_excessive_range(self):
        # range = 1500 / 2000 = 75% — exceeds 50% limit
        market = make_market(open_price=2000, high_price=3500, low_price=2000)
        cfg = RiskFilterConfig(max_intraday_range_pct=0.50)
        result = run_volatility_filter(market, cfg)
        assert result.decision == FilterDecision.FILTERED
        assert result.metric > 0.50

    def test_disabled(self):
        market = make_market(open_price=100, high_price=200, low_price=50)  # crazy range
        cfg = RiskFilterConfig(enabled=False)
        result = run_volatility_filter(market, cfg)
        assert result.decision == FilterDecision.PASS


class TestLiquidityFilter:
    def test_passes_normal_volume(self):
        market = make_market(volume=500_000, avg_volume_20d=400_000)
        cfg = RiskFilterConfig(min_volume_ratio=0.30, min_absolute_volume=100_000)
        result = run_liquidity_filter(market, cfg)
        assert result.decision == FilterDecision.PASS

    def test_fails_low_relative_volume(self):
        market = make_market(volume=100_000, avg_volume_20d=1_000_000)
        # ratio = 0.10 < 0.30
        cfg = RiskFilterConfig(min_volume_ratio=0.30, min_absolute_volume=50_000)
        result = run_liquidity_filter(market, cfg)
        assert result.decision == FilterDecision.FILTERED

    def test_fails_below_absolute_floor(self):
        market = make_market(volume=50_000, avg_volume_20d=10_000)
        # volume ratio > 1 BUT below absolute floor
        cfg = RiskFilterConfig(min_volume_ratio=0.30, min_absolute_volume=100_000)
        result = run_liquidity_filter(market, cfg)
        assert result.decision == FilterDecision.FILTERED


class TestStopLossFilter:
    def test_passes_tight_sl(self):
        sig = make_signal(entry_price=2500, stop_loss=2400)
        # SL distance = 100/2500 = 4% < 10%
        cfg = RiskFilterConfig(max_stop_loss_distance_pct=0.10)
        result = run_stop_loss_filter(sig, cfg)
        assert result.decision == FilterDecision.PASS

    def test_fails_wide_sl(self):
        sig = make_signal(entry_price=2500, stop_loss=2000)
        # SL distance = 500/2500 = 20% > 10%
        cfg = RiskFilterConfig(max_stop_loss_distance_pct=0.10)
        result = run_stop_loss_filter(sig, cfg)
        assert result.decision == FilterDecision.FILTERED
        assert result.metric == pytest.approx(0.20)

    def test_passes_when_no_sl_provided(self):
        sig = make_signal(stop_loss=None)
        cfg = RiskFilterConfig()
        result = run_stop_loss_filter(sig, cfg)
        assert result.decision == FilterDecision.PASS


# ---------------------------------------------------------------------------
# 4. Historical Performance Filter
# ---------------------------------------------------------------------------

class TestHistoricalPerformanceFilter:
    def test_known_strong_gets_boost(self):
        sig = make_signal(ticker="HINDUNILVR")
        cfg = HistoricalPerformanceConfig(
            known_strong=["HINDUNILVR"],
            strong_performer_size_mult=1.25,
        )
        result = run_historical_performance_filter(sig, {}, {}, cfg)
        assert result.decision == FilterDecision.PASS
        assert result.details["position_size_multiplier"] == pytest.approx(1.25)

    def test_known_weak_gets_reduced_size(self):
        sig = make_signal(ticker="WEAKCORP")
        cfg = HistoricalPerformanceConfig(
            known_weak=["WEAKCORP"],
            poor_performer_size_mult=0.50,
        )
        result = run_historical_performance_filter(sig, {}, {}, cfg)
        assert result.decision == FilterDecision.PASS
        assert result.details["position_size_multiplier"] == pytest.approx(0.50)

    def test_strong_historical_performer_boost(self):
        sig = make_signal(ticker="TCS")
        win_rates = {"TCS": 0.72}
        trade_counts = {"TCS": 10}
        cfg = HistoricalPerformanceConfig(
            strong_performer_threshold=0.60,
            strong_performer_size_mult=1.25,
            known_strong=[],
        )
        result = run_historical_performance_filter(sig, win_rates, trade_counts, cfg)
        assert result.details["position_size_multiplier"] == pytest.approx(1.25)

    def test_poor_historical_performer_reduced(self):
        sig = make_signal(ticker="BADCO")
        win_rates = {"BADCO": 0.30}
        trade_counts = {"BADCO": 8}
        cfg = HistoricalPerformanceConfig(
            poor_performer_threshold=0.40,
            poor_performer_size_mult=0.50,
            known_strong=[],
        )
        result = run_historical_performance_filter(sig, win_rates, trade_counts, cfg)
        assert result.details["position_size_multiplier"] == pytest.approx(0.50)

    def test_insufficient_history_default_size(self):
        sig = make_signal(ticker="NEWCO")
        cfg = HistoricalPerformanceConfig(min_trades_required=5)
        result = run_historical_performance_filter(sig, {}, {"NEWCO": 2}, cfg)
        assert result.details["position_size_multiplier"] == 1.0


# ---------------------------------------------------------------------------
# 5. Market Regime Filter
# ---------------------------------------------------------------------------

class TestMarketRegimeFilter:
    def test_passes_neutral_regime(self):
        sig = make_signal(signal="BUY")
        market = make_market(nifty50_change_pct=0.005)
        cfg = MarketRegimeConfig()
        result = run_market_regime_filter(sig, market, cfg)
        assert result.decision == FilterDecision.PASS

    def test_filters_sell_in_strong_bull(self):
        sig = make_signal(signal="SELL")
        market = make_market(nifty50_change_pct=0.025)  # strong bull
        cfg = MarketRegimeConfig(bull_threshold_pct=0.015, skip_uw_in_bull=True)
        result = run_market_regime_filter(sig, market, cfg)
        assert result.decision == FilterDecision.FILTERED

    def test_filters_buy_in_strong_bear(self):
        sig = make_signal(signal="BUY")
        market = make_market(nifty50_change_pct=-0.025)  # strong bear
        cfg = MarketRegimeConfig(bear_threshold_pct=-0.015, skip_ow_in_bear=True)
        result = run_market_regime_filter(sig, market, cfg)
        assert result.decision == FilterDecision.FILTERED

    def test_buy_passes_in_bull(self):
        sig = make_signal(signal="BUY")
        market = make_market(nifty50_change_pct=0.025)  # strong bull
        cfg = MarketRegimeConfig()
        result = run_market_regime_filter(sig, market, cfg)
        assert result.decision == FilterDecision.PASS

    def test_disabled(self):
        sig = make_signal(signal="SELL")
        market = make_market(nifty50_change_pct=0.10)  # extreme bull
        cfg = MarketRegimeConfig(enabled=False)
        result = run_market_regime_filter(sig, market, cfg)
        assert result.decision == FilterDecision.PASS


# ---------------------------------------------------------------------------
# 6. Dissent Analysis
# ---------------------------------------------------------------------------

class TestDissentAnalysis:
    def test_passes_no_dissenters(self):
        sig = make_signal(dissenting_agents=[])
        cfg = DissentAnalysisConfig()
        result = run_dissent_analysis(sig, cfg)
        assert result.decision == FilterDecision.PASS

    def test_logs_strong_dissent_but_no_veto(self):
        dissenters = [
            {"agent": "sentiment", "signal": "SELL", "confidence": 0.85, "reasoning": "News bad"}
        ]
        sig = make_signal(dissenting_agents=dissenters)
        cfg = DissentAnalysisConfig(
            strong_dissent_confidence=0.80,
            veto_confidence_threshold=0.90,
            veto_eligible_agents=["risk_manager"],
        )
        result = run_dissent_analysis(sig, cfg)
        # sentiment is not in veto_eligible_agents, so no veto
        assert result.decision == FilterDecision.PASS

    def test_veto_by_eligible_agent_with_high_confidence(self):
        dissenters = [
            {"agent": "risk_manager", "signal": "SELL", "confidence": 0.95, "reasoning": "Risk too high"}
        ]
        sig = make_signal(dissenting_agents=dissenters)
        cfg = DissentAnalysisConfig(
            veto_confidence_threshold=0.90,
            veto_eligible_agents=["risk_manager"],
        )
        result = run_dissent_analysis(sig, cfg)
        assert result.decision == FilterDecision.FILTERED
        assert "VETO" in result.reason

    def test_disabled(self):
        dissenters = [{"agent": "risk_manager", "signal": "SELL", "confidence": 0.99}]
        sig = make_signal(dissenting_agents=dissenters)
        cfg = DissentAnalysisConfig(enabled=False)
        result = run_dissent_analysis(sig, cfg)
        assert result.decision == FilterDecision.PASS


# ---------------------------------------------------------------------------
# Pipeline Integration Tests
# ---------------------------------------------------------------------------

class TestSignalFilterPipeline:
    def test_clean_signal_passes_all_stages(self):
        pipeline = SignalFilterPipeline(FilterConfig())
        sig = make_signal()
        market = make_market()
        result = pipeline.run(sig, market)
        assert result.passed
        assert result.final_signal == "BUY"
        assert result.position_size_multiplier > 0

    def test_hold_signal_passes_through_unfiltered(self):
        pipeline = SignalFilterPipeline(FilterConfig())
        sig = make_signal(signal="HOLD")
        result = pipeline.run(sig, None)
        assert result.passed
        assert result.final_signal == "HOLD"
        assert result.stage_results == []

    def test_low_confidence_gets_filtered(self):
        cfg = FilterConfig()
        cfg.confidence.min_avg_confidence = 0.70
        pipeline = SignalFilterPipeline(cfg)
        votes = [{"agent": a, "signal": "BUY", "confidence": 0.40} for a in ["bull", "bear"]]
        sig = make_signal(confidence=0.40, agent_votes=votes)
        result = pipeline.run(sig, make_market())
        assert not result.passed
        assert result.first_failure == FilterStage.CONFIDENCE

    def test_excessive_volatility_filtered(self):
        cfg = FilterConfig()
        pipeline = SignalFilterPipeline(cfg)
        sig = make_signal()
        # 80% intraday range
        market = make_market(open_price=1000, high_price=1800, low_price=1000)
        result = pipeline.run(sig, market)
        assert not result.passed
        assert result.first_failure == FilterStage.VOLATILITY

    def test_strong_bull_regime_filters_sell(self):
        pipeline = SignalFilterPipeline(FilterConfig())
        # Make all agents agree on SELL so consensus/confidence pass
        sell_votes = [
            {"agent": "bull", "signal": "SELL", "confidence": 0.85},
            {"agent": "technical", "signal": "SELL", "confidence": 0.80},
            {"agent": "fundamental", "signal": "SELL", "confidence": 0.75},
            {"agent": "sentiment", "signal": "SELL", "confidence": 0.80},
            {"agent": "risk_manager", "signal": "SELL", "confidence": 0.75},
        ]
        sig = make_signal(signal="SELL", agent_votes=sell_votes, confidence=0.80)
        sig.recommendation = "UNDERWEIGHT"
        market = make_market(nifty50_change_pct=0.03)  # strong bull
        result = pipeline.run(sig, market)
        assert not result.passed
        assert result.first_failure == FilterStage.MARKET_REGIME

    def test_historical_strong_performer_gets_size_boost(self):
        cfg = FilterConfig()
        pipeline = SignalFilterPipeline(cfg)
        sig = make_signal(ticker="HINDUNILVR")
        market = make_market()
        result = pipeline.run(sig, market)
        assert result.passed
        assert result.position_size_multiplier > 1.0

    def test_pipeline_without_market_data_skips_risk_stages(self):
        pipeline = SignalFilterPipeline(FilterConfig())
        sig = make_signal()
        result = pipeline.run(sig, market=None)
        assert result.passed
        stage_names = [sr.stage for sr in result.stage_results]
        assert FilterStage.VOLATILITY not in stage_names
        assert FilterStage.LIQUIDITY not in stage_names

    def test_veto_dissenter_overrides_everything(self):
        dissenters = [
            {"agent": "risk_manager", "signal": "SELL", "confidence": 0.95, "reasoning": "Extreme risk"}
        ]
        sig = make_signal(dissenting_agents=dissenters)
        pipeline = SignalFilterPipeline(FilterConfig())
        result = pipeline.run(sig, make_market())
        assert not result.passed
        assert result.first_failure == FilterStage.DISSENT_ANALYSIS

    def test_filter_result_to_dict(self):
        pipeline = SignalFilterPipeline(FilterConfig())
        result = pipeline.run(make_signal(), make_market())
        d = result.to_dict()
        assert "ticker" in d
        assert "decision" in d
        assert "stage_results" in d

    def test_filter_result_to_report(self):
        pipeline = SignalFilterPipeline(FilterConfig())
        result = pipeline.run(make_signal(), make_market())
        report = result.to_report()
        assert "RELIANCE" in report
        assert "BUY" in report


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestFilterConfig:
    def test_default_config(self):
        cfg = FilterConfig()
        assert cfg.consensus.min_agreement_pct == 0.60
        assert cfg.confidence.min_avg_confidence == 0.70
        assert cfg.risk.max_stop_loss_distance_pct == 0.10

    def test_from_yaml(self, tmp_path):
        yaml_content = """
consensus:
  min_agreement_pct: 0.70
confidence:
  min_avg_confidence: 0.75
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        cfg = FilterConfig.from_yaml(str(yaml_file))
        assert cfg.consensus.min_agreement_pct == 0.70
        assert cfg.confidence.min_avg_confidence == 0.75
        # Unchanged defaults
        assert cfg.risk.max_stop_loss_distance_pct == 0.10

    def test_to_dict_roundtrip(self):
        cfg = FilterConfig()
        d = cfg.to_dict()
        assert d["consensus"]["min_agreement_pct"] == 0.60
        assert d["risk"]["max_intraday_range_pct"] == 0.50
