"""
Individual filter implementations.

Each filter is a pure function (or small class) that takes a RawSignal +
MarketData + relevant config, and returns a StageResult.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .config import (
    ConsensusFilterConfig,
    ConfidenceFilterConfig,
    DissentAnalysisConfig,
    HistoricalPerformanceConfig,
    MarketRegimeConfig,
    RiskFilterConfig,
)
from .models import FilterDecision, FilterStage, MarketData, RawSignal, StageResult

logger = logging.getLogger(__name__)

RECOMMENDATION_MAP = {
    "BUY": "OVERWEIGHT",
    "SELL": "UNDERWEIGHT",
    "HOLD": "HOLD",
    "SKIP": "SKIP",
}


# ---------------------------------------------------------------------------
# 1. Consensus Filter
# ---------------------------------------------------------------------------

def run_consensus_filter(signal: RawSignal, cfg: ConsensusFilterConfig) -> StageResult:
    """Require 60%+ agent agreement for OW/UW signals."""
    if not cfg.enabled or signal.signal not in cfg.apply_to_signals:
        return StageResult(
            stage=FilterStage.CONSENSUS,
            decision=FilterDecision.PASS,
            reason=f"Filter disabled or signal '{signal.signal}' not subject to consensus check",
        )

    votes = signal.agent_votes
    if not votes:
        return StageResult(
            stage=FilterStage.CONSENSUS,
            decision=FilterDecision.PASS,
            reason="No individual agent votes available — skipping consensus check",
        )

    n_total = len(votes)
    n_agreeing = sum(1 for v in votes if v.get("signal") == signal.signal)
    agreement_pct = n_agreeing / n_total if n_total > 0 else 0.0

    if agreement_pct >= cfg.min_agreement_pct:
        return StageResult(
            stage=FilterStage.CONSENSUS,
            decision=FilterDecision.PASS,
            reason=f"Consensus {agreement_pct:.0%} >= {cfg.min_agreement_pct:.0%} required",
            metric=agreement_pct,
        )
    else:
        return StageResult(
            stage=FilterStage.CONSENSUS,
            decision=FilterDecision.HOLD_OVERRIDE,
            reason=(
                f"Consensus {agreement_pct:.0%} < {cfg.min_agreement_pct:.0%} required — "
                f"only {n_agreeing}/{n_total} agents agree on {signal.signal}"
            ),
            metric=agreement_pct,
            details={"n_agreeing": n_agreeing, "n_total": n_total},
        )


# ---------------------------------------------------------------------------
# 2. Confidence Threshold
# ---------------------------------------------------------------------------

def run_confidence_filter(signal: RawSignal, cfg: ConfidenceFilterConfig) -> StageResult:
    """Require avg confidence > 70 for actionable signals."""
    if not cfg.enabled:
        return StageResult(
            stage=FilterStage.CONFIDENCE,
            decision=FilterDecision.PASS,
            reason="Confidence filter disabled",
        )

    votes = signal.agent_votes
    if votes:
        confs = [v.get("confidence", 0.5) for v in votes if isinstance(v.get("confidence"), (int, float))]
        avg_conf = sum(confs) / len(confs) if confs else signal.confidence
    else:
        avg_conf = signal.confidence

    if avg_conf >= cfg.min_avg_confidence:
        return StageResult(
            stage=FilterStage.CONFIDENCE,
            decision=FilterDecision.PASS,
            reason=f"Avg confidence {avg_conf:.1%} >= {cfg.min_avg_confidence:.0%} threshold",
            metric=avg_conf,
        )
    else:
        return StageResult(
            stage=FilterStage.CONFIDENCE,
            decision=FilterDecision.FILTERED,
            reason=f"Avg confidence {avg_conf:.1%} < {cfg.min_avg_confidence:.0%} threshold",
            metric=avg_conf,
            details={"avg_confidence": avg_conf, "threshold": cfg.min_avg_confidence},
        )


# ---------------------------------------------------------------------------
# 3. Risk Filters
# ---------------------------------------------------------------------------

def run_volatility_filter(market: MarketData, cfg: RiskFilterConfig) -> StageResult:
    """Skip if intraday range > 50% of open price."""
    if not cfg.enabled or market.open_price <= 0:
        return StageResult(
            stage=FilterStage.VOLATILITY,
            decision=FilterDecision.PASS,
            reason="Volatility filter disabled or no open price",
        )

    intraday_range = market.high_price - market.low_price
    range_pct = intraday_range / market.open_price

    if range_pct <= cfg.max_intraday_range_pct:
        return StageResult(
            stage=FilterStage.VOLATILITY,
            decision=FilterDecision.PASS,
            reason=f"Intraday range {range_pct:.1%} <= {cfg.max_intraday_range_pct:.0%} limit",
            metric=range_pct,
        )
    else:
        return StageResult(
            stage=FilterStage.VOLATILITY,
            decision=FilterDecision.FILTERED,
            reason=f"Excessive volatility: intraday range {range_pct:.1%} > {cfg.max_intraday_range_pct:.0%}",
            metric=range_pct,
            details={
                "high": market.high_price,
                "low": market.low_price,
                "open": market.open_price,
                "range_pct": round(range_pct, 4),
            },
        )


def run_liquidity_filter(market: MarketData, cfg: RiskFilterConfig) -> StageResult:
    """Skip if volume < threshold * avg_volume or below absolute floor."""
    if not cfg.enabled:
        return StageResult(
            stage=FilterStage.LIQUIDITY,
            decision=FilterDecision.PASS,
            reason="Liquidity filter disabled",
        )

    # Absolute volume floor
    if market.volume < cfg.min_absolute_volume:
        return StageResult(
            stage=FilterStage.LIQUIDITY,
            decision=FilterDecision.FILTERED,
            reason=f"Volume {market.volume:,.0f} below absolute floor {cfg.min_absolute_volume:,.0f}",
            metric=market.volume,
        )

    # Relative volume check
    if market.avg_volume_20d > 0:
        vol_ratio = market.volume / market.avg_volume_20d
        if vol_ratio < cfg.min_volume_ratio:
            return StageResult(
                stage=FilterStage.LIQUIDITY,
                decision=FilterDecision.FILTERED,
                reason=(
                    f"Volume ratio {vol_ratio:.2f} < {cfg.min_volume_ratio:.2f} — "
                    f"only {vol_ratio:.0%} of 20d average"
                ),
                metric=vol_ratio,
                details={"volume": market.volume, "avg_20d": market.avg_volume_20d},
            )

    return StageResult(
        stage=FilterStage.LIQUIDITY,
        decision=FilterDecision.PASS,
        reason=f"Volume {market.volume:,.0f} passes liquidity check",
        metric=market.volume,
    )


def run_stop_loss_filter(signal: RawSignal, cfg: RiskFilterConfig) -> StageResult:
    """Reject if stop-loss distance > 10% from entry."""
    if not cfg.enabled:
        return StageResult(
            stage=FilterStage.STOP_LOSS,
            decision=FilterDecision.PASS,
            reason="Stop-loss filter disabled",
        )

    if signal.entry_price is None or signal.stop_loss is None:
        return StageResult(
            stage=FilterStage.STOP_LOSS,
            decision=FilterDecision.PASS,
            reason="No entry/stop-loss prices provided — skipping SL distance check",
        )

    sl_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price

    if sl_distance <= cfg.max_stop_loss_distance_pct:
        return StageResult(
            stage=FilterStage.STOP_LOSS,
            decision=FilterDecision.PASS,
            reason=f"SL distance {sl_distance:.1%} <= {cfg.max_stop_loss_distance_pct:.0%} limit",
            metric=sl_distance,
        )
    else:
        return StageResult(
            stage=FilterStage.STOP_LOSS,
            decision=FilterDecision.FILTERED,
            reason=(
                f"SL distance {sl_distance:.1%} > {cfg.max_stop_loss_distance_pct:.0%} — "
                f"entry={signal.entry_price:.2f}, sl={signal.stop_loss:.2f}"
            ),
            metric=sl_distance,
        )


# ---------------------------------------------------------------------------
# 4. Historical Performance Filter
# ---------------------------------------------------------------------------

def run_historical_performance_filter(
    signal: RawSignal,
    win_rates: Dict[str, float],
    trade_counts: Dict[str, int],
    cfg: HistoricalPerformanceConfig,
) -> StageResult:
    """Adjust position size based on historical win rate per ticker."""
    if not cfg.enabled:
        return StageResult(
            stage=FilterStage.HISTORICAL_PERFORMANCE,
            decision=FilterDecision.PASS,
            reason="Historical performance filter disabled",
        )

    ticker = signal.ticker
    n_trades = trade_counts.get(ticker, 0)

    # Check seeded strong/weak lists first (always have priority)
    if ticker in cfg.known_strong:
        multiplier = min(cfg.strong_performer_size_mult, cfg.max_position_size_mult)
        return StageResult(
            stage=FilterStage.HISTORICAL_PERFORMANCE,
            decision=FilterDecision.PASS,
            reason=f"{ticker} is a known strong performer — size boost {multiplier:.0%}",
            metric=multiplier,
            details={"position_size_multiplier": multiplier, "source": "known_strong"},
        )

    if ticker in cfg.known_weak:
        multiplier = cfg.poor_performer_size_mult
        return StageResult(
            stage=FilterStage.HISTORICAL_PERFORMANCE,
            decision=FilterDecision.PASS,
            reason=f"{ticker} is a known weak performer — size reduced to {multiplier:.0%}",
            metric=multiplier,
            details={"position_size_multiplier": multiplier, "source": "known_weak"},
        )

    if n_trades < cfg.min_trades_required:
        return StageResult(
            stage=FilterStage.HISTORICAL_PERFORMANCE,
            decision=FilterDecision.PASS,
            reason=f"Insufficient history ({n_trades} trades) for {ticker} — using default sizing",
            details={"position_size_multiplier": 1.0},
        )

    win_rate = win_rates.get(ticker, 0.5)

    if win_rate >= cfg.strong_performer_threshold:
        multiplier = min(cfg.strong_performer_size_mult, cfg.max_position_size_mult)
        return StageResult(
            stage=FilterStage.HISTORICAL_PERFORMANCE,
            decision=FilterDecision.PASS,
            reason=f"{ticker} strong performer ({win_rate:.0%} win rate, {n_trades} trades) — boost to {multiplier:.0%}",
            metric=win_rate,
            details={"position_size_multiplier": multiplier, "win_rate": win_rate, "n_trades": n_trades},
        )
    elif win_rate < cfg.poor_performer_threshold:
        multiplier = cfg.poor_performer_size_mult
        return StageResult(
            stage=FilterStage.HISTORICAL_PERFORMANCE,
            decision=FilterDecision.PASS,
            reason=f"{ticker} poor performer ({win_rate:.0%} win rate, {n_trades} trades) — reduced to {multiplier:.0%}",
            metric=win_rate,
            details={"position_size_multiplier": multiplier, "win_rate": win_rate, "n_trades": n_trades},
        )
    else:
        return StageResult(
            stage=FilterStage.HISTORICAL_PERFORMANCE,
            decision=FilterDecision.PASS,
            reason=f"{ticker} average performer ({win_rate:.0%} win rate) — default sizing",
            metric=win_rate,
            details={"position_size_multiplier": 1.0, "win_rate": win_rate, "n_trades": n_trades},
        )


# ---------------------------------------------------------------------------
# 5. Market Regime Filter
# ---------------------------------------------------------------------------

def classify_regime(market: MarketData, cfg: MarketRegimeConfig) -> str:
    """Returns 'bull', 'bear', or 'sideways'."""
    if market.nifty50_change_pct >= cfg.bull_threshold_pct:
        return "bull"
    elif market.nifty50_change_pct <= cfg.bear_threshold_pct:
        return "bear"
    return "sideways"


def run_market_regime_filter(
    signal: RawSignal,
    market: MarketData,
    cfg: MarketRegimeConfig,
) -> StageResult:
    """Detect market regime and skip conflicting signals."""
    if not cfg.enabled:
        return StageResult(
            stage=FilterStage.MARKET_REGIME,
            decision=FilterDecision.PASS,
            reason="Market regime filter disabled",
        )

    regime = classify_regime(market, cfg)
    high_vix = market.vix > cfg.high_vix_threshold if market.vix > 0 else False

    details = {
        "regime": regime,
        "nifty50_change_pct": market.nifty50_change_pct,
        "vix": market.vix,
        "high_vix": high_vix,
    }

    # Strong bull: skip UW/SELL signals
    if regime == "bull" and cfg.skip_uw_in_bull and signal.signal == "SELL":
        return StageResult(
            stage=FilterStage.MARKET_REGIME,
            decision=FilterDecision.FILTERED,
            reason=f"Strong bull regime ({market.nifty50_change_pct:+.2%}) — skipping SELL/UW signal",
            metric=market.nifty50_change_pct,
            details=details,
        )

    # Strong bear: skip OW/BUY signals
    if regime == "bear" and cfg.skip_ow_in_bear and signal.signal == "BUY":
        return StageResult(
            stage=FilterStage.MARKET_REGIME,
            decision=FilterDecision.FILTERED,
            reason=f"Strong bear regime ({market.nifty50_change_pct:+.2%}) — skipping BUY/OW signal",
            metric=market.nifty50_change_pct,
            details=details,
        )

    # High VIX — note but don't filter (just log)
    if high_vix:
        logger.warning(f"High VIX ({market.vix:.1f}) for {signal.ticker} — elevated risk environment")

    return StageResult(
        stage=FilterStage.MARKET_REGIME,
        decision=FilterDecision.PASS,
        reason=f"Regime={regime}, signal={signal.signal} — compatible",
        metric=market.nifty50_change_pct,
        details=details,
    )


# ---------------------------------------------------------------------------
# 6. Dissent Analysis
# ---------------------------------------------------------------------------

def run_dissent_analysis(signal: RawSignal, cfg: DissentAnalysisConfig) -> StageResult:
    """Analyze dissenting votes; optionally veto on high-confidence dissent."""
    if not cfg.enabled:
        return StageResult(
            stage=FilterStage.DISSENT_ANALYSIS,
            decision=FilterDecision.PASS,
            reason="Dissent analysis disabled",
        )

    dissenters = signal.dissenting_agents
    if not dissenters:
        return StageResult(
            stage=FilterStage.DISSENT_ANALYSIS,
            decision=FilterDecision.PASS,
            reason="No dissenting agents",
        )

    strong_dissenters = [
        d for d in dissenters
        if d.get("confidence", 0) >= cfg.strong_dissent_confidence
    ]

    veto_dissenters = [
        d for d in dissenters
        if d.get("confidence", 0) >= cfg.veto_confidence_threshold
        and d.get("agent") in cfg.veto_eligible_agents
    ]

    if veto_dissenters:
        vd = veto_dissenters[0]
        return StageResult(
            stage=FilterStage.DISSENT_ANALYSIS,
            decision=FilterDecision.FILTERED,
            reason=(
                f"VETO by {vd.get('agent')} (conf={vd.get('confidence', 0):.0%}): "
                f"{vd.get('reasoning', 'high-confidence dissent')[:200]}"
            ),
            metric=vd.get("confidence", 0),
            details={
                "veto_agent": vd.get("agent"),
                "veto_signal": vd.get("signal"),
                "all_dissenters": len(dissenters),
            },
        )

    if strong_dissenters:
        sd_names = ", ".join(d.get("agent", "?") for d in strong_dissenters)
        return StageResult(
            stage=FilterStage.DISSENT_ANALYSIS,
            decision=FilterDecision.PASS,
            reason=f"Strong dissent from {sd_names} — logged but no veto (below veto threshold)",
            metric=max(d.get("confidence", 0) for d in strong_dissenters),
            details={"strong_dissenters": [d.get("agent") for d in strong_dissenters]},
        )

    return StageResult(
        stage=FilterStage.DISSENT_ANALYSIS,
        decision=FilterDecision.PASS,
        reason=f"{len(dissenters)} dissenter(s) — below strong dissent threshold",
    )
