"""Multi-Timeframe Analysis Engine — V2.0 (Upgraded).

Orchestrates Smart Money Concepts analysis across four timeframes:

  • **4H** — Macro trend direction (bullish / bearish / neutral)
  • **1H** — Market structure validation (HH/HL, LH/LL, BOS, CHoCH)
  • **15M** — Setup detection (FVG, Order Blocks, Liquidity zones)
  • **5M** — Entry confirmation (confirmation candles, displacement)

Rules:
  • Lower-timeframe signals **never contradict** higher-timeframe direction.
  • If a contradiction is detected, the contradiction-handling mode
    determines whether to filter out the signal, reduce the score, or
    warn only.

Key improvements over V1:
  • Full multi-timeframe orchestration instead of single-timeframe analysis.
  • Timeframe-specific result dictionaries so each layer's findings are
    independently inspectable.
  • Contradiction detection and configurable handling.
  • Backward-compatible ``AnalysisResult`` dataclass and ``analyze()`` function
    (accepts a single DataFrame for simple V1-style calls).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Import V1 analysis sub-modules (they operate on DataFrames and add columns)
from src.analysis.market_structure import (
    find_swing_points,
    analyze_market_structure,
    
)
from src.analysis.liquidity import find_equal_highs_lows, find_liquidity_sweeps
from src.analysis.fvg import find_fair_value_gaps, track_fvg_fill
from src.analysis.order_blocks import find_order_blocks, find_breaker_blocks, find_mitigation_blocks
from src.analysis.zones import find_premium_discount_zones, find_imbalances
from src.analysis.candles import (
    find_displacement_candles,
    find_rejection_candles,
    find_confirmation_candles,
)

logger = logging.getLogger(__name__)


# ── Timeframe roles ───────────────────────────────────────────────────────────

TIMEFRAME_ROLES: Dict[str, str] = {
    "4h": "macro",       # Macro trend
    "1h": "structural",  # Market structure
    "15m": "setup",      # Setup zones
    "5m": "entry",       # Entry confirmation
}

# Timeframe hierarchy (highest to lowest)
TIMEFRAME_HIERARCHY: List[str] = ["4h", "1h", "15m", "5m"]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class TimeframeResult:
    """Results for a single timeframe analysis layer.

    Attributes:
        timeframe: The interval string (e.g. ``'4h'``).
        role: The role assigned (``'macro'``, ``'structural'``, ``'setup'``, ``'entry'``).
        direction: Detected direction for this TF (``'BUY'``, ``'SELL'``, or ``None``).
        confidence: Raw confidence for this TF alone (0.0–1.0).
        findings: Dict of boolean / numeric findings keyed by indicator name.
        df: The processed DataFrame with all indicator columns.
    """
    timeframe: str
    role: str
    direction: Optional[str]
    confidence: float
    findings: Dict[str, Any] = field(default_factory=dict)
    df: Optional[pd.DataFrame] = None


@dataclass
class AnalysisResult:
    """Comprehensive multi-timeframe analysis result.

    Attributes:
        symbol: Trading pair symbol.
        timeframe: Primary (setup) timeframe string.
        signal_direction: Overall signal direction (``'BUY'``, ``'SELL'``, or ``None``).
        entry_zone_start: Entry zone lower bound.
        entry_zone_end: Entry zone upper bound.
        stop_loss: Stop-loss price.
        take_profits: List of take-profit prices.
        risk_reward_ratios: List of R:R ratios.
        confidence_score: Overall confidence score (0–100).
        confidence_breakdown: Weighted breakdown dict for the scorer.
        human_explanation: Human-readable summary of the analysis.
        chart_path: Optional path to the generated chart image.
        raw_data: Processed OHLCV DataFrame (for charting).
        mtf_results: Dict of per-timeframe ``TimeframeResult`` objects.
        macro_direction: Macro trend from the 4H layer.
        contradictions: List of contradiction descriptions.
        is_aligned: Whether all timeframes are aligned.
    """
    symbol: str
    timeframe: str
    signal_direction: Optional[str]
    entry_zone_start: Optional[float]
    entry_zone_end: Optional[float]
    stop_loss: Optional[float]
    take_profits: List[float]
    risk_reward_ratios: List[float]
    confidence_score: float
    confidence_breakdown: Dict[str, float]
    human_explanation: str
    chart_path: Optional[str] = None
    raw_data: Optional[pd.DataFrame] = None
    mtf_results: Dict[str, TimeframeResult] = field(default_factory=dict)
    macro_direction: Optional[str] = None
    contradictions: List[str] = field(default_factory=list)
    is_aligned: bool = False


# ── Single-timeframe analysis ─────────────────────────────────────────────────

def _analyze_single_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    symbol: str,
    config: Optional[Dict] = None,
) -> TimeframeResult:
    """Runs the full SMC analysis pipeline on a single timeframe's DataFrame.

    The pipeline mirrors the V1 engine but returns structured findings.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        timeframe: Interval string (e.g. ``'15m'``).
        symbol: Trading symbol.
        config: Optional analysis config dict (for thresholds, windows, etc.).

    Returns:
        ``TimeframeResult`` with direction, findings, and processed DataFrame.
    """
    cfg = config or {}

    # Ensure DataFrame is sorted
    df = df.sort_index().copy()

    if df.empty or len(df) < 20:
        return TimeframeResult(
            timeframe=timeframe,
            role=TIMEFRAME_ROLES.get(timeframe, "unknown"),
            direction=None,
            confidence=0.0,
            findings={},
            df=df,
        )

    # ── 1. Market Structure ──
    swing_window = cfg.get("market_structure", {}).get("swing_window", 5)
    df = find_swing_points(df, window=swing_window)
    result = analyze_market_structure(df)
    df = result["df"]
    
    # Extract structure points for findings (HH, HL, LH, LL)
    structure = result.get("structure", {})
    for label in ["HH", "HL", "LH", "LL"]:
        df[label] = False
        if label in structure:
            for sp in structure[label]:
                if sp.index < len(df):
                    df.loc[df.index[sp.index], label] = True
   

    # ── 2. Liquidity ──
    df = find_equal_highs_lows(df)
    df, liquidity_sweeps = find_liquidity_sweeps(df) 

    # ── 3. Fair Value Gaps ──
    df = find_fair_value_gaps(df)
    df, fvg_list = track_fvg_fill(df)

    # ── 4. Order Blocks ──
    df, order_blocks = find_order_blocks(df)
    
    # Extract swing points for breaker blocks if available
    swing_highs = [sp.price for sp in result.get("swing_points", ([], []))[0]]
    swing_lows = [sp.price for sp in result.get("swing_points", ([], []))[1]]
    
    df = find_breaker_blocks(df, swing_highs=swing_highs, swing_lows=swing_lows)
    df = find_mitigation_blocks(df)

    # ── 5. Premium / Discount Zones ──
    last_sh_idx = df[df["swing_high"]].index[-1] if not df[df["swing_high"]].empty else None
    last_sl_idx = df[df["swing_low"]].index[-1] if not df[df["swing_low"]].empty else None
    swing_high = df["high"].loc[last_sh_idx] if last_sh_idx else float(df["high"].max())
    swing_low = df["low"].loc[last_sl_idx] if last_sl_idx else float(df["low"].min())

    prem_start, equilibrium, disc_end, prem_zone, disc_zone = find_premium_discount_zones(
        df, swing_high, swing_low
    )
    df = find_imbalances(df)

    # ── 6. Confirmation Candles ──
    df = find_displacement_candles(df)
    df = find_rejection_candles(df)
    # Only run confirmation if rejection column exists
    if "rejection_bullish" in df.columns:
        df = find_confirmation_candles(df.copy(), "rejection_bullish", "bullish")

    # ── Derive direction from recent candles ──
    direction = _derive_direction(df, timeframe, cfg)

    # ── Collect findings ──
    findings = _collect_findings(df, swing_high, swing_low, prem_start, disc_end, prem_zone, disc_zone)

    return TimeframeResult(
        timeframe=timeframe,
        role=TIMEFRAME_ROLES.get(timeframe, "unknown"),
        direction=direction,
        confidence=findings.get("raw_confidence", 0.0),
        findings=findings,
        df=df,
    )


def _derive_direction(df: pd.DataFrame, timeframe: str, cfg: Dict) -> Optional[str]:
    """Determine the trade direction from the latest indicators.

    Looks at the most recent bullish / bearish signals across BOS, liquidity
    sweeps, and FVGs to decide BUY / SELL / None.

    Args:
        df: Processed DataFrame.
        timeframe: Interval string.
        cfg: Analysis config.

    Returns:
        ``'BUY'``, ``'SELL'``, or ``None``.
    """
    recent_bullish = 0
    recent_bearish = 0

    # Check for recent BOS
    for _, row in df.tail(3).iterrows():
        if row.get("bos_bullish"):
            recent_bullish += 1
        if row.get("bos_bearish"):
            recent_bearish += 1
        if row.get("choch_bullish"):
            recent_bullish += 1
        if row.get("choch_bearish"):
            recent_bearish += 1

    # Check for recent liquidity sweeps
    for _, row in df.tail(5).iterrows():
        if row.get("liquidity_sweep_bullish"):
            recent_bullish += 1
        if row.get("liquidity_sweep_bearish"):
            recent_bearish += 1

    # Check for recent FVGs
    for _, row in df.tail(5).iterrows():
        if row.get("fvg_bullish") and not row.get("fvg_filled"):
            recent_bullish += 1
        if row.get("fvg_bearish") and not row.get("fvg_filled"):
            recent_bearish += 1

    if recent_bullish > recent_bearish:
        return "BUY"
    elif recent_bearish > recent_bullish:
        return "SELL"
    else:
        return None


def _collect_findings(
    df: pd.DataFrame,
    swing_high: float,
    swing_low: float,
    prem_start: float,
    disc_end: float,
    prem_zone: float,
    disc_zone: float,
) -> Dict[str, Any]:
    """Collect all indicator findings from the processed DataFrame.

    Args:
        df: Processed DataFrame with indicator columns.
        swing_high: Latest swing high price.
        swing_low: Latest swing low price.
        prem_start: Premium zone start price.
        disc_end: Discount zone end price.
        prem_zone: Premium zone midpoint.
        disc_zone: Discount zone midpoint.

    Returns:
        Dict with boolean flags and numeric values for each indicator.
    """
    latest = df.iloc[-1]
    current_price = float(latest["close"])

    findings: Dict[str, Any] = {
        "current_price": current_price,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "premium_start": prem_start,
        "discount_end": disc_end,
        "premium_zone": prem_zone,
        "discount_zone": disc_zone,
        # Swing points
        "has_swing_high": bool(df["swing_high"].any()),
        "has_swing_low": bool(df["swing_low"].any()),
        # Market structure flags
        "has_hh": bool(df.get("HH", pd.Series(dtype=bool)).any()),
        "has_hl": bool(df.get("HL", pd.Series(dtype=bool)).any()),
        "has_lh": bool(df.get("LH", pd.Series(dtype=bool)).any()),
        "has_ll": bool(df.get("LL", pd.Series(dtype=bool)).any()),
        # BOS
        "has_bos_bullish": bool(df.get("bos_bullish", pd.Series(dtype=bool)).any()),
        "has_bos_bearish": bool(df.get("bos_bearish", pd.Series(dtype=bool)).any()),
        # CHoCH
        "has_choch_bullish": bool(df.get("choch_bullish", pd.Series(dtype=bool)).any()),
        "has_choch_bearish": bool(df.get("choch_bearish", pd.Series(dtype=bool)).any()),
        # Liquidity
        "has_equal_highs": bool(df.get("equal_high", pd.Series(dtype=bool)).any()),
        "has_equal_lows": bool(df.get("equal_low", pd.Series(dtype=bool)).any()),
        "has_sweep_bullish": bool(df.get("liquidity_sweep_bullish", pd.Series(dtype=bool)).any()),
        "has_sweep_bearish": bool(df.get("liquidity_sweep_bearish", pd.Series(dtype=bool)).any()),
        # FVG
        "has_fvg_bullish": bool(df.get("fvg_bullish", pd.Series(dtype=bool)).any()),
        "has_fvg_bearish": bool(df.get("fvg_bearish", pd.Series(dtype=bool)).any()),
        "has_untouched_fvg": False,
        # Order Blocks
        "has_bullish_ob": bool(df.get("bullish_ob", pd.Series(dtype=bool)).any()),
        "has_bearish_ob": bool(df.get("bearish_ob", pd.Series(dtype=bool)).any()),
        # Candles
        "has_displacement_bullish": bool(df.get("displacement_bullish", pd.Series(dtype=bool)).any()),
        "has_displacement_bearish": bool(df.get("displacement_bearish", pd.Series(dtype=bool)).any()),
        "has_rejection_bullish": bool(df.get("rejection_bullish", pd.Series(dtype=bool)).any()),
        "has_rejection_bearish": bool(df.get("rejection_bearish", pd.Series(dtype=bool)).any()),
        "has_confirmation_bullish": bool(df.get("confirmation_bullish", pd.Series(dtype=bool)).any()),
        "has_confirmation_bearish": bool(df.get("confirmation_bearish", pd.Series(dtype=bool)).any()),
        # Premium / Discount
        "in_premium_zone": current_price > prem_start,
        "in_discount_zone": current_price < disc_end,
        # Raw confidence (placeholder — refined by scorer)
        "raw_confidence": 0.0,
    }

    # Check for untouched FVGs (FVG exists but not yet filled)
    if findings["has_fvg_bullish"] or findings["has_fvg_bearish"]:
        fvg_rows = df[df.get("fvg_bullish", pd.Series(dtype=bool)) | df.get("fvg_bearish", pd.Series(dtype=bool))]
        if "fvg_filled" in fvg_rows.columns:
            findings["has_untouched_fvg"] = bool((~fvg_rows["fvg_filled"]).any())
        else:
            findings["has_untouched_fvg"] = True

    # Calculate raw confidence based on findings
    findings["raw_confidence"] = _calculate_raw_confidence(findings)

    return findings


def _calculate_raw_confidence(findings: Dict[str, Any]) -> float:
    """Calculate a raw confidence score from the collected findings.

    This is a preliminary score that the ``ConfidenceScorer`` will refine
    using configurable weights.

    Args:
        findings: Dict of indicator findings.

    Returns:
        Float between 0.0 and 1.0.
    """
    score = 0.0
    total = 0.0

    # Market structure alignment (weight: 20)
    if findings.get("has_swing_high") or findings.get("has_swing_low"):
        score += 20
    total += 20

    # Liquidity sweep (weight: 20)
    if findings.get("has_sweep_bullish") or findings.get("has_sweep_bearish"):
        score += 20
    total += 20

    # BOS / CHoCH confirmation (weight: 15)
    if any([
        findings.get("has_bos_bullish"),
        findings.get("has_bos_bearish"),
        findings.get("has_choch_bullish"),
        findings.get("has_choch_bearish"),
    ]):
        score += 15
    total += 15

    # Fair Value Gap (weight: 15)
    if findings.get("has_fvg_bullish") or findings.get("has_fvg_bearish"):
        score += 15
    total += 15

    # Fresh Order Block (weight: 10)
    if findings.get("has_bullish_ob") or findings.get("has_bearish_ob"):
        score += 10
    total += 10

    # Premium / Discount Zone (weight: 5)
    if findings.get("in_premium_zone") or findings.get("in_discount_zone"):
        score += 5
    total += 5

    # Confirmation Candle (weight: 5)
    if any([
        findings.get("has_confirmation_bullish"),
        findings.get("has_confirmation_bearish"),
        findings.get("has_rejection_bullish"),
        findings.get("has_rejection_bearish"),
    ]):
        score += 5
    total += 5

    # Session quality (weight: 5) — handled externally by SessionFilter
    total += 5

    # News filter (weight: 5) — handled externally by NewsFilter
    total += 5

    return min(1.0, score / total) if total > 0 else 0.0


# ── Multi-timeframe orchestration ─────────────────────────────────────────────

def analyze_multi_timeframe(
    data_by_timeframe: Dict[str, pd.DataFrame],
    symbol: str,
    config: Optional[Dict] = None,
) -> AnalysisResult:
    """Orchestrates multi-timeframe SMC analysis.

    This is the primary V2 entry point.  It analyzes each timeframe in the
    hierarchy (4H → 1H → 15M → 5M), detects contradictions, and returns
    a comprehensive ``AnalysisResult``.

    Args:
        data_by_timeframe: Dict mapping interval strings to OHLCV DataFrames.
                           Expected keys: ``'4h'``, ``'1h'``, ``'15m'``, ``'5m'``.
        symbol: Trading pair symbol.
        config: Optional analysis configuration dict (from settings.yaml).

    Returns:
        ``AnalysisResult`` with per-timeframe findings and overall signal.
    """
    cfg = config or {}
    mtf_cfg = cfg.get("trading", {}).get("multi_timeframe", {})
    contradiction_handling = mtf_cfg.get("contradiction_handling", "filter_out")

    # Determine which timeframes are available
    available_tfs = [tf for tf in TIMEFRAME_HIERARCHY if tf in data_by_timeframe]

    if not available_tfs:
        return _empty_result(symbol)

    # ── Analyze each timeframe ──
    tf_results: Dict[str, TimeframeResult] = {}
    for tf in available_tfs:
        df = data_by_timeframe[tf]
        result = _analyze_single_timeframe(df, tf, symbol, cfg)
        tf_results[tf] = result

    # ── Determine macro direction (highest available TF) ──
    macro_tf = available_tfs[0]  # e.g. '4h'
    macro_result = tf_results[macro_tf]
    macro_direction = macro_result.direction

    # ── Detect contradictions ──
    contradictions: List[str] = []
    is_aligned = True

    if macro_direction:
        for tf in available_tfs[1:]:  # Skip the macro TF itself
            tf_result = tf_results[tf]
            if tf_result.direction and tf_result.direction != macro_direction:
                contradiction_msg = (
                    f"{tf.upper()} direction ({tf_result.direction}) contradicts "
                    f"macro direction ({macro_direction})"
                )
                contradictions.append(contradiction_msg)
                is_aligned = False

    # ── Handle contradictions ──
    if contradictions and macro_direction:
        if contradiction_handling == "filter_out":
            # If any contradiction exists, suppress the signal entirely
            return _empty_result(
                symbol,
                contradiction=contradictions[0],
                mtf_results=tf_results,
                macro_direction=macro_direction,
            )
        elif contradiction_handling == "reduce_score":
            # Signal is kept but will be penalised by the scorer
            logger.warning(f"MTF contradiction detected: {contradictions}")
        elif contradiction_handling == "warn_only":
            logger.warning(f"MTF contradiction detected: {contradictions}")
        # else: filter_out is the default and safest

    # ── Determine overall signal direction ──
    # Use the setup timeframe (15M) or the highest available TF below macro
    setup_tf = "15m" if "15m" in tf_results else available_tfs[-1]
    overall_direction = tf_results[setup_tf].direction

    # If setup TF has no direction but macro does, use macro
    if overall_direction is None and macro_direction:
        overall_direction = macro_direction

    # ── Build confidence breakdown for the scorer ──
    confidence_breakdown = _build_component_scores(tf_results, cfg)

    # ── Generate entry / SL / TP ──
    entry_start, entry_end, stop_loss, take_profits, rr_ratios = _generate_trade_levels(
        tf_results, overall_direction, symbol
    )

    # ── Build human explanation ──
    explanation = _build_explanation(tf_results, overall_direction, macro_direction, contradictions)

    # ── Determine raw data for charting ──
    # Use the setup timeframe's processed DataFrame
    raw_data = tf_results[setup_tf].df if setup_tf in tf_results else None

    return AnalysisResult(
        symbol=symbol,
        timeframe=setup_tf,
        signal_direction=overall_direction,
        entry_zone_start=entry_start,
        entry_zone_end=entry_end,
        stop_loss=stop_loss,
        take_profits=take_profits,
        risk_reward_ratios=rr_ratios,
        confidence_score=0.0,  # Populated by ConfidenceScorer
        confidence_breakdown=confidence_breakdown,
        human_explanation=explanation,
        raw_data=raw_data,
        mtf_results=tf_results,
        macro_direction=macro_direction,
        contradictions=contradictions,
        is_aligned=is_aligned,
    )


def _build_component_scores(
    tf_results: Dict[str, TimeframeResult],
    cfg: Dict,
) -> Dict[str, float]:
    """Builds the component scores dict consumed by ``ConfidenceScorer``.

    Each component maps to a raw score (0.0–1.0) derived from the per-TF
    findings.

    Args:
        tf_results: Dict of ``TimeframeResult`` by interval.
        cfg: Analysis config.

    Returns:
        Dict mapping component names to raw scores.
    """
    # Collect findings from all timeframes
    macro = tf_results.get("4h")
    structural = tf_results.get("1h")
    setup = tf_results.get("15m")
    entry = tf_results.get("5m")

    # Use setup TF findings as primary, fall back to structural
    primary = setup or structural or tf_results.get(list(tf_results.keys())[0])
    primary_findings = primary.findings if primary else {}

    scores: Dict[str, float] = {}

    # 1. Market Structure Alignment (20)
    # Check if macro TF direction aligns with structural TF direction
    if macro and structural and macro.direction and structural.direction:
        if macro.direction == structural.direction:
            scores["market_structure_alignment"] = 1.0
        else:
            scores["market_structure_alignment"] = 0.3
    elif primary_findings.get("has_swing_high") or primary_findings.get("has_swing_low"):
        scores["market_structure_alignment"] = 0.7
    else:
        scores["market_structure_alignment"] = 0.0

    # 2. Liquidity Sweep (20)
    if primary_findings.get("has_sweep_bullish") or primary_findings.get("has_sweep_bearish"):
        scores["liquidity_sweep"] = 1.0
    else:
        scores["liquidity_sweep"] = 0.0

    # 3. BOS / CHoCH Confirmation (15)
    has_bos = (
        primary_findings.get("has_bos_bullish")
        or primary_findings.get("has_bos_bearish")
        or primary_findings.get("has_choch_bullish")
        or primary_findings.get("has_choch_bearish")
    )
    scores["bos_choch_confirmation"] = 1.0 if has_bos else 0.0

    # 4. Fair Value Gap (15)
    has_fvg = primary_findings.get("has_fvg_bullish") or primary_findings.get("has_fvg_bearish")
    has_untouched = primary_findings.get("has_untouched_fvg", False)
    if has_fvg:
        scores["fair_value_gap"] = 1.0 if has_untouched else 0.6
    else:
        scores["fair_value_gap"] = 0.0

    # 5. Fresh Order Block (10)
    has_ob = primary_findings.get("has_bullish_ob") or primary_findings.get("has_bearish_ob")
    scores["fresh_order_block"] = 1.0 if has_ob else 0.0

    # 6. Premium / Discount Zone (5)
    in_zone = primary_findings.get("in_premium_zone") or primary_findings.get("in_discount_zone")
    scores["premium_discount_zone"] = 1.0 if in_zone else 0.0

    # 7. Confirmation Candle (5)
    has_confirmation = any([
        primary_findings.get("has_confirmation_bullish"),
        primary_findings.get("has_confirmation_bearish"),
        primary_findings.get("has_rejection_bullish"),
        primary_findings.get("has_rejection_bearish"),
    ])
    scores["confirmation_candle"] = 1.0 if has_confirmation else 0.0

    # 8. Trading Session Quality (5) — handled externally, default 0.5
    scores["trading_session_quality"] = 0.5

    # 9. News Filter (5) — handled externally, default 1.0
    scores["news_filter"] = 1.0

    return scores


def _generate_trade_levels(
    tf_results: Dict[str, TimeframeResult],
    direction: Optional[str],
    symbol: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], List[float], List[float]]:
    """Generate entry zone, stop loss, and take profit levels.

    Args:
        tf_results: Dict of ``TimeframeResult``.
        direction: ``'BUY'``, ``'SELL'``, or ``None``.
        symbol: Trading symbol.

    Returns:
        Tuple of (entry_start, entry_end, stop_loss, take_profits, rr_ratios).
    """
    if direction is None:
        return None, None, None, [], []

    # Get current price from the setup TF
    setup_tf = tf_results.get("15m") or tf_results.get("5m") or tf_results.get(list(tf_results.keys())[0])
    if setup_tf is None or not setup_tf.df is not None:
        return None, None, None, [], []

    df = setup_tf.df
    current_price = float(df.iloc[-1]["close"])
    swing_high = setup_tf.findings.get("swing_high", df["high"].max())
    swing_low = setup_tf.findings.get("swing_low", df["low"].min())

    # Calculate ATR-like range for SL/TP sizing
    price_range = swing_high - swing_low
    if price_range <= 0:
        price_range = current_price * 0.01  # 1% fallback

    if direction == "BUY":
        entry_start = current_price * 0.998
        entry_end = current_price * 1.002
        stop_loss = swing_low * 0.995
        tp1 = current_price + price_range * 0.5
        tp2 = current_price + price_range * 1.0
        tp3 = current_price + price_range * 1.5
    else:  # SELL
        entry_start = current_price * 1.002
        entry_end = current_price * 0.998
        stop_loss = swing_high * 1.005
        tp1 = current_price - price_range * 0.5
        tp2 = current_price - price_range * 1.0
        tp3 = current_price - price_range * 1.5

    take_profits = [tp1, tp2, tp3]

    # Calculate R:R ratios
    risk = abs(entry_start - stop_loss) if direction == "BUY" else abs(stop_loss - entry_start)
    rr_ratios = []
    for tp in take_profits:
        reward = abs(tp - entry_start) if direction == "BUY" else abs(entry_start - tp)
        rr_ratios.append(round(reward / risk, 2) if risk > 0 else 0.0)

    return entry_start, entry_end, stop_loss, take_profits, rr_ratios


def _build_explanation(
    tf_results: Dict[str, TimeframeResult],
    direction: Optional[str],
    macro_direction: Optional[str],
    contradictions: List[str],
) -> str:
    """Builds a human-readable explanation of the multi-timeframe analysis.

    Args:
        tf_results: Dict of ``TimeframeResult``.
        direction: Overall signal direction.
        macro_direction: Macro trend direction.
        contradictions: List of contradiction descriptions.

    Returns:
        Formatted explanation string.
    """
    if direction is None:
        return "No clear signal identified based on multi-timeframe analysis."

    lines = [f"Multi-timeframe analysis for {direction} signal:\n"]

    # Macro trend
    if macro_direction:
        lines.append(f"  Macro Trend (4H): {macro_direction}")

    # Per-timeframe summary
    for tf in TIMEFRAME_HIERARCHY:
        if tf in tf_results:
            result = tf_results[tf]
            role = result.role.title()
            dir_str = result.direction or "Neutral"
            lines.append(f"  {tf.upper()} ({role}): {dir_str}")

    # Contradictions
    if contradictions:
        lines.append("\n  ⚠ Contradictions:")
        for c in contradictions:
            lines.append(f"    - {c}")

    if not contradictions and tf_results:
        lines.append("\n  ✓ All timeframes aligned.")

    # Key findings from setup TF
    setup = tf_results.get("15m") or tf_results.get("5m")
    if setup:
        f = setup.findings
        findings_list = []
        if f.get("has_sweep_bullish") or f.get("has_sweep_bearish"):
            findings_list.append("Liquidity sweep detected")
        if f.get("has_bos_bullish") or f.get("has_bos_bearish"):
            findings_list.append("BOS confirmed")
        if f.get("has_choch_bullish") or f.get("has_choch_bearish"):
            findings_list.append("CHoCH detected")
        if f.get("has_untouched_fvg"):
            findings_list.append("Untouched FVG present")
        if f.get("has_bullish_ob") or f.get("has_bearish_ob"):
            findings_list.append("Order block identified")
        if f.get("in_discount_zone"):
            findings_list.append("Price in discount zone")
        elif f.get("in_premium_zone"):
            findings_list.append("Price in premium zone")

        if findings_list:
            lines.append("\n  Key Findings:")
            for finding in findings_list:
                lines.append(f"    - {finding}")

    return "\n".join(lines)


# ── Backward-compatible single-TF analyze() ───────────────────────────────────

def analyze(ohlcv_df: pd.DataFrame, symbol: str, timeframe: str, config: Optional[Dict] = None) -> AnalysisResult:
    """Backward-compatible single-timeframe analysis entry point.

    This is the function imported by ``main.py`` in V1.  It wraps the
    single-timeframe analysis pipeline and returns an ``AnalysisResult``.

    For multi-timeframe analysis, use ``analyze_multi_timeframe()`` instead.

    Args:
        ohlcv_df: OHLCV DataFrame.
        symbol: Trading symbol.
        timeframe: Interval string.
        config: Optional analysis config dict.

    Returns:
        ``AnalysisResult`` with signal and breakdown.
    """
    if ohlcv_df.empty:
        return AnalysisResult(
            symbol=symbol, timeframe=timeframe, signal_direction=None,
            entry_zone_start=None, entry_zone_end=None, stop_loss=None,
            take_profits=[], risk_reward_ratios=[], confidence_score=0.0,
            confidence_breakdown={}, human_explanation="No data available for analysis."
        )

    # Run single-TF analysis
    tf_result = _analyze_single_timeframe(ohlcv_df, timeframe, symbol, config)

    direction = tf_result.direction
    confidence_breakdown = _build_component_scores({timeframe: tf_result}, config or {})

    # Generate trade levels
    entry_start, entry_end, stop_loss, take_profits, rr_ratios = _generate_trade_levels(
        {timeframe: tf_result}, direction, symbol
    )

    # Build explanation
    explanation = f"Single-timeframe analysis on {timeframe}: "
    if direction:
        explanation += f"{direction} signal detected. "
        findings = tf_result.findings
        if findings.get("has_sweep_bullish") or findings.get("has_sweep_bearish"):
            explanation += "Liquidity sweep present. "
        if findings.get("has_bos_bullish") or findings.get("has_bos_bearish"):
            explanation += "BOS confirmed. "
        if findings.get("has_fvg_bullish") or findings.get("has_fvg_bearish"):
            explanation += "FVG detected. "
        if findings.get("has_bullish_ob") or findings.get("has_bearish_ob"):
            explanation += "Order block identified. "
    else:
        explanation += "No clear signal identified."

    return AnalysisResult(
        symbol=symbol,
        timeframe=timeframe,
        signal_direction=direction,
        entry_zone_start=entry_start,
        entry_zone_end=entry_end,
        stop_loss=stop_loss,
        take_profits=take_profits,
        risk_reward_ratios=rr_ratios,
        confidence_score=0.0,  # Populated by ConfidenceScorer
        confidence_breakdown=confidence_breakdown,
        human_explanation=explanation,
        raw_data=tf_result.df,
    )


def _empty_result(
    symbol: str,
    contradiction: str = "",
    mtf_results: Optional[Dict[str, TimeframeResult]] = None,
    macro_direction: Optional[str] = None,
) -> AnalysisResult:
    """Returns an empty AnalysisResult (no signal)."""
    explanation = "No signal detected."
    if contradiction:
        explanation = f"Signal filtered due to MTF contradiction: {contradiction}"

    return AnalysisResult(
        symbol=symbol,
        timeframe="",
        signal_direction=None,
        entry_zone_start=None,
        entry_zone_end=None,
        stop_loss=None,
        take_profits=[],
        risk_reward_ratios=[],
        confidence_score=0.0,
        confidence_breakdown={},
        human_explanation=explanation,
        mtf_results=mtf_results or {},
        macro_direction=macro_direction,
        contradictions=[contradiction] if contradiction else [],
        is_aligned=False,
    )


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    # Generate dummy OHLCV data for each timeframe
    def make_dummy_df(n: int = 100, base_price: float = 50000) -> pd.DataFrame:
        dates = pd.date_range("2026-07-01", periods=n, freq="15min")
        prices = base_price + np.cumsum(np.random.randn(n) * 50)
        highs = prices + np.abs(np.random.randn(n) * 30)
        lows = prices - np.abs(np.random.randn(n) * 30)
        opens = prices - np.random.randn(n) * 20
        volumes = np.random.rand(n) * 1000
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes,
        }, index=dates)
        return df

    # Single-TF test (backward compat)
    print("=== Single-TF Analysis (Backward Compat) ===\n")
    df_15m = make_dummy_df(100, 60000)
    result = analyze(df_15m, "BTCUSDT", "15m")
    print(f"Signal: {result.signal_direction}")
    print(f"Entry: {result.entry_zone_start:.2f} – {result.entry_zone_end:.2f}")
    print(f"SL: {result.stop_loss:.2f}")
    print(f"TPs: {result.take_profits}")
    print(f"R:R: {result.risk_reward_ratios}")
    print(f"Explanation:\n{result.human_explanation}")

    # Multi-TF test
    print("\n\n=== Multi-Timeframe Analysis ===\n")
    data = {
        "4h": make_dummy_df(100, 58000),
        "1h": make_dummy_df(100, 59000),
        "15m": make_dummy_df(100, 60000),
        "5m": make_dummy_df(100, 60200),
    }
    multi_result = analyze_multi_timeframe(data, "BTCUSDT")
    print(f"Macro Direction: {multi_result.macro_direction}")
    print(f"Signal: {multi_result.signal_direction}")
    print(f"Aligned: {multi_result.is_aligned}")
    print(f"Contradictions: {multi_result.contradictions}")
    print(f"Component Scores: {multi_result.confidence_breakdown}")
    print(f"Explanation:\n{multi_result.human_explanation}")
