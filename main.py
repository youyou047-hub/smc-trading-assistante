"""Professional AI Trading Analysis System V2.0 — Main Orchestrator.

This is the upgraded main entry point for the trading analysis system.
It orchestrates the complete scanning loop with V2.0 features:

1. Multi-timeframe analysis (4H macro → 1H structure → 15M setup → 5M entry)
2. Intelligent confidence scoring with debug-mode breakdown
3. Tiered alert system (Watchlist, Potential, High Probability, Premium)
4. Deduplication to prevent redundant alerts
5. Chart generation with professional annotations
6. Telegram alerts with HTML formatting
7. Heartbeat system for uptime monitoring
8. Statistics tracking for every detected setup
9. Error notifications for critical failures
10. Graceful shutdown on SIGINT/SIGTERM

The analysis engine remains independent and callable for external
integrations (n8n, custom scripts, etc.).

Usage:
    python main.py
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

# --- Core project imports (V1 modules preserved for compatibility) ---
from src.data.binance_client import BinanceClient, BinanceClientError
from src.analysis.engine import analyze, AnalysisResult
from src.scoring.confidence import ConfidenceScorer
from src.filters.news import NewsFilter
from src.filters.sessions import SessionFilter

# --- V2.0 upgraded imports ---
from src.utils.logger import setup_logger, log_score_breakdown
from src.utils.heartbeat import HeartbeatSystem
from src.utils.statistics import StatisticsTracker
from src.alerts.telegram import TelegramBot
from src.alerts.formatter import AlertFormatter
from src.alerts.deduplication import AlertDeduplicator
from src.charts.generator import ChartGenerator


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
running = True


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------
def signal_handler(signum: int, frame) -> None:
    """Handle graceful shutdown signals (SIGINT, SIGTERM)."""
    global running
    logging.getLogger(__name__).info(
        "Signal %d received. Initiating graceful shutdown...", signum
    )
    running = False


# ---------------------------------------------------------------------------
# Chart data extraction (evolved from V1)
# ---------------------------------------------------------------------------
def extract_chart_data(result: AnalysisResult) -> Dict:
    """Extract chart annotation data from analysis results.

    Processes the ``raw_data`` DataFrame to find indicators that should
    be annotated on the chart (BOS, CHoCH, FVGs, Order Blocks, Liquidity
    Sweeps).

    Args:
        result: Completed analysis result with populated ``raw_data``.

    Returns:
        Dictionary of chart annotation data ready for the chart generator.
    """
    chart_data: Dict = {
        "liquidity_sweeps": [],
        "bos_lines": [],
        "choch_lines": [],
        "fvgs": [],
        "order_blocks": [],
        "entry_zone_start": result.entry_zone_start,
        "entry_zone_end": result.entry_zone_end,
        "stop_loss": result.stop_loss,
        "take_profits": result.take_profits or [],
    }

    if result.raw_data is None or result.raw_data.empty:
        return chart_data

    df = result.raw_data

    for idx, row in df.iterrows():
        # FVGs
        if row.get("fvg_bullish"):
            chart_data["fvgs"].append({
                "type": "bullish",
                "fvg_start": row.get("fvg_start", 0),
                "fvg_end": row.get("fvg_end", 0),
                "index": idx,
            })
        elif row.get("fvg_bearish"):
            chart_data["fvgs"].append({
                "type": "bearish",
                "fvg_start": row.get("fvg_start", 0),
                "fvg_end": row.get("fvg_end", 0),
                "index": idx,
            })

        # Order Blocks
        if row.get("bullish_ob"):
            chart_data["order_blocks"].append({
                "type": "bullish",
                "ob_start": row.get("ob_start", 0),
                "ob_end": row.get("ob_end", 0),
                "index": idx,
            })
        elif row.get("bearish_ob"):
            chart_data["order_blocks"].append({
                "type": "bearish",
                "ob_start": row.get("ob_start", 0),
                "ob_end": row.get("ob_end", 0),
                "index": idx,
            })

        # Liquidity Sweeps
        if row.get("liquidity_sweep_bullish"):
            chart_data["liquidity_sweeps"].append((row["low"], idx))
        elif row.get("liquidity_sweep_bearish"):
            chart_data["liquidity_sweeps"].append((row["high"], idx))

        # BOS
        if row.get("bos_bullish"):
            chart_data["bos_lines"].append((row["high"], idx, "bullish"))
        elif row.get("bos_bearish"):
            chart_data["bos_lines"].append((row["low"], idx, "bearish"))

        # CHoCH
        if row.get("choch_bullish"):
            chart_data["choch_lines"].append((row["high"], idx, "bullish"))
        elif row.get("choch_bearish"):
            chart_data["choch_lines"].append((row["low"], idx, "bearish"))

    return chart_data


# ---------------------------------------------------------------------------
# Tier determination (V2)
# ---------------------------------------------------------------------------
def determine_tier(score: float, tiers_config: dict) -> str:
    """Determine the alert tier from a confidence score.

    Args:
        score: Confidence score (0-100).
        tiers_config: Alert tier thresholds from config.

    Returns:
        Tier label string.
    """
    premium_min = tiers_config.get("premium_min", 90)
    high_prob_min = tiers_config.get("high_probability_min", 80)
    potential_min = tiers_config.get("potential_setup_min", 70)
    watchlist_min = tiers_config.get("watchlist_min", 60)

    if score >= premium_min:
        return "PREMIUM"
    elif score >= high_prob_min:
        return "HIGH_PROBABILITY"
    elif score >= potential_min:
        return "POTENTIAL"
    elif score >= watchlist_min:
        return "WATCHLIST"
    else:
        return "IGNORED"


# ---------------------------------------------------------------------------
# Build reasons and missing conditions from score breakdown
# ---------------------------------------------------------------------------
def build_reasons(
    breakdown: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[List[str], List[str]]:
    """Classify score components into found and missing conditions.

    A condition is "found" if it contributes >= 60% of its maximum
    possible contribution (i.e., raw score >= 0.6).

    Args:
        breakdown: Weighted contribution per component.
        weights: Configured weights per component.

    Returns:
        Tuple of ``(found_reasons, missing_conditions)`` string lists.
    """
    found: List[str] = []
    missing: List[str] = []
    component_labels = {
        "market_structure_alignment": "Bullish/Bearish Market Structure",
        "liquidity_sweep": "Liquidity Sweep Detected",
        "bos_choch_confirmation": "BOS / CHoCH Confirmed",
        "fair_value_gap": "Fair Value Gap Present",
        "fresh_order_block": "Fresh Order Block",
        "premium_discount_zone": "Premium / Discount Zone Alignment",
        "confirmation_candle": "Confirmation Candle",
        "trading_session_quality": "Quality Trading Session",
        "news_filter": "No High-Impact News",
    }

    for comp, contribution in breakdown.items():
        label = component_labels.get(comp, comp.replace("_", " ").title())
        weight = weights.get(comp, 1)
        threshold = weight * 0.6
        if contribution >= threshold:
            found.append(label)
        else:
            missing.append(label)

    return found, missing


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main() -> None:
    """Main execution loop for the V2.0 trading analysis system."""
    global running

    # --- Load environment & config ---
    load_dotenv()
    config = load_config()

    # --- Setup Logger ---
    debug_mode = config.get("system", {}).get("debug_mode", False)
    log_file = config.get("logging", {}).get("file", "logs/trading_assistant.log")
    log_level = config.get("logging", {}).get("level", "INFO")
    logger = setup_logger(
        log_file=log_file,
        level=log_level,
        debug_mode=debug_mode,
    )

    logger.info("=" * 60)
    logger.info("Professional AI Trading Analysis System V2.0 — Starting")
    logger.info("=" * 60)
    logger.info(
        "Config: version=%s | debug=%s | scan_interval=%ds",
        config.get("system", {}).get("version", "2.0.0"),
        debug_mode,
        config.get("system", {}).get("scan_interval_seconds", 60),
    )

    # --- Initialise Components ---
    binance_url = config.get("binance", {}).get(
        "base_url", "https://data-api.binance.vision/api/v3"
    )
    binance_client = BinanceClient(base_url=binance_url)

    # Confidence scorer — adapt V1 thresholds to V2 config
    weights = config["scoring"]["weights"]
    alert_tiers = config.get("alert_tiers", {})
    scorer_thresholds = {
        "ignore_below": alert_tiers.get("ignore_below", 60),
        "record_only_min": alert_tiers.get("watchlist_min", 60),
        "high_probability_min": alert_tiers.get("high_probability_min", 80),
        "premium_min": alert_tiers.get("premium_min", 90),
    }
    confidence_scorer = ConfidenceScorer(
        weights=weights,
        thresholds=scorer_thresholds,
    )

    # News filter
    news_config = config.get("news_filter", {})
    news_filter = NewsFilter(
        enabled=news_config.get("enabled", True),
        buffer_minutes=news_config.get("buffer_minutes_before", 30),
        high_impact_events=news_config.get("high_impact_events", []),
    )

    # Session filter
    session_filter = SessionFilter(
        sessions_config=config.get("sessions", {}),
    )

    # Chart generator
    chart_config = config.get("charts", {})
    chart_generator = ChartGenerator(
        output_dir=chart_config.get("output_dir", "charts"),
        theme=chart_config.get("theme", "dark"),
        figscale=chart_config.get("figscale", 1.5),
    )

    # Telegram
    telegram_config = config.get("telegram", {})
    telegram_enabled = telegram_config.get("enabled", True)
    bot_token = os.getenv(
        "TELEGRAM_BOT_TOKEN", telegram_config.get("bot_token", "")
    )
    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID", telegram_config.get("chat_id", "")
    )
    telegram_bot = TelegramBot(bot_token, chat_id)
    alert_formatter = AlertFormatter()

    # Deduplication
    dedup_config = config.get("deduplication", {})
    dedup_stats_path = config.get("deduplication", {}).get(
        "persist_path", "logs/dedup_state.json"
    )
    deduplicator = AlertDeduplicator(
        score_change_threshold=dedup_config.get("score_change_threshold", 10),
        structure_change_resend=dedup_config.get("structure_change_resend", True),
        direction_change_resend=dedup_config.get("direction_change_resend", True),
        cooldown_minutes=dedup_config.get("cooldown_minutes", 15),
        max_alerts_per_hour=dedup_config.get("max_alerts_per_hour", 3),
        persist_path=dedup_stats_path if dedup_config.get("enabled", True) else None,
    )

    # Statistics tracker
    stats_config = config.get("statistics", {})
    stats_tracker = StatisticsTracker(
        store_path=stats_config.get("store_path", "logs/stats.json"),
        retention_days=stats_config.get("retention_days", 90),
        enabled=stats_config.get("enabled", True),
    )

    # Heartbeat system
    heartbeat_config = config.get("heartbeat", config.get("telegram", {}).get("heartbeat", {}))
    hb_symbols = config.get("trading", {}).get("symbols", [])
    mtf_config = config.get("trading", {}).get("multi_timeframe", {})
    hb_timeframes = {
        "4H": mtf_config.get("macro_tf", "4h"),
        "1H": mtf_config.get("structural_tf", "1h"),
        "15M": mtf_config.get("setup_tf", "15m"),
        "5M": mtf_config.get("entry_tf", "5m"),
    }
    heartbeat = HeartbeatSystem(
        telegram_bot=telegram_bot,
        formatter=alert_formatter,
        interval_hours=heartbeat_config.get("interval_hours", 5),
        send_startup=heartbeat_config.get("startup_message", True),
        symbols=hb_symbols,
        timeframes=hb_timeframes,
        include_stats=heartbeat_config.get("include_scan_count", True),
    )

    # --- Register signal handlers ---
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Start heartbeat ---
    heartbeat.start()

    # --- Extract trading parameters ---
    scan_interval = config.get("system", {}).get("scan_interval_seconds", 60)
    symbols = hb_symbols
    mtf_enabled = mtf_config.get("enabled", False)
    higher_timeframe = mtf_config.get("macro_tf", "4h") if mtf_enabled else "1h"
    lower_timeframe = mtf_config.get("setup_tf", "15m")
    kline_limit = config.get("binance", {}).get("kline_limit", 500)

    # Ensure output directories
    os.makedirs(config.get("charts", {}).get("output_dir", "charts"), exist_ok=True)
    os.makedirs(os.path.dirname(stats_config.get("store_path", "logs/stats.json")), exist_ok=True)

    logger.info(f"Monitoring symbols: {symbols}")
    logger.info(f"Multi-timeframe: {mtf_enabled}")
    if mtf_enabled:
        logger.info(
            "  4H: %s | 1H: %s | 15M: %s | 5M: %s",
            mtf_config.get("macro_tf"),
            mtf_config.get("structural_tf"),
            mtf_config.get("setup_tf"),
            mtf_config.get("entry_tf"),
        )
    else:
        logger.info(f"Timeframes: Higher={higher_timeframe}, Lower={lower_timeframe}")
    logger.info(f"Scan interval: {scan_interval}s")
    logger.info(f"Telegram: {'Enabled' if telegram_enabled else 'Disabled'}")
    logger.info(f"Deduplication: {'Enabled' if dedup_config.get('enabled') else 'Disabled'}")

    # --- Main Scanning Loop ---
    while running:
        current_utc = datetime.now(timezone.utc)
        logger.info(
            "--- Scan started at %s ---",
            current_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        for symbol in symbols:
            try:
                # 1. Fetch Data
                logger.debug("Fetching %s data...", symbol)

                if mtf_enabled:
                    # Multi-timeframe: fetch all 4 timeframes
                    ohlcv_macro = binance_client.get_ohlcv(
                        symbol, mtf_config.get("macro_tf", "4h"), limit=kline_limit
                    )
                    ohlcv_structural = binance_client.get_ohlcv(
                        symbol, mtf_config.get("structural_tf", "1h"), limit=kline_limit
                    )
                    ohlcv_setup = binance_client.get_ohlcv(
                        symbol, mtf_config.get("setup_tf", "15m"), limit=kline_limit
                    )
                    ohlcv_entry = binance_client.get_ohlcv(
                        symbol, mtf_config.get("entry_tf", "5m"), limit=kline_limit
                    )

                    # Run analysis on setup timeframe (primary)
                    analysis_result = analyze(
                        ohlcv_setup, symbol, mtf_config.get("setup_tf", "15m")
                    )

                    # Multi-timeframe alignment check
                    if analysis_result.signal_direction is not None:
                        # Validate that lower TF does not contradict higher TF
                        macro_analysis = analyze(
                            ohlcv_macro, symbol, mtf_config.get("macro_tf", "4h")
                        )
                        if macro_analysis.signal_direction:
                            macro_dir = macro_analysis.signal_direction
                            setup_dir = analysis_result.signal_direction
                            if macro_dir != setup_dir:
                                handling = mtf_config.get(
                                    "contradiction_handling", "filter_out"
                                )
                                if handling == "filter_out":
                                    logger.info(
                                        "%s: MTF contradiction (4H=%s, 15M=%s). "
                                        "Filtered out.",
                                        symbol, macro_dir, setup_dir,
                                    )
                                    heartbeat.increment_scans()
                                    continue
                                elif handling == "reduce_score":
                                    logger.info(
                                        "%s: MTF contradiction detected. "
                                        "Score will be reduced.",
                                        symbol,
                                    )
                                else:
                                    logger.info(
                                        "%s: MTF contradiction detected. "
                                        "Warning only.",
                                        symbol,
                                    )

                else:
                    # Legacy single-timeframe approach
                    ohlcv_setup = binance_client.get_ohlcv(
                        symbol, lower_timeframe, limit=kline_limit
                    )
                    analysis_result = analyze(
                        ohlcv_setup, symbol, lower_timeframe
                    )

                # If no signal, skip
                if analysis_result.signal_direction is None:
                    logger.info("%s: No signal detected.", symbol)
                    heartbeat.increment_scans()
                    continue

                # 2. Calculate Confidence Score
                confidence_score, weighted_breakdown = confidence_scorer.calculate_score(
                    analysis_result.confidence_breakdown
                )
                analysis_result.confidence_score = confidence_score
                analysis_result.confidence_breakdown = weighted_breakdown

                # Debug mode: log detailed breakdown
                if debug_mode:
                    log_score_breakdown(
                        logger, symbol, confidence_score,
                        weighted_breakdown, weights,
                    )

                # Determine V2 tier
                alert_tier = determine_tier(confidence_score, alert_tiers)

                logger.info(
                    "%s: %s signal | Score: %.1f%% | Tier: %s",
                    symbol,
                    analysis_result.signal_direction,
                    confidence_score,
                    alert_tier,
                )

                # 3. Apply Filters
                if news_filter.is_news_approaching(current_utc):
                    logger.warning(
                        "%s: Skipping — high-impact news approaching.", symbol
                    )
                    heartbeat.increment_scans()
                    continue

                session_quality = session_filter.get_session_quality(current_utc)
                logger.debug("Session quality: %s", session_quality)

                # 4. Check Deduplication
                market_structure = "bullish" if analysis_result.signal_direction == "BUY" else "bearish"

                if not deduplicator.should_send(
                    symbol=symbol,
                    signal_direction=analysis_result.signal_direction,
                    confidence_score=confidence_score,
                    alert_tier=alert_tier,
                    market_structure=market_structure,
                ):
                    logger.debug(
                        "%s: Alert suppressed by deduplication.", symbol
                    )
                    heartbeat.increment_scans()
                    continue

                # 5. Record Statistics
                reasons, missing_conditions = build_reasons(
                    weighted_breakdown, weights
                )
                stats_tracker.record_setup(
                    symbol=symbol,
                    direction=analysis_result.signal_direction,
                    confidence_score=confidence_score,
                    alert_tier=alert_tier,
                    timeframe=lower_timeframe,
                    reasons=reasons,
                    entry_zone_start=analysis_result.entry_zone_start,
                    entry_zone_end=analysis_result.entry_zone_end,
                    stop_loss=analysis_result.stop_loss,
                    take_profits=analysis_result.take_profits,
                    score_breakdown=weighted_breakdown,
                )

                # 6. Determine Action Based on Tier
                tier_config = telegram_config.get("alert_options", {})

                if alert_tier == "IGNORED" or confidence_score < alert_tiers.get("ignore_below", 60):
                    logger.debug(
                        "%s: Score too low (%.1f%%). Ignored.",
                        symbol, confidence_score,
                    )

                elif alert_tier == "WATCHLIST" and tier_config.get("send_watchlist", True):
                    _handle_tiered_alert(
                        symbol=symbol,
                        result=analysis_result,
                        confidence_score=confidence_score,
                        weighted_breakdown=weighted_breakdown,
                        alert_tier=alert_tier,
                        reasons=reasons,
                        missing_conditions=missing_conditions,
                        session_quality=session_quality,
                        telegram_bot=telegram_bot,
                        alert_formatter=alert_formatter,
                        chart_generator=chart_generator,
                        chart_config=chart_config,
                        tier_config=tier_config,
                        heartbeat=heartbeat,
                        weights=weights,
                        debug_mode=debug_mode,
                    )

                elif alert_tier == "POTENTIAL" and tier_config.get("send_potential", True):
                    _handle_tiered_alert(
                        symbol=symbol,
                        result=analysis_result,
                        confidence_score=confidence_score,
                        weighted_breakdown=weighted_breakdown,
                        alert_tier=alert_tier,
                        reasons=reasons,
                        missing_conditions=missing_conditions,
                        session_quality=session_quality,
                        telegram_bot=telegram_bot,
                        alert_formatter=alert_formatter,
                        chart_generator=chart_generator,
                        chart_config=chart_config,
                        tier_config=tier_config,
                        heartbeat=heartbeat,
                        weights=weights,
                        debug_mode=debug_mode,
                    )

                elif alert_tier == "HIGH_PROBABILITY" and tier_config.get("send_high_prob", True):
                    _handle_tiered_alert(
                        symbol=symbol,
                        result=analysis_result,
                        confidence_score=confidence_score,
                        weighted_breakdown=weighted_breakdown,
                        alert_tier=alert_tier,
                        reasons=reasons,
                        missing_conditions=missing_conditions,
                        session_quality=session_quality,
                        telegram_bot=telegram_bot,
                        alert_formatter=alert_formatter,
                        chart_generator=chart_generator,
                        chart_config=chart_config,
                        tier_config=tier_config,
                        heartbeat=heartbeat,
                        weights=weights,
                        debug_mode=debug_mode,
                    )

                elif alert_tier == "PREMIUM" and tier_config.get("send_premium", True):
                    _handle_tiered_alert(
                        symbol=symbol,
                        result=analysis_result,
                        confidence_score=confidence_score,
                        weighted_breakdown=weighted_breakdown,
                        alert_tier=alert_tier,
                        reasons=reasons,
                        missing_conditions=missing_conditions,
                        session_quality=session_quality,
                        telegram_bot=telegram_bot,
                        alert_formatter=alert_formatter,
                        chart_generator=chart_generator,
                        chart_config=chart_config,
                        tier_config=tier_config,
                        heartbeat=heartbeat,
                        weights=weights,
                        debug_mode=debug_mode,
                    )

                else:
                    logger.debug(
                        "%s: Tier %s not enabled for sending.",
                        symbol, alert_tier,
                    )

                heartbeat.increment_scans()

            except BinanceClientError as e:
                logger.error("Binance API error for %s: %s", symbol, e)
                if telegram_enabled:
                    telegram_bot.send_error(
                        f"Binance API Error for {symbol}: {e}\n"
                        "Please check API connectivity."
                    )
            except Exception as e:
                logger.error(
                    "Unexpected error for %s: %s", symbol, e, exc_info=True
                )
                if telegram_enabled:
                    telegram_bot.send_error(
                        f"Unexpected error processing {symbol}: {e}"
                    )

        # --- Sleep until next scan ---
        if running:
            logger.info(
                "Scan complete. Next scan in %d seconds.", scan_interval
            )
            for _ in range(scan_interval):
                if not running:
                    break
                time.sleep(1)

    # --- Graceful Shutdown ---
    logger.info("=" * 60)
    logger.info("Graceful shutdown initiated...")
    heartbeat.stop()
    logger.info(
        "Final stats: scans=%d, alerts=%d, uptime=%.0fs",
        heartbeat.scan_count,
        heartbeat.alert_count,
        heartbeat.uptime_seconds,
    )
    logger.info("=" * 60)
    logger.info("Professional AI Trading Analysis System V2.0 — Stopped")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Tiered alert handler (extracted for clarity)
# ---------------------------------------------------------------------------
def _handle_tiered_alert(
    symbol: str,
    result: AnalysisResult,
    confidence_score: float,
    weighted_breakdown: Dict[str, float],
    alert_tier: str,
    reasons: List[str],
    missing_conditions: List[str],
    session_quality: str,
    telegram_bot: TelegramBot,
    alert_formatter: AlertFormatter,
    chart_generator: ChartGenerator,
    chart_config: dict,
    tier_config: dict,
    heartbeat: HeartbeatSystem,
    weights: Dict[str, float],
    debug_mode: bool,
) -> None:
    """Process a tiered alert: generate chart, format message, send via Telegram.

    Args:
        symbol: Trading pair symbol.
        result: Analysis result with trade setup details.
        confidence_score: Calculated confidence score.
        weighted_breakdown: Per-component weighted scores.
        alert_tier: Tier label.
        reasons: List of found condition labels.
        missing_conditions: List of missing condition labels.
        session_quality: Current session quality descriptor.
        telegram_bot: TelegramBot instance.
        alert_formatter: AlertFormatter instance.
        chart_generator: ChartGenerator instance.
        chart_config: Chart configuration from YAML.
        tier_config: Telegram alert_options config.
        heartbeat: HeartbeatSystem instance for counter updates.
        weights: Scoring weights for breakdown formatting.
        debug_mode: Whether debug mode is active.
    """
    logger = logging.getLogger(__name__)

    # Determine risk level
    if confidence_score >= 90:
        risk_level = "Low"
    elif confidence_score >= 80:
        risk_level = "Medium-Low"
    elif confidence_score >= 70:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Generate chart if enabled
    chart_path = None
    if tier_config.get("include_chart", True) and result.raw_data is not None:
        try:
            chart_data = extract_chart_data(result)
            timeframe_label = (
                f"{result.timeframe}"
            )
            chart_path = chart_generator.generate_chart(
                ohlcv_df=result.raw_data,
                analysis_data=chart_data,
                symbol=symbol,
                timeframe=timeframe_label,
                signal_direction=result.signal_direction,
                confidence_score=confidence_score,
                reasons=reasons,
            )
            logger.info("Chart saved: %s", chart_path)
        except Exception as e:
            logger.error("Chart generation failed: %s", e)

    # Format alert message
    timeframe_str = result.timeframe
    alert_message = alert_formatter.format_alert_message(
        signal_direction=result.signal_direction,
        symbol=symbol,
        timeframe=timeframe_str,
        entry_zone_start=result.entry_zone_start or 0,
        entry_zone_end=result.entry_zone_end or 0,
        stop_loss=result.stop_loss or 0,
        take_profits=result.take_profits or [],
        risk_reward_ratios=result.risk_reward_ratios or [],
        confidence_score=confidence_score,
        confidence_breakdown=weighted_breakdown,
        human_explanation=result.human_explanation,
        alert_tier=alert_tier,
        score_weights=weights,
        component_raw_scores={
            k: v / weights.get(k, 1) if weights.get(k, 0) > 0 else 0
            for k, v in weighted_breakdown.items()
        },
        reasons=reasons,
        missing_conditions=missing_conditions,
        risk_level=risk_level,
    )

    # Send via Telegram
    if chart_path and os.path.isfile(chart_path):
        success = telegram_bot.send_photo(
            photo_path=chart_path,
            caption=alert_message,
        )
    else:
        success = telegram_bot.send_message(alert_message)

    if success:
        logger.info("Alert sent for %s (%s, score=%.1f%%).", symbol, alert_tier, confidence_score)
        heartbeat.increment_alerts()
    else:
        logger.error("Failed to send alert for %s.", symbol)

    if debug_mode:
        logger.info("Alert message (debug):\n%s", alert_message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
