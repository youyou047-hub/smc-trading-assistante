"""Professional AI Trading Analysis System - Main Runner

This is the main entry point for the trading analysis system.
It orchestrates the scanning loop: fetch data → analyze → score → filter → chart → alert.

The analysis logic is independent from scheduling. This module only handles
the scheduling loop and component coordination. The analysis engine can be
imported and called independently for n8n or other integrations.

Usage:
    python main.py
"""

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import yaml
from dotenv import load_dotenv

from src.utils.logger import setup_logger
from src.data.binance_client import BinanceClient, BinanceClientError
from src.analysis.engine import analyze, AnalysisResult
from src.scoring.confidence import ConfidenceScorer
from src.filters.news import NewsFilter
from src.filters.sessions import SessionFilter
from src.charts.generator import ChartGenerator
from src.alerts.telegram import TelegramBot
from src.alerts.formatter import AlertFormatter


# --- Global State ---
running = True


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Loads configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def signal_handler(signum, frame):
    """Handles graceful shutdown signals (SIGINT, SIGTERM)."""
    global running
    print(f"\n[!] Signal {signum} received. Initiating graceful shutdown...")
    running = False


def extract_chart_data(analysis_result: AnalysisResult) -> dict:
    """Extracts chart annotation data from analysis results.

    Args:
        analysis_result: The completed analysis result.

    Returns:
        dict: Chart annotation data for the chart generator.
    """
    chart_data = {
        "liquidity_sweeps": [],
        "bos_lines": [],
        "choch_lines": [],
        "fvgs": [],
        "order_blocks": [],
        "entry_zone_start": analysis_result.entry_zone_start,
        "entry_zone_end": analysis_result.entry_zone_end,
        "stop_loss": analysis_result.stop_loss,
        "take_profits": analysis_result.take_profits or [],
    }

    if analysis_result.raw_data is None:
        return chart_data

    df = analysis_result.raw_data

    for idx, row in df.iterrows():
        # Extract FVGs
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

        # Extract Order Blocks
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

        # Extract Liquidity Sweeps
        if row.get("liquidity_sweep_bullish"):
            chart_data["liquidity_sweeps"].append((row["low"], idx))
        elif row.get("liquidity_sweep_bearish"):
            chart_data["liquidity_sweeps"].append((row["high"], idx))

        # Extract BOS
        if row.get("bos_bullish"):
            chart_data["bos_lines"].append((row["high"], idx, "bullish"))
        elif row.get("bos_bearish"):
            chart_data["bos_lines"].append((row["low"], idx, "bearish"))

        # Extract CHoCH
        if row.get("choch_bullish"):
            chart_data["choch_lines"].append((row["high"], idx, "bullish"))
        elif row.get("choch_bearish"):
            chart_data["choch_lines"].append((row["low"], idx, "bearish"))

    return chart_data


def main():
    """Main execution loop for the trading analysis system."""
    global running

    # Load environment variables from .env file
    load_dotenv()

    # Load configuration
    config = load_config()

    # Setup Logger
    log_file = config["logging"]["file"]
    log_level = config["logging"]["level"]
    logger = setup_logger(log_file, log_level)
    logger.info("=" * 60)
    logger.info("Professional AI Trading Analysis System - Starting")
    logger.info("=" * 60)

    # Initialize components
    binance_url = config.get("binance", {}).get("base_url", "https://data-api.binance.vision/api/v3")
    binance_client = BinanceClient(base_url=binance_url)

    confidence_scorer = ConfidenceScorer(
        weights=config["scoring"]["weights"],
        thresholds=config["scoring"]["thresholds"],
    )

    news_filter = NewsFilter(
        enabled=config["news_filter"]["enabled"],
        buffer_minutes=config["news_filter"]["buffer_minutes"],
        high_impact_events=config["news_filter"]["high_impact_events"],
    )

    session_filter = SessionFilter(
        sessions_config=config["sessions"],
    )

    chart_output_dir = config.get("charts", {}).get("output_dir", "charts")
    chart_generator = ChartGenerator(output_dir=chart_output_dir)

    # Telegram setup (env vars override config)
    telegram_enabled = config["telegram"]["enabled"]
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", config["telegram"]["bot_token"])
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", config["telegram"]["chat_id"])
    telegram_bot = TelegramBot(telegram_bot_token, telegram_chat_id)
    alert_formatter = AlertFormatter()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Extract trading parameters
    scan_interval = config["trading"]["scan_interval_seconds"]
    symbols = config["trading"]["symbols"]
    higher_timeframe = config["trading"]["timeframes"]["higher"]
    lower_timeframe = config["trading"]["timeframes"]["lower"]
    stats_log_file = config["logging"]["stats_file"]
    kline_limit = config.get("binance", {}).get("kline_limit", 500)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(stats_log_file), exist_ok=True)
    os.makedirs(chart_output_dir, exist_ok=True)

    logger.info(f"Monitoring symbols: {symbols}")
    logger.info(f"Timeframes: Higher={higher_timeframe}, Lower={lower_timeframe}")
    logger.info(f"Scan interval: {scan_interval}s")
    logger.info(f"Telegram alerts: {'Enabled' if telegram_enabled else 'Disabled'}")

    # --- Main Scanning Loop ---
    while running:
        current_utc_time = datetime.now(timezone.utc)
        logger.info(f"--- Scan started at {current_utc_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")

        for symbol in symbols:
            try:
                # 1. Fetch Data (both timeframes)
                logger.debug(f"Fetching {symbol} data...")
                ohlcv_higher = binance_client.get_ohlcv(symbol, higher_timeframe, limit=kline_limit)
                ohlcv_lower = binance_client.get_ohlcv(symbol, lower_timeframe, limit=kline_limit)

                # 2. Run Analysis Engine
                # Higher timeframe for trend, lower timeframe for entry
                analysis_result = analyze(ohlcv_lower, symbol, lower_timeframe)

                # If no signal detected, skip
                if analysis_result.signal_direction is None:
                    logger.info(f"{symbol}: No signal detected.")
                    continue

                # 3. Calculate Confidence Score
                confidence_score, weighted_breakdown = confidence_scorer.calculate_score(
                    analysis_result.confidence_breakdown
                )
                alert_tier = confidence_scorer.get_alert_tier(confidence_score)
                analysis_result.confidence_score = confidence_score
                analysis_result.confidence_breakdown = weighted_breakdown

                logger.info(
                    f"{symbol}: {analysis_result.signal_direction} signal | "
                    f"Score: {confidence_score:.1f}% | Tier: {alert_tier}"
                )

                # 4. Apply Filters
                # News filter check
                if news_filter.is_news_approaching(current_utc_time):
                    logger.warning(f"{symbol}: Skipping - high-impact news approaching.")
                    continue

                # Session quality check
                session_quality = session_filter.get_session_quality(current_utc_time)
                logger.debug(f"Session quality: {session_quality}")

                # 5. Determine Action Based on Score
                thresholds = config["scoring"]["thresholds"]

                if confidence_score >= thresholds["high_probability_min"]:
                    # Score >= 80: Generate chart and send alert
                    logger.info(f"{symbol}: Alert threshold met! Generating chart...")

                    # Generate annotated chart
                    chart_data = extract_chart_data(analysis_result)
                    chart_path = chart_generator.generate_chart(
                        ohlcv_lower,
                        chart_data,
                        symbol,
                        lower_timeframe,
                        analysis_result.signal_direction,
                        confidence_score,
                    )
                    logger.info(f"Chart saved: {chart_path}")

                    # Format alert message
                    alert_message = alert_formatter.format_alert_message(
                        signal_direction=analysis_result.signal_direction,
                        symbol=symbol,
                        timeframe=f"{higher_timeframe} / {lower_timeframe}",
                        entry_zone_start=analysis_result.entry_zone_start,
                        entry_zone_end=analysis_result.entry_zone_end,
                        stop_loss=analysis_result.stop_loss,
                        take_profits=analysis_result.take_profits,
                        risk_reward_ratios=analysis_result.risk_reward_ratios,
                        confidence_score=confidence_score,
                        confidence_breakdown=weighted_breakdown,
                        human_explanation=analysis_result.human_explanation,
                        alert_tier=alert_tier,
                    )

                    # Send Telegram alert with chart
                    if telegram_enabled:
                        telegram_bot.send_photo(chart_path, caption=alert_message)
                        logger.info(f"Telegram alert sent for {symbol}.")
                    else:
                        logger.info(f"Telegram disabled. Alert logged only.")
                        logger.info(f"Alert message:\n{alert_message}")

                elif confidence_score >= thresholds["record_only_min"]:
                    # Score 70-79: Record for statistics only
                    logger.info(
                        f"{symbol}: Sub-threshold signal ({confidence_score:.1f}%). "
                        f"Recording for statistics."
                    )
                    stats_entry = {
                        "timestamp": current_utc_time.isoformat(),
                        "symbol": symbol,
                        "timeframe": lower_timeframe,
                        "signal_direction": analysis_result.signal_direction,
                        "confidence_score": round(confidence_score, 2),
                        "confidence_breakdown": {
                            k: round(v, 2) for k, v in weighted_breakdown.items()
                        },
                        "human_explanation": analysis_result.human_explanation,
                        "alert_tier": alert_tier,
                    }
                    with open(stats_log_file, "a") as f:
                        json.dump(stats_entry, f)
                        f.write("\n")

                else:
                    # Score < 70: Ignore completely
                    logger.debug(f"{symbol}: Score too low ({confidence_score:.1f}%). Ignored.")

            except BinanceClientError as e:
                logger.error(f"Binance API error for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error for {symbol}: {e}", exc_info=True)

        # Sleep until next scan
        if running:
            logger.info(f"Scan complete. Next scan in {scan_interval} seconds.")
            # Use interruptible sleep
            for _ in range(scan_interval):
                if not running:
                    break
                time.sleep(1)

    logger.info("=" * 60)
    logger.info("Professional AI Trading Analysis System - Stopped")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
