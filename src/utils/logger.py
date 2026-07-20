"""Logger V2.0 — Production-grade logging with debug mode toggle.

Features:
- Dual output: file (with rotation) and console (stderr)
- Configurable log level (DEBUG / INFO / WARNING / ERROR)
- Debug mode: when enabled, logs score breakdowns and raw analysis data
- Named logger ``trading_assistant`` for backward compatibility with V1
- Automatic directory creation for log files
- Duplicate-handler prevention (idempotent ``setup_logger`` calls)

Usage:
    from src.utils.logger import setup_logger
    logger = setup_logger(
        log_file="logs/trading_assistant.log",
        level="INFO",
        debug_mode=False,
        max_bytes=10 * 1024 * 1024,
        backup_count=5,
    )
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


# Module-level logger name — kept consistent with V1
LOGGER_NAME = "trading_assistant"

# Default format strings
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s"
DEBUG_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)


# ---------------------------------------------------------------------------
# setup_logger
# ---------------------------------------------------------------------------
def setup_logger(
    log_file: str = "logs/trading_assistant.log",
    level: str = "INFO",
    *,
    debug_mode: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure and return the ``trading_assistant`` logger.

    The logger is idempotent: calling ``setup_logger`` multiple times
    will not duplicate handlers.  If ``debug_mode`` is ``True``, the
    log level is forced to ``DEBUG`` regardless of the ``level``
    argument, and the file format includes function names.

    Args:
        log_file: Path to the log file.
        level: Base log level string (``DEBUG``, ``INFO``, ``WARNING``,
               ``ERROR``).  Overridden by ``debug_mode`` if ``True``.
        debug_mode: If ``True``, force ``DEBUG`` level and verbose format.
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of rotated backup files to keep.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    # Effective log level
    if debug_mode:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, level.upper(), logging.INFO)

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Get or create the named logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)
    logger.propagate = False  # Do not propagate to root logger

    # ---- Console handler (stdout/stderr) ----
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT))
        logger.addHandler(console_handler)

    # ---- File handler (rotating) ----
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        file_fmt = DEBUG_FILE_FORMAT if debug_mode else FILE_FORMAT
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(file_fmt))
        logger.addHandler(file_handler)

    logger.info(
        "Logger initialised | file=%s | level=%s | debug=%s",
        log_file, logging.getLevelName(log_level), debug_mode,
    )
    return logger


# ---------------------------------------------------------------------------
# Helper: log score breakdown in debug mode
# ---------------------------------------------------------------------------
def log_score_breakdown(
    logger: logging.Logger,
    symbol: str,
    score: float,
    breakdown: dict,
    weights: Optional[dict] = None,
) -> None:
    """Log a detailed score breakdown (only emitted at DEBUG level).

    Args:
        logger: The configured logger instance.
        symbol: Trading pair symbol.
        score: Final confidence score.
        breakdown: Weighted contribution per component.
        weights: *(optional)* Configured weights for raw-score recovery.
    """
    if logger.level > logging.DEBUG:
        return

    logger.debug("=" * 60)
    logger.debug("Score Breakdown — %s", symbol)
    logger.debug("-" * 60)
    for comp, contribution in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        weight = weights.get(comp, 0) if weights else 0
        raw_pct = (contribution / weight * 100) if weight > 0 else 0
        logger.debug(
            "  %-35s  %6.1f / %6.1f  (%5.1f%%)",
            comp.replace("_", " ").title(), contribution, weight, raw_pct,
        )
    logger.debug("-" * 60)
    logger.debug("  %-35s  %6.1f", "FINAL SCORE", score)
    logger.debug("=" * 60)


# ---------------------------------------------------------------------------
# Module-level self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Basic test
    logger = setup_logger(log_file="/tmp/v2_test.log", level="INFO")
    logger.info("This is an INFO message.")
    logger.debug("This DEBUG message should NOT appear.")

    # Debug mode test
    logger_debug = setup_logger(
        log_file="/tmp/v2_test_debug.log",
        level="DEBUG",
        debug_mode=True,
    )
    logger_debug.debug("This DEBUG message SHOULD appear.")
    logger_debug.info("INFO with debug mode.")

    # Score breakdown test
    log_score_breakdown(
        logger_debug,
        "BTCUSDT",
        92.5,
        {
            "market_structure_alignment": 20.0,
            "liquidity_sweep": 18.0,
            "bos_choch_confirmation": 15.0,
            "fair_value_gap": 13.5,
            "fresh_order_block": 8.0,
            "premium_discount_zone": 5.0,
            "confirmation_candle": 4.0,
            "trading_session_quality": 5.0,
            "news_filter": 4.0,
        },
        weights={
            "market_structure_alignment": 20,
            "liquidity_sweep": 20,
            "bos_choch_confirmation": 15,
            "fair_value_gap": 15,
            "fresh_order_block": 10,
            "premium_discount_zone": 5,
            "confirmation_candle": 5,
            "trading_session_quality": 5,
            "news_filter": 5,
        },
    )

    print("Logger tests complete. Check /tmp/v2_test*.log")
