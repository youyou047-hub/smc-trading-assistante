"""
Professional AI Trading Analysis System V2.0
Module: Confirmation Candle Detection

Upgraded from V1.0 with comprehensive candle pattern detection:
- Bullish / Bearish Engulfing patterns
- Pin Bars with proper wick-to-body ratio validation
- Rejection Candles (long wicks indicating price rejection)
- Strong Displacement Candles (impulsive institutional moves)
- Momentum Confirmation (consecutive strong candles)
- Quality scoring and weak candle rejection

Maintains backward-compatible DataFrame interface (pandas OHLCV as input).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================
# Data Structures
# ============================================================

class CandlePattern(Enum):
    """Types of confirmation candle patterns."""
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    BULLISH_PIN_BAR = "bullish_pin_bar"
    BEARISH_PIN_BAR = "bearish_pin_bar"
    BULLISH_REJECTION = "bullish_rejection"
    BEARISH_REJECTION = "bearish_rejection"
    STRONG_DISPLACEMENT_BULLISH = "displacement_bullish"
    STRONG_DISPLACEMENT_BEARISH = "displacement_bearish"
    MOMENTUM_BULLISH = "momentum_bullish"
    MOMENTUM_BEARISH = "momentum_bearish"
    DOJI = "doji"
    MARUBOZU_BULLISH = "marubozu_bullish"
    MARUBOZU_BEARISH = "marubozu_bearish"


@dataclass
class CandleSignal:
    """Represents a detected confirmation candle signal with quality metadata."""
    index: int
    pattern: CandlePattern
    strength: float  # 0-1 quality score
    body_size: float
    wick_upper: float
    wick_lower: float
    candle_range: float
    body_ratio: float  # body / range
    volume_ratio: float
    is_strong: bool = False  # Meets minimum quality threshold
    price_level: float = 0.0  # Close price for context

    def as_dict(self) -> Dict:
        return {
            "index": self.index,
            "pattern": self.pattern.value,
            "strength": self.strength,
            "body_ratio": self.body_ratio,
            "volume_ratio": self.volume_ratio,
            "is_strong": self.is_strong,
            "price_level": self.price_level,
        }


# ============================================================
# Helper Functions
# ============================================================

def _calculate_candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates common candle metrics for all pattern detection.

    Returns DataFrame with added columns:
        'candle_range', 'candle_body', 'body_ratio',
        'wick_upper', 'wick_lower', 'wick_upper_ratio', 'wick_lower_ratio',
        'is_bullish', 'is_bearish', 'volume_ratio', 'avg_range'
    """
    df = df.copy()
    df["candle_range"] = df["high"] - df["low"]
    df["candle_body"] = (df["close"] - df["open"]).abs()
    df["body_ratio"] = np.where(
        df["candle_range"] > 0,
        df["candle_body"] / df["candle_range"],
        0.0
    )
    df["wick_upper"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["wick_lower"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["wick_upper_ratio"] = np.where(
        df["candle_range"] > 0,
        df["wick_upper"] / df["candle_range"],
        0.0
    )
    df["wick_lower_ratio"] = np.where(
        df["candle_range"] > 0,
        df["wick_lower"] / df["candle_range"],
        0.0
    )
    df["is_bullish"] = df["close"] > df["open"]
    df["is_bearish"] = df["close"] < df["open"]

    # Volume ratio vs average
    avg_vol = df["volume"].mean() if "volume" in df.columns and df["volume"].mean() > 0 else 1.0
    df["volume_ratio"] = df["volume"] / avg_vol if "volume" in df.columns and avg_vol > 0 else 1.0

    # Average range for comparison
    df["avg_range"] = df["candle_range"].rolling(window=10, min_periods=1).mean().shift(1)
    df["avg_range"] = df["avg_range"].fillna(df["candle_range"].mean())

    return df


def _score_candle_signal(
    body_ratio: float,
    wick_ratio: float,
    volume_ratio: float,
    range_vs_avg: float,
) -> float:
    """Calculates a composite quality score for a candle signal.

    Higher score = stronger, more reliable candle pattern.

    Args:
        body_ratio: Body as fraction of total range (0-1).
        wick_ratio: Wick as fraction of total range (0-1).
        volume_ratio: Volume vs average (0+).
        range_vs_avg: Candle range vs average range (0+).

    Returns:
        Float score 0.0-1.0.
    """
    # Body strength component (0.35 weight)
    if body_ratio >= 0.8:
        body_score = 1.0
    elif body_ratio >= 0.6:
        body_score = 0.8
    elif body_ratio >= 0.4:
        body_score = 0.5
    else:
        body_score = 0.2

    # Wick quality component (0.25 weight) - relevant for rejection/pin bars
    if wick_ratio >= 0.6:
        wick_score = 1.0
    elif wick_ratio >= 0.5:
        wick_score = 0.8
    elif wick_ratio >= 0.3:
        wick_score = 0.5
    else:
        wick_score = 0.3

    # Volume confirmation (0.25 weight)
    if volume_ratio >= 2.0:
        vol_score = 1.0
    elif volume_ratio >= 1.5:
        vol_score = 0.8
    elif volume_ratio >= 1.0:
        vol_score = 0.6
    else:
        vol_score = 0.3

    # Range significance (0.15 weight)
    if range_vs_avg >= 2.0:
        range_score = 1.0
    elif range_vs_avg >= 1.5:
        range_score = 0.8
    elif range_vs_avg >= 1.0:
        range_score = 0.5
    else:
        range_score = 0.2

    return (
        body_score * 0.35 +
        wick_score * 0.25 +
        vol_score * 0.25 +
        range_score * 0.15
    )


# ============================================================
# Engulfing Patterns
# ============================================================

def find_engulfing_candles(
    df: pd.DataFrame,
    min_body_ratio: float = 1.1,
    min_range_multiplier: float = 1.2,
) -> pd.DataFrame:
    """Identifies Bullish and Bearish Engulfing patterns.

    Bullish Engulfing: A bullish candle that completely engulfs the previous
    bearish candle (body extends beyond both open and close of prior candle).

    Bearish Engulfing: A bearish candle that completely engulfs the previous
    bullish candle.

    V2.0 improvements over V1.0:
    - Configurable body and range thresholds
    - Proper engulfing validation (body must extend beyond prior body)
    - Volume confirmation requirement

    Args:
        df: OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume'.
        min_body_ratio: Engulfing candle body must be >= this x previous body.
        min_range_multiplier: Engulfing range must be >= this x previous range.

    Returns:
        DataFrame with 'bullish_engulfing' and 'bearish_engulfing' columns.
    """
    df = _calculate_candle_metrics(df)

    df["bullish_engulfing"] = False
    df["bearish_engulfing"] = False

    for i in range(1, len(df)):
        prev_open = df["open"].iloc[i - 1]
        prev_close = df["close"].iloc[i - 1]
        curr_open = df["open"].iloc[i]
        curr_close = df["close"].iloc[i]
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        prev_range = df["candle_range"].iloc[i - 1]
        curr_range = df["candle_range"].iloc[i]

        # --- Bullish Engulfing ---
        # Previous candle is bearish, current is bullish
        if prev_close < prev_open and curr_close > curr_open:
            # Current body must engulf previous body
            body_engulfs = curr_open <= prev_close and curr_close >= prev_open
            # Body size comparison
            body_ratio_ok = curr_body >= prev_body * min_body_ratio
            # Range comparison
            range_ok = curr_range >= prev_range * min_range_multiplier

            if body_engulfs and body_ratio_ok and range_ok:
                df.loc[df.index[i], "bullish_engulfing"] = True

        # --- Bearish Engulfing ---
        # Previous candle is bullish, current is bearish
        elif prev_close > prev_open and curr_close < curr_open:
            body_engulfs = curr_open >= prev_close and curr_close <= prev_open
            body_ratio_ok = curr_body >= prev_body * min_body_ratio
            range_ok = curr_range >= prev_range * min_range_multiplier

            if body_engulfs and body_ratio_ok and range_ok:
                df.loc[df.index[i], "bearish_engulfing"] = True

    return df.drop(
        columns=["candle_range", "candle_body", "body_ratio",
                 "wick_upper", "wick_lower", "wick_upper_ratio",
                 "wick_lower_ratio", "is_bullish", "is_bearish",
                 "volume_ratio", "avg_range"],
        errors='ignore'
    )


# ============================================================
# Pin Bar Detection
# ============================================================

def find_pin_bars(
    df: pd.DataFrame,
    min_wick_body_ratio: float = 2.5,
    min_total_range_pct: float = 0.005,
    wick_percentage: float = 0.67,
) -> pd.DataFrame:
    """Identifies Bullish and Bearish Pin Bars.

    A Pin Bar is a candle with a small body and a long wick on one side,
    indicating strong rejection from a price level.

    V2.0 improvements over V1.0:
    - Configurable wick-to-body ratio (default 2.5x vs V1.0's 2.0x)
    - Minimum total range requirement (ignores tiny candles)
    - Wick percentage of total range validation
    - Volume confirmation

    Bullish Pin Bar: Long lower wick, small body near the top of the range.
    Bearish Pin Bar: Long upper wick, small body near the bottom of the range.

    Args:
        df: OHLCV DataFrame.
        min_wick_body_ratio: Wick must be >= this x body length.
        min_total_range_pct: Minimum candle range as % of price.
        wick_percentage: Wick must be >= this fraction of total range.

    Returns:
        DataFrame with 'pin_bar_bullish' and 'pin_bar_bearish' columns.
    """
    df = _calculate_candle_metrics(df)

    df["pin_bar_bullish"] = False
    df["pin_bar_bearish"] = False

    avg_close = df["close"].mean() if len(df) > 0 else 1.0

    for i in range(len(df)):
        body = df["candle_body"].iloc[i]
        wick_upper = df["wick_upper"].iloc[i]
        wick_lower = df["wick_lower"].iloc[i]
        total_range = df["candle_range"].iloc[i]

        if total_range <= 0 or body == 0:
            continue

        # Minimum range check (ignore insignificant candles)
        if total_range / avg_close < min_total_range_pct:
            continue

        # --- Bullish Pin Bar: Long lower wick ---
        if wick_lower > 0 and body > 0:
            wick_body = wick_lower / body
            wick_range_pct = wick_lower / total_range

            if wick_body >= min_wick_body_ratio and wick_range_pct >= wick_percentage:
                df.loc[df.index[i], "pin_bar_bullish"] = True

        # --- Bearish Pin Bar: Long upper wick ---
        if wick_upper > 0 and body > 0:
            wick_body = wick_upper / body
            wick_range_pct = wick_upper / total_range

            if wick_body >= min_wick_body_ratio and wick_range_pct >= wick_percentage:
                df.loc[df.index[i], "pin_bar_bearish"] = True

    return df.drop(
        columns=["candle_range", "candle_body", "body_ratio",
                 "wick_upper", "wick_lower", "wick_upper_ratio",
                 "wick_lower_ratio", "is_bullish", "is_bearish",
                 "volume_ratio", "avg_range"],
        errors='ignore'
    )


# ============================================================
# Rejection Candles
# ============================================================

def find_rejection_candles(
    df: pd.DataFrame,
    min_wick_body_ratio: float = 2.0,
    min_range_pct: float = 0.003,
) -> pd.DataFrame:
    """Identifies Rejection Candles (long wicks indicating price rejection).

    A rejection candle has a long wick relative to its body, showing that
    price was aggressively pushed away from a certain level.

    V2.0 improvements over V1.0:
    - Configurable minimum range percentage
    - Better wick-to-body ratio calculation
    - Handles doji candles properly

    Args:
        df: OHLCV DataFrame.
        min_wick_body_ratio: Wick must be >= this x body length.
        min_range_pct: Minimum candle range as % of price.

    Returns:
        DataFrame with 'rejection_bullish' and 'rejection_bearish' columns.
    """
    df = _calculate_candle_metrics(df)

    df["rejection_bullish"] = False
    df["rejection_bearish"] = False

    avg_close = df["close"].mean() if len(df) > 0 else 1.0

    for i in range(len(df)):
        body = df["candle_body"].iloc[i]
        wick_upper = df["wick_upper"].iloc[i]
        wick_lower = df["wick_lower"].iloc[i]
        total_range = df["candle_range"].iloc[i]

        # Skip doji candles (no meaningful body)
        if body == 0:
            continue

        # Minimum range check
        if avg_close > 0 and total_range / avg_close < min_range_pct:
            continue

        # --- Bullish Rejection: Long lower wick (price rejected from below) ---
        if body > 0:
            lower_wick_body = wick_lower / body
            if lower_wick_body >= min_wick_body_ratio:
                df.loc[df.index[i], "rejection_bullish"] = True

        # --- Bearish Rejection: Long upper wick (price rejected from above) ---
        if body > 0:
            upper_wick_body = wick_upper / body
            if upper_wick_body >= min_wick_body_ratio:
                df.loc[df.index[i], "rejection_bearish"] = True

    return df.drop(
        columns=["candle_range", "candle_body", "body_ratio",
                 "wick_upper", "wick_lower", "wick_upper_ratio",
                 "wick_lower_ratio", "is_bullish", "is_bearish",
                 "volume_ratio", "avg_range"],
        errors='ignore'
    )


# ============================================================
# Displacement Candles
# ============================================================

def find_displacement_candles(
    df: pd.DataFrame,
    min_body_pct: float = 0.7,
    min_range_vs_avg: float = 1.5,
    require_volume: bool = True,
    volume_multiplier: float = 1.3,
) -> pd.DataFrame:
    """Identifies Strong Displacement Candles.

    A displacement candle is a large, impulsive candle that represents
    institutional order flow. It must have:
    - Large body relative to range (minimal wicks)
    - Range significantly larger than average
    - Above-average volume (optional but preferred)

    V2.0 improvements over V1.0:
    - Configurable body percentage threshold
    - Configurable range vs average multiplier
    - Volume confirmation (configurable)
    - Separate bullish and bearish displacement

    Args:
        df: OHLCV DataFrame.
        min_body_pct: Body must be >= this % of total range.
        min_range_vs_avg: Range must be >= this x average range.
        require_volume: Whether volume must exceed threshold.
        volume_multiplier: Volume must be >= this x average volume.

    Returns:
        DataFrame with 'displacement_bullish' and 'displacement_bearish' columns.
    """
    df = _calculate_candle_metrics(df)

    df["displacement_bullish"] = False
    df["displacement_bearish"] = False

    avg_vol = df["volume"].mean() if "volume" in df.columns and df["volume"].mean() > 0 else 1.0

    for i in range(len(df)):
        body_ratio = df["body_ratio"].iloc[i]
        range_val = df["candle_range"].iloc[i]
        avg_range = df["avg_range"].iloc[i]
        vol = df["volume"].iloc[i]

        # Must have significant body
        if body_ratio < min_body_pct:
            continue

        # Must be significantly larger than average
        if avg_range > 0 and range_val < avg_range * min_range_vs_avg:
            continue

        # Volume check
        if require_volume and avg_vol > 0 and vol < avg_vol * volume_multiplier:
            continue

        if df["is_bullish"].iloc[i]:
            df.loc[df.index[i], "displacement_bullish"] = True
        elif df["is_bearish"].iloc[i]:
            df.loc[df.index[i], "displacement_bearish"] = True

    return df.drop(
        columns=["candle_range", "candle_body", "body_ratio",
                 "wick_upper", "wick_lower", "wick_upper_ratio",
                 "wick_lower_ratio", "is_bullish", "is_bearish",
                 "volume_ratio", "avg_range"],
        errors='ignore'
    )


# ============================================================
# Momentum Confirmation
# ============================================================

def find_momentum_confirmation(
    df: pd.DataFrame,
    consecutive_count: int = 2,
    min_close_in_direction: float = 0.6,
) -> pd.DataFrame:
    """Identifies Momentum Confirmation (consecutive strong candles).

    Momentum is confirmed when multiple consecutive candles close
    strongly in the same direction, indicating sustained institutional
    participation.

    V2.0 improvement: New pattern type not present in V1.0.
    Requires configurable number of consecutive strong candles
    with close position validation.

    Args:
        df: OHLCV DataFrame.
        consecutive_count: Need this many consecutive strong candles.
        min_close_in_direction: Close must be in top 60% (bullish)
            or bottom 40% (bearish) of candle range.

    Returns:
        DataFrame with 'momentum_bullish' and 'momentum_bearish' columns.
    """
    df = _calculate_candle_metrics(df)

    df["momentum_bullish"] = False
    df["momentum_bearish"] = False

    avg_vol = df["volume"].mean() if "volume" in df.columns and df["volume"].mean() > 0 else 1.0

    for i in range(consecutive_count - 1, len(df)):
        # Check last N candles for consecutive momentum
        bullish_streak = True
        bearish_streak = True

        for j in range(i - consecutive_count + 1, i + 1):
            body = df["candle_body"].iloc[j]
            total_range = df["candle_range"].iloc[j]
            close_pos = 0.0
            vol = df["volume"].iloc[j]

            if total_range > 0 and body > 0:
                # Where does close sit in the range?
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    # Bullish: close should be in top portion
                    close_pos = (df["close"].iloc[j] - df["low"].iloc[j]) / total_range
                else:
                    # Bearish: close should be in bottom portion
                    close_pos = (df["high"].iloc[j] - df["close"].iloc[j]) / total_range

                # Volume check
                vol_ok = vol >= avg_vol * 0.8 if avg_vol > 0 else True
            else:
                close_pos = 0.0
                vol_ok = False

            # Bullish momentum: close in top 60% of range
            if not (df["close"].iloc[j] > df["open"].iloc[j] and close_pos >= min_close_in_direction and vol_ok):
                bullish_streak = False

            # Bearish momentum: close in bottom 60% of range
            if not (df["close"].iloc[j] < df["open"].iloc[j] and close_pos >= min_close_in_direction and vol_ok):
                bearish_streak = False

        if bullish_streak:
            df.loc[df.index[i], "momentum_bullish"] = True
        if bearish_streak:
            df.loc[df.index[i], "momentum_bearish"] = True

    return df.drop(
        columns=["candle_range", "candle_body", "body_ratio",
                 "wick_upper", "wick_lower", "wick_upper_ratio",
                 "wick_lower_ratio", "is_bullish", "is_bearish",
                 "volume_ratio", "avg_range"],
        errors='ignore'
    )


# ============================================================
# Confirmation Candle Detection
# ============================================================

def find_confirmation_candles(
    df: pd.DataFrame,
    prev_signal_col: str = "rejection_bullish",
    signal_type: str = "bullish",
) -> pd.DataFrame:
    """Identifies confirmation candles following a previous signal.

    A confirmation candle is one that closes in the expected direction
    after a signal candle (e.g., a bullish candle after a bullish rejection).

    V2.0 improvement: Validates that confirmation candle has meaningful
    body and range, rejecting weak confirmations.

    Args:
        df: DataFrame with a signal column (e.g., 'rejection_bullish').
        prev_signal_col: Column name of the previous signal.
        signal_type: 'bullish' or 'bearish'.

    Returns:
        DataFrame with 'confirmation_bullish' or 'confirmation_bearish' column.
    """
    df = _calculate_candle_metrics(df)

    if signal_type == "bullish":
        df["confirmation_bullish"] = False
        for i in range(1, len(df)):
            if df[prev_signal_col].iloc[i - 1]:
                # Current candle must be bullish with meaningful body
                if df["is_bullish"].iloc[i] and df["body_ratio"].iloc[i] >= 0.3:
                    df.loc[df.index[i], "confirmation_bullish"] = True

    elif signal_type == "bearish":
        df["confirmation_bearish"] = False
        for i in range(1, len(df)):
            if df[prev_signal_col].iloc[i - 1]:
                if df["is_bearish"].iloc[i] and df["body_ratio"].iloc[i] >= 0.3:
                    df.loc[df.index[i], "confirmation_bearish"] = True
    else:
        raise ValueError("signal_type must be 'bullish' or 'bearish'")

    return df.drop(
        columns=["candle_range", "candle_body", "body_ratio",
                 "wick_upper", "wick_lower", "wick_upper_ratio",
                 "wick_lower_ratio", "is_bullish", "is_bearish",
                 "volume_ratio", "avg_range"],
        errors='ignore'
    )


# ============================================================
# Comprehensive Candle Analysis
# ============================================================

def analyze_candles(
    df: pd.DataFrame,
    min_total_score: float = 0.3,
    engulfing_min_body: float = 1.1,
    pin_bar_min_wick_ratio: float = 2.5,
    rejection_min_wick_ratio: float = 2.0,
    displacement_min_body: float = 0.7,
    momentum_consecutive: int = 2,
) -> Dict[str, object]:
    """Complete confirmation candle analysis pipeline.

    Detects all candle patterns, scores them, and rejects weak signals.

    Args:
        df: OHLCV DataFrame.
        min_total_score: Minimum quality score to accept a signal.
        engulfing_min_body: Engulfing body ratio threshold.
        pin_bar_min_wick_ratio: Pin bar wick-to-body ratio.
        rejection_min_wick_ratio: Rejection wick-to-body ratio.
        displacement_min_body: Displacement body percentage.
        momentum_consecutive: Consecutive candles for momentum.

    Returns:
        Dict with:
            'df': Annotated DataFrame
            'signals': List of CandleSignal objects
            'strong_bullish_signals': Bullish signals above threshold
            'strong_bearish_signals': Bearish signals above threshold
            'latest_signal': Most recent strong signal (or None)
    """
    df = _calculate_candle_metrics(df)

    signals: List[CandleSignal] = []
    avg_vol = df["volume"].mean() if "volume" in df.columns and df["volume"].mean() > 0 else 1.0

    for i in range(len(df)):
        body_ratio = df["body_ratio"].iloc[i]
        wick_upper = df["wick_upper"].iloc[i]
        wick_lower = df["wick_lower"].iloc[i]
        total_range = df["candle_range"].iloc[i]
        vol_ratio = df["volume_ratio"].iloc[i]
        avg_range = df["avg_range"].iloc[i]
        close_price = df["close"].iloc[i]
        open_price = df["open"].iloc[i]

        if total_range <= 0:
            continue

        range_vs_avg = total_range / avg_range if avg_range > 0 else 1.0

        # --- Engulfing Patterns ---
        if i > 0:
            prev_open = df["open"].iloc[i - 1]
            prev_close = df["close"].iloc[i - 1]
            prev_body = abs(prev_close - prev_open)
            prev_range = df["high"].iloc[i - 1] - df["low"].iloc[i - 1]
            curr_body = df["candle_body"].iloc[i]

            # Bullish engulfing
            if prev_close < prev_open and close_price > open_price:
                body_engulfs = open_price <= prev_close and close_price >= prev_open
                if body_engulfs and curr_body >= prev_body * engulfing_min_body:
                    strength = _score_candle_signal(body_ratio, 0.0, vol_ratio, range_vs_avg)
                    sig = CandleSignal(
                        index=i, pattern=CandlePattern.BULLISH_ENGULFING,
                        strength=strength, body_size=curr_body,
                        wick_upper=wick_upper, wick_lower=wick_lower,
                        candle_range=total_range, body_ratio=body_ratio,
                        volume_ratio=vol_ratio, is_strong=strength >= min_total_score,
                        price_level=close_price,
                    )
                    signals.append(sig)

            # Bearish engulfing
            elif prev_close > prev_open and close_price < open_price:
                body_engulfs = open_price >= prev_close and close_price <= prev_open
                if body_engulfs and curr_body >= prev_body * engulfing_min_body:
                    strength = _score_candle_signal(body_ratio, 0.0, vol_ratio, range_vs_avg)
                    sig = CandleSignal(
                        index=i, pattern=CandlePattern.BEARISH_ENGULFING,
                        strength=strength, body_size=curr_body,
                        wick_upper=wick_upper, wick_lower=wick_lower,
                        candle_range=total_range, body_ratio=body_ratio,
                        volume_ratio=vol_ratio, is_strong=strength >= min_total_score,
                        price_level=close_price,
                    )
                    signals.append(sig)

        # --- Pin Bars ---
        if df["candle_body"].iloc[i] > 0:
            # Bullish pin bar
            if wick_lower > 0:
                wick_body = wick_lower / df["candle_body"].iloc[i]
                wick_range = wick_lower / total_range
                if wick_body >= pin_bar_min_wick_ratio and wick_range >= 0.67:
                    strength = _score_candle_signal(body_ratio, wick_lower / total_range, vol_ratio, range_vs_avg)
                    sig = CandleSignal(
                        index=i, pattern=CandlePattern.BULLISH_PIN_BAR,
                        strength=strength, body_size=df["candle_body"].iloc[i],
                        wick_upper=wick_upper, wick_lower=wick_lower,
                        candle_range=total_range, body_ratio=body_ratio,
                        volume_ratio=vol_ratio, is_strong=strength >= min_total_score,
                        price_level=close_price,
                    )
                    signals.append(sig)

            # Bearish pin bar
            if wick_upper > 0:
                wick_body = wick_upper / df["candle_body"].iloc[i]
                wick_range = wick_upper / total_range
                if wick_body >= pin_bar_min_wick_ratio and wick_range >= 0.67:
                    strength = _score_candle_signal(body_ratio, wick_upper / total_range, vol_ratio, range_vs_avg)
                    sig = CandleSignal(
                        index=i, pattern=CandlePattern.BEARISH_PIN_BAR,
                        strength=strength, body_size=df["candle_body"].iloc[i],
                        wick_upper=wick_upper, wick_lower=wick_lower,
                        candle_range=total_range, body_ratio=body_ratio,
                        volume_ratio=vol_ratio, is_strong=strength >= min_total_score,
                        price_level=close_price,
                    )
                    signals.append(sig)

        # --- Rejection Candles ---
        if df["candle_body"].iloc[i] > 0:
            if wick_lower / df["candle_body"].iloc[i] >= rejection_min_wick_ratio:
                strength = _score_candle_signal(body_ratio, wick_lower / total_range, vol_ratio, range_vs_avg)
                sig = CandleSignal(
                    index=i, pattern=CandlePattern.BULLISH_REJECTION,
                    strength=strength, body_size=df["candle_body"].iloc[i],
                    wick_upper=wick_upper, wick_lower=wick_lower,
                    candle_range=total_range, body_ratio=body_ratio,
                    volume_ratio=vol_ratio, is_strong=strength >= min_total_score,
                    price_level=close_price,
                )
                signals.append(sig)

            if wick_upper / df["candle_body"].iloc[i] >= rejection_min_wick_ratio:
                strength = _score_candle_signal(body_ratio, wick_upper / total_range, vol_ratio, range_vs_avg)
                sig = CandleSignal(
                    index=i, pattern=CandlePattern.BEARISH_REJECTION,
                    strength=strength, body_size=df["candle_body"].iloc[i],
                    wick_upper=wick_upper, wick_lower=wick_lower,
                    candle_range=total_range, body_ratio=body_ratio,
                    volume_ratio=vol_ratio, is_strong=strength >= min_total_score,
                    price_level=close_price,
                )
                signals.append(sig)

        # --- Displacement Candles ---
        if body_ratio >= displacement_min_body and range_vs_avg >= 1.5:
            if vol_ratio >= 1.3:
                if df["is_bullish"].iloc[i]:
                    strength = _score_candle_signal(body_ratio, 0.0, vol_ratio, range_vs_avg)
                    sig = CandleSignal(
                        index=i, pattern=CandlePattern.STRONG_DISPLACEMENT_BULLISH,
                        strength=strength, body_size=df["candle_body"].iloc[i],
                        wick_upper=wick_upper, wick_lower=wick_lower,
                        candle_range=total_range, body_ratio=body_ratio,
                        volume_ratio=vol_ratio, is_strong=strength >= min_total_score,
                        price_level=close_price,
                    )
                    signals.append(sig)
                elif df["is_bearish"].iloc[i]:
                    strength = _score_candle_signal(body_ratio, 0.0, vol_ratio, range_vs_avg)
                    sig = CandleSignal(
                        index=i, pattern=CandlePattern.STRONG_DISPLACEMENT_BEARISH,
                        strength=strength, body_size=df["candle_body"].iloc[i],
                        wick_upper=wick_upper, wick_lower=wick_lower,
                        candle_range=total_range, body_ratio=body_ratio,
                        volume_ratio=vol_ratio, is_strong=strength >= min_total_score,
                        price_level=close_price,
                    )
                    signals.append(sig)

    # Sort signals by index descending (most recent first)
    signals.sort(key=lambda s: s.index, reverse=True)

    # Separate by direction and strength
    strong_bullish = [s for s in signals if s.is_strong and s.pattern.value.startswith("bullish")]
    strong_bearish = [s for s in signals if s.is_strong and s.pattern.value.startswith("bearish")
                      or "displacement_bearish" in s.pattern.value or "pin_bar_bearish" in s.pattern.value]
    latest_signal = signals[0] if signals else None

    return {
        "df": df,
        "signals": signals,
        "strong_bullish_signals": strong_bullish,
        "strong_bearish_signals": strong_bearish,
        "latest_signal": latest_signal,
        "total_signals": len(signals),
        "strong_signals": len([s for s in signals if s.is_strong]),
    }


if __name__ == "__main__":
    # Example Usage
    data = {
        'open':  [10, 12, 9,  8,  11, 9,  8,  10, 13, 11, 14, 12, 11, 13, 16],
        'high':  [14, 16, 12, 11, 15, 13, 12, 14, 17, 15, 18, 16, 15, 17, 20],
        'low':   [9,  11, 7,  6,  10, 7,  6,  9,  12, 10, 13, 11, 10, 12, 15],
        'close': [13, 14, 8,  7,  14, 8,  7,  13, 16, 13, 17, 14, 13, 16, 19],
        'volume':[100,150,80, 60, 200,90, 70, 180,250,130,220,140,120,200,280],
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(pd.Series(range(len(df))), unit='s')

    print("=" * 60)
    print("V2.0 Confirmation Candle Analysis")
    print("=" * 60)

    result = analyze_candles(df.copy())

    print(f"\nTotal signals detected: {result['total_signals']}")
    print(f"Strong signals: {result['strong_signals']}")
    print(f"Strong Bullish: {len(result['strong_bullish_signals'])}")
    print(f"Strong Bearish: {len(result['strong_bearish_signals'])}")

    print("\nAll signals:")
    for sig in result['signals']:
        d = sig.as_dict()
        marker = "✓" if d['is_strong'] else "✗"
        print(f"  {marker} [{d['pattern']:30s}] strength={d['strength']:.2f} | "
              f"body={d['body_ratio']:.0%} | vol={d['volume_ratio']:.1f}x | "
              f"price={d['price_level']:.1f}")

    if result['latest_signal']:
        print(f"\nLatest Signal: {result['latest_signal'].pattern.value}")
        print(f"  Strength: {result['latest_signal'].strength:.2f}")
        print(f"  Is Strong: {result['latest_signal'].is_strong}")
