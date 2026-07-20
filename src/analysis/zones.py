"""
Professional AI Trading Analysis System V2.0
Module: Premium / Discount Zones

Upgraded from V1.0 with improved Premium/Discount calculation:
- Uses current dealing range (swing-to-swing) instead of simplistic midpoint
- Dynamic range updates on each new swing point
- Multiple range methods: swing-to-swing, session high/low, rolling range
- Proper zone classification with granularity levels

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

class ZonePosition(Enum):
    """Where price sits relative to the dealing range."""
    PREMIUM = "premium"
    EQUILIBRIUM = "equilibrium"
    DISCOUNT = "discount"
    OPTIMAL_PREMIUM = "optimal_premium"    # 75% level (ideal sell)
    OPTIMAL_DISCOUNT = "optimal_discount"  # 25% level (ideal buy)


@dataclass
class DealingRange:
    """Represents the current dealing range for P/D calculation."""
    high: float
    low: float
    equilibrium: float
    premium_start: float
    discount_end: float
    optimal_premium: float
    optimal_discount: float
    range_size: float
    method: str  # "swing_to_swing" | "session" | "rolling"
    current_position: ZonePosition = ZonePosition.EQUILIBRIUM

    def as_dict(self) -> Dict:
        return {
            "high": self.high,
            "low": self.low,
            "equilibrium": self.equilibrium,
            "premium_start": self.premium_start,
            "discount_end": self.discount_end,
            "optimal_premium": self.optimal_premium,
            "optimal_discount": self.optimal_discount,
            "range_size": self.range_size,
            "method": self.method,
            "current_position": self.current_position.value,
        }


# ============================================================
# Dealing Range Calculation
# ============================================================

def calculate_swing_dealing_range(
    swing_high: float,
    swing_low: float,
) -> DealingRange:
    """Calculates Premium/Discount zones from swing-to-swing dealing range.

    This is the primary method for institutional P/D analysis.
    The dealing range is defined by the most recent swing high and swing low
    that established the current trend structure.

    V2.0 improvement: Proper dealing range context instead of V1.0's
    simplistic midpoint calculation that didn't account for structure.

    Args:
        swing_high: Price of the most recent swing high.
        swing_low: Price of the most recent swing low.

    Returns:
        DealingRange object with all zone levels calculated.
    """
    if swing_high <= swing_low:
        # Invalid range; return defaults
        mid = (swing_high + swing_low) / 2.0
        return DealingRange(
            high=swing_high, low=swing_low, equilibrium=mid,
            premium_start=mid, discount_end=mid,
            optimal_premium=mid, optimal_discount=mid,
            range_size=0.0, method="swing_to_swing",
        )

    range_size = swing_high - swing_low
    equilibrium = swing_low + range_size * 0.50
    premium_start = swing_low + range_size * 0.50
    discount_end = swing_low + range_size * 0.50
    optimal_premium = swing_low + range_size * 0.75
    optimal_discount = swing_low + range_size * 0.25

    return DealingRange(
        high=swing_high,
        low=swing_low,
        equilibrium=equilibrium,
        premium_start=premium_start,
        discount_end=discount_end,
        optimal_premium=optimal_premium,
        optimal_discount=optimal_discount,
        range_size=range_size,
        method="swing_to_swing",
    )


def calculate_session_dealing_range(
    df: pd.DataFrame,
    session_start_idx: int,
    session_end_idx: int,
) -> DealingRange:
    """Calculates P/D zones from session high/low.

    Useful for intraday analysis where the session range defines the
    dealing range for that trading day.

    Args:
        df: OHLCV DataFrame.
        session_start_idx: Starting index of the session.
        session_end_idx: Ending index of the session.

    Returns:
        DealingRange object.
    """
    session_data = df.iloc[session_start_idx:session_end_idx + 1]
    if len(session_data) == 0:
        return calculate_swing_dealing_range(df["high"].iloc[-1], df["low"].iloc[-1])

    swing_high = session_data["high"].max()
    swing_low = session_data["low"].min()
    dr = calculate_swing_dealing_range(swing_high, swing_low)
    dr.method = "session"
    return dr


def calculate_rolling_dealing_range(
    df: pd.DataFrame,
    candles: int = 100,
) -> DealingRange:
    """Calculates P/D zones from a rolling window of candles.

    Useful when swing points are not available or for a broader
    contextual range.

    Args:
        df: OHLCV DataFrame.
        candles: Number of recent candles for the range.

    Returns:
        DealingRange object.
    """
    n = len(df)
    start = max(0, n - candles)
    window = df.iloc[start:n]

    swing_high = window["high"].max()
    swing_low = window["low"].min()
    dr = calculate_swing_dealing_range(swing_high, swing_low)
    dr.method = "rolling"
    return dr


# ============================================================
# Dynamic Range Updates
# ============================================================

def update_dealing_range(
    current_range: DealingRange,
    new_high: float,
    new_low: float,
    dynamic_update: bool = True,
) -> DealingRange:
    """Updates the dealing range when new swing points are identified.

    In SMC analysis, the dealing range is not static. As the market
    creates new swing highs or swing lows, the range expands or shifts.
    This function handles that dynamic update.

    V2.0 improvement: Dynamic range updates that adapt to new structure
    instead of V1.0's static single-calculation approach.

    Args:
        current_range: The current DealingRange.
        new_high: New swing high price (if None, keeps existing).
        new_low: New swing low price (if None, keeps existing).
        dynamic_update: Whether to apply the update.

    Returns:
        Updated DealingRange.
    """
    if not dynamic_update:
        return current_range

    updated_high = current_range.high
    updated_low = current_range.low

    if new_high is not None and new_high > current_range.high:
        updated_high = new_high
    if new_low is not None and new_low < current_range.low:
        updated_low = new_low

    return calculate_swing_dealing_range(updated_high, updated_low)


# ============================================================
# Current Price Position
# ============================================================

def classify_price_position(
    price: float,
    dealing_range: DealingRange,
) -> ZonePosition:
    """Classifies where the current price sits within the dealing range.

    Zones:
    - Optimal Discount: Below 25% of range (ideal buy zone)
    - Discount: 25%-50% of range
    - Equilibrium: Exactly at 50%
    - Premium: 50%-75% of range
    - Optimal Premium: Above 75% of range (ideal sell zone)

    Args:
        price: Current price to classify.
        dealing_range: The current DealingRange.

    Returns:
        ZonePosition enum value.
    """
    if dealing_range.range_size <= 0:
        return ZonePosition.EQUILIBRIUM

    # Calculate position as percentage of range
    position_pct = (price - dealing_range.low) / dealing_range.range_size

    if position_pct >= 0.75:
        return ZonePosition.OPTIMAL_PREMIUM
    elif position_pct >= 0.50:
        return ZonePosition.PREMIUM
    elif position_pct > 0.45 and position_pct < 0.55:
        return ZonePosition.EQUILIBRIUM
    elif position_pct >= 0.25:
        return ZonePosition.DISCOUNT
    else:
        return ZonePosition.OPTIMAL_DISCOUNT


def get_zone_score(
    price: float,
    dealing_range: DealingRange,
    direction: str = "bullish",
) -> float:
    """Calculates a zone quality score (0-1) for the current price.

    For bullish trades: lower in discount = higher score
    For bearish trades: higher in premium = higher score

    Args:
        price: Current price.
        dealing_range: Current DealingRange.
        direction: 'bullish' or 'bearish'.

    Returns:
        Float score 0.0-1.0.
    """
    position = classify_price_position(price, dealing_range)
    dr = dealing_range

    if direction == "bullish":
        # Bullish: want to buy in discount (lower = better)
        if dr.range_size <= 0:
            return 0.5
        position_pct = (price - dr.low) / dr.range_size

        # Optimal discount (0-25%): score 0.8-1.0
        if position_pct <= 0.25:
            return 0.8 + (0.25 - position_pct) * 0.8
        # Discount (25-50%): score 0.5-0.8
        elif position_pct <= 0.50:
            return 0.5 + (0.50 - position_pct) * 1.2
        # Equilibrium (50%): score 0.4
        elif abs(position_pct - 0.50) < 0.05:
            return 0.4
        # Premium: low score for bullish
        else:
            return 0.2
    else:
        # Bearish: want to sell in premium (higher = better)
        if dr.range_size <= 0:
            return 0.5
        position_pct = (price - dr.low) / dr.range_size

        # Optimal premium (75-100%): score 0.8-1.0
        if position_pct >= 0.75:
            return 0.8 + (position_pct - 0.75) * 1.0
        # Premium (50-75%): score 0.5-0.8
        elif position_pct >= 0.50:
            return 0.5 + (position_pct - 0.50) * 1.2
        # Equilibrium: score 0.4
        elif abs(position_pct - 0.50) < 0.05:
            return 0.4
        # Discount: low score for bearish
        else:
            return 0.2


# ============================================================
# Comprehensive Zone Analysis
# ============================================================

def find_premium_discount_zones(
    df: pd.DataFrame,
    swing_high: float,
    swing_low: float,
) -> Tuple[float, float, float, float, float]:
    """Backward-compatible interface matching V1.0.

    Returns: (premium_start, equilibrium, discount_end, premium_zone_price, discount_zone_price)

    Args:
        df: OHLCV DataFrame (context only).
        swing_high: Recent swing high price.
        swing_low: Recent swing low price.

    Returns:
        Tuple of 5 floats matching V1.0 signature.
    """
    dr = calculate_swing_dealing_range(swing_high, swing_low)
    return (
        dr.premium_start,
        dr.equilibrium,
        dr.discount_end,
        dr.optimal_premium,
        dr.optimal_discount,
    )


def analyze_zones(
    df: pd.DataFrame,
    swing_high: Optional[float] = None,
    swing_low: Optional[float] = None,
    method: str = "swing_to_swing",
    rolling_candles: int = 100,
    dynamic_update: bool = True,
) -> Dict[str, object]:
    """Complete Premium/Discount zone analysis.

    V2.0 pipeline:
    1. Determine the dealing range using the configured method
    2. Calculate all zone levels
    3. Classify current price position
    4. Score zone quality for both directions
    5. Optionally update range dynamically

    Args:
        df: OHLCV DataFrame.
        swing_high: Swing high for range (auto-detected if None).
        swing_low: Swing low for range (auto-detected if None).
        method: "swing_to_swing", "session", or "rolling".
        rolling_candles: Candles for rolling range method.
        dynamic_update: Whether to update range on new swings.

    Returns:
        Dict with:
            'dealing_range': DealingRange object
            'current_position': ZonePosition enum
            'bullish_zone_score': Float 0-1 for buy quality
            'bearish_zone_score': Float 0-1 for sell quality
            'current_price': Latest close price
            'price_range_pct': Where price sits as % of range
            'is_in_premium': bool
            'is_in_discount': bool
    """
    current_price = df["close"].iloc[-1] if len(df) > 0 else 0.0

    # Step 1: Calculate dealing range
    if method == "swing_to_swing":
        if swing_high is None:
            swing_high = df["high"].iloc[-50:].max() if len(df) >= 50 else df["high"].max()
        if swing_low is None:
            swing_low = df["low"].iloc[-50:].min() if len(df) >= 50 else df["low"].min()
        dealing_range = calculate_swing_dealing_range(swing_high, swing_low)

    elif method == "session":
        # Use last 24 hours as a session proxy
        session_end = len(df) - 1
        session_start = max(0, len(df) - 48)  # Approximate session
        dealing_range = calculate_session_dealing_range(df, session_start, session_end)

    elif method == "rolling":
        dealing_range = calculate_rolling_dealing_range(df, candles=rolling_candles)
    else:
        dealing_range = calculate_swing_dealing_range(
            df["high"].max() if len(df) > 0 else 0,
            df["low"].min() if len(df) > 0 else 0,
        )

    # Step 2: Classify current position
    position = classify_price_position(current_price, dealing_range)
    dealing_range.current_position = position

    # Step 3: Calculate zone scores
    bullish_score = get_zone_score(current_price, dealing_range, "bullish")
    bearish_score = get_zone_score(current_price, dealing_range, "bearish")

    # Step 4: Calculate price position percentage
    price_range_pct = 0.0
    if dealing_range.range_size > 0:
        price_range_pct = (current_price - dealing_range.low) / dealing_range.range_size

    # Step 5: Binary flags
    is_premium = position in (ZonePosition.PREMIUM, ZonePosition.OPTIMAL_PREMIUM)
    is_discount = position in (ZonePosition.DISCOUNT, ZonePosition.OPTIMAL_DISCOUNT)

    return {
        "dealing_range": dealing_range,
        "current_position": position,
        "bullish_zone_score": bullish_score,
        "bearish_zone_score": bearish_score,
        "current_price": current_price,
        "price_range_pct": price_range_pct,
        "is_in_premium": is_premium,
        "is_in_discount": is_discount,
    }


# ============================================================
# Imbalance Detection (from V1.0, improved)
# ============================================================

def find_imbalances(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies imbalances (similar to FVGs but focusing on candle body gaps).

    V2.0 improvement: Same interface as V1.0 for backward compatibility.

    Args:
        df: OHLCV DataFrame.

    Returns:
        DataFrame with 'imbalance_bullish', 'imbalance_bearish',
        'imbalance_start', 'imbalance_end' columns.
    """
    df = df.copy()
    df["imbalance_bullish"] = False
    df["imbalance_bearish"] = False
    df["imbalance_start"] = np.nan
    df["imbalance_end"] = np.nan

    for i in range(1, len(df)):
        # Bullish imbalance: Current low > Previous high
        if df["low"].iloc[i] > df["high"].iloc[i - 1]:
            df.loc[df.index[i], "imbalance_bullish"] = True
            df.loc[df.index[i], "imbalance_start"] = df["high"].iloc[i - 1]
            df.loc[df.index[i], "imbalance_end"] = df["low"].iloc[i]
        # Bearish imbalance: Current high < Previous low
        elif df["high"].iloc[i] < df["low"].iloc[i - 1]:
            df.loc[df.index[i], "imbalance_bearish"] = True
            df.loc[df.index[i], "imbalance_start"] = df["low"].iloc[i - 1]
            df.loc[df.index[i], "imbalance_end"] = df["high"].iloc[i]

    return df


if __name__ == "__main__":
    # Example Usage
    data = {
        'open':  [10, 12, 15, 13, 16, 14, 17, 15, 18, 16, 19, 17, 20, 18, 22, 20, 21, 19, 23, 21],
        'high':  [13, 16, 17, 15, 18, 17, 19, 17, 20, 18, 22, 20, 23, 21, 25, 23, 24, 22, 26, 24],
        'low':   [9,  11, 12, 11, 13, 12, 14, 13, 15, 14, 17, 15, 18, 16, 20, 18, 19, 17, 21, 19],
        'close': [12, 15, 13, 14, 17, 15, 18, 16, 19, 17, 21, 19, 22, 20, 24, 22, 23, 21, 25, 23],
        'volume':[100,120,110,130,140,120,150,130,160,140,180,160,200,170,220,180,220,190,280,200],
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(pd.Series(range(len(df))), unit='s')

    print("=" * 60)
    print("V2.0 Premium/Discount Zone Analysis")
    print("=" * 60)

    # Use V2.0 analysis
    result = analyze_zones(df.copy(), swing_high=26.0, swing_low=9.0)

    dr = result['dealing_range']
    print(f"\nDealing Range Method: {dr.method}")
    print(f"Range High: {dr.high:.2f}")
    print(f"Range Low: {dr.low:.2f}")
    print(f"Range Size: {dr.range_size:.2f}")
    print(f"Equilibrium: {dr.equilibrium:.2f}")
    print(f"Premium Start: {dr.premium_start:.2f}")
    print(f"Discount End: {dr.discount_end:.2f}")
    print(f"Optimal Premium (75%): {dr.optimal_premium:.2f}")
    print(f"Optimal Discount (25%): {dr.optimal_discount:.2f}")

    print(f"\nCurrent Price: {result['current_price']:.2f}")
    print(f"Position: {result['current_position'].value}")
    print(f"Range Percentage: {result['price_range_pct']:.1%}")
    print(f"In Premium: {result['is_in_premium']}")
    print(f"In Discount: {result['is_in_discount']}")
    print(f"Bullish Zone Score: {result['bullish_zone_score']:.2f}")
    print(f"Bearish Zone Score: {result['bearish_zone_score']:.2f}")

    # Compare with V1.0 interface
    print("\n--- V1.0 Compatible Interface ---")
    ps, eq, de, pp, dp = find_premium_discount_zones(df.copy(), 26.0, 9.0)
    print(f"Premium Start: {ps:.2f}")
    print(f"Equilibrium: {eq:.2f}")
    print(f"Discount End: {de:.2f}")
    print(f"Premium Zone (75%): {pp:.2f}")
    print(f"Discount Zone (25%): {dp:.2f}")

    # Test dynamic range update
    print("\n--- Dynamic Range Update ---")
    new_range = update_dealing_range(dr, new_high=28.0, new_low=None)
    print(f"Updated High: {new_range.high:.2f}")
    print(f"Updated Low: {new_range.low:.2f}")
    print(f"New Optimal Premium: {new_range.optimal_premium:.2f}")
