"""
Professional AI Trading Analysis System V2.0
Module: Order Block Detection

Upgraded from V1.0 with improved OB handling:
- Rank by strength (multi-criteria scoring)
- Detect fresh Order Blocks
- Ignore weak/invalid Order Blocks
- Consider mitigation status
- Consider displacement before creation
- Breaker Block and Mitigation Block detection

Maintains backward-compatible DataFrame interface (pandas OHLCV as input).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# Data Structures
# ============================================================

class OBState(Enum):
    """State of an Order Block."""
    FRESH = "fresh"
    PARTIALLY_MITIGATED = "partially_mitigated"
    MITIGATED = "mitigated"
    INVALID = "invalid"


class OBStrength(Enum):
    """Strength classification of an Order Block."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class OrderBlock:
    """Represents a detected Order Block with full metadata."""
    index: int
    ob_type: str  # "bullish" | "bearish"
    ob_start: float  # Lower price of the OB zone
    ob_end: float    # Upper price of the OB zone
    ob_midpoint: float
    timestamp: Optional[pd.Timestamp] = None
    # Quality metrics
    displacement_size: float = 0.0   # Size of the move after OB (in pips)
    displacement_pct: float = 0.0    # Displacement as % of price
    body_size_pct: float = 0.0       # OB candle body as % of its range
    volume_ratio: float = 1.0        # Volume at OB vs average
    freshness_age: int = 0           # Candles since OB creation
    structure_position: str = ""     # Position relative to market structure
    # State
    state: OBState = OBState.FRESH
    strength: OBStrength = OBStrength.MODERATE
    strength_score: float = 0.0      # Composite 0-1 score
    mitigation_pct: float = 0.0      # 0-100% how much OB has been tested
    is_valid: bool = True

    # Weights for strength scoring (can be configured)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "displacement_size": 0.30,
        "body_size": 0.20,
        "volume": 0.20,
        "freshness": 0.20,
        "position_in_structure": 0.10,
    })

    def as_dict(self) -> Dict:
        return {
            "index": self.index,
            "ob_type": self.ob_type,
            "ob_start": self.ob_start,
            "ob_end": self.ob_end,
            "ob_midpoint": self.ob_midpoint,
            "state": self.state.value,
            "strength": self.strength.value,
            "strength_score": self.strength_score,
            "displacement_pct": self.displacement_pct,
            "body_size_pct": self.body_size_pct,
            "volume_ratio": self.volume_ratio,
            "freshness_age": self.freshness_age,
            "mitigation_pct": self.mitigation_pct,
            "is_valid": self.is_valid,
        }


# ============================================================
# Displacement Detection
# ============================================================

def detect_displacement(
    df: pd.DataFrame,
    multiplier: float = 2.0,
    lookback: int = 10,
) -> pd.Series:
    """Detects impulsive displacement moves on each candle.

    Displacement is a strong, impulsive move that indicates institutional
    order flow. An OB must precede a displacement move to be valid.

    V2.0 improvement: Proper displacement calculation using ATR and
    configurable multiplier instead of V1.0's hardcoded 1.5x.

    Args:
        df: OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume'.
        multiplier: Minimum move as multiple of average candle range.
        lookback: Candles for calculating average range.

    Returns:
        Series of displacement scores (0.0 = no displacement, 1.0+ = strong).
    """
    df_range = df["high"] - df["low"]
    avg_range = df_range.rolling(window=lookback, min_periods=1).mean().shift(1)

    # Displacement = actual range / average range
    displacement = df_range / avg_range.replace(0, np.nan)
    displacement = displacement.fillna(0)

    # Direction: positive for bullish, negative for bearish
    direction = np.where(df["close"] > df["open"], 1.0, -1.0)
    displacement = displacement * direction

    return displacement


# ============================================================
# Order Block Detection
# ============================================================

def find_order_blocks(
    df: pd.DataFrame,
    displacement_multiplier: float = 2.0,
    min_body_pct: float = 0.5,
    freshness_window: int = 30,
    avg_volume: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[OrderBlock]]:
    """Identifies Order Blocks based on the last opposite candle before displacement.

    V2.0 improvements over V1.0:
    - Requires significant displacement after the OB candle (configurable)
    - Minimum body percentage requirement for OB candle
    - Freshness tracking (how old is the OB)
    - Mitigation status tracking
    - Structured OrderBlock objects with full metadata

    Bullish Order Block: Last bearish candle before an impulsive bullish move.
    Bearish Order Block: Last bullish candle before an impulsive bearish move.

    Args:
        df: OHLCV DataFrame.
        displacement_multiplier: Min displacement after OB (x avg range).
        min_body_pct: Minimum body/range ratio for OB candle.
        freshness_window: Candles within which OB is considered fresh.
        avg_volume: Pre-calculated average volume.

    Returns:
        Tuple of (annotated DataFrame, list of OrderBlock objects).
    """
    df = df.copy()
    n = len(df)

    # Initialize columns
    df["bullish_ob"] = False
    df["bearish_ob"] = False
    df["ob_start"] = np.nan
    df["ob_end"] = np.nan
    df["ob_strength_raw"] = 0.0

    # Calculate average volume
    if avg_volume is None:
        avg_volume = df["volume"].mean() if "volume" in df.columns and len(df) > 0 else 1.0

    # Detect displacement
    displacement = detect_displacement(df, multiplier=displacement_multiplier, lookback=10)

    # Calculate average range for body ratio
    df["candle_range"] = df["high"] - df["low"]
    df["candle_body"] = (df["close"] - df["open"]).abs()
    df["body_ratio"] = np.where(
        df["candle_range"] > 0,
        df["candle_body"] / df["candle_range"],
        0.0
    )

    order_blocks: List[OrderBlock] = []

    for i in range(1, n):
        # --- Bullish Order Block ---
        # Condition: Previous candle is bearish (close < open)
        if df["close"].iloc[i - 1] < df["open"].iloc[i - 1]:
            # Next candle (i) shows impulsive bullish displacement
            if displacement.iloc[i] >= displacement_multiplier:
                # Verify body ratio of the OB candle
                if df["body_ratio"].iloc[i - 1] >= min_body_pct:
                    df.loc[df.index[i - 1], "bullish_ob"] = True
                    df.loc[df.index[i - 1], "ob_start"] = df["low"].iloc[i - 1]
                    df.loc[df.index[i - 1], "ob_end"] = df["open"].iloc[i - 1]

                    ob = OrderBlock(
                        index=i - 1,
                        ob_type="bullish",
                        ob_start=df["low"].iloc[i - 1],
                        ob_end=df["open"].iloc[i - 1],
                        ob_midpoint=(df["low"].iloc[i - 1] + df["open"].iloc[i - 1]) / 2,
                        timestamp=df.index[i - 1],
                        displacement_pct=displacement.iloc[i],
                        body_size_pct=df["body_ratio"].iloc[i - 1],
                        volume_ratio=df["volume"].iloc[i] / avg_volume if avg_volume > 0 else 1.0,
                        freshness_age=n - i,
                    )
                    order_blocks.append(ob)

        # --- Bearish Order Block ---
        # Condition: Previous candle is bullish (close > open)
        elif df["close"].iloc[i - 1] > df["open"].iloc[i - 1]:
            # Next candle (i) shows impulsive bearish displacement
            if displacement.iloc[i] <= -displacement_multiplier:
                if df["body_ratio"].iloc[i - 1] >= min_body_pct:
                    df.loc[df.index[i - 1], "bearish_ob"] = True
                    df.loc[df.index[i - 1], "ob_start"] = df["open"].iloc[i - 1]
                    df.loc[df.index[i - 1], "ob_end"] = df["high"].iloc[i - 1]

                    ob = OrderBlock(
                        index=i - 1,
                        ob_type="bearish",
                        ob_start=df["open"].iloc[i - 1],
                        ob_end=df["high"].iloc[i - 1],
                        ob_midpoint=(df["open"].iloc[i - 1] + df["high"].iloc[i - 1]) / 2,
                        timestamp=df.index[i - 1],
                        displacement_pct=abs(displacement.iloc[i]),
                        body_size_pct=df["body_ratio"].iloc[i - 1],
                        volume_ratio=df["volume"].iloc[i] / avg_volume if avg_volume > 0 else 1.0,
                        freshness_age=n - i,
                    )
                    order_blocks.append(ob)

    # Clean up temporary columns
    df = df.drop(columns=["candle_range", "candle_body", "body_ratio"], errors='ignore')

    return df, order_blocks


# ============================================================
# OB Strength Ranking
# ============================================================

def rank_order_blocks(
    order_blocks: List[OrderBlock],
    exclude_mitigated: bool = True,
    freshness_window: int = 30,
    weights: Optional[Dict[str, float]] = None,
) -> List[OrderBlock]:
    """Ranks Order Blocks by strength using multi-criteria scoring.

    V2.0 improvement: Comprehensive strength ranking system instead of
    V1.0's simplistic boolean classification.

    Criteria (configurable weights):
    - Displacement Size (30%): Larger displacement after OB = stronger
    - Body Size (20%): Larger OB candle body = stronger
    - Volume (20%): Higher volume at OB = stronger
    - Freshness (20%): More recent OB = stronger
    - Position in Structure (10%): At key structural levels = stronger

    Args:
        order_blocks: List of OrderBlock objects to rank.
        exclude_mitigated: Whether to remove mitigated OBs from ranking.
        freshness_window: Maximum age for "fresh" OBs.
        weights: Optional custom weights dict.

    Returns:
        Sorted list of OrderBlocks (strongest first).
    """
    if weights is None:
        weights = {
            "displacement_size": 0.30,
            "body_size": 0.20,
            "volume": 0.20,
            "freshness": 0.20,
            "position_in_structure": 0.10,
        }

    for ob in order_blocks:
        # Skip mitigated OBs
        if exclude_mitigated and ob.state == OBState.MITIGATED:
            ob.is_valid = False
            ob.strength = OBStrength.WEAK
            ob.strength_score = 0.0
            continue

        scores = {}

        # 1. Displacement score (0-1)
        # Optimal displacement: 2-5x average range
        disp = ob.displacement_pct
        if disp >= 2.0:
            scores["displacement_size"] = 1.0
        elif disp >= 1.5:
            scores["displacement_size"] = 0.7
        elif disp >= 1.0:
            scores["displacement_size"] = 0.4
        else:
            scores["displacement_size"] = 0.2

        # 2. Body size score (0-1)
        body = ob.body_size_pct
        if body >= 0.7:
            scores["body_size"] = 1.0
        elif body >= 0.5:
            scores["body_size"] = 0.8
        elif body >= 0.4:
            scores["body_size"] = 0.5
        else:
            scores["body_size"] = 0.2

        # 3. Volume score (0-1)
        vol = ob.volume_ratio
        if vol >= 2.0:
            scores["volume"] = 1.0
        elif vol >= 1.5:
            scores["volume"] = 0.8
        elif vol >= 1.0:
            scores["volume"] = 0.6
        else:
            scores["volume"] = 0.3

        # 4. Freshness score (0-1)
        age = ob.freshness_age
        if age <= 5:
            scores["freshness"] = 1.0
        elif age <= freshness_window / 3:
            scores["freshness"] = 0.8
        elif age <= freshness_window * 2 / 3:
            scores["freshness"] = 0.5
        elif age <= freshness_window:
            scores["freshness"] = 0.3
        else:
            scores["freshness"] = 0.1

        # 5. Position in structure (default moderate if unknown)
        if ob.structure_position in ["swing_low", "swing_high"]:
            scores["position_in_structure"] = 1.0
        elif ob.structure_position in ["breakout_level"]:
            scores["position_in_structure"] = 0.7
        else:
            scores["position_in_structure"] = 0.5

        # Weighted composite score
        composite = sum(
            scores.get(k, 0.0) * weights.get(k, 0.0) for k in weights
        )
        ob.strength_score = min(1.0, max(0.0, composite))

        # Classify strength
        if ob.strength_score >= 0.75:
            ob.strength = OBStrength.STRONG
        elif ob.strength_score >= 0.45:
            ob.strength = OBStrength.MODERATE
        else:
            ob.strength = OBStrength.WEAK

    # Sort by strength score descending
    order_blocks.sort(key=lambda ob: ob.strength_score, reverse=True)
    return order_blocks


# ============================================================
# Mitigation Tracking
# ============================================================

def track_ob_mitigation(
    df: pd.DataFrame,
    order_blocks: List[OrderBlock],
) -> List[OrderBlock]:
    """Tracks whether Order Blocks have been mitigated by subsequent price action.

    An OB is mitigated when price returns to the OB zone.
    For bullish OBs: price drops into the OB zone.
    For bearish OBs: price rises into the OB zone.

    V2.0 improvement: Proper mitigation tracking with percentage fill
    instead of V1.0's binary only.

    Args:
        df: OHLCV DataFrame.
        order_blocks: List of OrderBlock objects to track.

    Returns:
        Updated OrderBlock list with state and mitigation_pct.
    """
    n = len(df)

    for ob in order_blocks:
        start_idx = ob.index + 2  # Start checking after displacement candle
        max_fill = 0.0
        mitigated = False

        for j in range(start_idx, n):
            candle_high = df["high"].iloc[j]
            candle_low = df["low"].iloc[j]

            ob_range = ob.ob_end - ob.ob_start
            if ob_range <= 0:
                continue

            if ob.ob_type == "bullish":
                # Bullish OB: mitigated when price drops into the zone
                if candle_low <= ob.ob_end and candle_high >= ob.ob_start:
                    # How deep did price go into the OB?
                    fill_depth = ob.ob_end - max(candle_low, ob.ob_start)
                    fill_pct = (fill_depth / ob_range) * 100.0
                    max_fill = max(max_fill, fill_pct)

                    if fill_pct >= 80:
                        mitigated = True
                        break

            elif ob.ob_type == "bearish":
                # Bearish OB: mitigated when price rises into the zone
                if candle_high >= ob.ob_start and candle_low <= ob.ob_end:
                    fill_depth = min(candle_high, ob.ob_end) - ob.ob_start
                    fill_pct = (fill_depth / ob_range) * 100.0
                    max_fill = max(max_fill, fill_pct)

                    if fill_pct >= 80:
                        mitigated = True
                        break

        ob.mitigation_pct = max_fill
        if mitigated:
            ob.state = OBState.MITIGATED
            ob.is_valid = False
        elif max_fill > 0:
            ob.state = OBState.PARTIALLY_MITIGATED
            ob.is_valid = True
        else:
            ob.state = OBState.FRESH
            ob.is_valid = True

    return order_blocks


# ============================================================
# Select Best Order Block
# ============================================================

def select_best_ob(
    order_blocks: List[OrderBlock],
    direction: str = "bullish",
    min_strength_score: float = 0.4,
    exclude_mitigated: bool = True,
) -> Optional[OrderBlock]:
    """Selects the best Order Block for a given trading direction.

    Args:
        order_blocks: Ranked list of OrderBlock objects.
        direction: 'bullish' or 'bearish'.
        min_strength_score: Minimum strength score to consider.
        exclude_mitigated: Whether to skip mitigated OBs.

    Returns:
        Best matching OrderBlock or None.
    """
    candidates = [
        ob for ob in order_blocks
        if ob.ob_type == direction
        and ob.strength_score >= min_strength_score
        and ob.is_valid
        and (not exclude_mitigated or ob.state != OBState.MITIGATED)
    ]

    if not candidates:
        return None

    # Already sorted by strength from rank_order_blocks
    return candidates[0]


# ============================================================
# Breaker and Mitigation Blocks
# ============================================================

def find_breaker_blocks(
    df: pd.DataFrame,
    swing_highs: Optional[List[float]] = None,
    swing_lows: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Identifies Breaker Blocks.

    A Breaker Block forms when a swing high/low that was expected to hold
    is broken, and then price retests that level. This represents a failed
    Order Block that was overcome by stronger opposing force.

    V2.0 improvement: More structured detection using actual swing points
    instead of V1.0's simplified arbitrary conditions.

    Args:
        df: OHLCV DataFrame.
        swing_highs: Optional list of recent swing high prices.
        swing_lows: Optional list of recent swing low prices.

    Returns:
        DataFrame with 'bullish_breaker' and 'bearish_breaker' columns.
    """
    df = df.copy()
    df["bullish_breaker"] = False
    df["bearish_breaker"] = False
    df["breaker_price"] = np.nan

    n = len(df)
    if n < 5:
        return df

    # Use provided swings or detect from data
    if swing_highs is None:
        swing_highs = []
        for i in range(2, n - 2):
            if df["high"].iloc[i] > df["high"].iloc[i-2] and df["high"].iloc[i] > df["high"].iloc[i+2]:
                swing_highs.append(df["high"].iloc[i])

    if swing_lows is None:
        swing_lows = []
        for i in range(2, n - 2):
            if df["low"].iloc[i] < df["low"].iloc[i-2] and df["low"].iloc[i] < df["low"].iloc[i+2]:
                swing_lows.append(df["low"].iloc[i])

    for i in range(3, n):
        # Bullish Breaker: Price broke below a swing low, then reclaimed it
        if swing_lows and len(swing_lows) > 0:
            last_low = swing_lows[-1]
            # Look for the pattern: break below -> reclamation above
            if df["close"].iloc[i] > last_low:
                # Check if any of the last 5 candles broke below this low
                broken = False
                for j in range(max(0, i - 5), i):
                    if df["low"].iloc[j] < last_low:
                        broken = True
                        break
                if broken:
                    df.loc[df.index[i], "bullish_breaker"] = True
                    df.loc[df.index[i], "breaker_price"] = last_low

        # Bearish Breaker: Price broke above a swing high, then reclaimed it
        if swing_highs and len(swing_highs) > 0:
            last_high = swing_highs[-1]
            if df["close"].iloc[i] < last_high:
                broken = False
                for j in range(max(0, i - 5), i):
                    if df["high"].iloc[j] > last_high:
                        broken = True
                        break
                if broken:
                    df.loc[df.index[i], "bearish_breaker"] = True
                    df.loc[df.index[i], "breaker_price"] = last_high

    return df


def find_mitigation_blocks(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Identifies Mitigation Blocks.

    A Mitigation Block forms when price returns to an area where an Order Block
    failed to hold, and then reverses. Similar to a breaker but occurs after
    a liquidity sweep.

    V2.0 improvement: More robust detection based on OB failure pattern.

    Args:
        df: OHLCV DataFrame.

    Returns:
        DataFrame with 'bullish_mitigation' and 'bearish_mitigation' columns.
    """
    df = df.copy()
    df["bullish_mitigation"] = False
    df["bearish_mitigation"] = False

    n = len(df)
    if n < 6:
        return df

    # Calculate average candle range
    df["range"] = df["high"] - df["low"]
    avg_range = df["range"].rolling(window=10, min_periods=1).mean().shift(1)

    for i in range(5, n):
        # Bullish Mitigation: Price swept below a low, returned, and reversed up
        if df["low"].iloc[i] < df["low"].iloc[i - 3]:
            # Check if price swept a significant low
            sweep_depth = df["low"].iloc[i - 1] - df["low"].iloc[i] if df["low"].iloc[i] < df["low"].iloc[i - 1] else 0
            if sweep_depth > avg_range.iloc[i] * 0.5:
                # Price returned above the swept low
                if df["close"].iloc[i] > df["low"].iloc[i - 3]:
                    df.loc[df.index[i], "bullish_mitigation"] = True

        # Bearish Mitigation: Price swept above a high, returned, and reversed down
        if df["high"].iloc[i] > df["high"].iloc[i - 3]:
            sweep_depth = df["high"].iloc[i] - df["high"].iloc[i - 1] if df["high"].iloc[i] > df["high"].iloc[i - 1] else 0
            if sweep_depth > avg_range.iloc[i] * 0.5:
                if df["close"].iloc[i] < df["high"].iloc[i - 3]:
                    df.loc[df.index[i], "bearish_mitigation"] = True

    return df.drop(columns=["range"], errors='ignore')


# ============================================================
# Convenience: Backward-Compatible Interface
# ============================================================

def analyze_order_blocks(
    df: pd.DataFrame,
    displacement_multiplier: float = 2.0,
    min_body_pct: float = 0.5,
    freshness_window: int = 30,
    exclude_mitigated: bool = True,
    min_strength_score: float = 0.4,
) -> Dict[str, object]:
    """Complete Order Block analysis pipeline.

    Args:
        df: OHLCV DataFrame.
        displacement_multiplier: Min displacement for OB validation.
        min_body_pct: Minimum body ratio for OB candle.
        freshness_window: Max age for fresh OBs.
        exclude_mitigated: Whether to exclude mitigated OBs.
        min_strength_score: Minimum strength to consider.

    Returns:
        Dict with:
            'df': Annotated DataFrame
            'all_obs': All detected OrderBlocks
            'ranked_obs': OBs sorted by strength
            'best_bullish_ob': Best bullish OB (or None)
            'best_bearish_ob': Best bearish OB (or None)
            'fresh_count': Number of fresh OBs
            'strong_count': Number of strong OBs
    """
    # Step 1: Detect OBs
    df, order_blocks = find_order_blocks(
        df,
        displacement_multiplier=displacement_multiplier,
        min_body_pct=min_body_pct,
        freshness_window=freshness_window,
    )

    # Step 2: Track mitigation
    order_blocks = track_ob_mitigation(df, order_blocks)

    # Step 3: Rank by strength
    ranked_obs = rank_order_blocks(
        order_blocks,
        exclude_mitigated=exclude_mitigated,
        freshness_window=freshness_window,
    )

    # Step 4: Select best for each direction
    best_bullish = select_best_ob(
        ranked_obs, "bullish", min_strength_score, exclude_mitigated
    )
    best_bearish = select_best_ob(
        ranked_obs, "bearish", min_strength_score, exclude_mitigated
    )

    fresh_count = sum(1 for ob in order_blocks if ob.state == OBState.FRESH)
    strong_count = sum(1 for ob in order_blocks if ob.strength == OBStrength.STRONG)

    return {
        "df": df,
        "all_obs": order_blocks,
        "ranked_obs": ranked_obs,
        "best_bullish_ob": best_bullish,
        "best_bearish_ob": best_bearish,
        "fresh_count": fresh_count,
        "strong_count": strong_count,
    }


if __name__ == "__main__":
    # Example Usage
    data = {
        'open':  [10, 12, 15, 13, 16, 14, 17, 15, 18, 16, 19, 17, 20, 18, 22, 20, 21, 19, 23, 21],
        'high':  [13, 16, 17, 15, 18, 17, 19, 17, 20, 18, 22, 20, 23, 21, 25, 23, 24, 22, 26, 24],
        'low':   [9,  11, 12, 11, 13, 12, 14, 13, 15, 14, 17, 15, 18, 16, 20, 18, 19, 17, 21, 19],
        'close': [12, 15, 13, 14, 17, 15, 18, 16, 19, 17, 21, 19, 22, 20, 24, 22, 23, 21, 25, 23],
        'volume':[100,120,110,130,200,120,150,130,180,140,250,160,200,170,300,180,220,190,280,200],
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(pd.Series(range(len(df))), unit='s')

    print("=" * 60)
    print("V2.0 Order Block Analysis")
    print("=" * 60)

    result = analyze_order_blocks(df.copy(), displacement_multiplier=1.5)

    print(f"\nTotal OBs detected: {len(result['all_obs'])}")
    print(f"Fresh OBs: {result['fresh_count']}")
    print(f"Strong OBs: {result['strong_count']}")

    print(f"\nBest Bullish OB:")
    if result['best_bullish_ob']:
        d = result['best_bullish_ob'].as_dict()
        print(f"  Range: {d['ob_start']:.2f} - {d['ob_end']:.2f}")
        print(f"  Displacement: {d['displacement_pct']:.2f}x")
        print(f"  Body: {d['body_size_pct']:.0%}")
        print(f"  Volume: {d['volume_ratio']:.1f}x avg")
        print(f"  Age: {d['freshness_age']} candles")
        print(f"  State: {d['state']}")
        print(f"  Strength: {d['strength']} (score: {d['strength_score']:.2f})")
    else:
        print("  None found")

    print(f"\nBest Bearish OB:")
    if result['best_bearish_ob']:
        d = result['best_bearish_ob'].as_dict()
        print(f"  Range: {d['ob_start']:.2f} - {d['ob_end']:.2f}")
        print(f"  Displacement: {d['displacement_pct']:.2f}x")
        print(f"  Strength: {d['strength']} (score: {d['strength_score']:.2f})")
    else:
        print("  None found")

    print("\nAll Ranked OBs:")
    for ob in result['ranked_obs']:
        d = ob.as_dict()
        print(f"  [{d['ob_type'].upper()}] {d['ob_start']:.2f}-{d['ob_end']:.2f} | "
              f"Strength: {d['strength']} ({d['strength_score']:.2f}) | "
              f"State: {d['state']} | Disp: {d['displacement_pct']:.1f}x")
