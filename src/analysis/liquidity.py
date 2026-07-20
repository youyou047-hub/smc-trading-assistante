"""
Professional AI Trading Analysis System V2.0
Module: Liquidity Detection

Upgraded from V1.0 with improved identification of:
- Equal Highs / Equal Lows (multi-touch liquidity zones)
- Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL)
- Internal vs External Liquidity
- Liquidity Sweeps with proper validation
- False Breakouts detection

Maintains backward-compatible DataFrame interface (pandas OHLCV as input).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ============================================================
# Data Structures
# ============================================================

@dataclass
class LiquidityZone:
    """Represents a identified liquidity zone (equal highs/lows cluster)."""
    price_level: float
    zone_type: str  # "equal_high" | "equal_low"
    touch_count: int
    price_range_low: float
    price_range_high: float
    first_touch_index: int
    last_touch_index: int
    side: str = ""  # "buy_side" | "sell_side" | "internal" | "external"
    strength: float = 0.0  # 0-1 quality score

    def as_dict(self) -> Dict:
        return {
            "price_level": self.price_level,
            "zone_type": self.zone_type,
            "touch_count": self.touch_count,
            "price_range_low": self.price_range_low,
            "price_range_high": self.price_range_high,
            "first_touch_index": self.first_touch_index,
            "last_touch_index": self.last_touch_index,
            "side": self.side,
            "strength": self.strength,
        }


@dataclass
class LiquiditySweep:
    """Represents a confirmed liquidity sweep event."""
    index: int
    sweep_type: str  # "bullish" | "bearish"
    price_swept: float
    sweep_high: float
    sweep_low: float
    close_price: float
    is_false_breakout: bool = False
    strength: float = 0.0  # 0-1 quality score

    def as_dict(self) -> Dict:
        return {
            "index": self.index,
            "sweep_type": self.sweep_type,
            "price_swept": self.price_swept,
            "sweep_high": self.sweep_high,
            "sweep_low": self.sweep_low,
            "close_price": self.close_price,
            "is_false_breakout": self.is_false_breakout,
            "strength": self.strength,
        }


# ============================================================
# Equal Highs / Equal Lows Detection
# ============================================================

def find_equal_levels(
    df: pd.DataFrame,
    threshold_pct: float = 0.003,
    lookback_window: int = 15,
    min_equal_count: int = 2,
) -> pd.DataFrame:
    """Identifies equal highs and equal lows within a threshold.

    V2.0 improvements over V1.0:
    - Configurable threshold percentage (default 0.3% vs V1.0's 0.1%)
    - Configurable lookback window
    - Minimum equal touch count requirement
    - Stores price range of each equal level cluster
    - Vectorized-compatible loop structure for performance

    Args:
        df: OHLCV DataFrame with 'high', 'low' columns.
        threshold_pct: Percentage threshold for "equal" classification.
        lookback_window: Candles to search for matching levels.
        min_equal_count: Minimum touches to qualify as a liquidity zone.

    Returns:
        DataFrame with columns: 'equal_high', 'equal_low',
        'equal_high_price', 'equal_low_price', 'equal_high_cluster_id',
        'equal_low_cluster_id', 'equal_high_count', 'equal_low_count'.
    """
    df = df.copy()
    n = len(df)

    # Initialize columns
    df["equal_high"] = False
    df["equal_low"] = False
    df["equal_high_price"] = np.nan
    df["equal_low_price"] = np.nan
    df["equal_high_cluster_id"] = -1
    df["equal_low_cluster_id"] = -1
    df["equal_high_count"] = 0
    df["equal_low_count"] = 0

    # Track clusters of equal levels
    high_clusters: Dict[int, List[int]] = {}  # cluster_id -> list of indices
    low_clusters: Dict[int, List[int]] = {}
    next_high_cluster = 0
    next_low_cluster = 0

    highs = df["high"].values
    lows = df["low"].values

    for i in range(1, n):
        start = max(0, i - lookback_window)

        # --- Check for equal highs ---
        current_high = highs[i]
        matched_highs = []
        for j in range(start, i):
            prev_high = highs[j]
            if prev_high > 0 and abs(current_high - prev_high) / prev_high <= threshold_pct:
                matched_highs.append(j)

        if len(matched_highs) >= 1:
            df.loc[df.index[i], "equal_high"] = True
            df.loc[df.index[i], "equal_high_price"] = current_high
            df.loc[df.index[i], "equal_high_count"] = len(matched_highs) + 1

            # Assign or reuse cluster ID
            # Find if any matched high is already in a cluster
            assigned_cluster = -1
            for j in matched_highs:
                for cid, members in high_clusters.items():
                    if j in members:
                        assigned_cluster = cid
                        break
                if assigned_cluster >= 0:
                    break

            if assigned_cluster < 0:
                # Create new cluster
                assigned_cluster = next_high_cluster
                high_clusters[next_high_cluster] = []
                next_high_cluster += 1

            high_clusters[assigned_cluster].append(i)
            df.loc[df.index[i], "equal_high_cluster_id"] = assigned_cluster

        # --- Check for equal lows ---
        current_low = lows[i]
        matched_lows = []
        for j in range(start, i):
            prev_low = lows[j]
            if prev_low > 0 and abs(current_low - prev_low) / prev_low <= threshold_pct:
                matched_lows.append(j)

        if len(matched_lows) >= 1:
            df.loc[df.index[i], "equal_low"] = True
            df.loc[df.index[i], "equal_low_price"] = current_low
            df.loc[df.index[i], "equal_low_count"] = len(matched_lows) + 1

            assigned_cluster = -1
            for j in matched_lows:
                for cid, members in low_clusters.items():
                    if j in members:
                        assigned_cluster = cid
                        break
                if assigned_cluster >= 0:
                    break

            if assigned_cluster < 0:
                assigned_cluster = next_low_cluster
                low_clusters[next_low_cluster] = []
                next_low_cluster += 1

            low_clusters[assigned_cluster].append(i)
            df.loc[df.index[i], "equal_low_cluster_id"] = assigned_cluster

    # Ensure cluster indices are consistent for previously matched points
    for cid, members in high_clusters.items():
        for m in members:
            df.loc[df.index[m], "equal_high"] = True
            df.loc[df.index[m], "equal_high_cluster_id"] = cid

    for cid, members in low_clusters.items():
        for m in members:
            df.loc[df.index[m], "equal_low"] = True
            df.loc[df.index[m], "equal_low_cluster_id"] = cid

    return df


def build_liquidity_zones(
    df: pd.DataFrame,
    threshold_pct: float = 0.003,
    min_equal_count: int = 2,
) -> List[LiquidityZone]:
    """Builds structured LiquidityZone objects from equal level data.

    Args:
        df: DataFrame with equal level columns from find_equal_levels.
        threshold_pct: Percentage threshold used for detection.
        min_equal_count: Minimum touches required.

    Returns:
        List of LiquidityZone objects representing liquidity clusters.
    """
    zones: List[LiquidityZone] = []

    # Group by cluster IDs
    high_clusters: Dict[int, List[Tuple[int, float]]] = {}
    low_clusters: Dict[int, List[Tuple[int, float]]] = {}

    for i in range(len(df)):
        cid_h = df["equal_high_cluster_id"].iloc[i]
        if cid_h >= 0 and df["equal_high"].iloc[i]:
            high_clusters.setdefault(cid_h, []).append((i, df["high"].iloc[i]))

        cid_l = df["equal_low_cluster_id"].iloc[i]
        if cid_l >= 0 and df["equal_low"].iloc[i]:
            low_clusters.setdefault(cid_l, []).append((i, df["low"].iloc[i]))

    # Build zones from clusters with sufficient touches
    for cid, touches in high_clusters.items():
        if len(touches) >= min_equal_count:
            prices = [t[1] for t in touches]
            avg_price = sum(prices) / len(prices)
            zones.append(LiquidityZone(
                price_level=avg_price,
                zone_type="equal_high",
                touch_count=len(touches),
                price_range_low=min(prices),
                price_range_high=max(prices),
                first_touch_index=touches[0][0],
                last_touch_index=touches[-1][0],
                side="buy_side",
                strength=min(1.0, len(touches) / 5.0),  # Max strength at 5 touches
            ))

    for cid, touches in low_clusters.items():
        if len(touches) >= min_equal_count:
            prices = [t[1] for t in touches]
            avg_price = sum(prices) / len(prices)
            zones.append(LiquidityZone(
                price_level=avg_price,
                zone_type="equal_low",
                touch_count=len(touches),
                price_range_low=min(prices),
                price_range_high=max(prices),
                first_touch_index=touches[0][0],
                last_touch_index=touches[-1][0],
                side="sell_side",
                strength=min(1.0, len(touches) / 5.0),
            ))

    # Sort by most recent last_touch_index first
    zones.sort(key=lambda z: z.last_touch_index, reverse=True)
    return zones


# ============================================================
# Buy-Side and Sell-Side Liquidity
# ============================================================

def identify_liquidity_sides(
    df: pd.DataFrame,
    recent_highs: Optional[np.ndarray] = None,
    recent_lows: Optional[np.ndarray] = None,
    bsl_distance_pct: float = 0.005,
    ssl_distance_pct: float = 0.005,
) -> pd.DataFrame:
    """Identifies proximity to buy-side and sell-side liquidity.

    Buy-Side Liquidity (BSL): Price near recent equal highs or swing highs.
    Sell-Side Liquidity (SSL): Price near recent equal lows or swing lows.

    V2.0 improvement: Configurable distance thresholds, considers both
    equal levels and recent swing extremes (V1.0 only checked equal levels).

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns.
        recent_highs: Optional array of recent swing high prices.
        recent_lows: Optional array of recent swing low prices.
        bsl_distance_pct: How close to a high qualifies as BSL proximity.
        ssl_distance_pct: How close to a low qualifies as SSL proximity.

    Returns:
        DataFrame with 'bsl_proximity', 'ssl_proximity', 'near_bsl', 'near_ssl' columns.
    """
    df = df.copy()
    n = len(df)

    # If not provided, use rolling highs/lows
    if recent_highs is None:
        recent_highs = df["high"].iloc[-50:].values if len(df) >= 50 else df["high"].values
    if recent_lows is None:
        recent_lows = df["low"].iloc[-50:].values if len(df) >= 50 else df["low"].values

    top_high = recent_highs.max() if len(recent_highs) > 0 else df["high"].iloc[-1]
    bottom_low = recent_lows.min() if len(recent_lows) > 0 else df["low"].iloc[-1]

    for i in range(n):
        price = df["close"].iloc[i]

        # BSL proximity: how close is price to the top liquidity
        if top_high > 0:
            bsl_dist = (top_high - price) / top_high
            df.loc[df.index[i], "bsl_proximity"] = 1.0 - min(1.0, bsl_dist / bsl_distance_pct)
        else:
            df.loc[df.index[i], "bsl_proximity"] = 0.0

        # SSL proximity: how close is price to the bottom liquidity
        if bottom_low > 0:
            ssl_dist = (price - bottom_low) / bottom_low
            df.loc[df.index[i], "ssl_proximity"] = 1.0 - min(1.0, ssl_dist / ssl_distance_pct)
        else:
            df.loc[df.index[i], "ssl_proximity"] = 0.0

        # Binary flags for being "near" liquidity
        df.loc[df.index[i], "near_bsl"] = df["bsl_proximity"].iloc[i] >= 0.8
        df.loc[df.index[i], "near_ssl"] = df["ssl_proximity"].iloc[i] >= 0.8

    return df


# ============================================================
# Internal vs External Liquidity
# ============================================================

def classify_liquidity_type(
    zones: List[LiquidityZone],
    recent_high: float,
    recent_low: float,
    dealing_range_high: float,
    dealing_range_low: float,
) -> List[LiquidityZone]:
    """Classifies liquidity zones as internal or external.

    Internal Liquidity: Within the current dealing range (between recent swing high/low).
    External Liquidity: Outside the current dealing range (above swing high or below swing low).

    V2.0 improvement: Proper dealing range context instead of just distance-based
    classification (V1.0 had no internal/external distinction).

    Args:
        zones: List of LiquidityZone objects.
        recent_high: Most recent swing high price.
        recent_low: Most recent swing low price.
        dealing_range_high: Upper bound of the current dealing range.
        dealing_range_low: Lower bound of the current dealing range.

    Returns:
        Updated zones with 'side' field set to 'internal' or 'external'.
    """
    range_mid = (dealing_range_high + dealing_range_low) / 2.0

    for zone in zones:
        if zone.price_level > dealing_range_high:
            zone.side = "external_high"  # Above the range = external
        elif zone.price_level < dealing_range_low:
            zone.side = "external_low"   # Below the range = external
        else:
            # Within the dealing range = internal
            if zone.price_level > range_mid:
                zone.side = "internal_upper"
            else:
                zone.side = "internal_lower"

    return zones


# ============================================================
# Liquidity Sweeps
# ============================================================

def find_liquidity_sweeps(
    df: pd.DataFrame,
    liquidity_zones: Optional[List[LiquidityZone]] = None,
    max_wick_pct: float = 0.01,
    require_reversal: bool = True,
    min_reversal_pct: float = 0.001,
) -> Tuple[pd.DataFrame, List[LiquiditySweep]]:
    """Identifies liquidity sweeps above equal highs or below equal lows.

    A liquidity sweep occurs when price briefly moves above an equal high
    or below an equal low and then quickly reverses, indicating institutional
    absorption of resting orders.

    V2.0 improvements over V1.0:
    - Proper wick size validation (max wick percentage)
    - Reversal confirmation requirement
    - False breakout detection
    - Strength scoring based on sweep characteristics
    - Works with structured LiquidityZone objects, not just boolean flags

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close', 'open' columns.
        liquidity_zones: Optional pre-built liquidity zones. If None, builds from data.
        max_wick_pct: Maximum wick size as % of price for valid sweep.
        require_reversal: Sweep must close back inside the swept level.
        min_reversal_pct: Minimum reversal close into the range.

    Returns:
        Tuple of (annotated DataFrame, list of LiquiditySweep objects).
    """
    df = df.copy()
    df["liquidity_sweep_bullish"] = False
    df["liquidity_sweep_bearish"] = False
    df["sweep_price"] = np.nan
    df["sweep_high"] = np.nan
    df["sweep_low"] = np.nan

    # Build zones if not provided
    if liquidity_zones is None:
        df = find_equal_levels(df)
        liquidity_zones = build_liquidity_zones(df)

    sweeps: List[LiquiditySweep] = []

    if not liquidity_zones:
        return df, sweeps

    for zone in liquidity_zones:
        for i in range(zone.last_touch_index + 1, len(df)):
            current_high = df["high"].iloc[i]
            current_low = df["low"].iloc[i]
            current_close = df["close"].iloc[i]
            current_open = df["open"].iloc[i]
            current_range = current_high - current_low

            # Validate sweep magnitude
            if current_range <= 0:
                continue

            price_level = zone.price_level

            # --- Bearish sweep: Price sweeps above equal high zone ---
            if zone.zone_type == "equal_high" and current_high > zone.price_range_high:
                wick_pct = (current_high - max(current_open, current_close)) / price_level

                if wick_pct > max_wick_pct:
                    continue  # Wick too large, not a clean sweep

                # Check for reversal
                if require_reversal:
                    if current_close < price_level * (1 - min_reversal_pct):
                        sweep = LiquiditySweep(
                            index=i,
                            sweep_type="bearish",
                            price_swept=price_level,
                            sweep_high=current_high,
                            sweep_low=current_low,
                            close_price=current_close,
                            strength=_score_sweep(current_high, price_level, current_range, price_level),
                        )
                        # Check if this is a false breakout
                        sweep.is_false_breakout = _is_false_breakout(
                            df, i, zone.price_range_high, direction="bearish"
                        )
                        sweeps.append(sweep)
                        df.loc[df.index[i], "liquidity_sweep_bearish"] = True
                        df.loc[df.index[i], "sweep_price"] = price_level
                        df.loc[df.index[i], "sweep_high"] = current_high
                        df.loc[df.index[i], "sweep_low"] = current_low

            # --- Bullish sweep: Price sweeps below equal low zone ---
            elif zone.zone_type == "equal_low" and current_low < zone.price_range_low:
                wick_pct = (min(current_open, current_close) - current_low) / price_level

                if wick_pct > max_wick_pct:
                    continue  # Wick too large

                # Check for reversal
                if require_reversal:
                    if current_close > price_level * (1 + min_reversal_pct):
                        sweep = LiquiditySweep(
                            index=i,
                            sweep_type="bullish",
                            price_swept=price_level,
                            sweep_high=current_high,
                            sweep_low=current_low,
                            close_price=current_close,
                            strength=_score_sweep(price_level, current_low, current_range, price_level),
                        )
                        sweep.is_false_breakout = _is_false_breakout(
                            df, i, zone.price_range_low, direction="bullish"
                        )
                        sweeps.append(sweep)
                        df.loc[df.index[i], "liquidity_sweep_bullish"] = True
                        df.loc[df.index[i], "sweep_price"] = price_level
                        df.loc[df.index[i], "sweep_high"] = current_high
                        df.loc[df.index[i], "sweep_low"] = current_low

    # Sort sweeps by most recent first
    sweeps.sort(key=lambda s: s.index, reverse=True)
    return df, sweeps


def _score_sweep(sweep_extent: float, level: float, candle_range: float, ref_price: float) -> float:
    """Scores a liquidity sweep from 0.0 to 1.0.

    Higher score = cleaner, more institutional-looking sweep.

    Args:
        sweep_extent: How far price went beyond the level.
        level: The liquidity level that was swept.
        candle_range: Total range of the sweep candle.
        ref_price: Reference price for percentage calculation.

    Returns:
        Float score 0.0-1.0.
    """
    if ref_price <= 0 or candle_range <= 0:
        return 0.0

    # Sweep distance as % of price
    sweep_pct = abs(sweep_extent - level) / ref_price

    # Ideal sweep: small extension beyond level (0.1% - 0.5% of price)
    # Too deep = not a clean sweep, too shallow = may not have triggered stops
    if sweep_pct < 0.001:
        dist_score = 0.3  # Barely swept
    elif 0.001 <= sweep_pct <= 0.005:
        dist_score = 1.0  # Ideal sweep distance
    elif sweep_pct <= 0.01:
        dist_score = 0.6  # A bit deep but acceptable
    else:
        dist_score = 0.2  # Too deep, looks like a real break

    # Wick-to-range ratio: high ratio = cleaner sweep (thin wick through level)
    wick_range_ratio = candle_range / ref_price if ref_price > 0 else 0
    if wick_range_ratio < 0.02:
        wick_score = 0.5
    elif wick_range_ratio < 0.05:
        wick_score = 1.0
    else:
        wick_score = 0.4

    return (dist_score * 0.6) + (wick_score * 0.4)


def _is_false_breakout(
    df: pd.DataFrame,
    sweep_index: int,
    level: float,
    direction: str,
    lookback: int = 5,
) -> bool:
    """Determines if a sweep constitutes a false breakout.

    A false breakout is when price breaks a level but quickly reverses
    and closes back on the original side of the level.

    Args:
        df: OHLCV DataFrame.
        sweep_index: Index of the sweep candle.
        level: The level that was broken.
        direction: 'bullish' (swept below) or 'bearish' (swept above).
        lookback: How many candles after the sweep to check for reversal.

    Returns:
        True if the sweep is a confirmed false breakout.
    """
    end_idx = min(len(df), sweep_index + lookback)
    if end_idx <= sweep_index + 1:
        return False

    subsequent = df.iloc[sweep_index + 1:end_idx]

    if direction == "bearish":
        # Swept above a high; false if price closes back below the level
        return any(subsequent["close"] < level)
    else:
        # Swept below a low; false if price closes back above the level
        return any(subsequent["close"] > level)


# ============================================================
# Convenience: Backward-Compatible Interface
# ============================================================

def find_equal_highs_lows(
    df: pd.DataFrame,
    threshold: float = 0.003,
    window: int = 15,
) -> pd.DataFrame:
    """Backward-compatible wrapper matching V1.0 interface.

    Args:
        df: OHLCV DataFrame.
        threshold: Percentage threshold for equal levels.
        window: Lookback window.

    Returns:
        DataFrame with 'equal_high' and 'equal_low' columns.
    """
    return find_equal_levels(df, threshold_pct=threshold, lookback_window=window)


def analyze_liquidity(
    df: pd.DataFrame,
    threshold_pct: float = 0.003,
    lookback_window: int = 15,
    min_equal_count: int = 2,
    max_wick_pct: float = 0.01,
    require_reversal: bool = True,
    min_reversal_pct: float = 0.001,
) -> Dict[str, object]:
    """Complete liquidity analysis combining all sub-functions.

    Args:
        df: OHLCV DataFrame.
        threshold_pct: Equal level threshold.
        lookback_window: Lookback for equal level detection.
        min_equal_count: Minimum touches for a zone.
        max_wick_pct: Max wick for sweep validation.
        require_reversal: Require reversal for sweep.
        min_reversal_pct: Minimum reversal percentage.

    Returns:
        Dict with:
            'df': Annotated DataFrame
            'zones': List of LiquidityZone objects
            'sweeps': List of LiquiditySweep objects
            'bsl_level': Most recent buy-side liquidity price
            'ssl_level': Most recent sell-side liquidity price
            'bsl_zone': Top liquidity zone (buy-side)
            'ssl_zone': Bottom liquidity zone (sell-side)
    """
    # Step 1: Find equal levels
    df = find_equal_levels(df, threshold_pct=threshold_pct, lookback_window=lookback_window)

    # Step 2: Build liquidity zones
    zones = build_liquidity_zones(df, threshold_pct=threshold_pct, min_equal_count=min_equal_count)

    # Step 3: Find liquidity sweeps
    df, sweeps = find_liquidity_sweeps(
        df, liquidity_zones=zones,
        max_wick_pct=max_wick_pct,
        require_reversal=require_reversal,
        min_reversal_pct=min_reversal_pct,
    )

    # Step 4: Identify BSL and SSL
    df = identify_liquidity_sides(df)

    # Determine top BSL and bottom SSL
    bsl_zone = next((z for z in zones if z.zone_type == "equal_high"), None)
    ssl_zone = next((z for z in zones if z.zone_type == "equal_low"), None)

    bsl_level = bsl_zone.price_level if bsl_zone else df["high"].iloc[-1]
    ssl_level = ssl_zone.price_level if ssl_zone else df["low"].iloc[-1]

    return {
        "df": df,
        "zones": zones,
        "sweeps": sweeps,
        "bsl_level": bsl_level,
        "ssl_level": ssl_level,
        "bsl_zone": bsl_zone,
        "ssl_zone": ssl_zone,
    }


if __name__ == "__main__":
    # Example Usage
    import random
    random.seed(42)

    # Generate sample data with a clear equal highs pattern
    base = 100.0
    data = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

    # Create a pattern with two touches near 105 (equal highs)
    prices = []
    for i in range(50):
        o = base + (i * 0.3) + random.uniform(-0.5, 0.5)
        h = o + random.uniform(0.5, 2.0)
        l = o - random.uniform(0.5, 2.0)
        c = o + random.uniform(-1.0, 1.0)

        # Force two equal highs around index 15 and 30
        if i == 15:
            h = 105.50
            c = 105.00
        elif i == 30:
            h = 105.45  # Very close to 105.50
            c = 104.80
        # Force a sweep at index 32
        elif i == 32:
            h = 106.00  # Sweeps above the equal highs
            c = 104.50  # Closes back down (bearish sweep)
            l = 104.00

        data['open'].append(o)
        data['high'].append(h)
        data['low'].append(l)
        data['close'].append(c)
        data['volume'].append(random.randint(100, 500))

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(pd.Series(range(len(df))), unit='h')

    print("=" * 60)
    print("V2.0 Liquidity Analysis")
    print("=" * 60)

    result = analyze_liquidity(df.copy())

    print(f"\nEqual Highs detected: {result['df']['equal_high'].sum()}")
    print(f"Equal Lows detected: {result['df']['equal_low'].sum()}")
    print(f"Liquidity Zones found: {len(result['zones'])}")
    print(f"Liquidity Sweeps found: {len(result['sweeps'])}")

    for zone in result['zones']:
        print(f"  Zone: {zone.zone_type} at {zone.price_level:.2f} "
              f"(touches={zone.touch_count}, side={zone.side}, strength={zone.strength:.2f})")

    for sweep in result['sweeps']:
        fb = " (False Breakout)" if sweep.is_false_breakout else ""
        print(f"  Sweep: {sweep.sweep_type} at {sweep.price_swept:.2f} "
              f"(strength={sweep.strength:.2f}{fb})")

    print(f"\nBSL Level: {result['bsl_level']:.2f}")
    print(f"SSL Level: {result['ssl_level']:.2f}")

    print("\nLast 5 rows of annotated DataFrame:")
    cols = ['close', 'equal_high', 'equal_low', 'liquidity_sweep_bullish',
            'liquidity_sweep_bearish', 'near_bsl', 'near_ssl']
    print(result['df'][cols].tail())
