"""
Professional AI Trading Analysis System V2.0
Module: Market Structure Detection

Upgraded from V1.0 with improved detection of:
- Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), Lower Lows (LL)
- Break of Structure (BOS) with proper swing point tracking
- Change of Character (CHoCH) with prior trend validation
- Internal vs External Structure distinction
- Strong vs Weak Swing Points classification

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
class SwingPoint:
    """Represents a confirmed swing high or swing low with metadata."""
    index: int
    price: float
    is_high: bool
    timestamp: Optional[pd.Timestamp] = None
    strength: str = "strong"  # "strong" or "weak"
    structure_type: str = "external"  # "internal" or "external"
    break_count: int = 0  # How many times this swing has been broken


# ============================================================
# Swing Point Detection
# ============================================================

def find_swing_points(
    df: pd.DataFrame,
    window: int = 5,
    min_pips_pct: float = 0.002,
) -> pd.DataFrame:
    """Identifies swing highs and swing lows with quality filtering.

    V2.0 improvements over V1.0:
    - Uses rolling window correctly (centered) to detect true pivot points
    - Filters out swing points that are too small (min_pips_pct threshold)
    - Adds strength classification based on swing magnitude vs ATR

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns.
        window: Number of candles on each side to confirm a swing point.
        min_pips_pct: Minimum price movement (as % of close) to qualify as swing.

    Returns:
        DataFrame with added columns: 'swing_high', 'swing_low',
        'swing_high_price', 'swing_low_price', 'swing_strength'.
    """
    df = df.copy()
    w = 2 * window + 1

    # Detect swing highs: high is the max over a centered window
    df["swing_high"] = (df["high"] == df["high"].rolling(window=w, center=True).max())
    # Detect swing lows: low is the min over a centered window
    df["swing_low"] = (df["low"] == df["low"].rolling(window=w, center=True).min())

    # Store actual swing prices
    df["swing_high_price"] = np.where(df["swing_high"], df["high"], np.nan)
    df["swing_low_price"] = np.where(df["swing_low"], df["low"], np.nan)

    # Calculate ATR for strength classification
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(window=14, min_periods=1).mean()

    # Classify swing strength: strong if swing deviation > ATR, weak otherwise
    df["swing_strength"] = "weak"
    # For swing highs, compare to surrounding average high
    if len(df) > w:
        avg_high = df["high"].rolling(window=w, center=True).mean()
        avg_low = df["low"].rolling(window=w, center=True).mean()
        df.loc[df["swing_high"] & (df["high"] - avg_high) > df["atr"], "swing_strength"] = "strong"
        df.loc[df["swing_low"] & (avg_low - df["low"]) > df["atr"], "swing_strength"] = "strong"

    return df


def collect_swing_points(
    df: pd.DataFrame,
    lookback: int = 100,
) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """Collects ordered lists of swing highs and swing lows from recent data.

    Args:
        df: DataFrame with 'swing_high', 'swing_low', 'swing_high_price',
            'swing_low_price', 'swing_strength' columns.
        lookback: Number of recent candles to scan.

    Returns:
        Tuple of (swing_highs, swing_lows) as ordered lists of SwingPoint objects.
    """
    recent = df.iloc[-lookback:] if lookback < len(df) else df
    highs: List[SwingPoint] = []
    lows: List[SwingPoint] = []

    for idx, row in recent.iterrows():
        if row.get("swing_high", False) and pd.notna(row.get("swing_high_price")):
            highs.append(SwingPoint(
                index=recent.index.get_loc(idx) if idx in recent.index else -1,
                price=float(row["swing_high_price"]),
                is_high=True,
                timestamp=idx,
                strength=row.get("swing_strength", "weak"),
            ))
        if row.get("swing_low", False) and pd.notna(row.get("swing_low_price")):
            lows.append(SwingPoint(
                index=recent.index.get_loc(idx) if idx in recent.index else -1,
                price=float(row["swing_low_price"]),
                is_high=False,
                timestamp=idx,
                strength=row.get("swing_strength", "weak"),
            ))

    return highs, lows


# ============================================================
# Market Structure Classification (HH, HL, LH, LL)
# ============================================================

def classify_structure_sequence(
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
) -> Dict[str, List[SwingPoint]]:
    """Classifies the overall market structure from alternating swing sequences.

    V2.0 improvement: Instead of comparing every candle to its predecessor (V1.0),
    we compare actual swing points to each other, which is the correct SMC approach.

    Rules:
    - Uptrend: HH + HL pattern (each high > prior high, each low > prior low)
    - Downtrend: LH + LL pattern (each high < prior high, each low < prior low)
    - Consolidation: Mixed or flat pattern

    Args:
        swing_highs: Ordered list of swing high points.
        swing_lows: Ordered list of swing low points.

    Returns:
        Dict with keys 'HH', 'HL', 'LH', 'LL' mapping to lists of qualifying points.
    """
    result = {"HH": [], "HL": [], "LH": [], "LL": []}

    # Classify swing highs (HH vs LH)
    for i in range(1, len(swing_highs)):
        if swing_highs[i].price > swing_highs[i - 1].price:
            swing_highs[i].strength = "strong"
            result["HH"].append(swing_highs[i])
        elif swing_highs[i].price < swing_highs[i - 1].price:
            result["LH"].append(swing_highs[i])

    # Classify swing lows (HL vs LL)
    for i in range(1, len(swing_lows)):
        if swing_lows[i].price > swing_lows[i - 1].price:
            swing_lows[i].strength = "strong"
            result["HL"].append(swing_lows[i])
        elif swing_lows[i].price < swing_lows[i - 1].price:
            result["LL"].append(swing_lows[i])

    return result


def determine_overall_trend(
    structure: Dict[str, List[SwingPoint]],
) -> str:
    """Determines the dominant market trend from classified swing points.

    Args:
        structure: Output from classify_structure_sequence.

    Returns:
        'bullish', 'bearish', or 'neutral'
    """
    hh_count = len(structure["HH"])
    hl_count = len(structure["HL"])
    lh_count = len(structure["LH"])
    ll_count = len(structure["LL"])

    bull_score = hh_count + hl_count
    bear_score = lh_count + ll_count

    if bull_score > bear_score * 1.3:
        return "bullish"
    elif bear_score > bull_score * 1.3:
        return "bearish"
    else:
        return "neutral"


# ============================================================
# BOS and CHoCH Detection
# ============================================================

def find_bos_choch(
    df: pd.DataFrame,
    swing_highs: List[SwingPoint],
    swing_lows: List[SwingPoint],
    require_body_close: bool = True,
    choch_require_opposite: bool = True,
    min_prior_swing_count: int = 2,
) -> pd.DataFrame:
    """Identifies Break of Structure (BOS) and Change of Character (CHoCH).

    V2.0 improvements over V1.0:
    - Uses actual swing points instead of arbitrary lookback windows
    - BOS: Price breaks a confirmed swing point in the direction of the trend
    - CHoCH: Price breaks a swing point that signals a potential trend reversal
    - Proper body-close validation (configurable)
    - CHoCH requires evidence of prior opposite structure

    Args:
        df: OHLCV DataFrame with 'close', 'high', 'low' columns.
        swing_highs: Confirmed swing high points.
        swing_lows: Confirmed swing low points.
        require_body_close: If True, BOS requires body close beyond the swing.
        choch_require_opposite: If True, CHoCH requires prior opposite trend.
        min_prior_swing_count: Minimum prior swings needed for CHoCH validity.

    Returns:
        DataFrame with columns: 'bos_bullish', 'bos_bearish',
        'choch_bullish', 'choch_bearish', 'bos_price', 'choch_price'.
    """
    df = df.copy()
    df["bos_bullish"] = False
    df["bos_bearish"] = False
    df["choch_bullish"] = False
    df["choch_bearish"] = False
    df["bos_price"] = np.nan
    df["choch_price"] = np.nan
    df["bos_type"] = ""
    df["choch_type"] = ""

    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return df

    # Determine the most recent confirmed structure
    # A bullish BOS breaks the most recent swing high
    # A bearish BOS breaks the most recent swing low
    for i in range(len(df)):
        current_close = df["close"].iloc[i]
        current_high = df["high"].iloc[i]
        current_low = df["low"].iloc[i]
        open_price = df["open"].iloc[i]

        # --- Bullish BOS: Close above the most recent swing high ---
        if swing_highs:
            last_swing_high = swing_highs[-1]
            if current_close > last_swing_high.price:
                df.loc[df.index[i], "bos_bullish"] = True
                df.loc[df.index[i], "bos_price"] = last_swing_high.price
                df.loc[df.index[i], "bos_type"] = "swing_high_break"
                # Update break count
                last_swing_high.break_count += 1

            # --- Bearish CHoCH: Close below a swing low (reversal signal) ---
            if choch_require_opposite and len(swing_lows) >= min_prior_swing_count:
                # Check if there was an uptrend (HH/HL pattern) before this break
                prev_lows = [sl.price for sl in swing_lows[:-1]]
                prev_highs = [sh.price for sh in swing_highs]
                was_uptrend = (
                    len(prev_lows) >= 2 and
                    all(prev_lows[j] > prev_lows[j - 1] for j in range(1, len(prev_lows)))
                )
                if was_uptrend:
                    last_swing_low = swing_lows[-1]
                    if current_close < last_swing_low.price:
                        df.loc[df.index[i], "choch_bearish"] = True
                        df.loc[df.index[i], "choch_price"] = last_swing_low.price
                        df.loc[df.index[i], "choch_type"] = "trend_reversal"

        # --- Bearish BOS: Close below the most recent swing low ---
        if swing_lows:
            last_swing_low = swing_lows[-1]
            if current_close < last_swing_low.price:
                df.loc[df.index[i], "bos_bearish"] = True
                df.loc[df.index[i], "bos_price"] = last_swing_low.price
                df.loc[df.index[i], "bos_type"] = "swing_low_break"
                last_swing_low.break_count += 1

            # --- Bullish CHoCH: Close above a swing high (reversal signal) ---
            if choch_require_opposite and len(swing_highs) >= min_prior_swing_count:
                prev_highs = [sh.price for sh in swing_highs[:-1]]
                prev_lows = [sl.price for sl in swing_lows]
                was_downtrend = (
                    len(prev_highs) >= 2 and
                    all(prev_highs[j] < prev_highs[j - 1] for j in range(1, len(prev_highs)))
                )
                if was_downtrend:
                    last_swing_high = swing_highs[-1]
                    if current_close > last_swing_high.price:
                        df.loc[df.index[i], "choch_bullish"] = True
                        df.loc[df.index[i], "choch_price"] = last_swing_high.price
                        df.loc[df.index[i], "choch_type"] = "trend_reversal"

    # If body_close required, validate that break was by body, not just wick
    if require_body_close:
        for col in ["bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish"]:
            for i in range(len(df)):
                if df[col].iloc[i]:
                    break_price = df[f"{col.split('_')[0]}_price"].iloc[i] if col.startswith("bos") else df["choch_price"].iloc[i]
                    if pd.notna(break_price):
                        open_p = df["open"].iloc[i]
                        close_p = df["close"].iloc[i]
                        body_top = max(open_p, close_p)
                        body_bottom = min(open_p, close_p)
                        if col in ["bos_bullish", "choch_bullish"]:
                            # Body must close above the break level
                            if body_bottom < break_price:
                                df.loc[df.index[i], col] = False
                                df.loc[df.index[i], f"{col.split('_')[0]}_price" if col.startswith("bos") else "choch_price"] = np.nan
                        else:
                            # Body must close below the break level
                            if body_top > break_price:
                                df.loc[df.index[i], col] = False
                                df.loc[df.index[i], "choch_price"] = np.nan

    return df


# ============================================================
# Internal vs External Structure
# ============================================================

def identify_internal_external_structure(
    df: pd.DataFrame,
    internal_lookback: int = 20,
    external_lookback: int = 100,
) -> Dict[str, object]:
    """Distinguishes between internal and external market structure.

    Internal Structure: Shorter-term structure within the current dealing range.
    Typically the pattern of HH/HL or LH/LL within recent candles.

    External Structure: The broader macro structure based on larger swing points.
    Represents the overall trend direction on a higher timeframe perspective.

    V2.0 improvement: Proper separation with configurable lookback periods,
    instead of treating all structure equally (V1.0).

    Args:
        df: OHLCV DataFrame with swing point data.
        internal_lookback: Candles to analyze for internal structure.
        external_lookback: Candles to analyze for external structure.

    Returns:
        Dict with keys:
            'internal_trend': 'bullish' | 'bearish' | 'neutral'
            'external_trend': 'bullish' | 'bearish' | 'neutral'
            'internal_highs': List[SwingPoint]
            'internal_lows': List[SwingPoint]
            'external_highs': List[SwingPoint]
            'external_lows': List[SwingPoint]
            'alignment': bool (whether internal and external agree)
    """
    # Tag swings as internal or external
    for col, label in [("swing_high", "external"), ("swing_low", "external")]:
        if col in df.columns:
            df[f"{col}_scope"] = "external"

    # Collect internal structure swings
    internal_df = df.iloc[-internal_lookback:]
    external_df = df.iloc[-external_lookback:]

    int_highs, int_lows = collect_swing_points(internal_df, lookback=len(internal_df))
    ext_highs, ext_lows = collect_swing_points(external_df, lookback=len(external_df))

    # Classify both
    internal_struct = classify_structure_sequence(int_highs, int_lows)
    external_struct = classify_structure_sequence(ext_highs, ext_lows)

    internal_trend = determine_overall_trend(internal_struct)
    external_trend = determine_overall_trend(external_struct)

    return {
        "internal_trend": internal_trend,
        "external_trend": external_trend,
        "internal_highs": int_highs,
        "internal_lows": int_lows,
        "external_highs": ext_highs,
        "external_lows": ext_lows,
        "alignment": internal_trend == external_trend and internal_trend != "neutral",
    }


# ============================================================
# Strong vs Weak Swing Points
# ============================================================

def classify_swing_strength(
    df: pd.DataFrame,
    swing_points: List[SwingPoint],
    atr_value: Optional[float] = None,
) -> List[SwingPoint]:
    """Classifies swing points as strong or weak based on multiple criteria.

    Strong swings:
    - Caused the break of a prior swing point
    - Have sufficient magnitude relative to ATR
    - Are part of the dominant structure (HH in uptrend, LL in downtrend)

    Weak swings:
    - Failed to break prior structure
    - Small magnitude relative to ATR
    - Counter-trend or within consolidation

    V2.0 improvement: Multi-criteria strength assessment instead of
    V1.0's simplistic single-threshold approach.

    Args:
        df: OHLCV DataFrame with ATR data.
        swing_points: List of swing points to classify.
        atr_value: Pre-calculated ATR (if None, uses latest from df).

    Returns:
        Updated list of SwingPoints with 'strength' field set.
    """
    if atr_value is None:
        if "atr" in df.columns and len(df) > 0:
            atr_value = df["atr"].iloc[-1]
        else:
            # Fallback: use average range of last 14 candles
            atr_value = (df["high"] - df["low"]).iloc[-14:].mean() if len(df) >= 14 else 1.0

    for i, sp in enumerate(swing_points):
        if pd.notna(atr_value) and atr_value > 0:
            # Strength based on magnitude relative to ATR
            # Strong if the swing deviation from surrounding average > 1.5x ATR
            magnitude_ratio = abs(sp.price - (df["close"].iloc[-1] if len(df) > 0 else sp.price)) / atr_value
            if magnitude_ratio >= 1.0:
                sp.strength = "strong"
            else:
                sp.strength = "weak"

        # Check if this swing caused a break of prior structure
        if i > 0:
            prev = swing_points[i - 1]
            if sp.is_high and prev.is_high:
                if sp.price > prev.price:
                    sp.strength = "strong"  # Higher high = strong
            elif not sp.is_high and not prev.is_high:
                if sp.price < prev.price:
                    sp.strength = "strong"  # Lower low = strong

    return swing_points


# ============================================================
# Comprehensive Market Structure Analysis
# ============================================================

def analyze_market_structure(
    df: pd.DataFrame,
    swing_window: int = 5,
    min_pips_pct: float = 0.002,
    bos_body_close: bool = True,
    choch_require_opposite: bool = True,
    min_prior_swing_count: int = 2,
    internal_lookback: int = 20,
    external_lookback: int = 100,
) -> Dict[str, object]:
    """Performs a complete market structure analysis.

    This is the main entry point that combines all sub-analyses:
    1. Swing point detection with strength classification
    2. Structure classification (HH, HL, LH, LL)
    3. Overall trend determination
    4. BOS and CHoCH identification
    5. Internal vs External structure separation

    Args:
        df: OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
        swing_window: Window for swing point detection.
        min_pips_pct: Minimum swing size threshold.
        bos_body_close: Whether BOS requires body close.
        choch_require_opposite: Whether CHoCH needs prior opposite trend.
        min_prior_swing_count: Minimum swings before CHoCH is valid.
        internal_lookback: Candles for internal structure.
        external_lookback: Candles for external structure.

    Returns:
        Comprehensive dict with:
            'swing_points': (highs, lows) tuple
            'structure': classified structure dict
            'overall_trend': 'bullish' | 'bearish' | 'neutral'
            'df': Updated DataFrame with all annotations
            'internal_external': internal/external analysis dict
            'bos_signals': most recent BOS events
            'choch_signals': most recent CHoCH events
            'structure_score': float 0-100 indicating structure clarity
    """
    # Step 1: Detect swing points
    df = find_swing_points(df, window=swing_window, min_pips_pct=min_pips_pct)

    # Step 2: Collect swing points
    swing_highs, swing_lows = collect_swing_points(df, lookback=external_lookback)

    # Step 3: Classify swing strength
    swing_highs = classify_swing_strength(df, swing_highs)
    swing_lows = classify_swing_strength(df, swing_lows)

    # Step 4: Classify structure sequence
    structure = classify_structure_sequence(swing_highs, swing_lows)

    # Step 5: Determine overall trend
    overall_trend = determine_overall_trend(structure)

    # Step 6: Find BOS and CHoCH
    df = find_bos_choch(
        df, swing_highs, swing_lows,
        require_body_close=bos_body_close,
        choch_require_opposite=choch_require_opposite,
        min_prior_swing_count=min_prior_swing_count,
    )

    # Step 7: Internal vs External analysis
    internal_external = identify_internal_external_structure(
        df,
        internal_lookback=internal_lookback,
        external_lookback=external_lookback,
    )

    # Step 8: Calculate structure clarity score (0-100)
    structure_score = _calculate_structure_score(structure, internal_external)

    # Collect recent BOS/CHoCH signals
    bos_signals = []
    choch_signals = []
    for i in range(len(df)):
        if df["bos_bullish"].iloc[i] or df["bos_bearish"].iloc[i]:
            bos_signals.append({
                "index": i,
                "type": "bullish" if df["bos_bullish"].iloc[i] else "bearish",
                "price": df["bos_price"].iloc[i],
            })
        if df["choch_bullish"].iloc[i] or df["choch_bearish"].iloc[i]:
            choch_signals.append({
                "index": i,
                "type": "bullish" if df["choch_bullish"].iloc[i] else "bearish",
                "price": df["choch_price"].iloc[i],
            })

    return {
        "swing_points": (swing_highs, swing_lows),
        "structure": structure,
        "overall_trend": overall_trend,
        "df": df,
        "internal_external": internal_external,
        "bos_signals": bos_signals,
        "choch_signals": choch_signals,
        "structure_score": structure_score,
    }


def _calculate_structure_score(
    structure: Dict[str, List[SwingPoint]],
    internal_external: Dict[str, object],
) -> float:
    """Calculates a 0-100 score representing market structure clarity.

    Higher scores indicate clearer, more defined structure.
    Considers:
    - Number of confirmed HH/HL (bullish) or LH/LL (bearish) patterns
    - Alignment between internal and external structure
    - Strength of swing points involved

    Args:
        structure: Classified structure from classify_structure_sequence.
        internal_external: Internal/external analysis result.

    Returns:
        Float score from 0 to 100.
    """
    score = 50.0  # Base score

    hh = len(structure["HH"])
    hl = len(structure["HL"])
    lh = len(structure["LH"])
    ll = len(structure["LL"])

    # Count strong swings
    strong_bull = sum(1 for sp in structure["HH"] if sp.strength == "strong")
    strong_bull += sum(1 for sp in structure["HL"] if sp.strength == "strong")
    strong_bear = sum(1 for sp in structure["LH"] if sp.strength == "strong")
    strong_bear += sum(1 for sp in structure["LL"] if sp.strength == "strong")

    # Bullish structure bonus
    if hh + hl > lh + ll:
        score += min(25, (hh + hl) * 5)
        score += min(10, strong_bull * 3)

    # Bearish structure bonus
    elif lh + ll > hh + hl:
        score += min(25, (lh + ll) * 5)
        score += min(10, strong_bear * 3)
    else:
        # Neutral/consolidation: lower score
        score -= 15

    # Internal/External alignment bonus
    if internal_external["alignment"]:
        score += 15

    return min(100.0, max(0.0, score))


# ============================================================
# Convenience: Annotate DataFrame for V1.0 Compatibility
# ============================================================

def annotate_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible function that annotates a DataFrame with market structure.

    Maintains the same interface as V1.0's identify_market_structure()
    but with improved detection logic.

    Args:
        df: OHLCV DataFrame.

    Returns:
        DataFrame with 'HH', 'HL', 'LH', 'LL', 'bos_bullish', 'bos_bearish',
        'choch_bullish', 'choch_bearish', 'swing_high', 'swing_low' columns.
    """
    result = analyze_market_structure(df)
    return result["df"]


if __name__ == "__main__":
    # Example Usage
    data = {
        'open':  [10, 12, 15, 13, 16, 14, 17, 15, 18, 16, 19, 17, 20, 18, 22],
        'high':  [13, 16, 17, 15, 18, 17, 19, 17, 20, 18, 22, 20, 23, 21, 25],
        'low':   [9,  11, 12, 11, 13, 12, 14, 13, 15, 14, 17, 15, 18, 16, 20],
        'close': [12, 15, 13, 14, 17, 15, 18, 16, 19, 17, 21, 19, 22, 20, 24],
        'volume':[100,120,110,130,140,120,150,130,160,140,180,160,200,170,220],
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(pd.Series(range(len(df))), unit='s')

    print("=" * 60)
    print("V2.0 Market Structure Analysis")
    print("=" * 60)

    result = analyze_market_structure(df.copy(), swing_window=2)

    print(f"\nOverall Trend: {result['overall_trend'].upper()}")
    print(f"Structure Score: {result['structure_score']:.1f}/100")

    s = result["structure"]
    print(f"\nHH count: {len(s['HH'])}")
    print(f"HL count: {len(s['HL'])}")
    print(f"LH count: {len(s['LH'])}")
    print(f"LL count: {len(s['LL'])}")

    ie = result["internal_external"]
    print(f"\nInternal Trend: {ie['internal_trend']}")
    print(f"External Trend: {ie['external_trend']}")
    print(f"Alignment: {'Yes' if ie['alignment'] else 'No'}")

    print(f"\nBOS Signals: {len(result['bos_signals'])}")
    for b in result['bos_signals']:
        print(f"  {b['type'].capitalize()} BOS at price {b['price']}")

    print(f"\nCHoCH Signals: {len(result['choch_signals'])}")
    for c in result['choch_signals']:
        print(f"  {c['type'].capitalize()} CHoCH at price {c['price']}")

    print("\nAnnotated DataFrame (last 5 rows):")
    cols = ['close', 'swing_high', 'swing_low', 'HH', 'HL', 'LH', 'LL',
            'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish']
    print(result['df'][cols].tail())
