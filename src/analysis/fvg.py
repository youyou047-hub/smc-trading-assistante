"""
Professional AI Trading Analysis System V2.0
Module: Fair Value Gap (FVG) Detection

Upgraded from V1.0 with improved FVG handling:
- Ignore insignificant gaps (minimum size threshold)
- Prefer fresh, untouched FVGs over mitigated ones
- Track mitigation status over time
- Measure FVG quality (not just existence)
- Expiration tracking for stale FVGs

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

class FVGState(Enum):
    """State of a Fair Value Gap."""
    FRESH = "fresh"           # Not yet touched by price
    PARTIALLY_FILLED = "partially_filled"  # Price entered the gap but didn't fully fill
    MITIGATED = "mitigated"   # Fully filled/mitigated
    EXPIRED = "expired"       # Too old to be relevant


@dataclass
class FairValueGap:
    """Represents a detected Fair Value Gap with quality metadata."""
    index: int
    fvg_type: str  # "bullish" | "bearish"
    gap_start: float  # Lower boundary of the gap
    gap_end: float    # Upper boundary of the gap
    gap_size: float   # Absolute size of the gap
    gap_size_pct: float  # Gap size as % of reference price
    timestamp: Optional[pd.Timestamp] = None
    state: FVGState = FVGState.FRESH
    fill_pct: float = 0.0  # Percentage of gap that has been filled (0-100)
    quality_score: float = 0.0  # 0-1 overall quality score
    is_relevant: bool = True  # Whether this FVG is still tradable
    age_candles: int = 0  # How many candles old this FVG is
    volume_ratio: float = 1.0  # Volume at creation vs average
    body_overlap: float = 0.0  # How much the creating candle's body overlaps the gap

    def as_dict(self) -> Dict:
        return {
            "index": self.index,
            "fvg_type": self.fvg_type,
            "gap_start": self.gap_start,
            "gap_end": self.gap_end,
            "gap_size": self.gap_size,
            "gap_size_pct": self.gap_size_pct,
            "state": self.state.value,
            "fill_pct": self.fill_pct,
            "quality_score": self.quality_score,
            "is_relevant": self.is_relevant,
            "age_candles": self.age_candles,
            "volume_ratio": self.volume_ratio,
            "body_overlap": self.body_overlap,
        }


# ============================================================
# FVG Detection
# ============================================================

def find_fair_value_gaps(
    df: pd.DataFrame,
    min_gap_pct: float = 0.003,
    max_gap_pct: float = 0.05,
) -> pd.DataFrame:
    """Identifies Fair Value Gaps based on the 3-candle pattern.

    V2.0 improvements over V1.0:
    - Minimum gap size filter (ignores insignificant gaps)
    - Maximum gap size filter (ignores anomalies)
    - Calculates gap size percentage for quality scoring
    - Uses vectorized operations where possible

    Bullish FVG (3-candle pattern):
        Low of candle 3 > High of candle 1
        The gap between candle 1 high and candle 3 low is the FVG.

    Bearish FVG (3-candle pattern):
        High of candle 3 < Low of candle 1
        The gap between candle 1 low and candle 3 high is the FVG.

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close', 'volume' columns.
        min_gap_pct: Minimum gap size as % of price (default 0.3%).
        max_gap_pct: Maximum gap size as % of price (default 5%).

    Returns:
        DataFrame with columns: 'fvg_bullish', 'fvg_bearish',
        'fvg_start', 'fvg_end', 'fvg_size_pct', 'fvg_quality_raw'.
    """
    df = df.copy()
    n = len(df)

    # Initialize columns
    df["fvg_bullish"] = False
    df["fvg_bearish"] = False
    df["fvg_start"] = np.nan
    df["fvg_end"] = np.nan
    df["fvg_size"] = np.nan
    df["fvg_size_pct"] = np.nan
    df["fvg_quality_raw"] = 0.0

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    vols = df["volume"].values
    avg_vol = vols.mean() if len(vols) > 0 and vols.mean() > 0 else 1.0

    for i in range(2, n):
        # Reference price for percentage calculation
        ref_price = closes[i - 1] if closes[i - 1] > 0 else 1.0

        # --- Bullish FVG: Low of candle i > High of candle i-2 ---
        if lows[i] > highs[i - 2]:
            gap_start = highs[i - 2]
            gap_end = lows[i]
            gap_size = gap_end - gap_start
            gap_size_pct = gap_size / ref_price

            # Filter: ignore insignificant gaps
            if gap_size_pct < min_gap_pct:
                continue
            # Filter: ignore anomaly gaps (too large)
            if gap_size_pct > max_gap_pct:
                continue

            df.loc[df.index[i], "fvg_bullish"] = True
            df.loc[df.index[i], "fvg_start"] = gap_start
            df.loc[df.index[i], "fvg_end"] = gap_end
            df.loc[df.index[i], "fvg_size"] = gap_size
            df.loc[df.index[i], "fvg_size_pct"] = gap_size_pct

            # Raw quality: based on gap size (moderate is best)
            # Optimal gap: 0.5% - 2% of price
            if 0.005 <= gap_size_pct <= 0.02:
                size_quality = 1.0
            elif 0.003 <= gap_size_pct < 0.005:
                size_quality = 0.6
            elif 0.02 < gap_size_pct <= 0.03:
                size_quality = 0.7
            else:
                size_quality = 0.4

            # Volume contribution
            if vols[i] > 0 and avg_vol > 0:
                vol_quality = min(1.0, vols[i] / avg_vol)
            else:
                vol_quality = 0.5

            df.loc[df.index[i], "fvg_quality_raw"] = (size_quality * 0.6) + (vol_quality * 0.4)

        # --- Bearish FVG: High of candle i < Low of candle i-2 ---
        elif highs[i] < lows[i - 2]:
            gap_start = highs[i]
            gap_end = lows[i - 2]
            gap_size = gap_end - gap_start
            gap_size_pct = gap_size / ref_price

            if gap_size_pct < min_gap_pct:
                continue
            if gap_size_pct > max_gap_pct:
                continue

            df.loc[df.index[i], "fvg_bearish"] = True
            df.loc[df.index[i], "fvg_start"] = gap_start
            df.loc[df.index[i], "fvg_end"] = gap_end
            df.loc[df.index[i], "fvg_size"] = gap_size
            df.loc[df.index[i], "fvg_size_pct"] = gap_size_pct

            if 0.005 <= gap_size_pct <= 0.02:
                size_quality = 1.0
            elif 0.003 <= gap_size_pct < 0.005:
                size_quality = 0.6
            elif 0.02 < gap_size_pct <= 0.03:
                size_quality = 0.7
            else:
                size_quality = 0.4

            if vols[i] > 0 and avg_vol > 0:
                vol_quality = min(1.0, vols[i] / avg_vol)
            else:
                vol_quality = 0.5

            df.loc[df.index[i], "fvg_quality_raw"] = (size_quality * 0.6) + (vol_quality * 0.4)

    return df


# ============================================================
# FVG Quality Scoring
# ============================================================

def score_fvg_quality(
    fvg: FairValueGap,
    min_quality: float = 0.4,
    prefer_fresh: bool = True,
    fresh_multiplier: float = 1.5,
    body_contribution: float = 0.6,
    volume_contribution: float = 0.4,
) -> float:
    """Calculates the overall quality score for a Fair Value Gap.

    Quality factors:
    - Gap size (moderate gaps are best)
    - Freshness (untouched FVGs score higher)
    - Volume at creation (above average = higher quality)
    - Body overlap (more overlap = stronger imbalance)

    Args:
        fvg: FairValueGap object to score.
        min_quality: Minimum quality threshold.
        prefer_fresh: Whether to prioritize fresh FVGs.
        fresh_multiplier: Multiplier applied to fresh FVGs.
        body_contribution: Weight for body overlap in quality.
        volume_contribution: Weight for volume in quality.

    Returns:
        Quality score from 0.0 to 1.0.
    """
    score = fvg.quality_score  # Base score from detection

    # Freshness bonus
    if fvg.state == FVGState.FRESH and prefer_fresh:
        score = min(1.0, score * fresh_multiplier)
    elif fvg.state == FVGState.PARTIALLY_FILLED:
        score *= 0.7  # Partially filled = reduced quality
    elif fvg.state == FVGState.MITIGATED:
        score *= 0.1  # Mitigated = essentially worthless
    elif fvg.state == FVGState.EXPIRED:
        score = 0.0

    # Volume contribution
    if fvg.volume_ratio > 1.0:
        score += min(0.1, (fvg.volume_ratio - 1.0) * volume_contribution * 0.1)

    # Body overlap contribution
    if fvg.body_overlap > 0:
        score += min(0.1, fvg.body_overlap * body_contribution * 0.1)

    return min(1.0, max(0.0, score))


# ============================================================
# Mitigation Tracking
# ============================================================

def track_fvg_mitigation(
    df: pd.DataFrame,
    fvg_list: List[FairValueGap],
    mitigate_timeout: int = 50,
) -> List[FairValueGap]:
    """Tracks whether FVGs have been mitigated (filled) by subsequent price action.

    V2.0 improvements over V1.0:
    - Tracks partial vs full mitigation
    - Calculates fill percentage
    - Timeout-based expiration (FVGs too old are marked expired)
    - Updates FVG state properly

    A bullish FVG is mitigated when price drops into the gap.
    A bearish FVG is mitigated when price rises into the gap.

    Args:
        df: OHLCV DataFrame with FVG annotations.
        fvg_list: List of FairValueGap objects to track.
        mitigate_timeout: Candles after which FVG is considered expired.

    Returns:
        Updated list of FairValueGap objects with state and fill_pct.
    """
    n = len(df)

    for fvg in fvg_list:
        # Calculate age
        fvg.age_candles = n - 1 - fvg.index
        if fvg.age_candles > mitigate_timeout:
            fvg.state = FVGState.EXPIRED
            fvg.is_relevant = False
            continue

        # Check subsequent candles for mitigation
        end_idx = min(n, fvg.index + mitigate_timeout + 1)
        max_fill = 0.0

        for j in range(fvg.index + 3, end_idx):  # Start after the FVG candle
            candle_high = df["high"].iloc[j]
            candle_low = df["low"].iloc[j]

            if fvg.fvg_type == "bullish":
                # Bullish FVG: filled when price drops into the gap
                # Gap is between fvg_start (lower) and fvg_end (upper)
                if candle_low <= fvg.gap_end and candle_high >= fvg.gap_start:
                    # Calculate how deep price went into the gap
                    fill_depth = fvg.gap_end - max(candle_low, fvg.gap_start)
                    if fvg.gap_size > 0:
                        fill_pct = (fill_depth / fvg.gap_size) * 100.0
                    else:
                        fill_pct = 100.0
                    max_fill = max(max_fill, fill_pct)

                    if fill_pct >= 90:
                        fvg.state = FVGState.MITIGATED
                        fvg.is_relevant = False
                        break
                    else:
                        fvg.state = FVGState.PARTIALLY_FILLED
                        fvg.is_relevant = True

            elif fvg.fvg_type == "bearish":
                # Bearish FVG: filled when price rises into the gap
                if candle_high >= fvg.gap_start and candle_low <= fvg.gap_end:
                    fill_depth = min(candle_high, fvg.gap_end) - fvg.gap_start
                    if fvg.gap_size > 0:
                        fill_pct = (fill_depth / fvg.gap_size) * 100.0
                    else:
                        fill_pct = 100.0
                    max_fill = max(max_fill, fill_pct)

                    if fill_pct >= 90:
                        fvg.state = FVGState.MITIGATED
                        fvg.is_relevant = False
                        break
                    else:
                        fvg.state = FVGState.PARTIALLY_FILLED
                        fvg.is_relevant = True

        if fvg.state == FVGState.FRESH:
            fvg.fill_pct = 0.0
            fvg.is_relevant = True
        else:
            fvg.fill_pct = max_fill

    return fvg_list


# ============================================================
# Build FVG Objects from DataFrame
# ============================================================

def build_fvg_objects(
    df: pd.DataFrame,
    avg_volume: Optional[float] = None,
) -> List[FairValueGap]:
    """Converts annotated DataFrame into structured FairValueGap objects.

    Args:
        df: DataFrame with FVG columns from find_fair_value_gaps.
        avg_volume: Pre-calculated average volume (auto if None).

    Returns:
        List of FairValueGap objects.
    """
    fvg_list: List[FairValueGap] = []

    if avg_volume is None:
        avg_volume = df["volume"].mean() if "volume" in df.columns and len(df) > 0 else 1.0

    for i in range(len(df)):
        is_bullish = df["fvg_bullish"].iloc[i]
        is_bearish = df["fvg_bearish"].iloc[i]

        if not (is_bullish or is_bearish):
            continue

        gap_start = df["fvg_start"].iloc[i]
        gap_end = df["fvg_end"].iloc[i]
        gap_size = df["fvg_size"].iloc[i] if pd.notna(df["fvg_size"].iloc[i]) else abs(gap_end - gap_start)

        # Determine fvg type and boundaries
        if is_bullish:
            fvg_type = "bullish"
        elif is_bearish:
            fvg_type = "bearish"
        else:
            continue

        # Volume ratio
        vol = df["volume"].iloc[i] if "volume" in df.columns else avg_volume
        vol_ratio = vol / avg_volume if avg_volume > 0 else 1.0

        fvg = FairValueGap(
            index=i,
            fvg_type=fvg_type,
            gap_start=gap_start,
            gap_end=gap_end,
            gap_size=gap_size,
            gap_size_pct=df["fvg_size_pct"].iloc[i] if pd.notna(df["fvg_size_pct"].iloc[i]) else 0.0,
            timestamp=df.index[i],
            quality_score=df["fvg_quality_raw"].iloc[i] if "fvg_quality_raw" in df.columns else 0.5,
            volume_ratio=vol_ratio,
        )
        fvg_list.append(fvg)

    return fvg_list


# ============================================================
# Select Best FVG for Analysis
# ============================================================

def select_best_fvg(
    fvg_list: List[FairValueGap],
    direction: str = "bullish",
    prefer_fresh: bool = True,
    min_quality: float = 0.4,
) -> Optional[FairValueGap]:
    """Selects the best FVG for a given trading direction.

    For bullish trades: selects the best bullish FVG (fresh, high quality, closest).
    For bearish trades: selects the best bearish FVG.

    V2.0 improvement: Multi-criteria selection instead of just finding any FVG.

    Args:
        fvg_list: List of FairValueGap objects.
        direction: 'bullish' or 'bearish'.
        prefer_fresh: Whether to prioritize fresh FVGs.
        min_quality: Minimum quality score to consider.

    Returns:
        Best matching FairValueGap or None if no suitable FVG found.
    """
    candidates = [f for f in fvg_list if f.fvg_type == direction and f.is_relevant]

    if not candidates:
        return None

    # Filter by minimum quality
    candidates = [f for f in candidates if f.quality_score >= min_quality]

    if not candidates:
        return None

    # Score each candidate
    for fvg in candidates:
        score = fvg.quality_score

        # Freshness bonus
        if prefer_fresh and fvg.state == FVGState.FRESH:
            score += 0.2

        # Proximity bonus (closer to current price = more relevant)
        # We'll sort by index (most recent first) as a proxy
        score += 0.05

        fvg.quality_score = min(1.0, score)

    # Sort by quality score descending, then by freshness
    candidates.sort(key=lambda f: (f.quality_score, 1 if f.state == FVGState.FRESH else 0), reverse=True)

    return candidates[0] if candidates else None


# ============================================================
# Convenience: Backward-Compatible Interface
# ============================================================

def track_fvg_fill(
    df: pd.DataFrame,
    mitigate_timeout: int = 50,
) -> Tuple[pd.DataFrame, List[FairValueGap]]:
    """Backward-compatible wrapper that tracks FVG fill status.

    Returns both the annotated DataFrame and the structured FVG objects
    for more detailed analysis.

    Args:
        df: DataFrame with FVG columns.
        mitigate_timeout: Candles before expiration.

    Returns:
        Tuple of (annotated DataFrame, list of FairValueGap objects).
    """
    df = df.copy()

    # Add mitigation tracking columns
    df["fvg_mitigated"] = False
    df["fvg_partial_fill"] = False
    df["fvg_fill_pct"] = 0.0
    df["fvg_state"] = ""

    # Build FVG objects
    fvg_list = build_fvg_objects(df)

    # Track mitigation
    fvg_list = track_fvg_mitigation(df, fvg_list, mitigate_timeout=mitigate_timeout)

    # Write back to DataFrame
    for fvg in fvg_list:
        idx = df.index[fvg.index]
        df.loc[idx, "fvg_mitigated"] = fvg.state == FVGState.MITIGATED
        df.loc[idx, "fvg_partial_fill"] = fvg.state == FVGState.PARTIALLY_FILLED
        df.loc[idx, "fvg_fill_pct"] = fvg.fill_pct
        df.loc[idx, "fvg_state"] = fvg.state.value

    return df, fvg_list


def analyze_fvg(
    df: pd.DataFrame,
    min_gap_pct: float = 0.003,
    max_gap_pct: float = 0.05,
    prefer_fresh: bool = True,
    fresh_multiplier: float = 1.5,
    mitigate_timeout: int = 50,
    min_quality: float = 0.4,
) -> Dict[str, object]:
    """Complete FVG analysis pipeline.

    Args:
        df: OHLCV DataFrame.
        min_gap_pct: Minimum gap size threshold.
        max_gap_pct: Maximum gap size threshold.
        prefer_fresh: Whether to prioritize fresh FVGs.
        fresh_multiplier: Quality multiplier for fresh FVGs.
        mitigate_timeout: Candles before FVG expiration.
        min_quality: Minimum quality score.

    Returns:
        Dict with:
            'df': Annotated DataFrame
            'all_fvgs': List of all FairValueGap objects
            'fresh_bullish_fvg': Best fresh bullish FVG (or None)
            'fresh_bearish_fvg': Best fresh bearish FVG (or None)
            'bullish_fvgs': List of all bullish FVGs
            'bearish_fvgs': List of all bearish FVGs
            'mitigated_count': Number of mitigated FVGs
            'fresh_count': Number of fresh FVGs
    """
    # Step 1: Detect FVGs
    df = find_fair_value_gaps(df, min_gap_pct=min_gap_pct, max_gap_pct=max_gap_pct)

    # Step 2: Build objects and track mitigation
    df, fvg_list = track_fvg_fill(df, mitigate_timeout=mitigate_timeout)

    # Step 3: Score quality
    for fvg in fvg_list:
        fvg.quality_score = score_fvg_quality(
            fvg, prefer_fresh=prefer_fresh, fresh_multiplier=fresh_multiplier
        )

    # Step 4: Select best for each direction
    best_bullish = select_best_fvg(fvg_list, "bullish", prefer_fresh, min_quality)
    best_bearish = select_best_fvg(fvg_list, "bearish", prefer_fresh, min_quality)

    # Count states
    fresh_count = sum(1 for f in fvg_list if f.state == FVGState.FRESH)
    mitigated_count = sum(1 for f in fvg_list if f.state == FVGState.MITIGATED)

    return {
        "df": df,
        "all_fvgs": fvg_list,
        "fresh_bullish_fvg": best_bullish,
        "fresh_bearish_fvg": best_bearish,
        "bullish_fvgs": [f for f in fvg_list if f.fvg_type == "bullish"],
        "bearish_fvgs": [f for f in fvg_list if f.fvg_type == "bearish"],
        "mitigated_count": mitigated_count,
        "fresh_count": fresh_count,
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
    print("V2.0 FVG Analysis")
    print("=" * 60)

    result = analyze_fvg(df.copy())

    print(f"\nTotal FVGs detected: {len(result['all_fvgs'])}")
    print(f"  Bullish FVGs: {len(result['bullish_fvgs'])}")
    print(f"  Bearish FVGs: {len(result['bearish_fvgs'])}")
    print(f"  Fresh FVGs: {result['fresh_count']}")
    print(f"  Mitigated FVGs: {result['mitigated_count']}")

    print(f"\nBest Bullish FVG: {result['fresh_bullish_fvg']}")
    if result['fresh_bullish_fvg']:
        d = result['fresh_bullish_fvg'].as_dict()
        print(f"  Gap: {d['gap_start']:.2f} - {d['gap_end']:.2f}")
        print(f"  Size: {d['gap_size_pct']:.4f}%")
        print(f"  Quality: {d['quality_score']:.2f}")
        print(f"  State: {d['state']}")
        print(f"  Fill: {d['fill_pct']:.1f}%")

    print(f"\nBest Bearish FVG: {result['fresh_bearish_fvg']}")
    if result['fresh_bearish_fvg']:
        d = result['fresh_bearish_fvg'].as_dict()
        print(f"  Gap: {d['gap_start']:.2f} - {d['gap_end']:.2f}")
        print(f"  Size: {d['gap_size_pct']:.4f}%")
        print(f"  Quality: {d['quality_score']:.2f}")
        print(f"  State: {d['state']}")

    print("\nAll FVGs:")
    for fvg in result['all_fvgs']:
        d = fvg.as_dict()
        print(f"  [{d['fvg_type'].upper()}] {d['gap_start']:.2f}-{d['gap_end']:.2f} | "
              f"Size: {d['gap_size_pct']:.4f}% | Quality: {d['quality_score']:.2f} | "
              f"State: {d['state']} | Fill: {d['fill_pct']:.1f}%")
