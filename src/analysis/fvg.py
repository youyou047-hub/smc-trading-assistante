
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def find_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies Fair Value Gaps (FVG) based on a 3-candle pattern.

    A bullish FVG occurs when the low of the current candle is higher than the high of the previous candle,
    and the low of the next candle is higher than the high of the current candle.
    (This is a common interpretation, but the prompt describes it as a gap between candle 1 high and candle 3 low for bullish FVG)

    Let's use the common interpretation: Bullish FVG is when current_low > prev_high, and Bearish FVG is when current_high < prev_low.
    A more precise definition for a 3-candle FVG:
    Bullish FVG: Low of candle 3 > High of candle 1 (with candle 2 in between)
    Bearish FVG: High of candle 3 < Low of candle 1 (with candle 2 in between)

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'high', 'low', 'close' columns.

    Returns:
        pd.DataFrame: Original DataFrame with 'fvg_bullish', 'fvg_bearish', 'fvg_start', 'fvg_end' columns.
                      'fvg_start' and 'fvg_end' define the price range of the FVG.
    """
    df["fvg_bullish"] = False
    df["fvg_bearish"] = False
    df["fvg_start"] = np.nan
    df["fvg_end"] = np.nan

    for i in range(2, len(df)):
        # Bullish FVG: Low of current candle (i) > High of candle (i-2)
        # The gap is between high of (i-2) and low of (i)
        if df["low"].iloc[i] > df["high"].iloc[i-2]:
            df.loc[df.index[i], "fvg_bullish"] = True
            df.loc[df.index[i], "fvg_start"] = df["high"].iloc[i-2]
            df.loc[df.index[i], "fvg_end"] = df["low"].iloc[i]

        # Bearish FVG: High of current candle (i) < Low of candle (i-2)
        # The gap is between low of (i-2) and high of (i)
        elif df["high"].iloc[i] < df["low"].iloc[i-2]:
            df.loc[df.index[i], "fvg_bearish"] = True
            df.loc[df.index[i], "fvg_start"] = df["low"].iloc[i-2]
            df.loc[df.index[i], "fvg_end"] = df["high"].iloc[i]

    return df

def track_fvg_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Tracks if Fair Value Gaps have been filled.

    Args:
        df (pd.DataFrame): DataFrame with 'fvg_bullish', 'fvg_bearish', 'fvg_start', 'fvg_end' columns.

    Returns:
        pd.DataFrame: Original DataFrame with 'fvg_filled' column (boolean).
    """
    df["fvg_filled"] = False

    for i in range(len(df)):
        if df["fvg_bullish"].iloc[i] or df["fvg_bearish"].iloc[i]:
            fvg_start = df["fvg_start"].iloc[i]
            fvg_end = df["fvg_end"].iloc[i]

            # Check subsequent candles for FVG fill
            for j in range(i + 1, len(df)):
                if df["fvg_bullish"].iloc[i]: # Bullish FVG, filled if price drops into the gap
                    if df["low"].iloc[j] <= fvg_end and df["high"].iloc[j] >= fvg_start:
                        df.loc[df.index[i], "fvg_filled"] = True
                        break
                elif df["fvg_bearish"].iloc[i]: # Bearish FVG, filled if price rises into the gap
                    if df["high"].iloc[j] >= fvg_end and df["low"].iloc[j] <= fvg_start:
                        df.loc[df.index[i], "fvg_filled"] = True
                        break
    return df

if __name__ == '__main__':
    # Example Usage
    data = {
        'open': [10, 12, 15, 13, 16, 14, 17, 15, 18, 16],
        'high': [13, 16, 17, 15, 18, 17, 19, 17, 20, 18],
        'low': [9, 11, 12, 11, 13, 12, 14, 13, 15, 14],
        'close': [12, 15, 13, 14, 17, 15, 18, 16, 19, 17],
        'volume': [100, 120, 110, 130, 140, 120, 150, 130, 160, 140]
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(pd.Series(range(len(df))), unit='s') # Dummy datetime index

    df_fvg = find_fair_value_gaps(df.copy())
    print("Fair Value Gaps:")
    print(df_fvg[["high", "low", "fvg_bullish", "fvg_bearish", "fvg_start", "fvg_end"]])

    df_fvg_filled = track_fvg_fill(df_fvg.copy())
    print("\nFair Value Gaps (Filled Status):")
    print(df_fvg_filled[df_fvg_filled["fvg_bullish"] | df_fvg_filled["fvg_bearish"]][["high", "low", "fvg_bullish", "fvg_bearish", "fvg_start", "fvg_end", "fvg_filled"]])
