
import pandas as pd
import numpy as np
from typing import Tuple, List

def find_equal_highs_lows(df: pd.DataFrame, threshold: float = 0.001, window: int = 10) -> pd.DataFrame:
    """Identifies equal highs and equal lows within a specified threshold.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'high' and 'low' columns.
        threshold (float): Percentage threshold to consider highs/lows as 'equal'.
        window (int): Number of previous candles to check for equal highs/lows.

    Returns:
        pd.DataFrame: Original DataFrame with 'equal_high' and 'equal_low' columns (boolean).
    """
    df["equal_high"] = False
    df["equal_low"] = False

    for i in range(window, len(df)):
        current_high = df["high"].iloc[i]
        current_low = df["low"].iloc[i]

        # Check for equal highs
        prev_highs = df["high"].iloc[i-window:i]
        if not prev_highs.empty:
            for prev_h in prev_highs:
                if abs(current_high - prev_h) / prev_h <= threshold:
                    df.loc[df.index[i], "equal_high"] = True
                    break

        # Check for equal lows
        prev_lows = df["low"].iloc[i-window:i]
        if not prev_lows.empty:
            for prev_l in prev_lows:
                if abs(current_low - prev_l) / prev_l <= threshold:
                    df.loc[df.index[i], "equal_low"] = True
                    break
    return df

def find_liquidity_sweeps(df: pd.DataFrame, threshold: float = 0.001, window: int = 20) -> pd.DataFrame:
    """Identifies liquidity sweeps above equal highs or below equal lows.

    A liquidity sweep occurs when price briefly moves above an equal high or below an equal low
    and then quickly reverses, indicating absorption of orders.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'high', 'low', 'close' columns, and 'equal_high', 'equal_low' from find_equal_highs_lows.
        threshold (float): Percentage threshold for the sweep magnitude.
        window (int): Number of candles to look back for equal highs/lows.

    Returns:
        pd.DataFrame: Original DataFrame with 'liquidity_sweep_bullish' and 'liquidity_sweep_bearish' columns (boolean).
    """
    df["liquidity_sweep_bullish"] = False # Price sweeps below a low and reverses up
    df["liquidity_sweep_bearish"] = False # Price sweeps above a high and reverses down

    # Ensure equal highs/lows are identified first
    df = find_equal_highs_lows(df.copy(), threshold=threshold, window=window)

    for i in range(1, len(df)):
        # Bearish sweep: price takes out an equal high and closes below it or significantly lower
        if df["equal_high"].iloc[i-1]: # If previous candle was an equal high
            # Current candle high goes above the equal high, but closes below it or significantly lower
            if df["high"].iloc[i] > df["high"].iloc[i-1] and df["close"].iloc[i] < df["high"].iloc[i-1] * (1 - threshold):
                df.loc[df.index[i], "liquidity_sweep_bearish"] = True

        # Bullish sweep: price takes out an equal low and closes above it or significantly higher
        if df["equal_low"].iloc[i-1]: # If previous candle was an equal low
            # Current candle low goes below the equal low, but closes above it or significantly higher
            if df["low"].iloc[i] < df["low"].iloc[i-1] and df["close"].iloc[i] > df["low"].iloc[i-1] * (1 + threshold):
                df.loc[df.index[i], "liquidity_sweep_bullish"] = True

    return df

if __name__ == '__main__':
    # Example Usage
    data = {
        'open': [100, 102, 101, 103, 102, 104, 103, 105, 104, 106],
        'high': [103, 104, 103, 105, 104, 106, 105, 107, 106, 108],
        'low': [99, 100, 99, 101, 100, 102, 101, 103, 102, 104],
        'close': [102, 101, 102, 104, 103, 105, 104, 106, 105, 107],
        'volume': [100, 120, 110, 130, 140, 120, 150, 130, 160, 140]
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(pd.Series(range(len(df))), unit='s') # Dummy datetime index

    df_liquidity = find_liquidity_sweeps(df.copy(), threshold=0.005, window=5)
    print("Liquidity Analysis:")
    print(df_liquidity[["high", "low", "equal_high", "equal_low", "liquidity_sweep_bullish", "liquidity_sweep_bearish"]])
