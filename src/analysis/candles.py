
import pandas as pd
import numpy as np
from typing import Tuple, List

def find_displacement_candles(df: pd.DataFrame, min_body_ratio: float = 0.6, min_range_multiplier: float = 1.5) -> pd.DataFrame:
    """Identifies displacement candles (strong, large-bodied candles).

    A displacement candle is characterized by a large body relative to its total range
    and a total range significantly larger than recent candles.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'open', 'high', 'low', 'close' columns.
        min_body_ratio (float): Minimum ratio of candle body to total range (e.g., 0.6 means body is 60% of range).
        min_range_multiplier (float): Minimum multiplier for current candle's range compared to average recent range.

    Returns:
        pd.DataFrame: Original DataFrame with 'displacement_bullish' and 'displacement_bearish' columns (boolean).
    """
    df["displacement_bullish"] = False
    df["displacement_bearish"] = False

    df["range"] = df["high"] - df["low"]
    df["body"] = abs(df["close"] - df["open"])
    df["body_ratio"] = df["body"] / df["range"]

    # Calculate average range of previous candles for comparison
    df["avg_range"] = df["range"].rolling(window=10, min_periods=1).mean().shift(1)

    for i in range(len(df)):
        if df["body_ratio"].iloc[i] >= min_body_ratio and df["range"].iloc[i] > df["avg_range"].iloc[i] * min_range_multiplier:
            if df["close"].iloc[i] > df["open"].iloc[i]: # Bullish candle
                df.loc[df.index[i], "displacement_bullish"] = True
            elif df["close"].iloc[i] < df["open"].iloc[i]: # Bearish candle
                df.loc[df.index[i], "displacement_bearish"] = True

    return df.drop(columns=["range", "body", "body_ratio", "avg_range"], errors='ignore')

def find_rejection_candles(df: pd.DataFrame, min_wick_body_ratio: float = 2.0) -> pd.DataFrame:
    """Identifies rejection candles (long wicks indicating price rejection).

    A rejection candle has a long wick (upper or lower) relative to its body,
    indicating that price was rejected from a certain level.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'open', 'high', 'low', 'close' columns.
        min_wick_body_ratio (float): Minimum ratio of wick length to candle body length.

    Returns:
        pd.DataFrame: Original DataFrame with 'rejection_bullish' and 'rejection_bearish' columns (boolean).
    """
    df["rejection_bullish"] = False # Long lower wick, price rejected from below
    df["rejection_bearish"] = False # Long upper wick, price rejected from above

    df["body"] = abs(df["close"] - df["open"])

    for i in range(len(df)):
        open_price = df["open"].iloc[i]
        close_price = df["close"].iloc[i]
        high_price = df["high"].iloc[i]
        low_price = df["low"].iloc[i]
        body = df["body"].iloc[i]

        if body == 0: # Avoid division by zero for doji candles
            continue

        # Bullish rejection: long lower wick
        lower_wick = min(open_price, close_price) - low_price
        if lower_wick / body >= min_wick_body_ratio:
            df.loc[df.index[i], "rejection_bullish"] = True

        # Bearish rejection: long upper wick
        upper_wick = high_price - max(open_price, close_price)
        if upper_wick / body >= min_wick_body_ratio:
            df.loc[df.index[i], "rejection_bearish"] = True

    return df.drop(columns=["body"], errors='ignore')

def find_confirmation_candles(df: pd.DataFrame, prev_signal_col: str, signal_type: str = "bullish") -> pd.DataFrame:
    """Identifies confirmation candles following a previous signal.

    A confirmation candle is a candle that closes in the direction of the expected move
    after a signal (e.g., a bullish candle after a bullish rejection).

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'open', 'close' columns and a column for previous signal.
        prev_signal_col (str): Name of the column indicating the previous signal (e.g., 'rejection_bullish').
        signal_type (str): 'bullish' or 'bearish' to indicate the expected direction of confirmation.

    Returns:
        pd.DataFrame: Original DataFrame with 'confirmation_bullish' or 'confirmation_bearish' column (boolean).
    """
    if signal_type == "bullish":
        df["confirmation_bullish"] = False
        for i in range(1, len(df)):
            if df[prev_signal_col].iloc[i-1] and df["close"].iloc[i] > df["open"].iloc[i]:
                df.loc[df.index[i], "confirmation_bullish"] = True
    elif signal_type == "bearish":
        df["confirmation_bearish"] = False
        for i in range(1, len(df)):
            if df[prev_signal_col].iloc[i-1] and df["close"].iloc[i] < df["open"].iloc[i]:
                df.loc[df.index[i], "confirmation_bearish"] = True
    else:
        raise ValueError("signal_type must be 'bullish' or 'bearish'")

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

    df_displacement = find_displacement_candles(df.copy())
    print("\nDisplacement Candles:")
    print(df_displacement[["open", "close", "displacement_bullish", "displacement_bearish"]])

    df_rejection = find_rejection_candles(df.copy())
    print("\nRejection Candles:")
    print(df_rejection[["open", "close", "rejection_bullish", "rejection_bearish"]])

    # Example for confirmation candles (after bullish rejection)
    df_rejection_copy = df_rejection.copy()
    df_confirmation = find_confirmation_candles(df_rejection_copy, 'rejection_bullish', 'bullish')
    print("\nConfirmation Candles (Bullish after Rejection):")
    print(df_confirmation[["open", "close", "rejection_bullish", "confirmation_bullish"]])
