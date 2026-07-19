
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def find_premium_discount_zones(df: pd.DataFrame, swing_high: float, swing_low: float) -> Tuple[float, float, float, float, float]:
    """Calculates premium, equilibrium, and discount zones based on a swing high and swing low.

    Args:
        df (pd.DataFrame): OHLCV DataFrame (not directly used for calculation but for context).
        swing_high (float): The price of the recent swing high.
        swing_low (float): The price of the recent swing low.

    Returns:
        Tuple[float, float, float, float, float]: (premium_start, equilibrium, discount_end, premium_zone_price, discount_zone_price)
                                                  where premium_zone_price is the 0.75 level and discount_zone_price is the 0.25 level.
    """
    range_size = swing_high - swing_low
    equilibrium = swing_low + (range_size * 0.5)
    premium_start = equilibrium # 50% to 100% is premium
    discount_end = equilibrium # 0% to 50% is discount

    premium_zone_price = swing_high - (range_size * 0.25) # 75% level
    discount_zone_price = swing_low + (range_size * 0.25) # 25% level

    return premium_start, equilibrium, discount_end, premium_zone_price, discount_zone_price

def find_imbalances(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies imbalances (similar to FVGs but focusing on candle body gaps).

    An imbalance occurs when there's a significant gap between the close of one candle
    and the open of the next, or when the high/low of adjacent candles don't overlap.
    This simplified version looks for gaps between candle highs/lows.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        pd.DataFrame: Original DataFrame with 'imbalance_bullish', 'imbalance_bearish',
                      'imbalance_start', 'imbalance_end' columns.
    """
    df["imbalance_bullish"] = False
    df["imbalance_bearish"] = False
    df["imbalance_start"] = np.nan
    df["imbalance_end"] = np.nan

    for i in range(1, len(df)):
        # Bullish imbalance: Current low > Previous high
        if df["low"].iloc[i] > df["high"].iloc[i-1]:
            df.loc[df.index[i], "imbalance_bullish"] = True
            df.loc[df.index[i], "imbalance_start"] = df["high"].iloc[i-1]
            df.loc[df.index[i], "imbalance_end"] = df["low"].iloc[i]

        # Bearish imbalance: Current high < Previous low
        elif df["high"].iloc[i] < df["low"].iloc[i-1]:
            df.loc[df.index[i], "imbalance_bearish"] = True
            df.loc[df.index[i], "imbalance_start"] = df["low"].iloc[i-1]
            df.loc[df.index[i], "imbalance_end"] = df["high"].iloc[i]

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

    # Example for Premium/Discount Zones
    swing_high_price = 19.0
    swing_low_price = 9.0
    premium_start, equilibrium, discount_end, premium_zone_price, discount_zone_price = find_premium_discount_zones(df, swing_high_price, swing_low_price)
    print(f"\nPremium/Discount Zones (Swing High: {swing_high_price}, Swing Low: {swing_low_price}):")
    print(f"  Equilibrium: {equilibrium}")
    print(f"  Premium Zone (above {premium_start:.2f}): 75% level at {premium_zone_price:.2f}")
    print(f"  Discount Zone (below {discount_end:.2f}): 25% level at {discount_zone_price:.2f}")

    # Example for Imbalances
    df_imbalance = find_imbalances(df.copy())
    print("\nImbalances:")
    print(df_imbalance[["high", "low", "imbalance_bullish", "imbalance_bearish", "imbalance_start", "imbalance_end"]])
