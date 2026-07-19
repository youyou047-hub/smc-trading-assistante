"""Binance Public API client for fetching OHLCV market data.

This module provides a robust client for the Binance Public API,
supporting multiple endpoint fallbacks and automatic retries.
No authentication or exchange account is required.
"""

import logging
from typing import List, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

# Fallback endpoints in priority order
DEFAULT_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3",
    "https://api.binance.com/api/v3",
    "https://api.binance.us/api/v3",
]


class BinanceClientError(Exception):
    """Custom exception for Binance client errors."""
    pass


class BinanceClient:
    """Client for fetching OHLCV data from Binance Public API.

    This client uses the public klines endpoint which requires no authentication.
    It supports automatic endpoint fallback if the primary endpoint is unavailable.

    Attributes:
        base_url (str): The primary base URL for the Binance API.
        fallback_urls (List[str]): Additional endpoints to try if primary fails.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        fallback_urls: Optional[List[str]] = None,
    ) -> None:
        """Initialize the Binance client.

        Args:
            base_url: Primary API base URL. Defaults to data-api.binance.vision.
            fallback_urls: List of fallback URLs to try if primary fails.
        """
        self.base_url = base_url or DEFAULT_ENDPOINTS[0]
        self.fallback_urls = fallback_urls or DEFAULT_ENDPOINTS[1:]
        self._active_url = self.base_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def _make_request(self, url: str, endpoint: str, params: dict) -> list:
        """Makes a robust request to the Binance API with retries.

        Args:
            url: The base URL to use for this request.
            endpoint: The API endpoint (e.g., 'klines').
            params: Query parameters for the request.

        Returns:
            list: JSON response from the API (klines returns a list).

        Raises:
            BinanceClientError: If the API returns an error.
            requests.exceptions.RequestException: If the request fails (triggers retry).
        """
        full_url = f"{url}/{endpoint}"
        response = requests.get(full_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Check for API-level errors
        if isinstance(data, dict) and "code" in data and "msg" in data:
            raise BinanceClientError(
                f"Binance API error: {data['msg']} (Code: {data['code']})"
            )
        return data

    def _request_with_fallback(self, endpoint: str, params: dict) -> list:
        """Attempts request with primary URL, falls back to alternatives on failure.

        Args:
            endpoint: The API endpoint.
            params: Query parameters.

        Returns:
            list: JSON response data.

        Raises:
            BinanceClientError: If all endpoints fail.
        """
        all_urls = [self._active_url] + [
            u for u in [self.base_url] + self.fallback_urls if u != self._active_url
        ]

        last_error = None
        for url in all_urls:
            try:
                data = self._make_request(url, endpoint, params)
                # If successful with a different URL, update active
                if url != self._active_url:
                    logger.info(f"Switched to endpoint: {url}")
                    self._active_url = url
                return data
            except (BinanceClientError, requests.exceptions.RequestException) as e:
                last_error = e
                logger.warning(f"Endpoint {url} failed: {e}")
                continue

        raise BinanceClientError(
            f"All Binance endpoints failed. Last error: {last_error}"
        )

    def get_ohlcv(
        self, symbol: str, interval: str, limit: int = 500
    ) -> pd.DataFrame:
        """Fetches OHLCV (candlestick) data for a given symbol and interval.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').
            interval: Kline interval (e.g., '1h', '15m', '4h', '1d').
            limit: Number of candles to fetch (1-1000, default 500).

        Returns:
            pd.DataFrame: DataFrame with DatetimeIndex and columns:
                          open, high, low, close, volume.

        Raises:
            ValueError: If limit is out of range.
            BinanceClientError: If data fetching fails after all retries.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("Limit must be between 1 and 1000.")

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }

        try:
            data = self._request_with_fallback("klines", params)

            df = pd.DataFrame(
                data,
                columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            # Convert types
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].astype(float)

            # Keep only essential columns and set index
            df = df[["open_time", "open", "high", "low", "close", "volume"]]
            df.set_index("open_time", inplace=True)

            logger.info(
                f"Fetched {len(df)} candles for {symbol} ({interval})"
            )
            return df

        except BinanceClientError:
            raise
        except Exception as e:
            raise BinanceClientError(
                f"Failed to fetch OHLCV data for {symbol} ({interval}): {e}"
            ) from e

    def get_current_price(self, symbol: str) -> float:
        """Fetches the current price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT').

        Returns:
            float: Current price.

        Raises:
            BinanceClientError: If the request fails.
        """
        params = {"symbol": symbol.upper()}
        try:
            data = self._request_with_fallback("ticker/price", params)
            return float(data["price"])
        except (KeyError, TypeError) as e:
            raise BinanceClientError(
                f"Failed to parse price for {symbol}: {e}"
            ) from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = BinanceClient()

    try:
        # Fetch BTCUSDT 1h candles
        btc_1h = client.get_ohlcv("BTCUSDT", "1h", limit=10)
        print("BTCUSDT 1h OHLCV data:")
        print(btc_1h)
        print(f"\nCurrent BTC price: ${client.get_current_price('BTCUSDT'):,.2f}")

    except BinanceClientError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
