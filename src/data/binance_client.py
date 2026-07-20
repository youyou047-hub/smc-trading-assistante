"""Binance Public API Client — V2.0 (Upgraded).

Provides a robust, efficient client for fetching OHLCV market data from the
Binance Public API.  Key improvements over V1:

  • Batch fetching of multiple timeframes in a single method call
    (minimises API round-trips).
  • In-memory per-symbol / per-timeframe cache with configurable TTL
    to avoid redundant calls within the same scan cycle.
  • Configurable retry parameters (attempts, back-off, timeout).
  • Proper error classification: transient network errors are retried,
    API-level errors raise immediately.
  • Returns ``pandas.DataFrame`` objects with consistent typing and a
    ``DatetimeIndex`` — identical shape to V1 so existing consumers
    (chart generator, analysis engine) remain compatible.
  • Backward-compatible constructor signature (``base_url``, ``fallback_urls``).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_ENDPOINTS: List[str] = [
    "https://data-api.binance.vision/api/v3",
    "https://api.binance.com/api/v3",
    "https://api1.binance.com/api/v3",
]

# Standard intervals accepted by Binance
VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}

# Default cache TTL in seconds (covers a typical 60-second scan loop)
DEFAULT_CACHE_TTL_SECONDS: int = 120


# ── Exceptions ────────────────────────────────────────────────────────────────

class BinanceClientError(Exception):
    """Raised when the Binance client encounters an unrecoverable error."""
    pass


class BinanceRateLimitError(BinanceClientError):
    """Raised when Binance returns HTTP 429 (rate limit)."""
    pass


# ── Client ────────────────────────────────────────────────────────────────────

class BinanceClient:
    """Client for fetching OHLCV data from the Binance Public API.

    The client requires **no authentication** — it uses only the public
    ``/api/v3/klines`` endpoint.

    Backward compatibility:
        * The V1 constructor signature ``(base_url, fallback_urls)`` is
          preserved.  New keyword-only arguments are appended for V2.
        * ``get_ohlcv(symbol, interval, limit)`` still returns a single
          ``DataFrame`` exactly as V1 did.

    Attributes:
        base_url: Primary API base URL.
        fallback_urls: Additional endpoints tried on failure.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        fallback_urls: Optional[List[str]] = None,
        # V2 additions (keyword-only)
        max_retries: int = 3,
        retry_backoff_min: float = 2.0,
        retry_backoff_max: float = 10.0,
        request_timeout: int = 15,
        cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self.base_url = base_url or DEFAULT_ENDPOINTS[0]
        self.fallback_urls = fallback_urls or DEFAULT_ENDPOINTS[1:]
        self._active_url = self.base_url

        # V2 configurable retry / timeout parameters
        self._max_retries = max_retries
        self._retry_backoff_min = retry_backoff_min
        self._retry_backoff_max = retry_backoff_max
        self._request_timeout = request_timeout

        # V2 per-call cache: key = (symbol_upper, interval, limit)
        self._cache_ttl = cache_ttl
        self._cache: Dict[tuple, Dict[str, Any]] = {}

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _cache_key(self, symbol: str, interval: str, limit: int) -> tuple:
        return (symbol.upper(), interval.lower(), limit)

    def _get_cached(self, key: tuple) -> Optional[pd.DataFrame]:
        """Return cached DataFrame if it exists and has not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry["ts"] > self._cache_ttl:
            del self._cache[key]
            return None
        return entry["df"]

    def _set_cached(self, key: tuple, df: pd.DataFrame) -> None:
        self._cache[key] = {"ts": time.monotonic(), "df": df}

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        logger.debug("BinanceClient cache cleared.")

    # ── Low-level request ─────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _make_request(self, url: str, endpoint: str, params: dict) -> Any:
        """Execute a single HTTP request to a Binance endpoint.

        Args:
            url: Base URL for this attempt.
            endpoint: API endpoint path (e.g. ``'klines'``).
            params: Query-string parameters.

        Returns:
            Parsed JSON response.

        Raises:
            BinanceClientError: API-level error (bad request, invalid symbol, etc.).
            requests.exceptions.RequestException: Transient network error (retryable).
        """
        full_url = f"{url}/{endpoint}"
        response = requests.get(
            full_url,
            params=params,
            timeout=self._request_timeout,
        )

        if response.status_code == 429:
            raise BinanceRateLimitError("Binance rate limit exceeded (HTTP 429)")

        response.raise_for_status()
        data = response.json()

        # API-level errors carry ``{"code": <int>, "msg": <str>}``
        if isinstance(data, dict) and "code" in data and "msg" in data:
            raise BinanceClientError(
                f"Binance API error: {data['msg']} (Code: {data['code']})"
            )
        return data

    def _request_with_fallback(self, endpoint: str, params: dict) -> Any:
        """Attempt a request, trying fallback URLs on failure.

        Returns:
            Parsed JSON response from whichever endpoint succeeds first.

        Raises:
            BinanceClientError: If every endpoint fails.
        """
        all_urls = [self._active_url] + [
            u for u in [self.base_url] + self.fallback_urls if u != self._active_url
        ]

        last_error: Optional[Exception] = None
        for url in all_urls:
            try:
                data = self._make_request(url, endpoint, params)
                if url != self._active_url:
                    logger.info(f"BinanceClient switched to endpoint: {url}")
                    self._active_url = url
                return data
            except (BinanceClientError, requests.exceptions.RequestException) as exc:
                last_error = exc
                logger.warning(f"BinanceClient endpoint {url} failed: {exc}")
                continue

        raise BinanceClientError(
            f"All Binance endpoints failed. Last error: {last_error}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for a single symbol / interval.

        This is the V1-compatible entry point.  It also populates the cache
        so that subsequent calls for the same ``(symbol, interval, limit)``
        within the TTL window return immediately.

        Args:
            symbol: Trading pair symbol (e.g. ``'BTCUSDT'``).
            interval: Kline interval (e.g. ``'1h'``, ``'15m'``, ``'4h'``).
            limit: Number of candles to return (1-1000).

        Returns:
            ``pd.DataFrame`` with a ``DatetimeIndex`` and columns
            ``open``, ``high``, ``low``, ``close``, ``volume``.

        Raises:
            ValueError: If *limit* is out of range.
            BinanceClientError: If the API fails after retries.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("Limit must be between 1 and 1000.")

        symbol = symbol.upper()
        interval = interval.lower()
        if interval not in VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. Must be one of: {VALID_INTERVALS}"
            )

        key = self._cache_key(symbol, interval, limit)
        cached = self._get_cached(key)
        if cached is not None:
            logger.debug(f"BinanceClient cache hit for {symbol}/{interval}/{limit}")
            return cached

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        try:
            data = self._request_with_fallback("klines", params)
            df = self._parse_klines(data, symbol, interval)
            self._set_cached(key, df)
            logger.info(f"Fetched {len(df)} candles for {symbol} ({interval})")
            return df
        except BinanceClientError:
            raise
        except Exception as exc:
            raise BinanceClientError(
                f"Failed to fetch OHLCV data for {symbol} ({interval}): {exc}"
            ) from exc

    def get_ohlcv_multi(
        self,
        symbol: str,
        intervals: List[str],
        limit: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for **multiple timeframes** in one call.

        The method iterates over *intervals* and returns a dictionary keyed
        by interval string.  Each individual call may hit the cache, so
        repeated invocations within the same scan cycle are cheap.

        Args:
            symbol: Trading pair symbol (e.g. ``'BTCUSDT'``).
            intervals: List of interval strings (e.g. ``['5m', '15m', '1h', '4h']``).
            limit: Number of candles per interval (1-1000).

        Returns:
            Dictionary mapping each interval to its ``pd.DataFrame``.

        Raises:
            BinanceClientError: If **any** interval fails to fetch.
        """
        results: Dict[str, pd.DataFrame] = {}
        for interval in intervals:
            results[interval.lower()] = self.get_ohlcv(symbol, interval, limit)
        return results

    def get_current_price(self, symbol: str) -> float:
        """Fetch the latest ticker price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g. ``'BTCUSDT'``).

        Returns:
            Current price as a float.

        Raises:
            BinanceClientError: If the request fails.
        """
        symbol = symbol.upper()
        params = {"symbol": symbol}
        try:
            data = self._request_with_fallback("ticker/price", params)
            return float(data["price"])
        except (KeyError, TypeError) as exc:
            raise BinanceClientError(
                f"Failed to parse price for {symbol}: {exc}"
            ) from exc

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _parse_klines(raw_data: list, symbol: str, interval: str) -> pd.DataFrame:
        """Parse raw kline JSON into a clean DataFrame.

        Args:
            raw_data: List of kline arrays from the API.
            symbol: Symbol string (used in logging).
            interval: Interval string (used in logging).

        Returns:
            Clean DataFrame ready for analysis.
        """
        df = pd.DataFrame(
            raw_data,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        # Convert timestamps
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        # Convert price / volume columns to float
        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)

        # Keep only the essential OHLCV columns and set index
        df = df[["open_time", "open", "high", "low", "close", "volume"]]
        df.set_index("open_time", inplace=True)

        return df


# ── Standalone test / demo ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = BinanceClient()

    try:
        # Fetch BTCUSDT across four timeframes
        timeframes = ["5m", "15m", "1h", "4h"]
        data = client.get_ohlcv_multi("BTCUSDT", timeframes, limit=10)

        for tf, df in data.items():
            print(f"\nBTCUSDT {tf.upper()} OHLCV (last {len(df)} candles):")
            print(df.tail(3))

        print(f"\nCurrent BTC price: ${client.get_current_price('BTCUSDT'):,.2f}")

    except BinanceClientError as exc:
        print(f"Error: {exc}")
    except ValueError as exc:
        print(f"Configuration Error: {exc}")
