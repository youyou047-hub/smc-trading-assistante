"""Telegram Client V2.0 — Institutional-grade message delivery.

Provides robust Telegram Bot API integration with:
- Exponential-backoff retry logic for transient failures
- Rate limiting to respect Telegram's 30 messages/second ceiling
- HTML message formatting support
- Photo (chart) delivery with caption
- Dedicated error-notification channel for critical failures
- Connection-health checks

Backward-compatible with the V1 TelegramBot interface so existing
callers (formatter, chart generator, main orchestrator) continue to
work without modification.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 15           # seconds per HTTP request
MAX_RETRIES = 3                # how many retry attempts before giving up
RETRY_BACKOFF = (1, 3, 10)     # seconds to wait between retries (exponential)
RATE_LIMIT_BURST = 20          # messages allowed per 1-second window
RATE_LIMIT_WINDOW = 1.0        # burst window in seconds
HEARTBEAT_TIMEOUT = 30         # seconds for heartbeat-specific requests


# ---------------------------------------------------------------------------
# TelegramBot
# ---------------------------------------------------------------------------
class TelegramBot:
    """Handles sending messages and photos via the Telegram Bot API.

    Supports retry logic, rate limiting, HTML formatting, and a dedicated
    ``send_error`` method for critical system notifications.

    Attributes:
        bot_token (str): Telegram bot token.
        chat_id (str): Destination chat or channel ID.
        base_url (str): Constructed Telegram API base URL.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # Build a requests Session with automatic retries
        self._session = requests.Session()
        retry_strategy = Retry(
            total=0,  # We handle retries ourselves for more control
            backoff_factor=0,
            status_forcelist=[],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Rate-limiting state
        self._rate_lock = threading.Lock()
        self._rate_timestamps: list[float] = []

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        """Block the calling thread until the rate limit allows a request.

        Telegram allows 30 messages per second to a single bot.  We use a
        conservative burst of 20 to leave headroom for heartbeat and
        concurrent sends.
        """
        now = time.monotonic()
        with self._rate_lock:
            # Discard timestamps outside the current window
            self._rate_timestamps = [
                ts for ts in self._rate_timestamps
                if now - ts < RATE_LIMIT_WINDOW
            ]
            if len(self._rate_timestamps) >= RATE_LIMIT_BURST:
                sleep_until = self._rate_timestamps[0] + RATE_LIMIT_WINDOW
                sleep_for = max(0, sleep_until - now)
                if sleep_for > 0:
                    logger.debug(
                        "Rate limit hit — sleeping %.1fs", sleep_for
                    )
                    time.sleep(sleep_for)
                    now = time.monotonic()
            self._rate_timestamps.append(now)

    # ------------------------------------------------------------------
    # Retry helpers
    # ------------------------------------------------------------------
    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs,
    ) -> Optional[dict]:
        """Execute an HTTP request with exponential-backoff retry.

        Args:
            method: HTTP method ('get' or 'post').
            url: Full API endpoint URL.
            timeout: Request timeout in seconds.
            **kwargs: Additional keyword arguments forwarded to
                      ``requests.Session.request``.

        Returns:
            Parsed JSON response dict on success, or ``None`` on permanent
            failure after exhausting retries.
        """
        request_fn = getattr(self._session, method)
        last_exception: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                resp = request_fn(url, timeout=timeout, **kwargs)
                resp.raise_for_status()
                result = resp.json()
                if result.get("ok"):
                    return result
                # API returned an error (e.g. bad_request, flood)
                description = result.get("description", "Unknown error")
                logger.warning(
                    "Telegram API error (attempt %d/%d): %s",
                    attempt + 1, MAX_RETRIES, description,
                )
                if "flood" in description.lower():
                    # Honour Telegram's flood-control wait parameter
                    retry_after = result.get("parameters", {}).get(
                        "retry_after", RETRY_BACKOFF[attempt]
                    )
                    logger.info(
                        "Telegram flood control — waiting %ds", retry_after
                    )
                    time.sleep(retry_after)
                    continue
                return result  # Non-retryable error — return as-is
            except (requests.exceptions.RequestException, ValueError) as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF[attempt]
                    logger.warning(
                        "Telegram request failed (attempt %d/%d): %s — "
                        "retrying in %ds",
                        attempt + 1, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Telegram request failed after %d attempts: %s",
                        MAX_RETRIES, exc,
                    )

        # Exhausted all retries
        if last_exception:
            logger.error(
                "Telegram connection permanently failed: %s", last_exception
            )
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
    ) -> bool:
        """Send a text message to the configured chat.

        Args:
            text: The message body (HTML formatting supported).
            parse_mode: Telegram parse mode (``HTML`` or ``MarkdownV2``).

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        endpoint = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        result = self._request_with_retry("post", endpoint, json=payload)
        if result and result.get("ok"):
            logger.info("Telegram message sent successfully.")
            return True
        logger.error("Failed to send Telegram message.")
        return False

    def send_photo(
        self,
        photo_path: str,
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
    ) -> bool:
        """Send a photo with an optional HTML caption.

        Args:
            photo_path: Absolute or relative path to the image file.
            caption: Optional caption text (HTML).
            parse_mode: Parse mode for the caption.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        endpoint = f"{self.base_url}/sendPhoto"
        payload = {
            "chat_id": self.chat_id,
            "parse_mode": parse_mode,
        }
        if caption:
            payload["caption"] = caption

        try:
            with open(photo_path, "rb") as photo_file:
                files = {"photo": photo_file}
                result = self._request_with_retry(
                    "post", endpoint, data=payload, files=files,
                    timeout=30,
                )
        except FileNotFoundError:
            logger.error("Photo file not found: %s", photo_path)
            return False

        if result and result.get("ok"):
            logger.info("Telegram photo '%s' sent successfully.", photo_path)
            return True
        logger.error("Failed to send Telegram photo '%s'.", photo_path)
        return False

    def send_error(self, message: str) -> bool:
        """Send a critical error notification to Telegram.

        This is a convenience wrapper that prepends a standard error
        header and uses the same retry logic as ``send_message``.

        Args:
            message: The error description.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        header = "🚨 <b>CRITICAL ERROR NOTIFICATION</b> 🚨\n"
        full_message = f"{header}\n{message}"
        return self.send_message(full_message)

    def test_connection(self) -> bool:
        """Verify that the bot token and chat ID are valid.

        Performs a lightweight ``getMe`` call to confirm the bot is
        reachable.

        Returns:
            ``True`` if the bot responded successfully.
        """
        try:
            resp = self._session.get(
                f"{self.base_url}/getMe", timeout=HEARTBEAT_TIMEOUT
            )
            resp.raise_for_status()
            result = resp.json()
            return result.get("ok", False)
        except Exception as exc:
            logger.error("Telegram connection test failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Module-level self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or CHAT_ID == "YOUR_CHAT_ID_HERE":
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars to test.")
    else:
        bot = TelegramBot(BOT_TOKEN, CHAT_ID)

        print("\n--- Testing connection ---")
        if bot.test_connection():
            print("Connection OK!")
        else:
            print("Connection FAILED.")

        print("\n--- Testing send_message ---")
        msg = (
            "<b>V2.0 Telegram Client Test</b>\n"
            "Retry logic, rate limiting, and HTML formatting."
        )
        bot.send_message(msg)

        print("\n--- Testing send_error ---")
        bot.send_error("This is a test error notification.")
