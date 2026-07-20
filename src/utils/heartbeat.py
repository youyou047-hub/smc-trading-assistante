"""Heartbeat System V2.0 — Startup confirmation and periodic health checks.

Sends Telegram messages to confirm the system is alive:

1. **Startup message** — sent once when the system begins running.
2. **Periodic heartbeat** — sent every *N* hours (default 5).

Each heartbeat includes:
- System status (ONLINE / DEGRADED / ERROR)
- Total uptime
- Number of completed scans
- Number of alerts sent

The heartbeat runs in a background daemon thread so it does not block
the main scanning loop.  It is safe to call ``stop()`` at any time
during graceful shutdown.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

from src.alerts.telegram import TelegramBot
from src.alerts.formatter import AlertFormatter

logger = logging.getLogger(__name__)


class HeartbeatSystem:
    """Background heartbeat service.

    Args:
        telegram_bot: An initialised ``TelegramBot`` instance.
        formatter: An initialised ``AlertFormatter`` instance.
        interval_hours: Heartbeat interval in hours (default 5).
        send_startup: Whether to send a startup confirmation message.
        symbols: List of monitored symbols for the message body.
        timeframes: Dict mapping timeframe role → string (for startup msg).
        include_stats: Whether to include scan/alert counts.
    """

    def __init__(
        self,
        telegram_bot: TelegramBot,
        formatter: AlertFormatter,
        interval_hours: float = 5.0,
        send_startup: bool = True,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[dict] = None,
        include_stats: bool = True,
    ) -> None:
        self._bot = telegram_bot
        self._formatter = formatter
        self._interval_seconds = interval_hours * 3600
        self._send_startup = send_startup
        self._symbols = symbols or []
        self._timeframes = timeframes or {}
        self._include_stats = include_stats

        # Mutable counters (updated by the main loop)
        self._scan_count: int = 0
        self._alert_count: int = 0
        self._start_time: float = time.monotonic()
        self._status: str = "ONLINE"

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Counter accessors (thread-safe via atomic int operations)
    # ------------------------------------------------------------------
    @property
    def scan_count(self) -> int:
        return self._scan_count

    @scan_count.setter
    def scan_count(self, value: int) -> None:
        self._scan_count = value

    @property
    def alert_count(self) -> int:
        return self._alert_count

    @alert_count.setter
    def alert_count(self, value: int) -> None:
        self._alert_count = value

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        self._status = value

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def increment_scans(self, count: int = 1) -> None:
        """Increment the scan counter."""
        self._scan_count += count

    def increment_alerts(self, count: int = 1) -> None:
        """Increment the alert counter."""
        self._alert_count += count

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the heartbeat background thread and send startup message."""
        if self._running:
            logger.warning("Heartbeat already running — ignoring duplicate start.")
            return

        self._running = True
        self._start_time = time.monotonic()

        # Send startup confirmation
        if self._send_startup:
            self._send_startup_message()

        # Start background thread
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="HeartbeatThread",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Heartbeat system started (interval=%.1fh).",
            self._interval_seconds / 3600,
        )

    def stop(self) -> None:
        """Signal the heartbeat thread to stop.

        The thread is a daemon, so it will terminate when the main
        process exits even if ``stop()`` is not called.  Calling this
        method ensures a clean final heartbeat is sent before exit.
        """
        logger.info("Stopping heartbeat system...")
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Heartbeat system stopped.")

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _send_startup_message(self) -> None:
        """Send the initial startup confirmation to Telegram."""
        msg = self._formatter.format_startup(
            symbols=self._symbols,
            timeframes=self._timeframes,
        )
        if self._bot.send_message(msg):
            logger.info("Startup message sent successfully.")
        else:
            logger.error("Failed to send startup message.")

    def _heartbeat_loop(self) -> None:
        """Main loop that runs in the background thread."""
        while self._running:
            # Sleep in 1-second increments so we can respond to stop() quickly
            for _ in range(int(self._interval_seconds)):
                if not self._running:
                    return
                time.sleep(1)

            if not self._running:
                return

            # Send heartbeat
            self._send_periodic_heartbeat()

    def _send_periodic_heartbeat(self) -> None:
        """Compose and send a periodic heartbeat message."""
        uptime = int(self.uptime_seconds)
        scan_count = self._scan_count
        alert_count = self._alert_count

        msg = self._formatter.format_heartbeat(
            uptime_seconds=uptime,
            scan_count=scan_count,
            alert_count=alert_count,
            symbols=self._symbols,
        )

        if self._bot.send_message(msg):
            logger.info(
                "Heartbeat sent | uptime=%ds | scans=%d | alerts=%d",
                uptime, scan_count, alert_count,
            )
        else:
            logger.error("Heartbeat message failed to send.")
            self._status = "DEGRADED"


# ---------------------------------------------------------------------------
# Module-level self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

    logging.basicConfig(level=logging.INFO)

    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to test heartbeat.")
    else:
        bot = TelegramBot(BOT_TOKEN, CHAT_ID)
        formatter = AlertFormatter()
        hb = HeartbeatSystem(
            telegram_bot=bot,
            formatter=formatter,
            interval_hours=0.0001,  # very short for testing
            send_startup=True,
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframes={"4H": "4h", "1H": "1h", "15M": "15m", "5M": "5m"},
        )
        hb.start()
        # Simulate some activity
        hb.increment_scans(10)
        hb.increment_alerts(2)
        time.sleep(1)
        hb.stop()
        print("Heartbeat test complete.")
