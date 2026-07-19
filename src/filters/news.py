"""Economic News Filter Module.

Filters trading signals based on upcoming high-impact USD economic events.
The system avoids generating alerts immediately before major events like
CPI, PPI, NFP, FOMC, Interest Rate Decisions, and Powell Speeches.

Note: This module uses a configurable event schedule approach. For production
use, integrate with a live economic calendar API (e.g., Forex Factory,
Investing.com, or TradingEconomics) for real-time event data.
"""

import datetime
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NewsFilter:
    """Filters trading signals based on upcoming high-impact economic events.

    The filter checks if any high-impact USD events are approaching within
    the configured buffer period. If so, alerts should be suppressed.

    Attributes:
        enabled (bool): Whether the news filter is active.
        buffer_minutes (int): Minutes before an event to suppress alerts.
        high_impact_events (List[str]): List of event types to watch for.
    """

    def __init__(
        self,
        enabled: bool = True,
        buffer_minutes: int = 30,
        high_impact_events: Optional[List[str]] = None,
    ) -> None:
        """Initialize the news filter.

        Args:
            enabled: Whether the filter is active.
            buffer_minutes: Minimum buffer before events (configurable).
            high_impact_events: List of high-impact event names to monitor.
        """
        self.enabled = enabled
        self.buffer_minutes = buffer_minutes
        self.high_impact_events = high_impact_events or [
            "CPI", "PPI", "NFP", "FOMC",
            "Interest Rate Decision", "Powell Speech",
        ]
        self._scheduled_events: List[Dict] = []

    def add_scheduled_event(
        self,
        event_time: datetime.datetime,
        event_name: str,
        currency: str = "USD",
    ) -> None:
        """Manually add a scheduled economic event.

        This allows pre-loading known events from a calendar or API.

        Args:
            event_time: UTC datetime of the event.
            event_name: Name of the event (e.g., 'FOMC').
            currency: Currency affected (default 'USD').
        """
        self._scheduled_events.append({
            "time": event_time,
            "event": event_name,
            "currency": currency,
        })
        logger.debug(f"Scheduled event added: {event_name} at {event_time}")

    def _get_upcoming_events(
        self, current_time: datetime.datetime
    ) -> List[Dict]:
        """Gets upcoming high-impact events within the buffer window.

        In production, this would query a live economic calendar API.
        Currently uses manually scheduled events.

        Args:
            current_time: Current UTC time.

        Returns:
            List of upcoming event dictionaries within the buffer window.
        """
        upcoming = []
        buffer_delta = datetime.timedelta(minutes=self.buffer_minutes)

        for event in self._scheduled_events:
            if event["currency"] != "USD":
                continue

            event_time = event["time"]
            time_until_event = event_time - current_time

            # Check if event is within buffer window (upcoming)
            if datetime.timedelta(0) <= time_until_event <= buffer_delta:
                upcoming.append(event)

        return upcoming

    def is_news_approaching(self, current_time: datetime.datetime) -> bool:
        """Checks if any high-impact news event is approaching within the buffer.

        This is the primary method called by the main loop to determine
        whether to suppress alerts.

        Args:
            current_time: Current UTC time.

        Returns:
            True if a high-impact event is approaching, False otherwise.
        """
        if not self.enabled:
            return False

        upcoming = self._get_upcoming_events(current_time)

        if upcoming:
            event_names = [e["event"] for e in upcoming]
            logger.warning(
                f"High-impact news approaching within {self.buffer_minutes}min: "
                f"{', '.join(event_names)}"
            )
            return True

        return False

    # Alias for backward compatibility
    def is_impactful_news_present(self, current_time: datetime.datetime) -> bool:
        """Alias for is_news_approaching()."""
        return self.is_news_approaching(current_time)

    def get_news_score(self, current_time: datetime.datetime) -> float:
        """Returns a score (0.0 or 1.0) for the confidence scoring system.

        Returns 1.0 (full score) if no news is approaching.
        Returns 0.0 if news is within the buffer period.

        Args:
            current_time: Current UTC time.

        Returns:
            float: 1.0 if safe, 0.0 if news is approaching.
        """
        if self.is_news_approaching(current_time):
            return 0.0
        return 1.0


if __name__ == "__main__":
    import datetime

    # Example usage
    news_filter = NewsFilter(
        enabled=True,
        buffer_minutes=30,
        high_impact_events=["CPI", "PPI", "NFP", "FOMC", "Interest Rate Decision"],
    )

    now = datetime.datetime.now(datetime.timezone.utc)

    # Add a test event 20 minutes from now
    news_filter.add_scheduled_event(
        event_time=now + datetime.timedelta(minutes=20),
        event_name="FOMC Meeting",
        currency="USD",
    )

    print(f"Current UTC Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"News approaching: {news_filter.is_news_approaching(now)}")
    print(f"News score: {news_filter.get_news_score(now)}")

    # Test with no events
    news_filter2 = NewsFilter(enabled=True, buffer_minutes=30)
    print(f"\nNo events - News approaching: {news_filter2.is_news_approaching(now)}")
    print(f"No events - News score: {news_filter2.get_news_score(now)}")
