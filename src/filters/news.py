"""Economic News Filter — V2.0 (Upgraded).

Filters trading signals based on upcoming high-impact USD economic events.
Avoids generating alerts immediately before or after major events such as
CPI, PPI, NFP, FOMC, Interest Rate Decisions, and Powell Speeches.

Key improvements over V1:
  • **Separate before / after buffers** (V1 had only a single ``buffer_minutes``).
  • **Impact-based scoring**: instead of a flat 0.0 / 1.0, the filter returns
    a graduated score so that medium-impact events produce a partial penalty
    rather than a complete suppression.
  • **``NewsResult`` dataclass** with structured output (decision, reason,
    time-until-event, impact level).
  • **Scheduled event management**: events can be added individually or in
    bulk; a simple in-memory schedule is maintained (production systems can
    plug in a live calendar API by extending ``_refresh_events``).
  • **Backward-compatible** constructor and method names so V1 callers
    (``main.py``) continue to work without modification.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class NewsResult:
    """Structured result from the news filter evaluation.

    Attributes:
        should_avoid: ``True`` if trading should be paused.
        score: Quality score for the confidence scorer (0.0–1.0).
              1.0 = safe, 0.0 = full penalty.
        reason: Human-readable explanation.
        next_event: Name of the nearest approaching event (or ``None``).
        minutes_until_event: Minutes until that event (or ``None``).
        impact_level: ``'high'``, ``'medium'``, or ``None``.
    """
    should_avoid: bool
    score: float
    reason: str
    next_event: Optional[str] = None
    minutes_until_event: Optional[float] = None
    impact_level: Optional[str] = None


# ── Default event lists ───────────────────────────────────────────────────────

DEFAULT_HIGH_IMPACT_EVENTS: List[str] = [
    "CPI",
    "PPI",
    "NFP",
    "Nonfarm Payrolls",
    "FOMC",
    "Interest Rate Decision",
    "Federal Funds Rate",
    "Powell Speech",
    "GDP",
    "GDP Final",
    "Unemployment Claims",
    "Unemployment Rate",
    "Retail Sales",
    "ISM Manufacturing PMI",
    "ISM Services PMI",
]

DEFAULT_MEDIUM_IMPACT_EVENTS: List[str] = [
    "ADP Employment",
    "JOLTS Job Openings",
    "Michigan Consumer Sentiment",
    "Core PCE",
    "Durable Goods",
]


# ── Filter ────────────────────────────────────────────────────────────────────

class NewsFilter:
    """Filters trading signals based on economic news events.

    The filter checks if any high- or medium-impact USD events are approaching
    within the configured buffer windows.  If so, the confidence score is
    reduced or the alert is suppressed entirely.

    Backward compatibility:
        * Constructor signature matches V1 (``enabled``, ``buffer_minutes``,
          ``high_impact_events``).
        * ``is_news_approaching()`` and ``get_news_score()`` retain identical
          signatures and return types.
        * ``add_scheduled_event()`` works the same way.

    Args:
        enabled: Whether the filter is active.
        buffer_minutes_before: Minutes before an event to start suppressing.
        buffer_minutes_after: Minutes after an event to keep suppressing.
        penalty_score: Score multiplier when a high-impact event is within
            the buffer (0.0 = full suppression, 1.0 = no penalty).
        high_impact_events: Event names that trigger full suppression.
        medium_impact_events: Event names that trigger a partial penalty.
    """

    def __init__(
        self,
        enabled: bool = True,
        buffer_minutes: int = 30,
        high_impact_events: Optional[List[str]] = None,
        # V2 additions (keyword-only)
        buffer_minutes_before: Optional[int] = None,
        buffer_minutes_after: Optional[int] = None,
        penalty_score: float = 0.0,
        medium_impact_events: Optional[List[str]] = None,
        medium_penalty_score: float = 0.5,
    ) -> None:
        self.enabled = enabled

        # V2: separate before/after buffers, fall back to legacy single buffer
        self._buffer_before = buffer_minutes_before if buffer_minutes_before is not None else buffer_minutes
        self._buffer_after = buffer_minutes_after if buffer_minutes_after is not None else 0  # V1 had no "after"

        self.penalty_score = penalty_score
        self.medium_penalty_score = medium_penalty_score

        self.high_impact_events = high_impact_events or DEFAULT_HIGH_IMPACT_EVENTS
        self.medium_impact_events = medium_impact_events or DEFAULT_MEDIUM_IMPACT_EVENTS

        # In-memory event schedule
        self._scheduled_events: List[Dict] = []

    # ── Event management ──────────────────────────────────────────────────

    def add_scheduled_event(
        self,
        event_time: datetime.datetime,
        event_name: str,
        currency: str = "USD",
    ) -> None:
        """Manually add a scheduled economic event.

        Args:
            event_time: UTC datetime of the event.
            event_name: Name of the event (e.g. ``'FOMC'``).
            currency: Currency affected (default ``'USD'``).
        """
        self._scheduled_events.append({
            "time": event_time,
            "event": event_name,
            "currency": currency,
        })
        logger.debug(f"NewsFilter: scheduled event added — {event_name} at {event_time}")

    def add_scheduled_events(self, events: List[Dict]) -> None:
        """Bulk-add scheduled events.

        Each event dict should contain ``time``, ``event``, and optionally
        ``currency`` (default ``'USD'``).

        Args:
            events: List of event dictionaries.
        """
        for evt in events:
            self.add_scheduled_event(
                event_time=evt["time"],
                event_name=evt["event"],
                currency=evt.get("currency", "USD"),
            )

    def clear_scheduled_events(self) -> None:
        """Remove all manually scheduled events."""
        self._scheduled_events.clear()

    def _refresh_events(self) -> None:
        """Hook for plugging in a live economic calendar API.

        Override this method in a subclass to fetch events from an external
        API (Forex Factory, Investing.com, etc.).  The default implementation
        is a no-op because the filter relies on the in-memory schedule.
        """
        pass

    # ── Event classification ──────────────────────────────────────────────

    def _classify_event(self, event_name: str) -> str:
        """Classify an event as high-impact, medium-impact, or unknown.

        Args:
            event_name: The event name string.

        Returns:
            ``'high'``, ``'medium'``, or ``'unknown'``.
        """
        name_upper = event_name.upper()
        for hi in self.high_impact_events:
            if hi.upper() in name_upper or name_upper in hi.upper():
                return "high"
        for mi in self.medium_impact_events:
            if mi.upper() in name_upper or name_upper in mi.upper():
                return "medium"
        return "unknown"

    # ── Upcoming event detection ──────────────────────────────────────────

    def _get_upcoming_events(
        self, current_time: datetime.datetime
    ) -> List[Dict]:
        """Finds USD events within the before/after buffer window.

        An event is "upcoming" if:
            * It is a USD event.
            * The current time is within ``buffer_before`` minutes *before*
              the event **or** within ``buffer_after`` minutes *after* the event.

        Args:
            current_time: Current UTC time.

        Returns:
            List of event dicts within the buffer window.
        """
        upcoming: List[Dict] = []
        before_delta = datetime.timedelta(minutes=self._buffer_before)
        after_delta = datetime.timedelta(minutes=self._buffer_after)

        for event in self._scheduled_events:
            if event.get("currency", "USD") != "USD":
                continue

            event_time = event["time"]
            time_diff = event_time - current_time

            # Within "before" buffer (event is in the future)
            in_before_buffer = datetime.timedelta(0) <= time_diff <= before_delta
            # Within "after" buffer (event just passed)
            in_after_buffer = -after_delta <= time_diff < datetime.timedelta(0)

            if in_before_buffer or in_after_buffer:
                upcoming.append(event)

        return upcoming

    def _get_nearest_event(
        self, current_time: datetime.datetime
    ) -> Optional[Dict]:
        """Returns the single closest upcoming USD event (within buffer).

        Returns:
            The event dict, or ``None`` if no event is within range.
        """
        upcoming = self._get_upcoming_events(current_time)
        if not upcoming:
            return None
        # Sort by time distance (ascending)
        upcoming.sort(key=lambda e: abs((e["time"] - current_time).total_seconds()))
        return upcoming[0]

    # ── Backward-compatible methods ───────────────────────────────────────

    def is_news_approaching(self, current_time: datetime.datetime) -> bool:
        """Checks if any high-impact news event is approaching within the buffer.

        Primary method called by ``main.py`` to decide whether to suppress alerts.

        Args:
            current_time: Current UTC time.

        Returns:
            ``True`` if a high-impact event is approaching, ``False`` otherwise.
        """
        if not self.enabled:
            return False

        self._refresh_events()
        upcoming = self._get_upcoming_events(current_time)

        # Check for high-impact events
        for event in upcoming:
            if self._classify_event(event["event"]) == "high":
                event_names = [e["event"] for e in upcoming if self._classify_event(e["event"]) == "high"]
                logger.warning(
                    f"NewsFilter: high-impact news approaching within {self._buffer_before}min: "
                    f"{', '.join(event_names)}"
                )
                return True

        return False

    def is_impactful_news_present(self, current_time: datetime.datetime) -> bool:
        """Alias for ``is_news_approaching()`` (V1 backward compat)."""
        return self.is_news_approaching(current_time)

    def get_news_score(self, current_time: datetime.datetime) -> float:
        """Returns a score (0.0–1.0) for the confidence scoring system.

        * High-impact event in buffer → ``penalty_score`` (default 0.0).
        * Medium-impact event in buffer → ``medium_penalty_score`` (default 0.5).
        * No event → ``1.0``.

        Args:
            current_time: Current UTC time.

        Returns:
            float: Score between 0.0 and 1.0.
        """
        if not self.enabled:
            return 1.0

        self._refresh_events()
        result = self.evaluate(current_time)
        return result.score

    # ── V2 structured evaluation ──────────────────────────────────────────

    def evaluate(self, current_time: datetime.datetime) -> NewsResult:
        """Performs full news filter evaluation and returns a structured result.

        This is the preferred V2 entry point.  It bundles the decision,
        score, reason, and event details into a single ``NewsResult``.

        Args:
            current_time: Current UTC time.

        Returns:
            ``NewsResult`` with all relevant information.
        """
        if not self.enabled:
            return NewsResult(
                should_avoid=False,
                score=1.0,
                reason="News filter disabled.",
            )

        self._refresh_events()
        upcoming = self._get_upcoming_events(current_time)

        if not upcoming:
            return NewsResult(
                should_avoid=False,
                score=1.0,
                reason="No high-impact events within buffer window.",
            )

        # Determine the highest impact level among upcoming events
        highest_impact = "unknown"
        highest_score = 1.0
        highest_event = None

        for event in upcoming:
            impact = self._classify_event(event["event"])
            if impact == "high":
                highest_impact = "high"
                highest_score = self.penalty_score
                highest_event = event
                break  # High impact always wins
            elif impact == "medium" and highest_impact != "high":
                highest_impact = "medium"
                highest_score = self.medium_penalty_score
                highest_event = event

        if highest_event is None:
            return NewsResult(
                should_avoid=False,
                score=1.0,
                reason="No classified events within buffer.",
            )

        # Calculate time until event
        time_diff = (highest_event["time"] - current_time).total_seconds() / 60.0
        minutes_until = time_diff

        # Build reason
        if highest_impact == "high":
            should_avoid = highest_score == 0.0
            reason = (
                f"High-impact event '{highest_event['event']}' in {minutes_until:.0f} min. "
                f"Trading {'suppressed' if should_avoid else 'penalised'}. "
                f"Buffer: {self._buffer_before}min before + {self._buffer_after}min after."
            )
        else:
            should_avoid = False
            reason = (
                f"Medium-impact event '{highest_event['event']}' in {minutes_until:.0f} min. "
                f"Score reduced to {highest_score:.1f}."
            )

        return NewsResult(
            should_avoid=should_avoid,
            score=highest_score,
            reason=reason,
            next_event=highest_event["event"],
            minutes_until_event=minutes_until,
            impact_level=highest_impact,
        )


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    now = datetime.datetime.now(datetime.timezone.utc)

    # --- Test 1: High-impact event approaching ---
    news = NewsFilter(
        enabled=True,
        buffer_minutes_before=30,
        buffer_minutes_after=30,
        high_impact_events=["CPI", "NFP", "FOMC"],
        medium_impact_events=["ADP Employment"],
    )

    news.add_scheduled_event(
        event_time=now + datetime.timedelta(minutes=20),
        event_name="CPI Release",
        currency="USD",
    )
    news.add_scheduled_event(
        event_time=now + datetime.timedelta(minutes=45),
        event_name="ADP Employment",
        currency="USD",
    )

    print("=== V2 News Filter Evaluation ===\n")
    result = news.evaluate(now)
    print(f"Should avoid: {result.should_avoid}")
    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")
    print(f"Next event: {result.next_event} ({result.minutes_until_event:.0f} min away)")
    print(f"Impact: {result.impact_level}")

    print(f"\nLegacy check — is_news_approaching: {news.is_news_approaching(now)}")
    print(f"Legacy check — get_news_score: {news.get_news_score(now)}")

    # --- Test 2: No events ---
    print("\n--- No events ---")
    news2 = NewsFilter(enabled=True, buffer_minutes_before=30)
    result2 = news2.evaluate(now)
    print(f"Should avoid: {result2.should_avoid} | Score: {result2.score} | Reason: {result2.reason}")
