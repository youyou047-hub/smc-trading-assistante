"""Trading Session Filter — V2.0 (Upgraded).

Detects the current trading session, assigns a quality score, and returns
structured results consumed by the confidence scorer.

Key improvements over V1:
  • Structured return type (``SessionResult`` dataclass) instead of three
    separate method calls.
  • Quality scoring is now a first-class concept with configurable per-session
    weights (Asian = lowest, London-NY overlap = highest).
  • Backward-compatible: the original ``SessionFilter`` class and method names
    still work so existing V1 callers (``main.py``) are unaffected.
  • Handles overlapping sessions correctly (London-NY overlap gets its own
    quality tier).
  • Supports custom session definitions via ``sessions_config`` dict.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class SessionResult:
    """Structured result from session analysis.

    Attributes:
        session_name: Human-readable name (e.g. ``'London-NY Overlap'``).
        active_sessions: List of raw session keys currently active.
        quality_score: Float 0.0–1.0 representing session quality.
        is_high_quality: Whether the session meets a configurable threshold.
        quality_multiplier: Same as quality_score (alias for backward compat).
    """
    session_name: str
    active_sessions: List[str]
    quality_score: float
    is_high_quality: bool
    quality_multiplier: float = 0.0  # populated after __post_init__

    def __post_init__(self) -> None:
        self.quality_multiplier = self.quality_score


# ── Session definitions (default times in UTC) ───────────────────────────────

DEFAULT_SESSIONS: Dict[str, Tuple[str, str]] = {
    "asian": ("00:00", "08:00"),
    "london": ("08:00", "16:00"),
    "new_york": ("13:00", "21:00"),
}


class SessionFilter:
    """Detects active trading sessions and assigns quality scores.

    The system prioritises London and New York sessions as specified in the
    Smart Money Concepts trading methodology.  The London-NY overlap (13:00–16:00 UTC)
    is the highest-quality window.

    Backward compatibility:
        * Constructor accepts the same ``sessions_config`` dict as V1.
        * ``get_active_sessions()``, ``get_current_session_name()``, and
          ``get_session_quality()`` are preserved with identical signatures.

    Args:
        sessions_config: Session configuration dictionary from settings.yaml.
            Expected keys: ``timezone``, ``london``, ``new_york``, ``asian``,
            ``quality``, ``session_confluence``.
        high_quality_threshold: Minimum quality score (0.0–1.0) to consider
            a session "high quality" (default 0.9).
    """

    def __init__(
        self,
        sessions_config: Dict,
        high_quality_threshold: float = 0.9,
    ) -> None:
        self.timezone: str = sessions_config.get("timezone", "UTC")
        self._high_quality_threshold = high_quality_threshold

        # ── Parse session times ──
        self._session_times: Dict[str, Dict[str, datetime.time]] = {}
        for session_name in ["london", "new_york", "asian"]:
            session_data = sessions_config.get(session_name, {})
            if session_data:
                start_parts = session_data.get("start", "00:00").split(":")
                end_parts = session_data.get("end", "00:00").split(":")
                self._session_times[session_name] = {
                    "start": datetime.time(int(start_parts[0]), int(start_parts[1])),
                    "end": datetime.time(int(end_parts[0]), int(end_parts[1])),
                }
            else:
                # Fall back to defaults
                start_str, end_str = DEFAULT_SESSIONS.get(session_name, ("00:00", "00:00"))
                s_parts = start_str.split(":")
                e_parts = end_str.split(":")
                self._session_times[session_name] = {
                    "start": datetime.time(int(s_parts[0]), int(s_parts[1])),
                    "end": datetime.time(int(e_parts[0]), int(e_parts[1])),
                }

        # ── Quality multipliers ──
        quality_cfg = sessions_config.get("quality", {})
        self.quality: Dict[str, float] = {
            "london": quality_cfg.get("london", 1.0),
            "new_york": quality_cfg.get("new_york", 1.0),
            "overlap": quality_cfg.get("overlap", 1.0),
            "asian": quality_cfg.get("asian", 0.7),
            "off_session": quality_cfg.get("off_session", 0.5),
        }

        # ── Confluence bonus ──
        confluence_cfg = sessions_config.get("session_confluence", {})
        self._overlap_bonus: float = confluence_cfg.get("overlap_bonus", 0.05)

    # ── Backward-compatible methods ──────────────────────────────────────

    def get_active_sessions(self, current_time_utc: datetime.datetime) -> List[str]:
        """Determines which trading sessions are currently active.

        Args:
            current_time_utc: The current UTC time.

        Returns:
            List of active session names (e.g. ``['london', 'new_york']``).
        """
        active: List[str] = []
        current_time_only = current_time_utc.time()

        for session_name, times in self._session_times.items():
            start = times["start"]
            end = times["end"]

            if start <= end:
                # Normal session (does not cross midnight)
                if start <= current_time_only < end:
                    active.append(session_name)
            else:
                # Session crosses midnight (e.g. Asian: 00:00–08:00)
                if current_time_only >= start or current_time_only < end:
                    active.append(session_name)

        return active

    def get_current_session_name(self, current_time_utc: datetime.datetime) -> str:
        """Returns a human-readable description of the current session.

        Args:
            current_time_utc: The current UTC time.

        Returns:
            String describing the current session state.
        """
        active = self.get_active_sessions(current_time_utc)

        if "london" in active and "new_york" in active:
            return "London-NY Overlap"
        elif "london" in active:
            return "London"
        elif "new_york" in active:
            return "New York"
        elif "asian" in active:
            return "Asian"
        else:
            return "Off-Session"

    def get_session_quality(self, current_time_utc: datetime.datetime) -> float:
        """Returns the session quality multiplier (0.0–1.0).

        London and New York sessions get full quality (1.0).
        London-NY overlap gets 1.0 + overlap bonus (capped at 1.0).
        Asian session gets reduced quality (0.7).
        Off-session gets lowest quality (0.5).

        Args:
            current_time_utc: The current UTC time.

        Returns:
            float: Quality multiplier between 0.0 and 1.0.
        """
        active = self.get_active_sessions(current_time_utc)

        if "london" in active and "new_york" in active:
            # Overlap: base quality + confluence bonus, capped at 1.0
            base = self.quality.get("overlap", 1.0)
            return min(1.0, base + self._overlap_bonus)
        elif "london" in active:
            return self.quality.get("london", 1.0)
        elif "new_york" in active:
            return self.quality.get("new_york", 1.0)
        elif "asian" in active:
            return self.quality.get("asian", 0.7)
        else:
            return self.quality.get("off_session", 0.5)

    def get_session_quality_score(self, current_time_utc: datetime.datetime) -> float:
        """Returns session quality as a percentage score (0–100).

        Args:
            current_time_utc: The current UTC time.

        Returns:
            float: Score between 0 and 100.
        """
        return self.get_session_quality(current_time_utc) * 100.0

    # ── V2 structured method ─────────────────────────────────────────────

    def evaluate(self, current_time_utc: datetime.datetime) -> SessionResult:
        """Performs full session evaluation and returns a structured result.

        This is the preferred V2 entry point.  It bundles all session
        information into a single ``SessionResult`` object.

        Args:
            current_time_utc: The current UTC time.

        Returns:
            ``SessionResult`` with session name, active list, quality score,
            and ``is_high_quality`` flag.
        """
        active = self.get_active_sessions(current_time_utc)
        session_name = self.get_current_session_name(current_time_utc)
        quality = self.get_session_quality(current_time_utc)
        is_high_quality = quality >= self._high_quality_threshold

        return SessionResult(
            session_name=session_name,
            active_sessions=active,
            quality_score=quality,
            is_high_quality=is_high_quality,
        )


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {
        "timezone": "UTC",
        "london": {"start": "08:00", "end": "16:00"},
        "new_york": {"start": "13:00", "end": "21:00"},
        "asian": {"start": "00:00", "end": "08:00"},
        "quality": {
            "london": 1.0,
            "new_york": 1.0,
            "overlap": 1.0,
            "asian": 0.7,
            "off_session": 0.5,
        },
        "session_confluence": {"overlap_bonus": 0.05},
    }

    session_filter = SessionFilter(sessions_config=config)

    test_times = [
        datetime.datetime(2026, 7, 19, 3, 0),   # Asian
        datetime.datetime(2026, 7, 19, 9, 0),   # London
        datetime.datetime(2026, 7, 19, 14, 0),  # London-NY overlap
        datetime.datetime(2026, 7, 19, 19, 0),  # New York
        datetime.datetime(2026, 7, 19, 22, 0),  # Off-session
    ]

    print("=== V2 Session Analysis ===\n")
    for dt in test_times:
        result = session_filter.evaluate(dt)
        print(
            f"Time: {dt.strftime('%H:%M UTC')}  |  "
            f"Session: {result.session_name:<20}  |  "
            f"Quality: {result.quality_score:.2f}  |  "
            f"High-Quality: {result.is_high_quality}  |  "
            f"Active: {result.active_sessions}"
        )
