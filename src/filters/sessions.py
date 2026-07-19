"""Trading Session Filter Module.

Determines the current trading session and assigns a quality score.
Prioritizes London and New York sessions as specified in the trading methodology.
"""

import datetime
from typing import Dict, List, Optional, Tuple


class SessionFilter:
    """Detects active trading sessions and assigns quality scores.

    The system prioritizes London and New York sessions. Confidence is
    reduced outside major trading sessions unless the setup is exceptionally strong.

    Attributes:
        sessions_config (dict): Session configuration from settings.yaml.
    """

    def __init__(self, sessions_config: Dict) -> None:
        """Initialize the session filter.

        Args:
            sessions_config: Session configuration dictionary from settings.yaml.
                Expected keys: timezone, london, new_york, asian, quality.
        """
        self.timezone = sessions_config.get("timezone", "UTC")

        # Parse session times
        self.sessions = {}
        for session_name in ["london", "new_york", "asian"]:
            session_data = sessions_config.get(session_name, {})
            if session_data:
                start_parts = session_data.get("start", "00:00").split(":")
                end_parts = session_data.get("end", "00:00").split(":")
                self.sessions[session_name] = {
                    "start": datetime.time(int(start_parts[0]), int(start_parts[1])),
                    "end": datetime.time(int(end_parts[0]), int(end_parts[1])),
                }

        # Quality multipliers
        self.quality = sessions_config.get("quality", {
            "london": 1.0,
            "new_york": 1.0,
            "overlap": 1.0,
            "asian": 0.7,
            "off_session": 0.5,
        })

    def get_active_sessions(self, current_time_utc: datetime.datetime) -> List[str]:
        """Determines which trading sessions are currently active.

        Args:
            current_time_utc: The current UTC time.

        Returns:
            List of active session names (e.g., ['london', 'new_york']).
        """
        active = []
        current_time_only = current_time_utc.time()

        for session_name, times in self.sessions.items():
            start = times["start"]
            end = times["end"]

            if start <= end:
                # Session does not cross midnight
                if start <= current_time_only < end:
                    active.append(session_name)
            else:
                # Session crosses midnight
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
        """Returns the session quality multiplier (0.0 - 1.0).

        Used by the confidence scoring system to adjust the final score.
        London and New York sessions get full quality (1.0).
        Asian session gets reduced quality (0.7).
        Off-session gets lowest quality (0.5).

        Args:
            current_time_utc: The current UTC time.

        Returns:
            float: Quality multiplier between 0.0 and 1.0.
        """
        active = self.get_active_sessions(current_time_utc)

        if "london" in active and "new_york" in active:
            return self.quality.get("overlap", 1.0)
        elif "london" in active:
            return self.quality.get("london", 1.0)
        elif "new_york" in active:
            return self.quality.get("new_york", 1.0)
        elif "asian" in active:
            return self.quality.get("asian", 0.7)
        else:
            return self.quality.get("off_session", 0.5)

    def get_session_quality_score(self, current_time_utc: datetime.datetime) -> float:
        """Returns session quality as a percentage score (0-100).

        Args:
            current_time_utc: The current UTC time.

        Returns:
            float: Score between 0 and 100.
        """
        return self.get_session_quality(current_time_utc) * 100.0


if __name__ == "__main__":
    # Example usage
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
    }

    session_filter = SessionFilter(sessions_config=config)

    test_times = [
        datetime.datetime(2026, 7, 19, 3, 0),   # Asian
        datetime.datetime(2026, 7, 19, 9, 0),   # London
        datetime.datetime(2026, 7, 19, 14, 0),  # London-NY overlap
        datetime.datetime(2026, 7, 19, 19, 0),  # New York
        datetime.datetime(2026, 7, 19, 22, 0),  # Off-session
    ]

    print("--- Session Analysis ---")
    for dt in test_times:
        session_name = session_filter.get_current_session_name(dt)
        quality = session_filter.get_session_quality(dt)
        print(f"Time: {dt.strftime('%H:%M UTC')} | Session: {session_name} | Quality: {quality:.1f}")
