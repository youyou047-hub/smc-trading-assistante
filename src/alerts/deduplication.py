"""Deduplication Module V2.0 — Prevent duplicate alerts.

Tracks recently sent alerts and suppresses re-sends unless one of the
following conditions is met:

1. The confidence score changes by more than the configured threshold.
2. The market structure direction changes (e.g. bullish → bearish).
3. The setup becomes invalid (score drops below the alert tier minimum).
4. The setup becomes considerably stronger (jumps to a higher tier).
5. The cooldown period has elapsed and the alert is genuinely new.

Persistence:
  - In-memory by default (fast, zero I/O).
  - Optional JSON file persistence for crash recovery.

All public methods are thread-safe via an internal ``threading.Lock``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AlertDeduplicator:
    """Prevents sending duplicate or redundant alerts.

    The deduplicator maintains a record of recently sent alerts keyed by
    a compound identifier ``(symbol, signal_direction)``.  Each record
    stores the last timestamp, score, and market-structure direction.

    Args:
        score_change_threshold: Minimum absolute score change to re-alert.
        structure_change_resend: Re-alert when market structure flips.
        direction_change_resend: Re-alert when signal direction changes.
        cooldown_minutes: Minimum time between identical alerts.
        max_alerts_per_hour: Hard cap on alerts per hour per symbol.
        persist_path: Optional JSON file path for persistence across
                      restarts.  Set to ``None`` for in-memory only.
    """

    def __init__(
        self,
        score_change_threshold: float = 10.0,
        structure_change_resend: bool = True,
        direction_change_resend: bool = True,
        cooldown_minutes: float = 15.0,
        max_alerts_per_hour: int = 3,
        persist_path: Optional[str] = None,
    ) -> None:
        self._score_threshold = score_change_threshold
        self._structure_resend = structure_change_resend
        self._direction_resend = direction_change_resend
        self._cooldown = cooldown_minutes * 60  # convert to seconds
        self._max_per_hour = max_alerts_per_hour
        self._persist_path = persist_path

        # Internal state
        self._lock = threading.Lock()
        # Keyed by (symbol, signal_direction)
        self._last_alerts: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Keyed by symbol — rolling hourly window of alert timestamps
        self._hourly_counts: Dict[str, List[float]] = {}

        # Restore persisted state if available
        if persist_path and os.path.isfile(persist_path):
            self._load_state()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        """Restore deduplication state from a JSON file."""
        try:
            with open(self._persist_path, "r") as f:
                state = json.load(f)
            for key, val in state.get("last_alerts", {}).items():
                # Reconstruct tuple key
                symbol, direction = val.get("symbol", ""), val.get(
                    "direction", ""
                )
                self._last_alerts[(symbol, direction)] = val
            self._hourly_counts = state.get("hourly_counts", {})
            logger.info(
                "Deduplication state restored from %s", self._persist_path
            )
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning(
                "Failed to load deduplication state: %s — starting fresh",
                exc,
            )

    def _save_state(self) -> None:
        """Persist deduplication state to a JSON file."""
        if not self._persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            payload: Dict[str, Any] = {
                "last_alerts": {
                    f"{k[0]}:{k[1]}": {
                        "symbol": k[0],
                        "direction": k[1],
                        "timestamp": v.get("timestamp"),
                        "score": v.get("score"),
                        "structure": v.get("structure"),
                        "tier": v.get("tier"),
                    }
                    for k, v in self._last_alerts.items()
                },
                "hourly_counts": self._hourly_counts,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(self._persist_path, "w") as f:
                json.dump(payload, f, indent=2)
        except IOError as exc:
            logger.warning("Failed to save deduplication state: %s", exc)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def should_send(
        self,
        symbol: str,
        signal_direction: str,
        confidence_score: float,
        alert_tier: str,
        market_structure: Optional[str] = None,
    ) -> bool:
        """Decide whether a new alert should be sent.

        Args:
            symbol: Trading pair symbol.
            signal_direction: ``BUY`` or ``SELL``.
            confidence_score: Current confidence score (0-100).
            alert_tier: Current tier label (e.g. ``HIGH_PROBABILITY``).
            market_structure: Current market structure label
                              (e.g. ``bullish``, ``bearish``).

        Returns:
            ``True`` if the alert is new or qualifies for re-send,
            ``False`` if it should be suppressed.
        """
        with self._lock:
            key = (symbol, signal_direction)
            now = time.monotonic()

            # --- Hourly rate limit per symbol ---
            self._prune_hourly_counts(symbol, now)
            if len(self._hourly_counts.get(symbol, [])) >= self._max_per_hour:
                logger.info(
                    "Dedup: %s %s suppressed — hourly cap (%d) reached.",
                    symbol, signal_direction, self._max_per_hour,
                )
                return False

            # --- First alert ever for this key — always send ---
            if key not in self._last_alerts:
                logger.debug(
                    "Dedup: First alert for %s %s — sending.",
                    symbol, signal_direction,
                )
                self._record_alert(
                    key, symbol, signal_direction, now,
                    confidence_score, alert_tier, market_structure,
                )
                return True

            last = self._last_alerts[key]

            # --- Cooldown check ---
            elapsed = now - last["timestamp"]
            if elapsed < self._cooldown:
                logger.debug(
                    "Dedup: %s %s suppressed — cooldown not elapsed "
                    "(%.0fs / %.0fs required).",
                    symbol, signal_direction, elapsed, self._cooldown,
                )
                return False

            # --- Tier upgrade (jump to higher tier = resend) ---
            tier_order = [
                "IGNORED",
                "WATCHLIST",
                "SUB-THRESHOLD (RECORD ONLY)",
                "POTENTIAL",
                "HIGH_PROBABILITY",
                "HIGH PROBABILITY ALERT",
                "PREMIUM",
                "PREMIUM INSTITUTIONAL SETUP",
            ]
            last_tier_idx = tier_order.index(last.get("tier", "IGNORED"))
            current_tier_idx = tier_order.index(alert_tier)
            if current_tier_idx > last_tier_idx:
                logger.info(
                    "Dedup: %s %s — tier upgrade (%s → %s). Sending.",
                    symbol, signal_direction,
                    last.get("tier"), alert_tier,
                )
                self._record_alert(
                    key, symbol, signal_direction, now,
                    confidence_score, alert_tier, market_structure,
                )
                return True

            # --- Direction change (e.g. BUY → SELL for same symbol) ---
            if self._direction_resend:
                opposite_key = (
                    symbol,
                    "SELL" if signal_direction == "BUY" else "BUY",
                )
                if opposite_key in self._last_alerts:
                    opposite_last = self._last_alerts[opposite_key]
                    if (now - opposite_last["timestamp"]) < self._cooldown:
                        # Opposite direction was recently sent — the new one
                        # is a genuine reversal, allow it.
                        logger.info(
                            "Dedup: %s — direction reversal (%s → %s). "
                            "Sending.",
                            symbol, opposite_key[1], signal_direction,
                        )
                        self._record_alert(
                            key, symbol, signal_direction, now,
                            confidence_score, alert_tier,
                            market_structure,
                        )
                        return True

            # --- Score change ---
            last_score = last.get("score", 0.0)
            score_diff = abs(confidence_score - last_score)
            if score_diff >= self._score_threshold:
                logger.info(
                    "Dedup: %s %s — score changed by %.1f (threshold %.1f). "
                    "Sending.",
                    symbol, signal_direction, score_diff,
                    self._score_threshold,
                )
                self._record_alert(
                    key, symbol, signal_direction, now,
                    confidence_score, alert_tier, market_structure,
                )
                return True

            # --- Market structure change ---
            if self._structure_resend and market_structure:
                last_structure = last.get("structure")
                if last_structure and last_structure != market_structure:
                    logger.info(
                        "Dedup: %s %s — structure changed (%s → %s). "
                        "Sending.",
                        symbol, signal_direction,
                        last_structure, market_structure,
                    )
                    self._record_alert(
                        key, symbol, signal_direction, now,
                        confidence_score, alert_tier, market_structure,
                    )
                    return True

            # --- Setup invalidation (score dropped below tier) ---
            last_tier_min = self._tier_minimum(last.get("tier", "IGNORED"))
            if confidence_score < last_tier_min:
                logger.info(
                    "Dedup: %s %s — setup invalidated "
                    "(score %.1f < tier min %.1f). Sending.",
                    symbol, signal_direction,
                    confidence_score, last_tier_min,
                )
                self._record_alert(
                    key, symbol, signal_direction, now,
                    confidence_score, alert_tier, market_structure,
                )
                return True

            # --- Default: suppress duplicate ---
            logger.debug(
                "Dedup: %s %s — duplicate suppressed (no significant change).",
                symbol, signal_direction,
            )
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_alert(
        self,
        key: Tuple[str, str],
        symbol: str,
        direction: str,
        timestamp: float,
        score: float,
        tier: str,
        structure: Optional[str],
    ) -> None:
        """Record that an alert was sent."""
        self._last_alerts[key] = {
            "symbol": symbol,
            "direction": direction,
            "timestamp": timestamp,
            "score": score,
            "tier": tier,
            "structure": structure,
        }
        # Update hourly count
        self._hourly_counts.setdefault(symbol, []).append(timestamp)
        self._prune_hourly_counts(symbol, timestamp)
        self._save_state()

    def _prune_hourly_counts(self, symbol: str, now: float) -> None:
        """Remove entries older than 1 hour from the hourly tracker."""
        cutoff = now - 3600
        if symbol in self._hourly_counts:
            self._hourly_counts[symbol] = [
                ts for ts in self._hourly_counts[symbol] if ts >= cutoff
            ]

    @staticmethod
    def _tier_minimum(tier: str) -> float:
        """Return the minimum score for a given tier."""
        tier_map = {
            "WATCHLIST": 60,
            "SUB-THRESHOLD (RECORD ONLY)": 70,
            "POTENTIAL": 70,
            "HIGH_PROBABILITY": 80,
            "HIGH PROBABILITY ALERT": 80,
            "PREMIUM": 90,
            "PREMIUM INSTITUTIONAL SETUP": 90,
        }
        return tier_map.get(tier, 0)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_last_alert(
        self, symbol: str, direction: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the last alert record for a symbol/direction pair.

        Returns ``None`` if no record exists.
        """
        return self._last_alerts.get((symbol, direction))

    def clear(self) -> None:
        """Reset all deduplication state (useful for testing)."""
        with self._lock:
            self._last_alerts.clear()
            self._hourly_counts.clear()
            self._save_state()


# ---------------------------------------------------------------------------
# Module-level self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dedup = AlertDeduplicator(score_change_threshold=10, cooldown_minutes=0)

    # First alert — should always send
    assert dedup.should_send(
        "BTCUSDT", "BUY", 85.0, "HIGH_PROBABILITY", "bullish"
    )
    print("✔ First alert sent")

    # Duplicate within cooldown — suppressed
    assert not dedup.should_send(
        "BTCUSDT", "BUY", 85.5, "HIGH_PROBABILITY", "bullish"
    )
    print("✔ Duplicate suppressed within cooldown")

    # Score changed enough — resend
    assert dedup.should_send(
        "BTCUSDT", "BUY", 92.0, "PREMIUM", "bullish"
    )
    print("✔ Resent due to score change + tier upgrade")

    # Direction change — resend
    assert dedup.should_send(
        "BTCUSDT", "SELL", 78.0, "POTENTIAL", "bearish"
    )
    print("✔ Resent due to direction change")

    print("\nAll deduplication tests passed!")
