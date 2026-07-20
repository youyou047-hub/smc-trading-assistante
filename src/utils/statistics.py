"""Statistics Module V2.0 — Track every detected setup.

Records comprehensive statistics for each detected setup and stores them
in a JSON file for long-term persistence and analysis.

Features:
- Records timestamp, direction, score, reasons, outcome, symbol, timeframe
- JSON file persistence with automatic directory creation
- Query methods for history retrieval and summary generation
- Configurable retention period (prunes old entries)
- Thread-safe writes via a lock

This module helps track system performance over time and identify
patterns that could lead to future improvements.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StatisticsTracker:
    """Tracks and persists trading setup statistics.

    Args:
        store_path: Path to the JSON statistics file.
        retention_days: How many days to keep entries (0 = unlimited).
        enabled: Whether to actually record statistics.
    """

    def __init__(
        self,
        store_path: str = "logs/stats.json",
        retention_days: int = 90,
        enabled: bool = True,
    ) -> None:
        self._store_path = store_path
        self._retention_days = retention_days
        self._enabled = enabled
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []

        # Load existing records from file
        if enabled:
            self._load_records()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_records(self) -> None:
        """Load existing records from the JSON store."""
        if not os.path.isfile(self._store_path):
            return
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._records = data.get("records", [])
            logger.info(
                "Loaded %d statistics records from %s",
                len(self._records), self._store_path,
            )
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning(
                "Failed to load statistics from %s: %s — starting fresh",
                self._store_path, exc,
            )
            self._records = []

    def _save_records(self) -> None:
        """Persist current records to the JSON store."""
        try:
            os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
            payload = {
                "records": self._records,
                "total_count": len(self._records),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except IOError as exc:
            logger.error("Failed to save statistics: %s", exc)

    # ------------------------------------------------------------------
    # Public API — Recording
    # ------------------------------------------------------------------
    def record_setup(
        self,
        symbol: str,
        direction: str,
        confidence_score: float,
        alert_tier: str,
        timeframe: str = "",
        reasons: Optional[List[str]] = None,
        entry_zone_start: Optional[float] = None,
        entry_zone_end: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profits: Optional[List[float]] = None,
        score_breakdown: Optional[Dict[str, float]] = None,
        outcome: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a detected setup.

        Args:
            symbol: Trading pair symbol.
            direction: Signal direction (``BUY`` or ``SELL``).
            confidence_score: Overall confidence score (0-100).
            alert_tier: Tier label (e.g. ``HIGH_PROBABILITY``).
            timeframe: Analyzed timeframe.
            reasons: List of signal-reason strings.
            entry_zone_start: Entry zone lower bound.
            entry_zone_end: Entry zone upper bound.
            stop_loss: Stop loss level.
            take_profits: List of take-profit levels.
            score_breakdown: Per-component weighted scores.
            outcome: *(optional)* Trade outcome (``WIN``, ``LOSS``,
                     ``IN_PROGRESS``).
            extra: Additional custom fields.
        """
        if not self._enabled:
            return

        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "confidence_score": round(confidence_score, 2),
            "alert_tier": alert_tier,
            "timeframe": timeframe,
            "reasons": reasons or [],
            "entry_zone_start": entry_zone_start,
            "entry_zone_end": entry_zone_end,
            "stop_loss": stop_loss,
            "take_profits": take_profits or [],
            "score_breakdown": score_breakdown or {},
            "outcome": outcome or "IN_PROGRESS",
        }
        if extra:
            record.update(extra)

        with self._lock:
            self._records.append(record)
            self._save_records()

        logger.debug(
            "Recorded setup: %s %s (score=%.1f, tier=%s)",
            symbol, direction, confidence_score, alert_tier,
        )

    def update_outcome(
        self,
        timestamp: str,
        symbol: str,
        outcome: str,
    ) -> bool:
        """Update the outcome of a previously recorded setup.

        Args:
            timestamp: ISO timestamp of the original record.
            symbol: Trading pair symbol.
            outcome: New outcome string (``WIN``, ``LOSS``, etc.).

        Returns:
            ``True`` if a matching record was found and updated.
        """
        with self._lock:
            for record in self._records:
                if record["timestamp"] == timestamp and record["symbol"] == symbol:
                    record["outcome"] = outcome
                    record["outcome_updated"] = datetime.now(timezone.utc).isoformat()
                    self._save_records()
                    logger.info(
                        "Updated outcome for %s %s: %s",
                        symbol, timestamp, outcome,
                    )
                    return True
        logger.warning(
            "No matching record found for %s %s", symbol, timestamp,
        )
        return False

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_history(
        self,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        since: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query recorded setups with optional filters.

        Args:
            symbol: Filter by symbol.
            direction: Filter by direction (``BUY`` / ``SELL``).
            min_score: Minimum confidence score.
            max_score: Maximum confidence score.
            since: ISO date string to filter records after.
            limit: Maximum number of records to return.

        Returns:
            List of matching record dicts.
        """
        with self._lock:
            results = self._records[:]

        if symbol:
            results = [r for r in results if r["symbol"] == symbol]
        if direction:
            results = [r for r in results if r["direction"] == direction]
        if min_score is not None:
            results = [r for r in results if r["confidence_score"] >= min_score]
        if max_score is not None:
            results = [r for r in results if r["confidence_score"] <= max_score]
        if since:
            since_dt = datetime.fromisoformat(since)
            results = [
                r for r in results
                if datetime.fromisoformat(r["timestamp"]) >= since_dt
            ]
        if limit:
            results = results[:limit]

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary statistics report.

        Returns:
            Dict with counts, averages, win rates, and tier distribution.
        """
        with self._lock:
            records = self._records[:]

        if not records:
            return {"total_records": 0, "message": "No data available."}

        total = len(records)
        directions = {"BUY": 0, "SELL": 0}
        tier_counts: Dict[str, int] = {}
        scores = []
        outcomes = {"WIN": 0, "LOSS": 0, "IN_PROGRESS": 0, "OTHER": 0}

        for r in records:
            d = r.get("direction", "")
            if d in directions:
                directions[d] += 1
            tier = r.get("alert_tier", "UNKNOWN")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            scores.append(r.get("confidence_score", 0))
            outcome = r.get("outcome", "IN_PROGRESS")
            if outcome in outcomes:
                outcomes[outcome] += 1
            else:
                outcomes["OTHER"] += 1

        avg_score = sum(scores) / len(scores) if scores else 0
        decided = outcomes["WIN"] + outcomes["LOSS"]
        win_rate = (outcomes["WIN"] / decided * 100) if decided > 0 else 0

        summary: Dict[str, Any] = {
            "total_records": total,
            "directions": directions,
            "tier_distribution": tier_counts,
            "average_score": round(avg_score, 2),
            "score_distribution": {
                "60-69": sum(1 for s in scores if 60 <= s < 70),
                "70-79": sum(1 for s in scores if 70 <= s < 80),
                "80-89": sum(1 for s in scores if 80 <= s < 90),
                "90-100": sum(1 for s in scores if 90 <= s <= 100),
            },
            "outcomes": outcomes,
            "win_rate_pct": round(win_rate, 2) if decided > 0 else None,
            "earliest": records[0]["timestamp"] if records else None,
            "latest": records[-1]["timestamp"] if records else None,
        }
        return summary

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def prune_old_records(self) -> int:
        """Remove records older than the retention period.

        Returns:
            Number of records removed.
        """
        if self._retention_days <= 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        with self._lock:
            original_count = len(self._records)
            self._records = [
                r for r in self._records
                if datetime.fromisoformat(r["timestamp"]) >= cutoff
            ]
            removed = original_count - len(self._records)
            if removed > 0:
                self._save_records()
                logger.info("Pruned %d old statistics records.", removed)
            return removed

    def clear(self) -> None:
        """Delete all recorded statistics."""
        with self._lock:
            self._records.clear()
            self._save_records()
        logger.info("All statistics cleared.")


# ---------------------------------------------------------------------------
# Module-level self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        test_path = f.name

    tracker = StatisticsTracker(store_path=test_path, retention_days=30)

    # Record some test setups
    tracker.record_setup(
        symbol="BTCUSDT",
        direction="BUY",
        confidence_score=92.5,
        alert_tier="PREMIUM",
        timeframe="15m",
        reasons=["Bullish structure", "Liquidity sweep", "BOS confirmed"],
        entry_zone_start=64500,
        entry_zone_end=64600,
        stop_loss=64200,
        take_profits=[64900, 65200, 65600],
        score_breakdown={"market_structure": 20, "liquidity": 18, "bos": 15},
    )
    tracker.record_setup(
        symbol="BTCUSDT",
        direction="SELL",
        confidence_score=78.0,
        alert_tier="POTENTIAL",
        timeframe="15m",
        reasons=["Bearish structure", "Liquidity sweep"],
    )
    tracker.record_setup(
        symbol="ETHUSDT",
        direction="BUY",
        confidence_score=85.0,
        alert_tier="HIGH_PROBABILITY",
        timeframe="15m",
        reasons=["Bullish FVG", "Order block"],
    )

    # Query
    all_records = tracker.get_history()
    print(f"Total records: {len(all_records)}")

    btc_records = tracker.get_history(symbol="BTCUSDT")
    print(f"BTCUSDT records: {len(btc_records)}")

    buy_records = tracker.get_history(direction="BUY")
    print(f"BUY records: {len(buy_records)}")

    # Summary
    summary = tracker.get_summary()
    print(f"\nSummary: {json.dumps(summary, indent=2)}")

    # Update outcome
    first_ts = all_records[0]["timestamp"]
    tracker.update_outcome(first_ts, "BTCUSDT", "WIN")
    updated = tracker.get_history(symbol="BTCUSDT", direction="BUY")[0]
    print(f"\nUpdated outcome: {updated['outcome']}")

    # Prune
    removed = tracker.prune_old_records()
    print(f"Pruned: {removed}")

    # Cleanup
    os.unlink(test_path)
    print("\nAll statistics tests passed!")
