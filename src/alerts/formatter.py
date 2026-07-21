"""Alert Formatter V2.0 — Tiered, intelligent HTML alerts for Telegram.

Produces professional-grade HTML messages with:
- Four alert tiers: Watchlist, Potential, High Probability, Premium
- Intelligent explanation showing ✔ found conditions and ✘ missing ones
- Detailed score breakdown per component
- Entry zone, stop loss, take profits with R:R ratios
- Risk-level indicator
- Clean, readable HTML formatting optimised for Telegram

Backward-compatible: the ``format_alert_message`` signature is preserved
so V1 callers (main orchestrator) continue to work unchanged.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------
# Each tier is a tuple of (emoji_icon, label, color)
TIER_MAP: Dict[str, Tuple[str, str, str]] = {
    "WATCHLIST": (
        "📋",
        "WATCHLIST ALERT",
        "#FFA500",   # orange
    ),
    "POTENTIAL": (
        "⚡",
        "POTENTIAL SETUP",
        "#1E90FF",   # dodgerblue
    ),
    "HIGH_PROBABILITY": (
        "🎯",
        "HIGH PROBABILITY SETUP",
        "#00FF7F",   # springgreen
    ),
    "PREMIUM": (
        "🔥",
        "PREMIUM INSTITUTIONAL SETUP",
        "#FFD700",   # gold
    ),
    # V1 legacy compatibility
    "HIGH PROBABILITY ALERT": (
        "🎯",
        "HIGH PROBABILITY SETUP",
        "#00FF7F",
    ),
    "PREMIUM INSTITUTIONAL SETUP": (
        "🔥",
        "PREMIUM INSTITUTIONAL SETUP",
        "#FFD700",
    ),
    "SUB-THRESHOLD (RECORD ONLY)": (
        "📋",
        "WATCHLIST ALERT",
        "#FFA500",
    ),
}

# Direction colours
BUY_COLOR = "#00b060"
SELL_COLOR = "#ff3333"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _escape_html(text: str) -> str:
    """Escape characters that would break Telegram HTML parsing.

    Telegram's HTML parser is strict; unescaped ``<`` or ``&`` cause
    silent failures.  We only escape what Telegram does not accept.
    """
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def _classify_condition(
    component_name: str,
    score: float,
    weight: float,
) -> Tuple[str, str]:
    """Determine whether a condition is met or missing.

    Args:
        component_name: Human-readable component label.
        score: Raw component score (0.0-1.0).
        weight: Configured weight for the component.

    Returns:
        Tuple of ``(label, status)`` where status is ``✔`` or ``✘``.
    """
    contribution = score * weight
    # A condition is "found" if it contributes >= 60% of its max possible
    threshold = weight * 0.6
    label = component_name.replace("_", " ").title()
    if contribution >= threshold:
        return (label, "✔")
    return (label, "✘")


# ---------------------------------------------------------------------------
# AlertFormatter
# ---------------------------------------------------------------------------
class AlertFormatter:
    """Formats professional tiered alert messages in HTML for Telegram.

    The formatter can handle both the new V2 tier names
    (``WATCHLIST``, ``POTENTIAL``, ``HIGH_PROBABILITY``, ``PREMIUM``) and
    the V1 legacy names (``HIGH PROBABILITY ALERT``, etc.).
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def format_alert_message(
        self,
        signal_direction: str,
        symbol: str,
        timeframe: str,
        entry_zone_start: float,
        entry_zone_end: float,
        stop_loss: float,
        take_profits: List[float],
        risk_reward_ratios: List[float],
        confidence_score: float,
        confidence_breakdown: Dict[str, float],
        human_explanation: str,
        alert_tier: str,
        *,
        # V2 optional fields — gracefully ignored if absent
        score_weights: Optional[Dict[str, float]] = None,
        component_raw_scores: Optional[Dict[str, float]] = None,
        reasons: Optional[List[str]] = None,
        missing_conditions: Optional[List[str]] = None,
        risk_level: Optional[str] = None,
    ) -> str:
        """Format a complete tiered alert message in HTML.

        Args:
            signal_direction: ``BUY`` or ``SELL``.
            symbol: Trading pair symbol.
            timeframe: Analyzed timeframe(s).
            entry_zone_start: Entry zone lower bound.
            entry_zone_end: Entry zone upper bound.
            stop_loss: Stop loss level.
            take_profits: List of take-profit levels.
            risk_reward_ratios: Matching R:R for each TP.
            confidence_score: Overall confidence (0-100).
            confidence_breakdown: Weighted contribution per component.
            human_explanation: Free-text explanation of the setup.
            alert_tier: Tier label (e.g. ``HIGH_PROBABILITY``).
            score_weights: *(V2)* Configured weights for each component.
            component_raw_scores: *(V2)* Raw scores (0.0-1.0) per component.
            reasons: *(V2)* Pre-classified ``✔`` reasons list.
            missing_conditions: *(V2)* Pre-classified ``✘`` missing list.
            risk_level: *(V2)* Human-readable risk level string.

        Returns:
            Fully formatted HTML string ready for Telegram.
        """
        emoji, label, color = TIER_MAP.get(
            alert_tier, ("📊", alert_tier, "#FFFFFF")
        )
        direction_color = BUY_COLOR if signal_direction == "BUY" else SELL_COLOR

        # --- Header ---
        parts: List[str] = [
            f"{emoji} <b>{label}</b> {emoji}",
            "",
            f"<b>Signal:</b> <b>{signal_direction}</b>",
            f"<b>Symbol:</b> {_escape_html(symbol)} | "
            f"<b>TF:</b> {_escape_html(timeframe)}",
        ]

        # --- Trade Setup ---
        parts.append("")
        parts.append("<b>───────── TRADE SETUP ─────────</b>")
        parts.append(
            f"<b>Entry Zone:</b> {entry_zone_start:.2f} – {entry_zone_end:.2f}"
        )
        parts.append(f"<b>Stop Loss:</b> {stop_loss:.2f}")

        if take_profits:
            parts.append("<b>Take Profits:</b>")
            for i, tp in enumerate(take_profits):
                rr = (
                    risk_reward_ratios[i]
                    if i < len(risk_reward_ratios)
                    else 0.0
                )
                parts.append(f"  TP{i+1}: {tp:.2f} (R:R {rr:.2f})")

        # --- Score & Breakdown ---
        parts.append("")
        parts.append("<b>───────── SCORE ─────────</b>")
        parts.append(
            f"<b>Confidence:</b> {confidence_score:.1f}% ({label})"
        )

        # Intelligent breakdown: show ✔ / ✘ when we have raw data
        if score_weights and component_raw_scores:
            found: List[str] = []
            missing: List[str] = []
            for comp, weight in score_weights.items():
                raw = component_raw_scores.get(comp, 0.0)
                raw_score, weight_val = _classify_condition(comp, raw, weight)
                contribution = confidence_breakdown.get(comp, 0.0)
                line = f"{weight_val} {raw_score}: {contribution:.1f}/{weight}"
                if weight_val == "✔":
                    found.append(line)
                else:
                    missing.append(line)
            if found:
                parts.append("<b>Conditions Found:</b>")
                parts.extend(found)
            if missing:
                parts.append("<b>Missing Conditions:</b>")
                parts.extend(missing)
        else:
            # Fallback to V1-style breakdown
            parts.append("<b>Breakdown:</b>")
            for component, score in sorted(
                confidence_breakdown.items(), key=lambda x: x[1], reverse=True
            ):
                parts.append(
                    f"  - {component.replace('_', ' ').title()}: {score:.2f}"
                )

        # --- Risk Level ---
        if risk_level:
            parts.append("")
            parts.append(f"<b>Risk Level:</b> {risk_level}")

        # --- Explanation / Reasons ---
        parts.append("")
        parts.append("<b>───────── ANALYSIS ─────────</b>")

        if reasons or missing_conditions:
            if reasons:
                parts.append("<b>Reasons Found:</b>")
                parts.extend(f"  {r}" for r in reasons)
            if missing_conditions:
                parts.append("<b>Missing Conditions:</b>")
                parts.extend(f"  {m}" for m in missing_conditions)
        elif human_explanation:
            parts.append(f"{_escape_html(human_explanation)}")

        # --- Footer ---
        parts.append("")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        parts.append(
            f"<i>SMC Analysis System V2.0 | {ts}</i>"
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Convenience: format a heartbeat message
    # ------------------------------------------------------------------
    def format_heartbeat(
        self,
        uptime_seconds: int,
        scan_count: int,
        alert_count: int,
        symbols: List[str],
    ) -> str:
        """Format a periodic heartbeat status message.

        Args:
            uptime_seconds: System uptime in seconds.
            scan_count: Total completed scans since startup.
            alert_count: Total alerts sent since startup.
            symbols: List of monitored symbols.

        Returns:
            HTML-formatted heartbeat message.
        """
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        parts = [
            "💓 <b>SYSTEM HEARTBEAT</b> 💓",
            "",
            f"<b>Status:</b> ONLINE",
            f"<b>Monitoring:</b> {_escape_html(', '.join(symbols))}",
            f"<b>Uptime:</b> {uptime_str}",
            f"<b>Scans Completed:</b> {scan_count}",
            f"<b>Alerts Sent:</b> {alert_count}",
            "",
            f"<i>System operating normally. {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Convenience: format a startup confirmation
    # ------------------------------------------------------------------
    def format_startup(
        self,
        symbols: List[str],
        timeframes: Dict[str, str],
    ) -> str:
        """Format a startup confirmation message.

        Args:
            symbols: Monitored trading symbols.
            timeframes: Mapping of TF role → timeframe string.

        Returns:
            HTML-formatted startup message.
        """
        tf_str = " | ".join(
            f"{k}: {v}" for k, v in sorted(timeframes.items())
        )
        parts = [
            "🟢 <b>SYSTEM STARTED SUCCESSFULLY</b> 🟢",
            "",
            f"<b>Monitoring:</b> {_escape_html(', '.join(symbols))}",
            f"<b>Timeframes:</b> {tf_str}",
            f"<b>Status:</b> ONLINE",
            "",
            "Scanner running normally.",
            f"<i>SMC Analysis System V2.0 | "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Convenience: format an error notification
    # ------------------------------------------------------------------
    def format_error(
        self,
        error_type: str,
        error_message: str,
        symbol: Optional[str] = None,
    ) -> str:
        """Format a critical error notification.

        Args:
            error_type: Short error category (e.g. ``Binance API Error``).
            error_message: Detailed error description.
            symbol: *(optional)* Symbol associated with the error.

        Returns:
            HTML-formatted error message.
        """
        symbol_line = (
            f"\n<b>Symbol:</b> {_escape_html(symbol)}" if symbol else ""
        )
        parts = [
            "🚨 <b>ERROR NOTIFICATION</b> 🚨",
            "",
            f"<b>Type:</b> {_escape_html(error_type)}",
            f"{symbol_line}",
            f"<b>Details:</b> {_escape_html(error_message)}",
            "",
            f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>",
        ]
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Module-level self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    formatter = AlertFormatter()

    # --- Example: Premium BUY setup ---
    msg = formatter.format_alert_message(
        signal_direction="BUY",
        symbol="BTCUSDT",
        timeframe="1H / 15M",
        entry_zone_start=64500.00,
        entry_zone_end=64600.00,
        stop_loss=64200.00,
        take_profits=[64900.00, 65200.00, 65600.00],
        risk_reward_ratios=[1.33, 2.33, 3.67],
        confidence_score=92.5,
        confidence_breakdown={
            "market_structure_alignment": 20.0,
            "liquidity_sweep": 18.0,
            "bos_choch_confirmation": 15.0,
            "fair_value_gap": 13.5,
            "fresh_order_block": 8.0,
            "premium_discount_zone": 5.0,
            "confirmation_candle": 4.0,
            "trading_session_quality": 5.0,
            "news_filter": 4.0,
        },
        human_explanation=(
            "Price swept liquidity below equal lows at $64,350, "
            "followed by a bullish BOS on the 15M timeframe. "
            "Price retraced into a fresh bullish FVG within the "
            "discount zone, forming a strong rejection candle."
        ),
        alert_tier="PREMIUM",
        score_weights={
            "market_structure_alignment": 20,
            "liquidity_sweep": 20,
            "bos_choch_confirmation": 15,
            "fair_value_gap": 15,
            "fresh_order_block": 10,
            "premium_discount_zone": 5,
            "confirmation_candle": 5,
            "trading_session_quality": 5,
            "news_filter": 5,
        },
        component_raw_scores={
            "market_structure_alignment": 1.0,
            "liquidity_sweep": 0.9,
            "bos_choch_confirmation": 1.0,
            "fair_value_gap": 0.9,
            "fresh_order_block": 0.8,
            "premium_discount_zone": 1.0,
            "confirmation_candle": 0.8,
            "trading_session_quality": 1.0,
            "news_filter": 0.8,
        },
        risk_level="Medium",
    )
    print("--- Premium BUY Alert ---")
    print(msg)

    # --- Example: Watchlist SELL setup ---
    msg2 = formatter.format_alert_message(
        signal_direction="SELL",
        symbol="ETHUSDT",
        timeframe="1H / 15M",
        entry_zone_start=3500.00,
        entry_zone_end=3490.00,
        stop_loss=3530.00,
        take_profits=[3460.00, 3430.00],
        risk_reward_ratios=[1.0, 2.0],
        confidence_score=63.0,
        confidence_breakdown={
            "market_structure_alignment": 10.0,
            "liquidity_sweep": 12.0,
            "bos_choch_confirmation": 10.0,
            "fair_value_gap": 5.0,
            "fresh_order_block": 0.0,
            "premium_discount_zone": 5.0,
            "confirmation_candle": 2.0,
            "trading_session_quality": 3.5,
            "news_filter": 5.0,
        },
        human_explanation=(
            "Bearish structure detected but confirmation candle "
            "and order block are missing. Watch for entry."
        ),
        alert_tier="WATCHLIST",
        risk_level="High",
    )
    print("\n--- Watchlist SELL Alert ---")
    print(msg2)

    # --- Heartbeat example ---
    hb = formatter.format_heartbeat(
        uptime_seconds=18000,
        scan_count=300,
        alert_count=5,
        symbols=["BTCUSDT", "ETHUSDT"],
    )
    print("\n--- Heartbeat ---")
    print(hb)