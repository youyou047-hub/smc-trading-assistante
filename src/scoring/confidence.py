"""Confidence Scoring System — V2.0 (Upgraded).

Calculates a confidence score (0–100) for Smart Money Concepts trading setups
based on configurable weights for each of the 9 analysis components:

  1. Market Structure Alignment
  2. Liquidity Sweep
  3. BOS / CHoCH Confirmation
  4. Fair Value Gap (FVG)
  5. Fresh Order Block
  6. Premium / Discount Zone
  7. Confirmation Candle
  8. Trading Session Quality
  9. News Filter

Key improvements over V1:
  • **Quality-based scoring**: each component can contribute a fractional
    score (not just binary yes/no).  The scorer accepts raw scores (0.0–1.0)
    and applies weights.
  • **Alert tiers**: five-tier system (ignore / watchlist / potential /
    high-probability / premium).
  • **Debug mode**: optional verbose logging of per-component breakdown
    (e.g. ``"Market Structure: +20"``).
  • **Structured result**: ``ScoreResult`` dataclass carrying the final score,
    breakdown dict, reasons found, missing conditions, risk level, and tier.
  • **Multi-timeframe alignment bonus / penalty** applied on top of the
    base score.
  • **Backward-compatible** constructor and method signatures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Component names (must match keys in ``component_scores`` and ``weights``)
COMPONENT_NAMES: List[str] = [
    "market_structure_alignment",
    "liquidity_sweep",
    "bos_choch_confirmation",
    "fair_value_gap",
    "fresh_order_block",
    "premium_discount_zone",
    "confirmation_candle",
    "trading_session_quality",
    "news_filter",
]

# Default weights (must sum to 100)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "market_structure_alignment": 20,
    "liquidity_sweep": 20,
    "bos_choch_confirmation": 15,
    "fair_value_gap": 15,
    "fresh_order_block": 10,
    "premium_discount_zone": 5,
    "confirmation_candle": 5,
    "trading_session_quality": 5,
    "news_filter": 5,
}

# Default thresholds
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "ignore_below": 60,
    "watchlist_min": 60,
    "potential_setup_min": 70,
    "high_probability_min": 80,
    "premium_min": 90,
}

# Human-readable component labels for debug output
COMPONENT_LABELS: Dict[str, str] = {
    "market_structure_alignment": "Market Structure",
    "liquidity_sweep": "Liquidity Sweep",
    "bos_choch_confirmation": "BOS / CHoCH",
    "fair_value_gap": "FVG",
    "fresh_order_block": "Order Block",
    "premium_discount_zone": "Premium / Discount",
    "confirmation_candle": "Confirmation Candle",
    "trading_session_quality": "Session",
    "news_filter": "News",
}


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """Structured result from the confidence scoring system.

    Attributes:
        final_score: The total confidence score (0–100).
        breakdown: Dict mapping component names to their weighted contributions.
        reasons_found: List of component names that scored above a threshold.
        missing_conditions: List of component names that scored below threshold.
        risk_level: ``'Low'``, ``'Medium'``, ``'High'`` based on score tier.
        alert_tier: Human-readable tier string.
        tier_level: Numeric tier (1–5) for programmatic use.
        debug_lines: List of formatted debug strings (populated in debug mode).
    """
    final_score: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    reasons_found: List[str] = field(default_factory=list)
    missing_conditions: List[str] = field(default_factory=list)
    risk_level: str = "Medium"
    alert_tier: str = ""
    tier_level: int = 0
    debug_lines: List[str] = field(default_factory=list)


# ── Scorer ────────────────────────────────────────────────────────────────────

class ConfidenceScorer:
    """Calculates confidence scores and determines alert tiers.

    The scorer takes raw component scores (0.0–1.0) from the analysis engine
    and applies configurable weights to produce a final score (0–100).

    Backward compatibility:
        * Constructor accepts the same ``(weights, thresholds)`` arguments as V1.
        * ``calculate_score(component_scores)`` returns the same tuple shape
          ``(total_score, weighted_breakdown)``.
        * ``get_alert_tier(confidence_score)`` returns the same string format.
        * ``get_score_summary(confidence_score, breakdown)`` works as before.

    Args:
        weights: Dict mapping component names to their weights (must sum to 100).
        thresholds: Dict with threshold values for alert tiers.
        debug_mode: If ``True``, debug lines are generated for each score calculation.
        mtf_alignment_bonus: Bonus added when lower TF aligns with higher TF.
        mtf_contradiction_penalty: Penalty applied when lower TF contradicts higher TF.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        thresholds: Dict[str, float],
        # V2 additions (keyword-only)
        debug_mode: bool = False,
        mtf_alignment_bonus: float = 5.0,
        mtf_contradiction_penalty: float = -10.0,
    ) -> None:
        self.weights = dict(weights)
        self.thresholds = dict(thresholds)
        self.debug_mode = debug_mode
        self._mtf_bonus = mtf_alignment_bonus
        self._mtf_penalty = mtf_contradiction_penalty

        # Validate weights sum
        total_weight = sum(self.weights.values())
        if abs(total_weight - 100.0) > 0.01:
            logger.warning(
                f"ConfidenceScorer: weights sum to {total_weight}, expected 100. "
                f"Scores may not be properly normalized."
            )

    # ── Core scoring ──────────────────────────────────────────────────────

    def calculate_score(
        self,
        component_scores: Dict[str, float],
        # V2 optional keyword arguments
        mtf_aligned: Optional[bool] = None,
        mtf_contradicted: Optional[bool] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculates the total confidence score with detailed breakdown.

        Each component score should be between 0.0 and 1.0, representing
        how strongly that condition is met.

        The final score is:
            ``sum(component_score * weight) + mtf_bonus - mtf_penalty``

        Args:
            component_scores: Dict mapping component names to raw scores (0.0–1.0).
            mtf_aligned: ``True`` if lower TF aligns with higher TF.
            mtf_contradicted: ``True`` if lower TF contradicts higher TF.

        Returns:
            Tuple of ``(total_score, weighted_breakdown)`` — identical shape to V1.
        """
        total_score = 0.0
        weighted_breakdown: Dict[str, float] = {}

        for component_name, weight in self.weights.items():
            raw_score = component_scores.get(component_name, 0.0)
            raw_score = max(0.0, min(1.0, raw_score))
            contribution = raw_score * weight
            total_score += contribution
            weighted_breakdown[component_name] = round(contribution, 2)

        # MTF alignment adjustments
        if mtf_aligned and not mtf_contradicted:
            total_score += self._mtf_bonus
            weighted_breakdown["mtf_alignment_bonus"] = round(self._mtf_bonus, 2)
        if mtf_contradicted:
            total_score += self._mtf_penalty
            weighted_breakdown["mtf_contradiction_penalty"] = round(self._mtf_penalty, 2)

        # Clamp to [0, 100]
        total_score = max(0.0, min(100.0, total_score))

        if self.debug_mode:
            logger.debug(
                f"ConfidenceScorer: score={total_score:.1f} | "
                f"breakdown={weighted_breakdown}"
            )

        return round(total_score, 2), weighted_breakdown

    def calculate_score_v2(
        self,
        component_scores: Dict[str, float],
        mtf_aligned: Optional[bool] = None,
        mtf_contradicted: Optional[bool] = None,
        direction: str = "BUY",
    ) -> ScoreResult:
        """V2 structured scoring method.

        Returns a full ``ScoreResult`` with breakdown, reasons, missing
        conditions, risk level, and tier information.

        Args:
            component_scores: Dict mapping component names to raw scores (0.0–1.0).
            mtf_aligned: ``True`` if lower TF aligns with higher TF.
            mtf_contradicted: ``True`` if lower TF contradicts higher TF.
            direction: Trade direction (``'BUY'`` or ``'SELL'``).

        Returns:
            ``ScoreResult`` dataclass.
        """
        total_score, weighted_breakdown = self.calculate_score(
            component_scores,
            mtf_aligned=mtf_aligned,
            mtf_contradicted=mtf_contradicted,
        )

        # ── Reasons found / missing conditions ──
        reasons_found: List[str] = []
        missing_conditions: List[str] = []
        debug_lines: List[str] = []

        # Threshold for considering a component "found" (raw score >= 0.5)
        found_threshold = 0.5

        for component_name, weight in self.weights.items():
            raw = component_scores.get(component_name, 0.0)
            contribution = weighted_breakdown.get(component_name, 0.0)

            label = COMPONENT_LABELS.get(component_name, component_name)

            if raw >= found_threshold:
                reasons_found.append(component_name)
            else:
                missing_conditions.append(component_name)

            if self.debug_mode:
                debug_lines.append(f"  {label}: +{contribution:.1f} (raw={raw:.2f} × {weight})")

        # MTF info in debug
        if mtf_aligned:
            if self.debug_mode:
                debug_lines.append(f"  MTF Alignment: +{self._mtf_bonus}")
            reasons_found.append("mtf_alignment")
        if mtf_contradicted:
            if self.debug_mode:
                debug_lines.append(f"  MTF Contradiction: {self._mtf_penalty}")
            missing_conditions.append("mtf_contradiction")

        # ── Alert tier ──
        tier_level, alert_tier = self._get_tier(total_score)

        # ── Risk level ──
        risk_level = self._determine_risk_level(tier_level)

        if self.debug_mode:
            debug_lines.append(f"  Final Score: {total_score:.1f}")
            debug_lines.append(f"  Tier: {alert_tier}")
            debug_lines.append(f"  Risk: {risk_level}")

        return ScoreResult(
            final_score=total_score,
            breakdown=weighted_breakdown,
            reasons_found=reasons_found,
            missing_conditions=missing_conditions,
            risk_level=risk_level,
            alert_tier=alert_tier,
            tier_level=tier_level,
            debug_lines=debug_lines,
        )

    # ── Alert tiers ───────────────────────────────────────────────────────

    def _get_tier(self, confidence_score: float) -> Tuple[int, str]:
        """Returns ``(tier_level, tier_string)``.

        Tier levels:
            1 = Ignore (below 60)
            2 = Watchlist (60–69)
            3 = Potential (70–79)
            4 = High Probability (80–89)
            5 = Premium (90–100)
        """
        thresholds = self.thresholds
        ignore_below = thresholds.get("ignore_below", 60)
        watchlist_min = thresholds.get("watchlist_min", 60)
        potential_min = thresholds.get("potential_setup_min", 70)
        high_prob_min = thresholds.get("high_probability_min", 80)
        premium_min = thresholds.get("premium_min", 90)

        if confidence_score >= premium_min:
            return 5, "PREMIUM INSTITUTIONAL SETUP"
        elif confidence_score >= high_prob_min:
            return 4, "HIGH PROBABILITY SETUP"
        elif confidence_score >= potential_min:
            return 3, "POTENTIAL SETUP"
        elif confidence_score >= watchlist_min:
            return 2, "WATCHLIST ALERT"
        else:
            return 1, "IGNORED"

    def get_alert_tier(self, confidence_score: float) -> str:
        """Returns the alert tier string for a given confidence score.

        Backward-compatible with V1.

        Args:
            confidence_score: The calculated confidence score (0–100).

        Returns:
            String representing the alert tier.
        """
        _, tier_string = self._get_tier(confidence_score)
        return tier_string

    def _determine_risk_level(self, tier_level: int) -> str:
        """Maps a tier level to a risk level string.

        Args:
            tier_level: Numeric tier (1–5).

        Returns:
            ``'Low'``, ``'Medium'``, or ``'High'``.
        """
        if tier_level >= 4:
            return "Low"
        elif tier_level >= 3:
            return "Medium"
        else:
            return "High"

    # ── Score summary ─────────────────────────────────────────────────────

    def get_score_summary(
        self, confidence_score: float, breakdown: Dict[str, float]
    ) -> str:
        """Generates a human-readable score summary.

        Backward-compatible with V1.

        Args:
            confidence_score: The total confidence score.
            breakdown: The weighted breakdown dictionary.

        Returns:
            Formatted string showing the score breakdown.
        """
        tier = self.get_alert_tier(confidence_score)
        lines = [
            f"Confidence Score: {confidence_score:.1f}/100 ({tier})",
            "Score Breakdown:",
        ]

        for component, contribution in sorted(
            breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            weight = self.weights.get(component, 0)
            raw = contribution / weight if weight > 0 else 0
            label = COMPONENT_LABELS.get(component, component.replace("_", " ").title())
            lines.append(
                f"  {label}: {contribution:.1f}/{weight} ({raw*100:.0f}%)"
            )

        return "\n".join(lines)

    def format_alert_explanation(
        self,
        score_result: ScoreResult,
        direction: str = "BUY",
    ) -> str:
        """Formats a structured alert explanation for Telegram / logging.

        Includes reasons found, missing conditions, and risk level — exactly
        as specified in the V2.0 requirements.

        Args:
            score_result: The ``ScoreResult`` from ``calculate_score_v2()``.
            direction: Trade direction string (``'BUY'`` or ``'SELL'``).

        Returns:
            Formatted explanation string.
        """
        lines = [
            f"Direction: {direction}",
            f"Confidence Score: {score_result.final_score:.1f}/100",
            f"Alert Tier: {score_result.alert_tier}",
            "",
            "Reasons Found:",
        ]

        for reason in score_result.reasons_found:
            label = COMPONENT_LABELS.get(reason, reason.replace("_", " ").title())
            lines.append(f"  ✔ {label}")

        lines.append("")
        lines.append("Missing Conditions:")

        for missing in score_result.missing_conditions:
            label = COMPONENT_LABELS.get(missing, missing.replace("_", " ").title())
            lines.append(f"  ✘ {label}")

        lines.append("")
        lines.append(f"Risk Level: {score_result.risk_level}")

        if score_result.debug_lines:
            lines.append("")
            lines.append("Debug Breakdown:")
            lines.extend(score_result.debug_lines)

        return "\n".join(lines)

    def log_debug_breakdown(self, score_result: ScoreResult) -> None:
        """Logs the detailed score breakdown in debug mode.

        Args:
            score_result: The ``ScoreResult`` to log.
        """
        if not self.debug_mode:
            return

        logger.info("=" * 50)
        logger.info("Confidence Score Debug Breakdown")
        logger.info("=" * 50)
        for line in score_result.debug_lines:
            logger.info(line)
        logger.info("=" * 50)


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scorer = ConfidenceScorer(
        weights=DEFAULT_WEIGHTS,
        thresholds=DEFAULT_THRESHOLDS,
        debug_mode=True,
    )

    # ── Simulate a strong premium setup ──
    print("\n=== Strong Premium Setup ===\n")
    strong = {
        "market_structure_alignment": 1.0,
        "liquidity_sweep": 0.95,
        "bos_choch_confirmation": 1.0,
        "fair_value_gap": 0.85,
        "fresh_order_block": 0.9,
        "premium_discount_zone": 1.0,
        "confirmation_candle": 0.8,
        "trading_session_quality": 1.0,
        "news_filter": 1.0,
    }

    result = scorer.calculate_score_v2(strong, mtf_aligned=True, direction="BUY")
    print(scorer.format_alert_explanation(result, direction="BUY"))
    scorer.log_debug_breakdown(result)

    # ── Simulate a watchlist setup ──
    print("\n\n=== Watchlist Setup ===\n")
    weak = {
        "market_structure_alignment": 0.7,
        "liquidity_sweep": 0.4,
        "bos_choch_confirmation": 0.5,
        "fair_value_gap": 0.3,
        "fresh_order_block": 0.0,
        "premium_discount_zone": 0.5,
        "confirmation_candle": 0.2,
        "trading_session_quality": 0.7,
        "news_filter": 1.0,
    }

    result2 = scorer.calculate_score_v2(weak, mtf_aligned=False, direction="SELL")
    print(scorer.format_alert_explanation(result2, direction="SELL"))
    scorer.log_debug_breakdown(result2)

    # ── Backward-compat test ──
    print("\n\n=== Backward Compatibility Test ===\n")
    score, breakdown = scorer.calculate_score(strong)
    tier = scorer.get_alert_tier(score)
    print(f"Legacy score: {score} | Tier: {tier}")
    print(scorer.get_score_summary(score, breakdown))
