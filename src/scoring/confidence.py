"""Confidence Scoring Module.

Calculates a confidence score (0-100) for trading setups based on
configurable weights from settings.yaml. Each component of the
Smart Money Concepts analysis contributes to the final score.

Default weights (must sum to 100):
- Market Structure Alignment: 20
- Liquidity Sweep: 20
- BOS/CHoCH Confirmation: 15
- Fair Value Gap: 15
- Fresh Order Block: 10
- Premium/Discount Zone: 5
- Confirmation Candle: 5
- Trading Session Quality: 5
- News Filter: 5
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Calculates confidence scores and determines alert tiers.

    The scorer takes raw component scores (0.0 to 1.0) from the analysis
    engine and applies configurable weights to produce a final score (0-100).

    Alert Tiers:
        - Below 70: Ignore completely
        - 70-79: Record internally for statistics only
        - 80-89: Send High Probability Alert
        - 90-100: Send Premium Institutional Setup Alert

    Attributes:
        weights (Dict[str, float]): Weights for each scoring component.
        thresholds (Dict[str, float]): Score thresholds for alert tiers.
    """

    def __init__(self, weights: Dict[str, float], thresholds: Dict[str, float]) -> None:
        """Initialize the confidence scorer.

        Args:
            weights: Dictionary mapping component names to their weights.
                     Weights should sum to 100.
            thresholds: Dictionary with threshold values:
                       - ignore_below: Score below which signals are ignored
                       - record_only_min: Minimum score for recording stats
                       - high_probability_min: Minimum for high probability alert
                       - premium_min: Minimum for premium institutional alert
        """
        self.weights = weights
        self.thresholds = thresholds

        # Validate weights sum
        total_weight = sum(weights.values())
        if abs(total_weight - 100) > 0.01:
            logger.warning(
                f"Confidence weights sum to {total_weight}, expected 100. "
                f"Scores may not be properly normalized."
            )

    def calculate_score(
        self, component_scores: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculates the total confidence score with detailed breakdown.

        Each component score should be between 0.0 and 1.0, representing
        how strongly that condition is met (0 = not met, 1 = fully met).

        The final score is: sum(component_score * weight) for all components.

        Args:
            component_scores: Dictionary mapping component names to their
                            raw scores (0.0 to 1.0).

        Returns:
            Tuple of (total_score, weighted_breakdown):
                - total_score: Final confidence score (0-100)
                - weighted_breakdown: Dict showing each component's contribution
        """
        total_score = 0.0
        weighted_breakdown = {}

        for component_name, weight in self.weights.items():
            # Get the raw score for this component (0.0 to 1.0)
            raw_score = component_scores.get(component_name, 0.0)

            # Clamp to valid range
            raw_score = max(0.0, min(1.0, raw_score))

            # Calculate weighted contribution
            contribution = raw_score * weight
            total_score += contribution
            weighted_breakdown[component_name] = round(contribution, 2)

        # Cap at 100
        total_score = min(100.0, total_score)

        logger.debug(
            f"Confidence score: {total_score:.1f} | "
            f"Breakdown: {weighted_breakdown}"
        )

        return round(total_score, 2), weighted_breakdown

    def get_alert_tier(self, confidence_score: float) -> str:
        """Determines the alert tier based on the confidence score.

        Args:
            confidence_score: The calculated confidence score (0-100).

        Returns:
            String representing the alert tier.
        """
        premium_min = self.thresholds.get("premium_min", 90)
        high_prob_min = self.thresholds.get("high_probability_min", 80)
        record_min = self.thresholds.get("record_only_min", 70)

        if confidence_score >= premium_min:
            return "PREMIUM INSTITUTIONAL SETUP"
        elif confidence_score >= high_prob_min:
            return "HIGH PROBABILITY ALERT"
        elif confidence_score >= record_min:
            return "SUB-THRESHOLD (RECORD ONLY)"
        else:
            return "IGNORED"

    def get_score_summary(
        self, confidence_score: float, breakdown: Dict[str, float]
    ) -> str:
        """Generates a human-readable score summary.

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
            lines.append(
                f"  {component.replace('_', ' ').title()}: "
                f"{contribution:.1f}/{weight} ({raw*100:.0f}%)"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    # Example with spec-compliant weights
    weights = {
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

    thresholds = {
        "ignore_below": 70,
        "record_only_min": 70,
        "high_probability_min": 80,
        "premium_min": 90,
    }

    scorer = ConfidenceScorer(weights, thresholds)

    # Simulate a strong setup (raw scores 0.0 to 1.0)
    strong_setup = {
        "market_structure_alignment": 1.0,  # Fully aligned
        "liquidity_sweep": 0.9,             # Strong sweep detected
        "bos_choch_confirmation": 1.0,      # BOS confirmed
        "fair_value_gap": 0.8,              # FVG present
        "fresh_order_block": 0.7,           # Fresh OB nearby
        "premium_discount_zone": 1.0,       # In discount zone
        "confirmation_candle": 0.8,         # Strong rejection
        "trading_session_quality": 1.0,     # London session
        "news_filter": 1.0,                 # No news approaching
    }

    score, breakdown = scorer.calculate_score(strong_setup)
    tier = scorer.get_alert_tier(score)
    summary = scorer.get_score_summary(score, breakdown)

    print(f"\n=== Strong Setup ===")
    print(summary)
    print(f"\nAlert Tier: {tier}")

    # Simulate a weak setup
    weak_setup = {
        "market_structure_alignment": 0.5,
        "liquidity_sweep": 0.3,
        "bos_choch_confirmation": 0.4,
        "fair_value_gap": 0.2,
        "fresh_order_block": 0.0,
        "premium_discount_zone": 0.5,
        "confirmation_candle": 0.3,
        "trading_session_quality": 0.5,
        "news_filter": 1.0,
    }

    score2, breakdown2 = scorer.calculate_score(weak_setup)
    tier2 = scorer.get_alert_tier(score2)
    print(f"\n=== Weak Setup ===")
    print(scorer.get_score_summary(score2, breakdown2))
    print(f"\nAlert Tier: {tier2}")
