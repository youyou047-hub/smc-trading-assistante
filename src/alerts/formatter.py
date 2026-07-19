from typing import Dict, List, Optional
import pandas as pd

class AlertFormatter:
    """Formats professional alert messages in HTML for Telegram.
    """

    def __init__(self) -> None:
        pass

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
        alert_tier: str
    ) -> str:
        """Formats the alert message into an HTML string.

        Args:
            signal_direction (str): BUY or SELL.
            symbol (str): Trading pair symbol.
            timeframe (str): Timeframe analyzed.
            entry_zone_start (float): Start price of the entry zone.
            entry_zone_end (float): End price of the entry zone.
            stop_loss (float): Stop Loss level.
            take_profits (List[float]): List of Take Profit levels.
            risk_reward_ratios (List[float]): List of Risk:Reward ratios for each TP.
            confidence_score (float): Overall confidence score.
            confidence_breakdown (Dict[str, float]): Detailed breakdown of confidence scores.
            human_explanation (str): Human-readable explanation of the setup.
            alert_tier (str): The determined alert tier (e.g., PREMIUM, HIGH PROBABILITY).

        Returns:
            str: HTML formatted alert message.
        """
        # Alert Tier Header
        header = ""
        if alert_tier == "PREMIUM INSTITUTIONAL SETUP":
            header = "<b>🔥 PREMIUM INSTITUTIONAL SETUP 🔥</b>\n"
        elif alert_tier == "HIGH PROBABILITY ALERT":
            header = "<b>⚡ HIGH PROBABILITY ALERT ⚡</b>\n"
        else:
            header = f"<b>{alert_tier}</b>\n"

        message = f"{header}"
        message += f"\n<b>Signal:</b> <span style=\"color: {"#00ff00" if signal_direction == "BUY" else "#ff0000"}\">{signal_direction}</span>\n"
        message += f"<b>Symbol:</b> {symbol} | <b>Timeframe:</b> {timeframe}\n"
        message += f"<b>Entry Zone:</b> {entry_zone_start:.2f} - {entry_zone_end:.2f}\n"
        message += f"<b>Stop Loss:</b> {stop_loss:.2f}\n"

        # Take Profits
        message += "<b>Take Profits:</b>\n"
        for i, tp in enumerate(take_profits):
            rr = risk_reward_ratios[i] if i < len(risk_reward_ratios) else 0.0
            message += f"  TP{i+1}: {tp:.2f} (R:R {rr:.2f})\n"

        # Confidence Score
        message += f"\n<b>Confidence Score:</b> {confidence_score:.2f}%\n"
        message += "<b>Breakdown:</b>\n"
        for component, score in confidence_breakdown.items():
            message += f"  - {component.replace("_", " ").title()}: {score:.2f}%\n"

        # Human Explanation
        message += f"\n<b>Explanation:</b>\n{human_explanation}\n"

        # Timestamp
        message += f"\n<i>Timestamp: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</i>"

        return message

if __name__ == '__main__':
    formatter = AlertFormatter()

    # Example Analysis Result (dummy data)
    signal_direction = "BUY"
    symbol = "BTCUSDT"
    timeframe = "1h"
    entry_zone_start = 29500.00
    entry_zone_end = 29600.00
    stop_loss = 29300.00
    take_profits = [29800.00, 30000.00, 30200.00]
    risk_reward_ratios = [1.0, 2.0, 3.0]
    confidence_score = 88.5
    confidence_breakdown = {
        "market_structure": 25.0,
        "liquidity": 18.0,
        "fvg": 15.0,
        "order_block": 12.0,
        "displacement": 8.0,
        "session_confluence": 10.0
    }
    human_explanation = (
        "Price swept liquidity below equal lows at $29450, followed by a bullish BOS on the 15M timeframe. "
        "Price retraced into a fresh bullish FVG within the discount zone, forming a strong rejection candle. "
        "This indicates strong institutional buying pressure."
    )
    alert_tier = "PREMIUM INSTITUTIONAL SETUP"

    formatted_message = formatter.format_alert_message(
        signal_direction, symbol, timeframe, entry_zone_start, entry_zone_end,
        stop_loss, take_profits, risk_reward_ratios, confidence_score,
        confidence_breakdown, human_explanation, alert_tier
    )

    print("--- Example Formatted Alert Message (BUY) ---")
    print(formatted_message)

    # Example Bearish Signal
    signal_direction_sell = "SELL"
    entry_zone_start_sell = 30500.00
    entry_zone_end_sell = 30400.00
    stop_loss_sell = 30700.00
    take_profits_sell = [30200.00, 30000.00, 29800.00]
    risk_reward_ratios_sell = [1.0, 2.0, 3.0]
    confidence_score_sell = 78.2
    confidence_breakdown_sell = {
        "market_structure": 20.0,
        "liquidity": 15.0,
        "fvg": 12.0,
        "order_block": 10.0,
        "displacement": 7.0,
        "session_confluence": 8.0
    }
    human_explanation_sell = (
        "Price swept liquidity above equal highs at $30550, followed by a bearish BOS on the 15M timeframe. "
        "Price retraced into a fresh bearish FVG within the premium zone, forming a strong rejection candle. "
        "This indicates strong institutional selling pressure."
    )
    alert_tier_sell = "HIGH PROBABILITY ALERT"

    formatted_message_sell = formatter.format_alert_message(
        signal_direction_sell, symbol, timeframe, entry_zone_start_sell, entry_zone_end_sell,
        stop_loss_sell, take_profits_sell, risk_reward_ratios_sell, confidence_score_sell,
        confidence_breakdown_sell, human_explanation_sell, alert_tier_sell
    )

    print("\n--- Example Formatted Alert Message (SELL) ---")
    print(formatted_message_sell)

    # Example Sub-threshold Signal
    confidence_score_sub = 72.0
    alert_tier_sub = "SUB-THRESHOLD SIGNAL"
    formatted_message_sub = formatter.format_alert_message(
        signal_direction, symbol, timeframe, entry_zone_start, entry_zone_end,
        stop_loss, take_profits, risk_reward_ratios, confidence_score_sub,
        confidence_breakdown, human_explanation, alert_tier_sub
    )
    print("\n--- Example Formatted Alert Message (SUB-THRESHOLD) ---")
    print(formatted_message_sub)
