
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import os

# Assuming AnalysisResult is defined in src/analysis/engine.py
# We will import it directly or define a placeholder if circular dependency issues arise.
# For now, let's assume it's available or we'll pass the necessary parts.

class ChartGenerator:
    """Generates professional annotated candlestick charts using mplfinance.
    """

    def __init__(self, output_dir: str = "charts") -> None:
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_chart(
        self,
        ohlcv_df: pd.DataFrame,
        analysis_data: Dict[str, Any], # Using Dict for flexibility, can be AnalysisResult object
        symbol: str,
        timeframe: str,
        signal_direction: Optional[str],
        confidence_score: float,
        file_name: Optional[str] = None
    ) -> str:
        """Generates an annotated candlestick chart and saves it as a PNG.

        Args:
            ohlcv_df (pd.DataFrame): OHLCV DataFrame with DatetimeIndex.
            analysis_data (Dict[str, Any]): Dictionary containing analysis results for annotations.
                                           Expected keys: 'liquidity_sweeps', 'bos_lines', 'choch_lines',
                                           'fvgs', 'order_blocks', 'entry_zone', 'stop_loss', 'take_profits'.
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            timeframe (str): Timeframe of the chart (e.g., '1h').
            signal_direction (Optional[str]): 'BUY' or 'SELL' or None.
            confidence_score (float): The confidence score of the signal.
            file_name (Optional[str]): Name of the output PNG file. If None, a default name is generated.

        Returns:
            str: The file path to the generated chart image.
        """
        if ohlcv_df.empty:
            raise ValueError("OHLCV DataFrame cannot be empty.")

        # --- Dark professional theme ---
        mc = mpf.make_marketcolors(up="#00b060", down="#ff3333", edge="inherit", wick="inherit", volume="#2a2a2a")
        s = mpf.make_mpf_style(base_mpf_style="yahoo", marketcolors=mc, gridcolor="#303030", facecolor="#1a1a1a",
                                figcolor="#1a1a1a", y_on_right=False,
                                rc={
                                    "axes.labelcolor": "white",
                                    "axes.edgecolor": "white",
                                    "ytick.color": "white",
                                    "xtick.color": "white",
                                    "text.color": "white",
                                    "axes.linewidth": 0.8
                                })

        apds = [] # Additional plots
        hlines = [] # Horizontal lines for mplfinance
        hlines_colors = []
        hlines_labels = []
        hlines_styles = []

        # --- Annotate Liquidity Sweeps (dashed horizontal lines) ---
        # Assuming analysis_data["liquidity_sweeps"] is a list of tuples (price, index)
        for sweep in analysis_data.get("liquidity_sweeps", []):
            price, idx = sweep
            hlines.append(price)
            hlines_colors.append("purple")
            hlines_labels.append("Liq Sweep")
            hlines_styles.append("--")

        # --- Annotate BOS/CHoCH lines (solid lines with labels and arrows) ---
        # For simplicity, we'll add horizontal lines. Arrows will be handled as text annotations post-plot.
        for bos in analysis_data.get("bos_lines", []):
            price, idx, direction = bos
            hlines.append(price)
            hlines_colors.append("cyan" if direction == "bullish" else "magenta")
            hlines_labels.append("BOS")
            hlines_styles.append("-")

        for choch in analysis_data.get("choch_lines", []):
            price, idx, direction = choch
            hlines.append(price)
            hlines_colors.append("lime" if direction == "bullish" else "orange")
            hlines_labels.append("CHoCH")
            hlines_styles.append("-")

        # --- Annotate Fair Value Gaps (semi-transparent colored rectangles) ---
        # For FVGs and OBs, we'll use `fill_between` on a separate axis or create custom `addplot`.
        # A simpler approach for `mplfinance` is to use `fill_between` on the main axis after it's created.
        # Or, we can create `addplot` dataframes for fills.
        # Let's prepare data for `fill_between` to be applied after `mpf.plot` returns the figure.
        fvg_fills = [] # List of (index, fvg_start, fvg_end, color)
        for fvg in analysis_data.get("fvgs", []):
            fvg_type = fvg["type"]
            fvg_start_price = fvg["fvg_start"]
            fvg_end_price = fvg["fvg_end"]
            fvg_index = fvg["index"]
            color = "#00ff0050" if fvg_type == "bullish" else "#ff000050"
            fvg_fills.append((fvg_index, fvg_start_price, fvg_end_price, color))

        # --- Annotate Order Blocks (semi-transparent colored rectangles) ---
        ob_fills = [] # List of (index, ob_start, ob_end, color)
        for ob in analysis_data.get("order_blocks", []):
            ob_type = ob["type"]
            ob_start_price = ob["ob_start"]
            ob_end_price = ob["ob_end"]
            ob_index = ob["index"]
            color = "#0000ff50" if ob_type == "bullish" else "#ffa50050"
            ob_fills.append((ob_index, ob_start_price, ob_end_price, color))

        # --- Annotate Entry Zone (highlighted horizontal band in green) ---
        entry_zone_start = analysis_data.get("entry_zone_start")
        entry_zone_end = analysis_data.get("entry_zone_end")
        if entry_zone_start is not None and entry_zone_end is not None:
            hlines.extend([entry_zone_start, entry_zone_end])
            hlines_colors.extend(["#00ff00", "#00ff00"])
            hlines_labels.extend(["Entry Zone", "Entry Zone"])
            hlines_styles.extend(["-", "-"])

        # --- Annotate Stop Loss (red dashed horizontal line) ---
        stop_loss = analysis_data.get("stop_loss")
        if stop_loss is not None:
            hlines.append(stop_loss)
            hlines_colors.append("red")
            hlines_labels.append("SL")
            hlines_styles.append("--")

        # --- Annotate Take Profit targets (green dashed horizontal lines) ---
        take_profits = analysis_data.get("take_profits", [])
        for i, tp in enumerate(take_profits):
            hlines.append(tp)
            hlines_colors.append("green")
            hlines_labels.append(f"TP{i+1}")
            hlines_styles.append("--")

        # --- Plotting ----
        fig, axes = mpf.plot(ohlcv_df, type="candle", style=s, volume=True, returnfig=True,
                             title=f"{symbol} {timeframe} - Signal: {signal_direction or 'NONE'} (Conf: {confidence_score:.2f}%)",
                             hlines=dict(hlines=hlines, colors=hlines_colors, linestyle=hlines_styles, alpha=0.7, linewidths=0.8),
                             addplot=apds,
                             figscale=1.5)

        ax = axes[0] # Candlestick plot axis

        # Add labels to hlines (mplfinance doesn't directly support labels for hlines in the legend)
        # We can add text annotations for labels near the lines.
        for i, price in enumerate(hlines):
            # Avoid duplicate labels for start/end of zones
            if hlines_labels[i] not in [hlines_labels[j] for j in range(i)]:
                ax.annotate(hlines_labels[i], xy=(ohlcv_df.index[-1], price), xytext=(5, 0), textcoords="offset points",
                            color=hlines_colors[i], fontsize=8, verticalalignment="center", horizontalalignment="left")

        # Add FVG and OB rectangles using fill_between on the main axis
        for fvg_idx, fvg_start, fvg_end, color in fvg_fills:
            # Find the x-coordinates for the FVG candle
            x_start = ohlcv_df.index.get_loc(fvg_idx)
            # Assuming the FVG lasts for one candle for simplicity in plotting
            # For a persistent FVG, you'd need to define its duration.
            x_end = x_start + 1 # Represents the width of one candle

            # Convert datetime index to numerical for fill_between
            x_coords = ax.get_xticks()
            # This is a simplification. A more accurate way would be to map datetime to plot coordinates.
            # For now, let's just draw a horizontal span for the FVG.
            ax.axhspan(fvg_start, fvg_end, facecolor=color, alpha=0.3, linewidth=0)

        for ob_idx, ob_start, ob_end, color in ob_fills:
            x_start = ohlcv_df.index.get_loc(ob_idx)
            x_end = x_start + 1
            ax.axhspan(ob_start, ob_end, facecolor=color, alpha=0.3, linewidth=0)

        # --- Arrows pointing to key levels ---
        # Example: Arrow for BOS/CHoCH
        for bos in analysis_data.get("bos_lines", []):
            price, idx, direction = bos
            # Find the x-coordinate for the index
            x_pos = ohlcv_df.index.get_loc(idx)
            arrow_color = "cyan" if direction == "bullish" else "magenta"
            if direction == "bullish":
                ax.annotate("", xy=(x_pos, price * 1.005), xytext=(x_pos, price * 0.995),
                            arrowprops=dict(facecolor=arrow_color, shrink=0.05, width=1, headwidth=5),)
            else:
                ax.annotate("", xy=(x_pos, price * 0.995), xytext=(x_pos, price * 1.005),
                            arrowprops=dict(facecolor=arrow_color, shrink=0.05, width=1, headwidth=5),)

        for choch in analysis_data.get("choch_lines", []):
            price, idx, direction = choch
            x_pos = ohlcv_df.index.get_loc(idx)
            arrow_color = "lime" if direction == "bullish" else "orange"
            if direction == "bullish":
                ax.annotate("", xy=(x_pos, price * 1.005), xytext=(x_pos, price * 0.995),
                            arrowprops=dict(facecolor=arrow_color, shrink=0.05, width=1, headwidth=5),)
            else:
                ax.annotate("", xy=(x_pos, price * 0.995), xytext=(x_pos, price * 1.005),
                            arrowprops=dict(facecolor=arrow_color, shrink=0.05, width=1, headwidth=5),)

        if file_name is None:
            file_name = f"{symbol}_{timeframe}_{signal_direction or 'NONE'}_{confidence_score:.0f}.png"
        chart_path = os.path.join(self.output_dir, file_name)

        fig.savefig(chart_path)
        plt.close(fig) # Close the figure to free memory
        return chart_path

if __name__ == '__main__':
    # Example Usage with dummy data
    data = {
        'open': [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109, 108, 110, 109, 111],
        'high': [103, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109, 108, 110, 109, 111, 110, 112, 111, 113],
        'low': [99, 100, 99, 101, 100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109],
        'close': [102, 101, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107, 109, 108, 110, 109, 111, 110, 112],
        'volume': [100, 120, 110, 130, 140, 120, 150, 130, 160, 140, 170, 150, 180, 160, 190, 170, 200, 180, 210, 190]
    }
    index = pd.to_datetime(pd.Series(range(len(data['open']))), unit='m', origin='2023-01-01')
    dummy_df = pd.DataFrame(data, index=index)

    # Dummy analysis data
    dummy_analysis_data = {
        "liquidity_sweeps": [(100.5, dummy_df.index[5]), (108.5, dummy_df.index[15])],
        "bos_lines": [(104.0, dummy_df.index[7], "bullish"), (109.0, dummy_df.index[17], "bullish")],
        "choch_lines": [(101.0, dummy_df.index[3], "bearish")],
        "fvgs": [
            {"type": "bullish", "fvg_start": 101.5, "fvg_end": 102.5, "index": dummy_df.index[4]},
            {"type": "bearish", "fvg_start": 106.5, "fvg_end": 107.5, "index": dummy_df.index[14]}
        ],
        "order_blocks": [
            {"type": "bullish", "ob_start": 99.5, "ob_end": 100.0, "index": dummy_df.index[2]},
            {"type": "bearish", "ob_start": 105.0, "ob_end": 105.5, "index": dummy_df.index[12]}
        ],
        "entry_zone_start": 106.0,
        "entry_zone_end": 106.5,
        "stop_loss": 105.0,
        "take_profits": [107.5, 108.5, 109.5]
    }

    generator = ChartGenerator()
    try:
        chart_file = generator.generate_chart(
            ohlcv_df=dummy_df,
            analysis_data=dummy_analysis_data,
            symbol="DUMMYUSDT",
            timeframe="15m",
            signal_direction="BUY",
            confidence_score=88.5,
            file_name="dummy_chart.png"
        )
        print(f"Chart generated: {chart_file}")
    except ValueError as e:
        print(f"Error generating chart: {e}")


    # Test with no signal
    dummy_analysis_data_no_signal = {
        "liquidity_sweeps": [], "bos_lines": [], "choch_lines": [],
        "fvgs": [], "order_blocks": [], "take_profits": []
    }
    try:
        chart_file_no_signal = generator.generate_chart(
            ohlcv_df=dummy_df,
            analysis_data=dummy_analysis_data_no_signal,
            symbol="DUMMYUSDT",
            timeframe="15m",
            signal_direction=None,
            confidence_score=50.0,
            file_name="dummy_chart_no_signal.png"
        )
        print(f"Chart generated (no signal): {chart_file_no_signal}")
    except ValueError as e:
        print(f"Error generating chart (no signal): {e}")
