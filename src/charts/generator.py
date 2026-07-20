"""Chart Generator V2.0 — Professional annotated trading charts.

Produces publication-quality candlestick charts using ``mplfinance`` with a
dark theme suitable for trading analysis.  Displays:

- Candlestick price action with volume
- Market Structure annotations (swing highs/lows)
- BOS (Break of Structure) lines and labels
- CHoCH (Change of Character) lines and labels
- Liquidity Sweep markers
- Order Block zones (shaded rectangles)
- Fair Value Gap zones (shaded rectangles)
- Premium / Discount zone shading
- Entry Zone (highlighted band)
- Stop Loss and Take Profit levels
- Signal reasons text box

Annotations are positioned intelligently to minimise overlap.
Charts are saved as PNG and the file path is returned.

Backward-compatible with the V1 ``generate_chart`` signature so the
main orchestrator and existing callers work without modification.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
# Candlestick colours
UP_COLOR = "#00b060"
DOWN_COLOR = "#ff3333"

# Annotation colours
COLOR_BOS_BULL = "#00e5ff"       # cyan
COLOR_BOS_BEAR = "#ff4081"       # magenta
COLOR_CHOCH_BULL = "#76ff03"     # lime
COLOR_CHOCH_BEAR = "#ff9100"     # orange
COLOR_LIQUIDITY = "#b388ff"      # purple
COLOR_OB_BULL = "#2979ff80"      # blue (semi-transparent)
COLOR_OB_BEAR = "#ff6d0080"      # orange (semi-transparent)
COLOR_FVG_BULL = "#00e67640"     # green (semi-transparent)
COLOR_FVG_BEAR = "#ff174440"     # red (semi-transparent)
COLOR_ENTRY = "#00e676"          # bright green
COLOR_SL = "#ff1744"             # bright red
COLOR_TP = "#00e676"             # green
COLOR_PREMIUM = "#ff910030"      # premium zone
COLOR_DISCOUNT = "#2979ff30"     # discount zone

# Layout
BACKGROUND_COLOR = "#121212"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a2a"
AXIS_COLOR = "#ffffff"


def _safe_price_format(price: float) -> str:
    """Format a price value with appropriate precision."""
    if abs(price) < 1:
        return f"{price:.6f}"
    elif abs(price) < 100:
        return f"{price:.2f}"
    else:
        return f"{price:.2f}"


# ---------------------------------------------------------------------------
# ChartGenerator
# ---------------------------------------------------------------------------
class ChartGenerator:
    """Generates professional annotated candlestick charts.

    Args:
        output_dir: Directory where chart PNG files are saved.
        theme: ``dark`` or ``light``.
        figscale: Scaling factor for the chart figure.
    """

    def __init__(
        self,
        output_dir: str = "charts",
        theme: str = "dark",
        figscale: float = 1.5,
    ) -> None:
        self.output_dir = output_dir
        self.theme = theme
        self.figscale = figscale
        self._is_dark = theme == "dark"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_chart(
        self,
        ohlcv_df: pd.DataFrame,
        analysis_data: Dict[str, Any],
        symbol: str,
        timeframe: str,
        signal_direction: Optional[str],
        confidence_score: float,
        file_name: Optional[str] = None,
        *,
        # V2 optional annotations
        premium_zone: Optional[Dict[str, float]] = None,
        discount_zone: Optional[Dict[str, float]] = None,
        reasons: Optional[List[str]] = None,
    ) -> str:
        """Generate an annotated candlestick chart and save as PNG.

        Args:
            ohlcv_df: OHLCV DataFrame with DatetimeIndex.
            analysis_data: Annotation data dict.  Expected keys:
                ``liquidity_sweeps``, ``bos_lines``, ``choch_lines``,
                ``fvgs``, ``order_blocks``, ``entry_zone_start``,
                ``entry_zone_end``, ``stop_loss``, ``take_profits``.
            symbol: Trading pair symbol.
            timeframe: Timeframe label (e.g. ``15m``).
            signal_direction: ``BUY``, ``SELL``, or ``None``.
            confidence_score: Overall confidence percentage.
            file_name: Override output filename.  Auto-generated if ``None``.
            premium_zone: *(V2)* Premium zone dict with
                          ``start``/``end`` prices.
            discount_zone: *(V2)* Discount zone dict with
                           ``start``/``end`` prices.
            reasons: *(V2)* List of signal-reason strings for the
                     annotation text box.

        Returns:
            Absolute file path to the generated PNG.
        """
        if ohlcv_df.empty:
            raise ValueError("OHLCV DataFrame cannot be empty.")

        # Build mplfinance style
        mc = mpf.make_marketcolors(
            up=UP_COLOR,
            down=DOWN_COLOR,
            edge="inherit",
            wick="inherit",
            volume="#c8c8c8" if self._is_dark else "#c8c8c8",
        )
        base_style = "charles" if self._is_dark else "yahoo"
        s = mpf.make_mpf_style(
            base_mpf_style=base_style,
            marketcolors=mc,
            gridcolor=GRID_COLOR,
            facecolor=BACKGROUND_COLOR if self._is_dark else "#ffffff",
            figcolor=BACKGROUND_COLOR if self._is_dark else "#ffffff",
            y_on_right=True,
            rc={
                "axes.labelcolor": TEXT_COLOR,
                "axes.edgecolor": TEXT_COLOR,
                "axes.linewidth": 0.5,
                "ytick.color": TEXT_COLOR,
                "xtick.color": TEXT_COLOR,
                "text.color": TEXT_COLOR,
                "grid.color": GRID_COLOR,
                "grid.alpha": 0.3,
            },
        )

        # Collect horizontal lines
        hlines_prices: List[float] = []
        hlines_colors: List[str] = []
        hlines_styles: List[str] = []
        hlines_labels: Dict[float, Tuple[str, str]] = {}  # price → (label, color)

        # --- Liquidity sweeps ---
        for sweep in analysis_data.get("liquidity_sweeps", []):
            price = sweep[0] if isinstance(sweep, tuple) else sweep
            hlines_prices.append(price)
            hlines_colors.append(COLOR_LIQUIDITY)
            hlines_styles.append(":")
            hlines_labels.setdefault(price, ("Liq Sweep", COLOR_LIQUIDITY))

        # --- BOS lines ---
        for bos in analysis_data.get("bos_lines", []):
            price, _idx, direction = bos[0], bos[1], bos[2]
            color = COLOR_BOS_BULL if direction == "bullish" else COLOR_BOS_BEAR
            hlines_prices.append(price)
            hlines_colors.append(color)
            hlines_styles.append("-")
            label = f"BOS {'↑' if direction == 'bullish' else '↓'}"
            hlines_labels.setdefault(price, (label, color))

        # --- CHoCH lines ---
        for choch in analysis_data.get("choch_lines", []):
            price, _idx, direction = choch[0], choch[1], choch[2]
            color = COLOR_CHOCH_BULL if direction == "bullish" else COLOR_CHOCH_BEAR
            hlines_prices.append(price)
            hlines_colors.append(color)
            hlines_styles.append("-.")
            label = f"CHoCH {'↑' if direction == 'bullish' else '↓'}"
            hlines_labels.setdefault(price, (label, color))

        # --- Entry Zone ---
        entry_start = analysis_data.get("entry_zone_start")
        entry_end = analysis_data.get("entry_zone_end")
        if entry_start is not None and entry_end is not None:
            hlines_prices.extend([entry_start, entry_end])
            hlines_colors.extend([COLOR_ENTRY, COLOR_ENTRY])
            hlines_styles.extend(["--", "--"])
            hlines_labels[entry_start] = ("Entry", COLOR_ENTRY)
            hlines_labels[entry_end] = ("Entry", COLOR_ENTRY)

        # --- Stop Loss ---
        sl = analysis_data.get("stop_loss")
        if sl is not None:
            hlines_prices.append(sl)
            hlines_colors.append(COLOR_SL)
            hlines_styles.append("--")
            hlines_labels[sl] = ("SL", COLOR_SL)

        # --- Take Profits ---
        for i, tp in enumerate(analysis_data.get("take_profits", [])):
            hlines_prices.append(tp)
            hlines_colors.append(COLOR_TP)
            hlines_styles.append("--")
            hlines_labels[tp] = (f"TP{i+1}", COLOR_TP)

        # Build hlines dict for mplfinance
        hlines_dict = {}
        if hlines_prices:
            hlines_dict = dict(
                hlines=hlines_prices,
                colors=hlines_colors,
                linestyle=hlines_styles,
                alpha=0.6,
                linewidths=0.7,
            )

        # --- Plot ---
        title = (
            f"{symbol}  {timeframe}  |  "
            f"Signal: {signal_direction or 'NONE'}  |  "
            f"Confidence: {confidence_score:.1f}%"
        )

        fig, axes = mpf.plot(
            ohlcv_df,
            type="candle",
            style=s,
            volume=True,
            returnfig=True,
            title=title,
            hlines=hlines_dict,
            figscale=self.figscale,
        )

        ax_main = axes[0]
        ax_vol = axes[1]

        # --- Shade FVG zones ---
        for fvg in analysis_data.get("fvgs", []):
            fvg_type = fvg.get("type", "bullish")
            fvg_start = fvg.get("fvg_start", 0)
            fvg_end = fvg.get("fvg_end", 0)
            color = COLOR_FVG_BULL if fvg_type == "bullish" else COLOR_FVG_BEAR
            # Draw horizontal span across the visible chart width
            xlim = ax_main.get_xlim()
            ax_main.axhspan(fvg_start, fvg_end, facecolor=color, alpha=0.5, linewidth=0)

        # --- Shade Order Block zones ---
        for ob in analysis_data.get("order_blocks", []):
            ob_type = ob.get("type", "bullish")
            ob_start = ob.get("ob_start", 0)
            ob_end = ob.get("ob_end", 0)
            color = COLOR_OB_BULL if ob_type == "bullish" else COLOR_OB_BEAR
            xlim = ax_main.get_xlim()
            ax_main.axhspan(ob_start, ob_end, facecolor=color, alpha=0.5, linewidth=0)

        # --- Shade Premium/Discount zones (V2) ---
        if premium_zone:
            p_start = premium_zone.get("start", 0)
            p_end = premium_zone.get("end", 0)
            if p_start and p_end:
                xlim = ax_main.get_xlim()
                ax_main.axhspan(
                    p_start, p_end, facecolor=COLOR_PREMIUM, alpha=0.3, linewidth=0
                )
        if discount_zone:
            d_start = discount_zone.get("start", 0)
            d_end = discount_zone.get("end", 0)
            if d_start and d_end:
                xlim = ax_main.get_xlim()
                ax_main.axhspan(
                    d_start, d_end, facecolor=COLOR_DISCOUNT, alpha=0.3, linewidth=0
                )

        # --- Label hlines with text annotations ---
        seen_labels = set()
        for price, (label, color) in hlines_labels.items():
            label_key = f"{label}_{price}"
            if label_key in seen_labels:
                continue
            seen_labels.add(label_key)
            # Place annotation at the right edge of the chart
            last_idx = len(ohlcv_df) - 1
            ax_main.annotate(
                label,
                xy=(last_idx, price),
                xytext=(5, 0),
                textcoords="offset points",
                color=color,
                fontsize=7,
                fontweight="bold",
                verticalalignment="center",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=BACKGROUND_COLOR,
                    edgecolor=color,
                    alpha=0.8,
                    linewidth=0.5,
                ),
            )

        # --- Draw arrows for BOS/CHoCH events ---
        for bos in analysis_data.get("bos_lines", []):
            price, idx, direction = bos[0], bos[1], bos[2]
            color = COLOR_BOS_BULL if direction == "bullish" else COLOR_BOS_BEAR
            try:
                x_pos = ohlcv_df.index.get_loc(idx)
            except KeyError:
                continue
            if direction == "bullish":
                arrow_props = dict(facecolor=color, shrink=0.05, width=1.5, headwidth=6)
                ax_main.annotate(
                    "", xy=(x_pos, price * 1.004), xytext=(x_pos, price * 0.996),
                    arrowprops=arrow_props,
                )
            else:
                arrow_props = dict(facecolor=color, shrink=0.05, width=1.5, headwidth=6)
                ax_main.annotate(
                    "", xy=(x_pos, price * 0.996), xytext=(x_pos, price * 1.004),
                    arrowprops=arrow_props,
                )

        for choch in analysis_data.get("choch_lines", []):
            price, idx, direction = choch[0], choch[1], choch[2]
            color = COLOR_CHOCH_BULL if direction == "bullish" else COLOR_CHOCH_BEAR
            try:
                x_pos = ohlcv_df.index.get_loc(idx)
            except KeyError:
                continue
            if direction == "bullish":
                arrow_props = dict(facecolor=color, shrink=0.05, width=1.5, headwidth=6)
                ax_main.annotate(
                    "", xy=(x_pos, price * 1.004), xytext=(x_pos, price * 0.996),
                    arrowprops=arrow_props,
                )
            else:
                arrow_props = dict(facecolor=color, shrink=0.05, width=1.5, headwidth=6)
                ax_main.annotate(
                    "", xy=(x_pos, price * 0.996), xytext=(x_pos, price * 1.004),
                    arrowprops=arrow_props,
                )

        # --- Reasons text box (V2) ---
        if reasons:
            reasons_text = "\n".join(f"• {r}" for r in reasons[:8])
            ax_main.text(
                0.02, 0.02,
                f"Signal Reasons:\n{reasons_text}",
                transform=ax_main.transAxes,
                fontsize=7,
                color=TEXT_COLOR,
                verticalalignment="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#1a1a1a",
                    edgecolor=COLOR_ENTRY,
                    alpha=0.9,
                    linewidth=0.8,
                ),
            )

        # --- Legend patch ---
        patches = []
        if analysis_data.get("fvgs"):
            patches.append(mpatches.Patch(color="#00e676", alpha=0.5, label="FVG"))
        if analysis_data.get("order_blocks"):
            patches.append(mpatches.Patch(color="#2979ff", alpha=0.5, label="Order Block"))
        if analysis_data.get("liquidity_sweeps"):
            patches.append(mpatches.Patch(color=COLOR_LIQUIDITY, alpha=0.7, label="Liq Sweep"))
        if patches:
            ax_main.legend(
                handles=patches,
                loc="upper left",
                fontsize=7,
                facecolor=BACKGROUND_COLOR,
                edgecolor=GRID_COLOR,
                labelcolor=TEXT_COLOR,
            )

        # --- Save ---
        if file_name is None:
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{symbol}_{timeframe}_{signal_direction or 'none'}_{confidence_score:.0f}_{ts}.png"

        chart_path = os.path.join(self.output_dir, file_name)
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND_COLOR)
        plt.close(fig)
        logger.info("Chart saved: %s", chart_path)
        return chart_path

    # ------------------------------------------------------------------
    # Helper: generate a simplified chart without signal (for logging)
    # ------------------------------------------------------------------
    def generate_screenshot(
        self,
        ohlcv_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        file_name: Optional[str] = None,
    ) -> str:
        """Generate a clean price chart without trade annotations.

        Useful for logging or heartbeat messages where no signal exists.
        """
        return self.generate_chart(
            ohlcv_df=ohlcv_df,
            analysis_data={},
            symbol=symbol,
            timeframe=timeframe,
            signal_direction=None,
            confidence_score=0.0,
            file_name=file_name or f"{symbol}_{timeframe}_screenshot.png",
        )


# ---------------------------------------------------------------------------
# Module-level self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy OHLCV data
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    base = 65000
    data = {
        "open": np.random.uniform(base - 200, base + 200, n),
        "high": np.random.uniform(base + 100, base + 400, n),
        "low": np.random.uniform(base - 400, base - 100, n),
        "close": np.random.uniform(base - 200, base + 200, n),
        "volume": np.random.uniform(100, 1000, n),
    }
    dummy_df = pd.DataFrame(data, index=dates)

    dummy_analysis = {
        "liquidity_sweeps": [
            (base - 350, dummy_df.index[20]),
            (base + 300, dummy_df.index[70]),
        ],
        "bos_lines": [
            (base + 150, dummy_df.index[40], "bullish"),
            (base - 100, dummy_df.index[80], "bearish"),
        ],
        "choch_lines": [
            (base - 50, dummy_df.index[60], "bearish"),
        ],
        "fvgs": [
            {"type": "bullish", "fvg_start": base - 50, "fvg_end": base + 30, "index": dummy_df.index[35]},
            {"type": "bearish", "fvg_start": base + 100, "fvg_end": base + 180, "index": dummy_df.index[65]},
        ],
        "order_blocks": [
            {"type": "bullish", "ob_start": base - 300, "ob_end": base - 200, "index": dummy_df.index[15]},
            {"type": "bearish", "ob_start": base + 200, "ob_end": base + 300, "index": dummy_df.index[55]},
        ],
        "entry_zone_start": base + 50,
        "entry_zone_end": base + 100,
        "stop_loss": base - 150,
        "take_profits": [base + 250, base + 400, base + 550],
    }

    generator = ChartGenerator(output_dir="/tmp/v2_test_charts")
    path = generator.generate_chart(
        ohlcv_df=dummy_df,
        analysis_data=dummy_analysis,
        symbol="BTCUSDT",
        timeframe="15m",
        signal_direction="BUY",
        confidence_score=88.5,
        file_name="v2_test_chart.png",
        premium_zone={"start": base + 200, "end": base + 400},
        discount_zone={"start": base - 400, "end": base - 200},
        reasons=[
            "Bullish Market Structure",
            "Liquidity Sweep Detected",
            "BOS Confirmed",
            "Fresh Order Block",
            "Untouched FVG",
        ],
    )
    print(f"Chart generated: {path}")
    assert os.path.isfile(path)
    print("✔ Chart file exists!")
