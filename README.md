# Professional AI Trading Analysis System

## Overview

A production-ready, modular Python system that monitors cryptocurrency markets using **Smart Money Concepts (SMC)** methodology. The system detects high-probability institutional trading setups, generates professional annotated charts, and sends alerts via Telegram.

**This is NOT a trading bot.** The system never executes trades or connects to any exchange with trading permissions. Its sole purpose is market analysis and alert generation.

---

## Features

- **Smart Money Concepts Analysis** — Market Structure (HH/HL/LH/LL), BOS, CHoCH, Liquidity Sweeps, Fair Value Gaps, Order Blocks, Breaker Blocks, Mitigation Blocks, Premium/Discount Zones, Displacement, and Confirmation Candles.
- **Multi-Timeframe Analysis** — 1H for trend direction, 15M for entry confirmation.
- **Confidence Scoring (0-100)** — Configurable weighted scoring system with detailed breakdowns.
- **Professional Chart Generation** — Dark-themed annotated candlestick charts with FVGs, Order Blocks, BOS/CHoCH lines, entry zones, SL/TP levels, and arrows.
- **Telegram Alerts** — HTML-formatted messages with charts, score breakdowns, and human-readable explanations.
- **Economic News Filter** — Suppresses alerts before high-impact USD events (CPI, NFP, FOMC, etc.).
- **Trading Session Awareness** — Prioritizes London and New York sessions.
- **Modular Architecture** — Analysis engine is fully independent and callable for n8n integration.
- **Configurable** — All settings in YAML (weights, thresholds, symbols, timeframes, intervals).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
│              (Orchestrator / Scheduling Loop)                 │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┘
       │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Binance │ │ Analysis │ │ Scoring  │ │ Filters  │ │  Alerts  │
│  Client  │ │  Engine  │ │  System  │ │News/Sess.│ │Telegram  │
└──────────┘ └────┬─────┘ └──────────┘ └──────────┘ └──────────┘
                  │
    ┌─────────────┼─────────────────────────┐
    │             │                         │
    ▼             ▼                         ▼
┌────────┐  ┌──────────┐  ┌─────────────────────────┐
│Market  │  │Liquidity │  │ FVG / Order Blocks /    │
│Struct. │  │Detection │  │ Zones / Candles         │
└────────┘  └──────────┘  └─────────────────────────┘
```

---

## Project Structure

```
smc_trading_assistant/
├── config/
│   └── settings.yaml          # All configurable settings
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── binance_client.py  # Binance Public API data fetcher
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── market_structure.py  # HH/HL/LH/LL, BOS, CHoCH
│   │   ├── liquidity.py        # Liquidity sweeps, equal highs/lows
│   │   ├── fvg.py              # Fair Value Gaps
│   │   ├── order_blocks.py     # Order Blocks, Breaker, Mitigation
│   │   ├── zones.py            # Premium/Discount zones
│   │   ├── candles.py          # Displacement, rejection, confirmation
│   │   └── engine.py           # Main analysis engine (n8n integration point)
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── confidence.py       # Confidence scoring (0-100)
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── news.py             # Economic news filter
│   │   └── sessions.py         # Trading session filter
│   ├── charts/
│   │   ├── __init__.py
│   │   └── generator.py        # Annotated chart generation
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── telegram.py         # Telegram bot integration
│   │   └── formatter.py        # Alert message formatting
│   └── utils/
│       ├── __init__.py
│       └── logger.py           # Structured logging
├── charts/                     # Generated chart images
├── logs/                       # Log files
├── main.py                     # Main runner/scheduler
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md                   # This file
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Internet connection (for Binance API access)

### Setup

```bash
# 1. Clone or extract the project
cd smc_trading_assistant

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your Telegram Bot Token and Chat ID

# 5. Review and adjust configuration
# Edit config/settings.yaml as needed
```

### Dependencies

- `requests` — HTTP client for Binance API
- `pandas` — Data manipulation and analysis
- `numpy` — Numerical computations
- `tenacity` — Retry logic for API calls
- `PyYAML` — Configuration file parsing
- `matplotlib` — Chart rendering
- `mplfinance` — Candlestick chart generation
- `python-dotenv` — Environment variable management

---

## Configuration

All settings are in `config/settings.yaml`. Key sections:

### Trading Settings
```yaml
trading:
  symbols: ["BTCUSDT"]           # Symbols to monitor
  timeframes:
    higher: "1h"                  # Trend determination
    lower: "15m"                  # Entry confirmation
  scan_interval_seconds: 60       # Scan frequency
```

### Confidence Weights (must sum to 100)
```yaml
scoring:
  weights:
    market_structure_alignment: 20
    liquidity_sweep: 20
    bos_choch_confirmation: 15
    fair_value_gap: 15
    fresh_order_block: 10
    premium_discount_zone: 5
    confirmation_candle: 5
    trading_session_quality: 5
    news_filter: 5
```

### Alert Thresholds
```yaml
  thresholds:
    ignore_below: 70              # Below 70: Ignore
    record_only_min: 70           # 70-79: Record for stats
    high_probability_min: 80      # 80-89: High Probability Alert
    premium_min: 90               # 90-100: Premium Institutional Alert
```

### Telegram
```yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN_HERE"
  chat_id: "YOUR_CHAT_ID_HERE"
```

### News Filter
```yaml
news_filter:
  enabled: true
  buffer_minutes: 30
  high_impact_events: [CPI, PPI, NFP, FOMC, ...]
```

---

## Usage

### Running the System

```bash
# Activate virtual environment
source venv/bin/activate

# Run the analysis system
python main.py
```

The system will:
1. Fetch OHLCV data from Binance for configured symbols
2. Run SMC analysis on both timeframes
3. Calculate confidence scores
4. Apply news and session filters
5. Generate annotated charts for qualifying setups
6. Send Telegram alerts (if score >= 80)
7. Log sub-threshold signals (70-79) for statistics
8. Sleep and repeat at the configured interval

### Graceful Shutdown

Press `Ctrl+C` or send `SIGTERM` to stop the system gracefully.

---

## API Documentation (n8n Integration)

The analysis engine is designed to be called independently:

```python
from src.data.binance_client import BinanceClient
from src.analysis.engine import analyze, AnalysisResult

# Fetch data
client = BinanceClient(base_url="https://data-api.binance.vision/api/v3")
ohlcv_df = client.get_ohlcv("BTCUSDT", "15m", limit=200)

# Run analysis
result: AnalysisResult = analyze(ohlcv_df, "BTCUSDT", "15m")

# Access results
print(result.signal_direction)    # "BUY", "SELL", or None
print(result.entry_zone_start)    # Entry price start
print(result.entry_zone_end)      # Entry price end
print(result.stop_loss)           # Stop loss level
print(result.take_profits)        # [TP1, TP2, TP3]
print(result.confidence_score)    # 0-100
print(result.confidence_breakdown)  # Component scores
print(result.human_explanation)   # Why the setup is valid
```

### AnalysisResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | str | Trading pair (e.g., "BTCUSDT") |
| `timeframe` | str | Timeframe analyzed |
| `signal_direction` | str/None | "BUY", "SELL", or None |
| `entry_zone_start` | float | Entry zone lower bound |
| `entry_zone_end` | float | Entry zone upper bound |
| `stop_loss` | float | Suggested stop loss level |
| `take_profits` | List[float] | [TP1, TP2, TP3] |
| `risk_reward_ratios` | List[float] | R:R for each TP |
| `confidence_score` | float | Overall confidence (0-100) |
| `confidence_breakdown` | Dict | Per-component scores |
| `human_explanation` | str | Why the setup is valid |
| `raw_data` | DataFrame | Processed OHLCV with indicators |

### For n8n Integration

Use the "Execute Python Code" node in n8n to call the `analyze()` function directly. The analysis logic is completely independent from the scheduling loop.

---

## Alert Example

```
🔥 PREMIUM INSTITUTIONAL SETUP 🔥

Signal: BUY
Symbol: BTCUSDT | Timeframe: 1h / 15m
Entry Zone: 64,500.00 - 64,600.00
Stop Loss: 64,200.00

Take Profits:
  TP1: 64,900.00 (R:R 1.33)
  TP2: 65,200.00 (R:R 2.33)
  TP3: 65,600.00 (R:R 3.67)

Confidence Score: 92.5%
Breakdown:
  - Market Structure Alignment: 20.0/20
  - Liquidity Sweep: 18.0/20
  - BOS/CHoCH Confirmation: 15.0/15
  - Fair Value Gap: 13.5/15
  - Fresh Order Block: 8.0/10
  - Premium/Discount Zone: 5.0/5
  - Confirmation Candle: 4.0/5
  - Trading Session Quality: 5.0/5
  - News Filter: 4.0/5

Explanation:
Price swept liquidity below equal lows at $64,350, followed by a
bullish BOS on the 15M timeframe. Price retraced into a fresh
bullish FVG within the discount zone, forming a strong rejection
candle. This indicates strong institutional buying pressure with
confluence across multiple SMC factors.
```

---

## Adding More Symbols

To monitor additional symbols, simply add them to `config/settings.yaml`:

```yaml
trading:
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "SOLUSDT"
```

No code changes required.

---

## Disclaimer

⚠️ **This system is for educational and analysis purposes only.**

- It does NOT provide financial advice.
- It does NOT execute trades.
- It does NOT connect to exchange accounts.
- Trading cryptocurrencies carries significant risk.
- Past performance does not guarantee future results.
- Always do your own research and consult a qualified financial advisor.

The developers are not responsible for any financial losses incurred from using this system's analysis.

---

## License

MIT License
