# Professional AI Trading Analysis System V2.0

## Overview

An institutional-grade, modular Python system that monitors cryptocurrency markets using **Smart Money Concepts (SMC)** methodology. The system detects high-probability institutional trading setups, generates professional annotated charts, and sends tiered alerts via Telegram.

**This is NOT a trading bot.** The system never executes trades or connects to any exchange with trading permissions. Its sole purpose is market analysis and alert generation.

---

## What's New in V2.0

| Feature | V1.0 | V2.0 |
|---------|------|------|
| **Alert Tiers** | 2-tier (ignore/alert) | 4-tier (Watchlist, Potential, High Probability, Premium) |
| **Deduplication** | None | Smart dedup with cooldown, score-change detection, direction-change detection |
| **Heartbeat** | None | Startup confirmation + periodic 5h health checks via Telegram |
| **Statistics** | File append only | Structured JSON store with query, summary, and outcome tracking |
| **Chart Annotations** | Basic lines | FVG zones, OB zones, Premium/Discount shading, reasons text box |
| **Debug Mode** | None | Full score breakdown logging with per-component analysis |
| **Error Notifications** | None | Critical error alerts via Telegram |
| **Multi-Timeframe** | 2 TF (higher/lower) | 4 TF (4H macro → 1H structure → 15M setup → 5M entry) |
| **Retry Logic** | Basic tenacity | Exponential backoff with flood-control handling |
| **Rate Limiting** | None | Burst-aware rate limiting for Telegram API |

---

## Features

- **Smart Money Concepts Analysis** — Market Structure (HH/HL/LH/LL), BOS, CHoCH, Liquidity Sweeps, Fair Value Gaps, Order Blocks, Breaker Blocks, Mitigation Blocks, Premium/Discount Zones, Displacement, and Confirmation Candles.
- **Multi-Timeframe Analysis** — 4H for macro trend, 1H for structure validation, 15M for setup location, 5M for precise entry confirmation. Lower TF signals never contradict higher TF direction.
- **Intelligent Confidence Scoring (0-100)** — Configurable weighted scoring with detailed per-component breakdowns. Debug mode shows exactly why each point was awarded.
- **Tiered Alert System** — Four professional tiers with intelligent explanations showing found conditions (✔) and missing conditions (✘).
- **Professional Chart Generation** — Dark-themed annotated candlestick charts with FVG zones, Order Block zones, BOS/CHoCH lines, Liquidity Sweeps, Premium/Discount shading, Entry Zone, SL/TP levels, and signal reasons.
- **Telegram Alerts** — HTML-formatted messages with charts, score breakdowns, risk levels, and human-readable explanations.
- **Deduplication** — Prevents redundant alerts. Only re-sends when score changes significantly, market structure changes, direction flips, or setup invalidates.
- **Heartbeat System** — Startup confirmation and periodic (5h) health checks with uptime, scan count, and alert count.
- **Statistics Tracking** — Records every detected setup with timestamp, direction, score, reasons, and outcome. Queryable history with summary generation.
- **Error Notifications** — Immediate Telegram alerts for critical failures (API down, unexpected exceptions, scanner stopped).
- **Economic News Filter** — Suppresses alerts before high-impact USD events (CPI, NFP, FOMC, etc.) with configurable before/after buffers.
- **Trading Session Awareness** — Prioritizes London, New York, and overlap sessions.
- **Modular Architecture** — Analysis engine is fully independent and callable for n8n or custom integrations.
- **Fully Configurable** — All settings in YAML (weights, thresholds, symbols, timeframes, intervals, dedup rules, retention).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          main.py                                     │
│                (Orchestrator / Scheduling Loop)                      │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────┘
       │          │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Binance │ │ Analysis │ │ Scoring  │ │ Filters  │ │  Alerts  │ │  Stats   │
│  Client  │ │  Engine  │ │  System  │ │News/Sess.│ │Telegram  │ │ Tracker  │
└──────────┘ └────┬─────┘ └──────────┘ └──────────┘ └────┬─────┘ └──────────┘
                  │                                       │
    ┌─────────────┼─────────────────────────┐   ┌────────┼────────┐
    │             │                         │   │        │        │
    ▼             ▼                         ▼   ▼        ▼        ▼
┌────────┐  ┌──────────┐  ┌────────────────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Market  │  │Liquidity │  │ FVG / OB /     │ │Dedup │ │Chart │ │Heart │
│Struct. │  │Detection │  │ Zones / Candles│ │      │ │Gen   │ │beat  │
└────────┘  └──────────┘  └────────────────┘ └──────┘ └──────┘ └──────┘
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
│   ├── alerts/
│   │   ├── __init__.py
│   │   ├── telegram.py         # Telegram bot (retry, rate limit, error)
│   │   ├── formatter.py        # Tiered HTML alert formatter
│   │   └── deduplication.py    # Duplicate alert prevention
│   ├── charts/
│   │   ├── __init__.py
│   │   └── generator.py        # Annotated chart generation
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Structured logging with debug mode
│       ├── heartbeat.py        # Startup + periodic heartbeat
│       └── statistics.py       # Setup tracking & persistence
├── charts/                     # Generated chart images
├── logs/                       # Log files & statistics
├── main.py                     # Main runner/scheduler (V2.0 orchestrator)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── Procfile                    # Railway deployment config
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

| Package | Purpose |
|---------|---------|
| `requests` | HTTP client for Binance API and Telegram |
| `urllib3` | Connection pooling and retry configuration |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `tenacity` | Retry logic for API calls |
| `PyYAML` | Configuration file parsing |
| `matplotlib` | Chart rendering |
| `mplfinance` | Candlestick chart generation |
| `python-dotenv` | Environment variable management |

---

## Configuration

All settings are in `config/settings.yaml`. Key sections:

### System
```yaml
system:
  version: "2.0.0"
  debug_mode: false
  scan_interval_seconds: 60
  heartbeat_interval_hours: 5
```

### Trading & Multi-Timeframe
```yaml
trading:
  symbols: ["BTCUSDT", "ETHUSDT"]
  primary_timeframe: "15m"
  multi_timeframe:
    enabled: true
    macro_tf: "4h"
    structural_tf: "1h"
    setup_tf: "15m"
    entry_tf: "5m"
    contradiction_handling: "filter_out"
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

### Alert Tiers
```yaml
alert_tiers:
  ignore_below: 60
  watchlist_min: 60
  potential_setup_min: 70
  high_probability_min: 80
  premium_min: 90
```

### Telegram
```yaml
telegram:
  enabled: true
  bot_token: "YOUR_BOT_TOKEN_HERE"
  chat_id: "YOUR_CHAT_ID_HERE"
  alert_options:
    send_watchlist: true
    send_potential: true
    send_high_prob: true
    send_premium: true
    include_chart: true
    include_breakdown: true
```

### Deduplication
```yaml
deduplication:
  enabled: true
  score_change_threshold: 10
  structure_change_resend: true
  direction_change_resend: true
  cooldown_minutes: 15
  max_alerts_per_hour: 3
```

### Heartbeat
```yaml
heartbeat:
  startup_message: true
  interval_hours: 5
  include_scan_count: true
  include_alert_count: true
  include_uptime: true
```

### Error Notifications
```yaml
error_notifications:
  enabled: true
  events:
    - "binance_api_unavailable"
    - "telegram_connection_failed"
    - "unexpected_exception"
    - "scanner_stopped"
```

### Statistics
```yaml
statistics:
  enabled: true
  store_path: "logs/stats.json"
  retention_days: 90
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
1. Send a startup confirmation to Telegram
2. Fetch OHLCV data from Binance for all configured symbols
3. Run multi-timeframe SMC analysis
4. Calculate confidence scores with detailed breakdowns
5. Apply news and session filters
6. Check deduplication (suppress duplicates)
7. Generate annotated charts for qualifying setups
8. Send tiered Telegram alerts (Watchlist / Potential / High Probability / Premium)
9. Record statistics for every detected setup
10. Send periodic heartbeat messages every 5 hours
11. Log sub-threshold signals for statistics
12. Sleep and repeat at the configured interval

### Debug Mode

Enable debug mode in `config/settings.yaml` to see detailed score breakdowns:

```yaml
system:
  debug_mode: true
```

Debug logs show:
```
============================================================
Score Breakdown — BTCUSDT
------------------------------------------------------------
  Market Structure Alignment       20.0 /   20.0  (100.0%)
  Liquidity Sweep                  18.0 /   20.0  ( 90.0%)
  BOS / CHoCH Confirmation         15.0 /   15.0  (100.0%)
  Fair Value Gap                   13.5 /   15.0  ( 90.0%)
  Fresh Order Block                 8.0 /   10.0  ( 80.0%)
  Premium / Discount Zone           5.0 /    5.0  (100.0%)
  Confirmation Candle               4.0 /    5.0  ( 80.0%)
  Trading Session Quality           5.0 /    5.0  (100.0%)
  News Filter                       4.0 /    5.0  ( 80.0%)
------------------------------------------------------------
  FINAL SCORE                      92.5
============================================================
```

### Graceful Shutdown

Press `Ctrl+C` or send `SIGTERM` to stop the system gracefully. The system will:
1. Complete the current scan
2. Stop the heartbeat thread
3. Log final statistics
4. Exit cleanly

---

## Alert Tiers Explained

| Tier | Score Range | Description |
|------|-------------|-------------|
| **Ignored** | Below 60 | Setup is too weak. No alert sent. |
| **Watchlist** | 60–69 | Setup is incomplete. Clearly explains what conditions are missing. Warns that probability is relatively low. |
| **Potential** | 70–79 | Setup has strengths but needs additional confirmations. Explains both strengths and weaknesses. |
| **High Probability** | 80–89 | Complete alert with chart, full explanation, and all trade levels. |
| **Premium Institutional** | 90–100 | Highest confidence. Full alert with detailed explanation, chart, and all trade levels. |

---

## Alert Example

```
🔥 PREMIUM INSTITUTIONAL SETUP 🔥

Signal: BUY
Symbol: BTCUSDT | TF: 15m

───────── TRADE SETUP ─────────
Entry Zone: 64,500.00 – 64,600.00
Stop Loss: 64,200.00

Take Profits:
  TP1: 64,900.00 (R:R 1.33)
  TP2: 65,200.00 (R:R 2.33)
  TP3: 65,600.00 (R:R 3.67)

───────── SCORE ─────────
Confidence: 92.5% (PREMIUM INSTITUTIONAL SETUP)

Conditions Found:
  ✔ Market Structure Alignment: 20.0/20
  ✔ Liquidity Sweep: 18.0/20
  ✔ BOS / CHoCH Confirmation: 15.0/15
  ✔ Fair Value Gap: 13.5/15
  ✔ Fresh Order Block: 8.0/10
  ✔ Premium / Discount Zone: 5.0/5
  ✔ Confirmation Candle: 4.0/5
  ✔ Trading Session Quality: 5.0/5
  ✔ News Filter: 4.0/5

───────── ANALYSIS ─────────
Reasons Found:
  • Bullish Market Structure
  • Liquidity Sweep Detected
  • BOS Confirmed
  • Fresh Order Block
  • Untouched FVG

Risk Level: Low

SMC Analysis System V2.0 | 2024-01-15 14:30:00 UTC
```

---

## Statistics

The system records every detected setup to `logs/stats.json`. Use the `StatisticsTracker` class to query:

```python
from src.utils.statistics import StatisticsTracker

tracker = StatisticsTracker(store_path="logs/stats.json")

# Get all BTCUSDT records
btc = tracker.get_history(symbol="BTCUSDT")

# Get all BUY signals above 80
buys = tracker.get_history(direction="BUY", min_score=80)

# Generate summary report
summary = tracker.get_summary()
print(summary["win_rate_pct"])
print(summary["tier_distribution"])
```

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

## Deploying to Railway

1. Connect your GitHub repository to Railway.
2. Set environment variables in Railway's dashboard:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
3. Railway will automatically detect the `Procfile` and run:
   ```
   worker: python main.py
   ```
4. The system runs continuously as a background worker.

---

## Heartbeat Example

```
💓 SYSTEM HEARTBEAT 💓

Status: ONLINE
Monitoring: BTCUSDT, ETHUSDT
Uptime: 18h 23m 45s
Scans Completed: 1103
Alerts Sent: 7

System operating normally. 14:30:00 UTC
```

---

## Disclaimer

This system is for educational and analysis purposes only.

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
