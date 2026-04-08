# Stock Screener and MT5 Trading Bots

This repository contains:

1. MT5 trading bots (`robot_trade.py`, `robot_trade_sr.py`, `robot_crt_po3.py`)
2. Indonesian stock screener pipeline (`screener_id.py`, `command.py`, `manager.py`)

## Current Status

### `robot_trade.py` (trend-following bot)
- Environment-driven runtime config (required vars for core risk/runtime settings)
- Drawdown protection and emergency close logic
- Trend-following signal engine with per-symbol adjustments (XAUUSD, EURUSD, optional JPY tuning)
- Trading mode profiles: `conservative`, `normal`, `extreme`
- Clear timeframe separation:
  - `entry_timeframe` for entries/position decisions
  - `trend_timeframe` for higher timeframe trend filter
- Position sync against MT5 and close-reason/profit reconciliation
- Telegram summary reporting

### `robot_trade_sr.py` (support/resistance bot)
- Strategy profile runner (`SUPPORT_RESISTANCE_SWING`, `...SCALP_1M`, `...SCALP_5M`, `...SCALP_15M`, `CUSTOM`)
- Multi-timeframe scalping filter support (`SR_ENABLE_MULTI_TIMEFRAME_SCALPING`)
- Configurable run/report windows via environment variables

### `screener_id.py` (Indonesian daily screener)
- Symbol universe loaded from `stocklist.csv` each run (with built-in fallback list if CSV is missing/too small)
- Daily technical + volume scoring and ranking
- Action labels: `ADD`, `WATCH`, `MONITOR`
- Exports report files:
  - `screener_report.csv`
  - `screener_report.json`
- Telegram-ready summary + file attachments
- SSL fail-fast preflight and stricter symbol sanitization (prevents noisy repeated ticker errors)

### `command.py`
- Thin runner for the latest screener flow:
  - refresh universe (`stocklist.csv` -> fallback list)
  - screen
  - display
  - export
  - send Telegram

### `manager.py`
- Utility for loading current screener universe and exporting it to `idx_universe.csv`

## Key Files

- `robot_trade.py`
- `robot_trade_sr.py`
- `robot_crt_po3.py`
- `robot_runtime.py`
- `screener_id.py`
- `command.py`
- `manager.py`
- `.env.example`
- `requirements.txt`

## Requirements

- Windows
- Python 3.9+
- MetaTrader 5 desktop terminal installed
- Valid MT5 account session/credentials (depending on script)

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Setup

Copy environment template:

```powershell
Copy-Item .env.example .env
```

Then fill `.env` values.

## Environment Variables (Important)

### Shared
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `MT5_LOGIN`
- `MT5_PASSWORD`
- `MT5_SERVER`

### `screener_id.py`
- Uses shared Telegram variables above for report delivery.
- Universe source file: `stocklist.csv` (one symbol per line, e.g. `BBCA`; comments allowed with `#`).
- If `stocklist.csv` has too few valid symbols, screener falls back to built-in IDX universe.

### `robot_trade.py`
- `MT5_SYMBOLS`
- `MT5_RISK_PER_TRADE`
- `MT5_MAGIC_NUMBER`
- `MT5_MAX_DRAWDOWN_PERCENT`
- `MT5_DRAWDOWN_PERIOD_HOURS`
- `MT5_RUN_DURATION_MINUTES`
- `MT5_REPORT_PERIOD_HOURS`
- `MT5_ENTRY_TIMEFRAME` (example: `M1`)
- `MT5_TREND_TIMEFRAME` (example: `M5`)
- `MT5_ENABLE_JPY_TUNING`
- `MT5_TRADING_MODE` (`conservative` | `normal` | `extreme`)
- `ROBOT_LOG_LEVEL`
- `ROBOT_LOG_FILE`

### `robot_trade_sr.py`
- `SR_STRATEGY`
- `SR_SYMBOLS`
- `SR_RUN_DURATION_MINUTES`
- `SR_REPORT_PERIOD_DAYS`
- `SR_SHOW_LEVELS`
- `SR_ENABLE_MULTI_TIMEFRAME_SCALPING`
- `SR_LOG_LEVEL`
- `SR_LOG_FILE`

### `robot_crt_po3.py`
- `CRT_SYMBOL`
- `CRT_TIMEFRAME`
- `CRT_MAGIC_NUMBER`
- `CRT_RISK_PERCENT`
- `CRT_MAX_SPREAD`
- `CRT_ANALYSIS_INTERVAL_SECONDS`
- `CRT_REPORT_PERIOD_DAYS`
- `CRT_LOG_LEVEL`
- `CRT_LOG_FILE`

## How to Run

### Trend bot
```powershell
python robot_trade.py
```

### Support/Resistance bot
```powershell
python robot_trade_sr.py
```

### CRT PO3 bot
```powershell
python robot_crt_po3.py
```

### Daily screener pipeline
```powershell
python command.py
```

### Refresh/export IDX universe only
```powershell
python manager.py
```

## Logs and Reports

- MT5 bot logs are written to `logs/*.jsonl`
- Screener exports:
  - `screener_report.csv`
  - `screener_report.json`

## Notes

- This project is experimental and not financial advice.
- Always validate behavior on demo accounts first.
- MT5 and market-data connectivity issues can affect execution/reporting.
