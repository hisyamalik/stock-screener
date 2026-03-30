# Stock Screener and MT5 Trading Bots

This repository contains:

1. A Yahoo Finance based Indonesian stock screener.
2. Multiple MetaTrader 5 trading bots with different strategies.

The actively maintained MT5 scripts are:

- [`robot_trade.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade.py): trend-following MT5 bot with SMA, RSI, ATR, drawdown protection, position sync, Telegram reporting, and configurable trade-cycle reporting.
- [`robot_trade_sr.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade_sr.py): support/resistance trading bot with configurable strategy profiles and configurable stats lookback.
- [`robot_crt_po3.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_crt_po3.py): CRT Power of 3 MT5 bot with interactive menu, multi-timeframe analysis, env-based MT5 login, and configurable stats lookback.

## Project Contents

- [`robot_trade.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade.py)
- [`robot_trade_sr.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade_sr.py)
- [`robot_crt_po3.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_crt_po3.py)
- [`robot_runtime.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_runtime.py): shared runtime utilities for `.env` loading, MT5 credential loading, and structured logging.
- [`robot_trading_scalping.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trading_scalping.py): older scalping bot variant.
- [`robot_trading_scalping_ob_confirm.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trading_scalping_ob_confirm.py): older scalping bot variant with order-block confirmation.
- [`command.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/command.py): Indonesian stock screener using `yfinance`.
- [`manager.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/manager.py): stock symbol scraper prototype.
- [`stocklist.csv`](/abs/path/c:/Dev/Stock%20Price/stock-screener/stocklist.csv): sample Indonesian tickers.
- [`requirements.txt`](/abs/path/c:/Dev/Stock%20Price/stock-screener/requirements.txt): Python dependencies.
- [`.env.example`](/abs/path/c:/Dev/Stock%20Price/stock-screener/.env.example): example environment variables for all maintained robot scripts.

## What Changed

The maintained MT5 scripts now use:

- cleaner `main()` entrypoints instead of large inline startup blocks
- environment variables for runtime configuration and secrets
- structured rotating JSON logs written to `logs/*.jsonl`
- configurable report/statistics lookback windows
- shared runtime helpers in [`robot_runtime.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_runtime.py)

## Requirements

- Windows
- Python 3.9+ recommended
- MetaTrader 5 desktop terminal installed
- A demo or live MT5 account

Python packages:

- `MetaTrader5`
- `pandas`
- `numpy`
- `requests`
- `urllib3`
- `python-dotenv`
- `yfinance`
- `beautifulsoup4`

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create your local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Then update the values in `.env` for your MT5 account, Telegram bot, symbols, and runtime preferences.

## Environment Variables

Shared MT5 credentials for scripts that require explicit MT5 login:

- `MT5_LOGIN`
- `MT5_PASSWORD`
- `MT5_SERVER`

`robot_trade.py` settings:

- `MT5_SYMBOLS`
- `MT5_RISK_PER_TRADE`
- `MT5_MAGIC_NUMBER`
- `MT5_MAX_DRAWDOWN_PERCENT`
- `MT5_DRAWDOWN_PERIOD_HOURS`
- `MT5_RUN_DURATION_MINUTES`
- `MT5_REPORT_PERIOD_HOURS`
- `MT5_ENABLE_JPY_TUNING`
- `ROBOT_LOG_LEVEL`
- `ROBOT_LOG_FILE`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

`robot_trade_sr.py` settings:

- `SR_STRATEGY`
- `SR_SYMBOLS`
- `SR_RUN_DURATION_MINUTES`
- `SR_REPORT_PERIOD_DAYS`
- `SR_SHOW_LEVELS`
- `SR_ENABLE_MULTI_TIMEFRAME_SCALPING`
- `SR_LOG_LEVEL`
- `SR_LOG_FILE`

`robot_crt_po3.py` settings:

- `CRT_SYMBOL`
- `CRT_TIMEFRAME`
- `CRT_MAGIC_NUMBER`
- `CRT_RISK_PERCENT`
- `CRT_MAX_SPREAD`
- `CRT_ANALYSIS_INTERVAL_SECONDS`
- `CRT_REPORT_PERIOD_DAYS`
- `CRT_LOG_LEVEL`
- `CRT_LOG_FILE`

## Structured Logging

Each maintained MT5 script now writes JSON line logs:

- [`robot_trade.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade.py) -> `logs/robot_trade.jsonl`
- [`robot_trade_sr.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade_sr.py) -> `logs/robot_trade_sr.jsonl`
- [`robot_crt_po3.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_crt_po3.py) -> `logs/robot_crt_po3.jsonl`

Console logs are still shown during execution.

## Running the Bots

### Trend-following bot

```powershell
python robot_trade.py
```

Highlights:

- uses current MT5 terminal session
- detects manual closes from MT5 app
- supports Telegram reporting
- uses configurable report period in hours

### Support/resistance bot

```powershell
python robot_trade_sr.py
```

Highlights:

- supports multiple predefined S/R strategy profiles
- uses env-driven session duration and symbol overrides
- trading stats now accept configurable `report_period_days`

### CRT Power of 3 bot

```powershell
python robot_crt_po3.py
```

Highlights:

- reads MT5 login credentials from `.env`
- keeps the interactive menu workflow
- supports configurable stats lookback with `CRT_REPORT_PERIOD_DAYS`
- writes structured logs instead of plain file logging only

## Reporting Notes

[`robot_trade.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade.py):

- `get_trading_cycle(report_period_hours=...)` uses MT5 deal history grouped by `position_id`
- manual closes can still be included if the position originally belonged to the robot

[`robot_trade_sr.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade_sr.py):

- `get_trading_statistics(report_period_days=...)` now uses a configurable lookback period

[`robot_crt_po3.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_crt_po3.py):

- `get_trading_stats(period_days=...)` now uses a configurable lookback period

## Using the Stock Screener

Run the stock screener with:

```powershell
python command.py
```

It will:

- load tickers from [`stocklist.csv`](/abs/path/c:/Dev/Stock%20Price/stock-screener/stocklist.csv)
- fetch Yahoo Finance data
- calculate moving average crossover signals
- print symbols with fresh buy signals

## Known Limitations

- The strategy code in older experimental robot files is still not standardized.
- There is no automated test suite yet.
- The CRT script still uses an interactive CLI menu instead of command-line flags.
- The bots depend on a running MT5 desktop terminal and valid broker access.
- `manager.py` is still a prototype scraper with a placeholder target URL.

## Disclaimer

This repository is for educational and experimental use. Trading forex, commodities, and stocks carries significant risk. Test on demo accounts first and validate all strategy behavior yourself before using any bot on a live account.
