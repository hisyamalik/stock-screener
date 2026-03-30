# Stock Screener and MT5 Trading Bot

This repository contains two related but separate trading utilities:

1. An Indonesian stock screener built on Yahoo Finance data.
2. A MetaTrader 5 forex/gold trading bot with risk controls and Telegram reporting.

The current main trading script is [`robot_trade.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade.py).

## Project Contents

- [`robot_trade.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade.py): MT5 trading bot using SMA, RSI, ATR, higher-timeframe trend filtering, drawdown protection, position sync, and Telegram reporting.
- [`robot_trade_sr.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade_sr.py): Alternative robot version focused on support/resistance logic.
- [`robot_crt_po3.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_crt_po3.py): Alternative robot/strategy experiment.
- [`robot_trading_scalping.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trading_scalping.py): Earlier scalping bot version.
- [`robot_trading_scalping_ob_confirm.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trading_scalping_ob_confirm.py): Earlier scalping bot variant with order-block confirmation.
- [`command.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/command.py): Indonesian stock screener using moving average crossover logic and `yfinance`.
- [`manager.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/manager.py): Simple stock symbol scraper prototype.
- [`stocklist.csv`](/abs/path/c:/Dev/Stock%20Price/stock-screener/stocklist.csv): Sample list of Indonesian stock tickers.

## Features

### MT5 bot

- Connects directly to a locally installed MetaTrader 5 terminal.
- Generates trade signals from:
  - higher-timeframe trend filter
  - lower-timeframe SMA crossover
  - RSI confirmation
  - ATR-based stop loss and take profit
- Calculates position size from account balance and risk-per-trade.
- Tracks equity and stops trading when drawdown limit is breached.
- Syncs robot state with live MT5 positions, including manual closes from the MT5 app.
- Produces a recent-cycle trading report and can send it to Telegram.

### Stock screener

- Loads Indonesian tickers from CSV.
- Downloads market data with `yfinance`.
- Checks for moving average crossover buy signals.
- Prints tickers that currently meet the strategy rule.

## Requirements

- Windows with MetaTrader 5 installed for MT5 bot usage
- Python 3.9+ recommended
- A funded or demo MT5 account logged in through the terminal

Python packages used in this repository include:

- `MetaTrader5`
- `pandas`
- `numpy`
- `requests`
- `urllib3`
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
pip install MetaTrader5 pandas numpy requests urllib3 yfinance beautifulsoup4
```

## Using the MT5 Trading Bot

### 1. Prepare MetaTrader 5

- Install and open MetaTrader 5 desktop terminal.
- Log in to your broker account.
- Make sure the symbols you want to trade are available and visible in Market Watch.
- Enable algorithmic trading in MT5 if your broker and setup require it.

### 2. Review runtime settings in `robot_trade.py`

The executable section is at the bottom of [`robot_trade.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade.py).

Important values to review before running:

- `risk_per_trade`
- `magic_number`
- `max_drawdown_percent`
- `drawdown_period_hours`
- `symbols`
- `run_duration_minutes`

Current example configuration:

```python
robot = MT5ForexRobot(
    risk_per_trade=0.001,
    magic_number=19910,
    max_drawdown_percent=25.0,
    drawdown_period_hours=4
)

symbols = ['XAUUSD']
robot.run_trading_bot(symbols, run_duration_minutes=360)
```

### 3. Run the bot

```powershell
python robot_trade.py
```

### 4. What the bot does during runtime

- initializes MT5 connection
- reads account balance and equity
- checks drawdown protection each cycle
- monitors current positions
- syncs local state with MT5 so manual closes are detected
- analyzes each configured symbol
- places trades only when no robot-owned position is already open for that symbol
- reports the latest 4-hour trading summary at the end

## Telegram Reporting

[`robot_trade.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade.py) includes a `send_telegram_report()` function.

Before using it, replace the hardcoded values with your own:

- Telegram bot token
- Telegram chat ID

Important:

- The current implementation keeps the token and chat ID directly in source code.
- For safety, move these to environment variables before using this project in production or sharing the repository.

Example approach:

```powershell
$env:TELEGRAM_BOT_TOKEN="your-token"
$env:TELEGRAM_CHAT_ID="your-chat-id"
```

Then update the code to read from `os.environ`.

## Trading Report Notes

The `get_trading_cycle()` report in [`robot_trade.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/robot_trade.py) now:

- uses the last 4 hours of MT5 deal history
- groups history by `position_id`
- counts one closed position as one trade
- includes net profit using profit, commission, swap, and fee
- can still recognize manual closes as part of a robot-owned position

Returned report fields include:

- `Period`
- `Total Trades`
- `Winning Trades`
- `Losing Trades`
- `Breakeven Trades`
- `Win Rate`
- `Net Profit`
- `Open Positions`
- `Initial Balance`
- `Current Balance`
- `Last Closed Trade`

## Using the Stock Screener

The simpler stock screener is in [`command.py`](/abs/path/c:/Dev/Stock%20Price/stock-screener/command.py).

Run it with:

```powershell
python command.py
```

This script:

- loads symbols from [`stocklist.csv`](/abs/path/c:/Dev/Stock%20Price/stock-screener/stocklist.csv)
- downloads historical data from Yahoo Finance
- calculates short and long moving averages
- prints tickers with a fresh bullish crossover

## Known Limitations

- There is no central config file yet; strategy values are still edited directly in Python files.
- Telegram credentials are currently hardcoded in the MT5 bot.
- The bot assumes a local MetaTrader 5 desktop session is already active.
- Some older strategy files in the repo are experimental and may not match the current main bot behavior.
- There is no automated test suite yet.
- `manager.py` is a prototype scraper and uses a placeholder URL.

## Recommended Next Improvements

- move secrets to environment variables
- add a `requirements.txt`
- add a `.env.example`
- add structured logging to file
- make report period configurable
- add backtesting or dry-run mode
- add automated tests around reporting and position synchronization

## Disclaimer

This project is for educational and experimental use. Trading stocks, forex, and commodities involves significant risk. Use a demo account first, validate all strategy logic yourself, and do not rely on the software without your own testing and risk review.
