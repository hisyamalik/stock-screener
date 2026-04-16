# Project Structure

This document outlines the architecture for the Quantitative Trading Dashboard.
The project is divided into an isolated Python backend (FastAPI) and a modern Vite/React frontend.

```text
stock-screener/
├── .env                        # Environment variables (API tokens, MT5 credentials, etc.)
├── .env.example
├── bug-trace.md                # Bug tracking and resolution history
├── README.md                   # Legacy documentation
├── requirements.txt            # Python dependencies (FastAPI, pandas, MetaTrader5, etc.)
├── start.bat                   # Windows 1-click startup script
├── start.sh                    # Mac/Linux/WSL 1-click startup script
|
├── backend/                    # Python API and Trading Logic
│   ├── main.py                 # FastAPI orchestration server (port 8000)
│   ├── logs/                   # Directory for storing live stdout/stderr of local processes
│   |
│   ├── forex_robot/            # MT5 Algorithmic Trading Bots
│   │   ├── robot_trade.py      # Trend-following bot
│   │   ├── robot_trade_sr.py   # Support/Resistance scalper bot
│   │   ├── robot_crt_po3.py    # PO3 Strategy bot
│   │   └── robot_runtime.py    # Shared utilities and logging
│   |
│   └── stock_screener/         # Indonesian IDX Stock Screener
│       ├── screener_id.py      # Core technical and volume analysis engine
│       ├── command.py          # Runner script
│       ├── manager.py          # Universe definition generator
│       ├── stocklist.csv       # Custom defined universe fallbacks
│       └── screener_report.*   # Generated JSON/CSV reports
|
└── web/                        # React Frontend Dashboard
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx            # Entry point
        ├── App.jsx             # Main dashboard UI logic and API calls
        └── index.css           # Premium styling (glassmorphism, animations)
```
