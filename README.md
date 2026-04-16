# QuantDash: Stock Screener & MT5 Trading Bots Dashboard

Welcome to the Quantitative Trading Dashboard! 
This project has recently been refactored from a collection of standalone Python CLI scripts into a clean, modern web application architecture consisting of a standalone Python FastAPI backend and a beautiful React + Vite frontend.

## Overview
This repository manages two entirely distinct quantitative domains:

1. **Forex Algorithmic Trading Bots (`backend/forex_robot/`)**: Automated MT5 trading scripts designed for various market strategies (`robot_trade.py` for trend following, `robot_trade_sr.py` for Support/Resistance, and `robot_crt_po3.py`).
2. **Indonesian Stock Exchange (IDX) Screener (`backend/stock_screener/`)**: A dedicated daily technical and volume screener pipeline (`screener_id.py`, `command.py`).

Both pipelines are seamlessly managed, monitored, and executed via a unified web dashboard.

## Architecture

The project is thoughtfully split into isolated services:

- **`backend/`**: A native Python FastAPI server that acts as the orchestration layer. It exposes HTTP endpoints to trigger bot execution and screener pipelines asynchronously, whilst streaming their logs.
- **`web/`**: A React/Vite-powered modern UI with premium glassmorphism aesthetics and entrance animations, providing you a 10,000-foot view of your automated trading strategies.

## Requirements

- **OS**: Windows (Required for MetaTrader 5 terminal)
- **Python**: 3.9+ 
- **Node.js**: v16+ (or newer, to run the Vite React app)
- **MetaTrader 5**: Desktop terminal installed with a valid account session

## Installation & Setup

1. **Install Backend Dependencies**:
   Navigate to the repository root:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Install Frontend Dependencies**:
   ```powershell
   cd web
   npm install
   ```

3. **Environment Configuration**:
   Copy the `env` file to set up necessary variables:
   ```powershell
   Copy-Item .env.example .env
   ```
   *Make sure to configure your actual MT5 login, tokens, and thresholds in `.env` before running.*

## How to Run

You can boot up the entire application stack using the provided 1-click startup scripts.

- **For Windows**:
  Run `start.bat`
- **For Mac / Linux / WSL** (Development & Stock Screener Only):
  Run `start.sh`

The bootloader will launch two separate background processes:
1. FastAPI Backend at `http://localhost:8000`
2. React Web Interface at `http://localhost:5173`

Navigate to `http://localhost:5173` in your favorite web browser to see the dashboard. You will now be able to start/stop trading bots dynamically natively pressing buttons on the UI!

## Logs & Execution
- All backend background process logs and direct `stdout/stderr` streams are captured securely inside the `backend/logs/` directory.
- The web app endpoints live-poll these log payloads. They are streamed actively into the embedded terminal-equivalent UI inside the React dashboard, allowing real-time trading monitoring directly from your web browser!

## Disclaimer
This project is experimental and not financial advice. MT5 API wrappers and algorithmic scripts carry high risks. Always validate behavior rigorously on demo accounts first.
