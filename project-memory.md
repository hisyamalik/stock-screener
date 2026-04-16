# Project Memory

## Context & Background
This repository originally contained a mixture of standalone Python CLI scripts tailored towards MetaTrader 5 (Forex) algorithmic trading, and data analysis using `yfinance` for the Indonesian Stock Exchange (IDX). 

## What We Accomplished in the Last Milestone
1. **Architectural Redesign & Separation of Concerns**
   - We decoupled the monolithic directory structure into two heavily isolated domains: `forex_robot` and `stock_screener`.
   - We successfully enclosed them in a `backend/` directory so they wouldn't bleed dependencies onto each other, allowing them to scale independently.

2. **Web Dashboard Integration (FastAPI + React)**
   - **Backend API (`backend/main.py`)**: We wrapped the CLI tasks in a `FastAPI` instance. It manages subprocesses, letting us start/stop MT5 bots asynchronously and fetch up-to-date IDX screener JSON files.
   - **Frontend UI (`web/`)**: We initialized a `Vite` + `React` web application. It consumes the FastAPI endpoints to render a unified "Main App."
   - **Aesthetics**: We injected a highly premium, modern styling using generic Vanilla CSS (glassmorphism overlays, smooth entrance animations, dark mode gradients).

3. **Ease of Use Enhancements**
   - Built 1-click bootloader scripts (`start.bat` and `start.sh`) configured to simultaneously launch both the Uvicorn python server and the Node Vite server in separate terminal streams.

4. **Live Execution Output & Log Terminal UI**
   - Implemented real-time stdout/stderr redirection for `subprocess.Popen` in the FastAPI backend. All stdout is routed to a structured `logs/` directory.
   - Built endpoints to stream these logs to the frontend.
   - Added embedded, real-time logging screens to both `ScreenerPanel` and `ForexPanel` in the React dashboard, paired with explicit button "loading" visual feedback states to fix earlier bugs where the UI would appear dead.

## Next Steps / Future Work
- Implementing configuration editing (modifying `.env` variables) directly from the React dashboard.

## Bug Tracking
- A centralized bug-tracking file (`bug-trace.md`) was created to properly isolate bugs and trace their debugging workflows and resolutions.
- The recent `uvicorn` startup bug, as well as unresponsive UI buttons in the React app (caused by an uncaptured synchronous thread delay + no feedback states), have been successfully diagnosed and resolved in `bug-trace.md`.