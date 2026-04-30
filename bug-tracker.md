# Bug Tracing

## Bug 1
- **Description**: `run start.bat -> 'uvicorn' is not recognized as an internal or external command`
- **Location**: `start.bat` and `start.sh`
- **Root Cause**: The script was running `uvicorn` directly, which requires the `uvicorn` executable to be explicitly registered in the environment `PATH`.
- **Resolution**: Changed the execution command from `uvicorn main:app --reload` to `python -m uvicorn main:app --reload`. Calling it as a Python module ensures it resolves correctly.
- **Status**: **RESOLVED**

## Bug 2
- **Description**: `stock_screener/screener_id.py -> there is no response after click the button. Improve the UI to show the loading state (or show log console running) and the result.)`
- **Location**: `backend/main.py` and `web/src/App.jsx`
- **Root Cause**: The backend was running processes silently using `subprocess.Popen` without piping standard output or standard error to any accessible interface. Consequently, when the button is pressed on the UI, it provides no real-time feedback because there are no logs to poll.
- **Resolution**:
    1. Introduced a `logs` directory in the backend structure to capture output in local log files.
    2. Modified `subprocess.Popen` to redirect `stdout` and `stderr` to `logs/screener.log`.
    3. Created a new FastAPI endpoint (`/api/screener/logs`) to read and return the logs.
    4. Updated the React UI `ScreenerPanel` to show immediate visual feedback (`"Scanning..."`) when triggered.
    5. Added a live polling terminal console below the button to stream screener logs via API.
- **Status**: **RESOLVED**

## Bug 3
- **Description**: `forex_robot/robot_trade.py -> there is no response after click the button. Improve the UI to show the loading state (or show log console running) and the result.)`
- **Location**: `backend/main.py` and `web/src/App.jsx`
- **Root Cause**: Similar to the screener, trading robots were executed silently in the background, making it impossible to see logs. In addition, when executing the API to start the robot, the UI didn't show a `Loading` state, making the button appear unresponsive.
- **Resolution**:
    1. Modified `backend/main.py` forex endpoints to redirect process `stdout` and `stderr` to individual robot log files (`logs/{bot_type}.log`).
    2. Created a new endpoint (`/api/forex/{bot_type}/logs`) for querying bot logs.
    3. Updated the UI `ForexPanel` in `App.jsx` to show an immediate `"Starting..."` state prior to the process becoming active.
    4. Integrated local logging console windows embedded in each bot's display card to feed live logs.
- **Status**: **RESOLVED**

## Bug 4
- **Description**: `StockChart cannot load the chart. Error: 1 Failed download: ['TOOL.JK.JK']: YFPricesMissingError(...)`
- **Location**: `backend/main.py`
- **Root Cause**: The API endpoint `/api/screener/chart/{symbol}` unconditionally appended `.JK` to the `symbol` parameter (`ticker = f"{symbol}.JK"`). However, the `screener_report.json` data and the frontend already include the `.JK` suffix (e.g., `TOOL.JK`), resulting in the `yfinance` download attempting to fetch `TOOL.JK.JK`.
- **Resolution**: Updated `get_screener_chart` in `backend/main.py` to conditionally append `.JK` only if it is not already present: `ticker = symbol if symbol.endswith(".JK") else f"{symbol}.JK"`.
- **Status**: **RESOLVED**
