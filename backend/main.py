from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os
import json
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import csv

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep track of subprocesses
bot_processes = {}
screener_process = None

class BotAction(BaseModel):
    action: str

@app.get("/api/ping")
def ping():
    return {"status": "ok"}

# ====================
# STOCK SCREENER API
# ====================
@app.get("/api/screener/status")
def get_screener_status():
    global screener_process
    status = "idle"
    if screener_process and screener_process.poll() is None:
        status = "running"
    return {"status": status}

@app.get("/api/screener/report")
def get_screener_report():
    report_path = os.path.join(os.path.dirname(__file__), "stock_screener", "screener_report.json")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report not generated yet")

@app.post("/api/screener/run")
def run_screener():
    global screener_process
    if screener_process and screener_process.poll() is None:
        return {"status": "already_running"}
    
    screener_dir = os.path.join(os.path.dirname(__file__), "stock_screener")
    # Run in background
    log_file = open(os.path.join(LOG_DIR, "screener.log"), "w")
    screener_process = subprocess.Popen(["python", "command.py"], cwd=screener_dir, stdout=log_file, stderr=subprocess.STDOUT)
    return {"status": "started"}

@app.get("/api/screener/logs")
def get_screener_logs():
    log_path = os.path.join(LOG_DIR, "screener.log")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return {"logs": f.read()}
    except FileNotFoundError:
        return {"logs": "No logs available yet."}

@app.get("/api/screener/chart/{symbol}")
def get_screener_chart(symbol: str, period: str = "6mo"):
    try:
        ticker = symbol if symbol.endswith(".JK") else f"{symbol}.JK"
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, axis=1, level=-1)
            else:
                df.columns = df.columns.get_level_values(0)

        df = df.loc[:, ~df.columns.duplicated()].copy()
        df = df.dropna()
        
        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Not enough data")
        
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI14"] = (100 - (100 / (1 + rs))).fillna(50)
        
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal
        
        df["MACD"] = macd
        df["MACD_Signal"] = signal
        df["MACD_Hist"] = macd_hist
        
        df = df.reset_index()
        # Ensure we handle DatetimeIndex correctly, it might be called 'Date' or 'Datetime'
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
        
        df = df.dropna(subset=["SMA50"])
        # Replace inf/-inf with nan, then nan with None for JSON serialization
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        
        # Rename date_col to "Date" if it wasn't already
        if date_col != "Date":
            df = df.rename(columns={date_col: "Date"})
            
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ====================
# FOREX ROBOTS API
# ====================

VALID_BOTS = {
    "trend": "robot_trade.py",
    "sr": "robot_trade_sr.py",
    "po3": "robot_crt_po3.py"
}

@app.get("/api/forex/status")
def get_forex_status():
    status = {}
    for bot, script in VALID_BOTS.items():
        if bot in bot_processes and bot_processes[bot].poll() is None:
            status[bot] = "running"
        else:
            status[bot] = "stopped"
    return status

@app.post("/api/forex/{bot_type}/start")
def start_bot(bot_type: str):
    if bot_type not in VALID_BOTS:
        raise HTTPException(status_code=404, detail="Bot not found")
        
    if bot_type in bot_processes and bot_processes[bot_type].poll() is None:
        return {"status": "already_running"}
        
    script = VALID_BOTS[bot_type]
    bot_dir = os.path.join(os.path.dirname(__file__), "forex_robot")
    
    log_file = open(os.path.join(LOG_DIR, f"{bot_type}.log"), "w")
    p = subprocess.Popen(["python", script], cwd=bot_dir, stdout=log_file, stderr=subprocess.STDOUT)
    bot_processes[bot_type] = p
    
    return {"status": "started"}

@app.get("/api/forex/{bot_type}/logs")
def get_forex_logs(bot_type: str):
    log_path = os.path.join(LOG_DIR, f"{bot_type}.log")
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return {"logs": f.read()}
    except FileNotFoundError:
        return {"logs": "No logs available yet."}

@app.get("/api/forex/{bot_type}/performance")
def get_forex_performance(bot_type: str):
    if bot_type not in VALID_BOTS:
        raise HTTPException(status_code=404, detail="Bot not found")
        
    tsf_path = os.path.join(os.path.dirname(__file__), "forex_robot", "data", f"{bot_type}.tsf")
    if not os.path.exists(tsf_path):
        return []
        
    data = []
    try:
        with open(tsf_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "Timestamp": row.get("Timestamp", ""),
                    "Balance": float(row.get("Balance", 0)),
                    "Equity": float(row.get("Equity", 0)),
                    "Profit": float(row.get("Profit", 0))
                })
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forex/{bot_type}/stop")
def stop_bot(bot_type: str):
    if bot_type in bot_processes and bot_processes[bot_type].poll() is None:
        bot_processes[bot_type].terminate()
        return {"status": "stopped"}
        
    return {"status": "not_running"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
