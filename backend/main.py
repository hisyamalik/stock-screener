from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os
import json
import logging

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

@app.post("/api/forex/{bot_type}/stop")
def stop_bot(bot_type: str):
    if bot_type in bot_processes and bot_processes[bot_type].poll() is None:
        bot_processes[bot_type].terminate()
        return {"status": "stopped"}
        
    return {"status": "not_running"}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
