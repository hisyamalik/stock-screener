#!/bin/bash
echo "========================================="
echo "   STARTING QUANTS DASHBOARD"
echo "========================================="

echo "\n[1/2] Starting FastAPI Backend API..."
(cd backend && python -m uvicorn main:app --reload) &
BACKEND_PID=$!

echo "[2/2] Starting React Web Interface..."
(cd web && npm run dev) &
FRONTEND_PID=$!

echo "\nBoth servers are running!"
echo "Press Ctrl+C to stop both servers."

# Wait for process exit, then kill background processes
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM
wait
