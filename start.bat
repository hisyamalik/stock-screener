@echo off
echo =========================================
echo    STARTING QUANTS DASHBOARD
echo =========================================

echo.
echo [1/2] Starting FastAPI Backend API...
start "Backend API" cmd /k "cd backend && python -m uvicorn main:app --reload"

echo [2/2] Starting React Web Interface...
start "Web Frontend" cmd /k "cd web && npm run dev"

echo.
echo Both servers are launching in separate windows!
echo Once the Vite server is ready, check the terminal for the localhost URL (usually http://localhost:5173).
echo You can safely close this launcher window.
pause
