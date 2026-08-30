@echo off
echo =======================================================
echo     SIH26077 IMD AI-Driven Early Warning System
echo =======================================================
echo.
echo Which mode do you want to run?
echo   [1] Full Dashboard  (React frontend + FastAPI backend)
echo   [2] Risk Map Viewer (Streamlit — simpler, no backend needed)
echo.
set /p CHOICE="Enter 1 or 2: "

if "%CHOICE%"=="2" goto STREAMLIT

:FULL_DASHBOARD
echo.
echo [0/3] Verifying Python dependencies...
pip install scipy fastapi uvicorn -q
echo      [OK] Python deps verified.
echo.
echo [1/3] Starting PyTorch FastAPI Backend (Port 8000)...
start "AI Backend" cmd /k "title AI Backend && python api.py"

echo [2/3] Starting React Vite Frontend...
start "React Frontend" cmd /k "title React Frontend && cd frontend\TheMorphs-SIH2026 && npm run dev -- --open"

echo.
echo All servers started! Browser will open automatically.
goto END

:STREAMLIT
echo.
echo [0/2] Verifying Python dependencies...
pip install streamlit folium streamlit-folium scipy -q
echo      [OK] Python deps verified.
echo.
echo [1/2] Launching Streamlit Risk Map Viewer (Port 8501)...
start "Streamlit App" cmd /k "title Streamlit Risk Map && streamlit run streamlit_app.py"
echo.
echo Streamlit will open automatically in your browser at http://localhost:8501

:END
echo.
echo (Keep the terminal window(s) open to keep the app running)
