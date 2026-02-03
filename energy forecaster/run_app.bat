@echo off
echo Starting Energy Forecaster App...
echo.

cd /d "%~dp0"

python -m streamlit run app.py

pause
