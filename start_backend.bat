@echo off
title OceanEmbed FastAPI Backend (SIH26066)
echo ========================================================
echo Starting OceanEmbed FastAPI Backend Server...
echo Project: SIH26066
echo Target: http://127.0.0.1:8000 and Network LAN IP
echo ========================================================
cd /d "%~dp0"
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
pause
