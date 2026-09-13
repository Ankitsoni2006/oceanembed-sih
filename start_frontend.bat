@echo off
title OceanEmbed Frontend Dashboard (SIH26066)
echo ========================================================
echo Starting OceanEmbed Vite Frontend...
echo Target: http://localhost:3000 or http://localhost:5173
echo ========================================================
cd /d "%~dp0"
npm run dev
pause
