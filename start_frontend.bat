@echo off
title OceanEmbed Frontend Dashboard (SIH26066)
echo ========================================================
echo Starting OceanEmbed Vite Frontend...
echo Target: http://localhost:3000 (LAN: http://^<this-PC-LAN-IP^>:3000)
echo ========================================================
cd /d "%~dp0"
npm run dev
pause
