@echo off
setlocal EnableExtensions
title Vidlore Diagnostic
cd /d "%~dp0"

set "PYEXE="
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not defined PYEXE ( where py >nul 2>nul && set "PYEXE=py -3" )
if not defined PYEXE ( where python >nul 2>nul && set "PYEXE=python" )
if not defined PYEXE (
  echo  [X] Python nahi mila. Pehle run-windows.bat chala kar setup karein.
  pause
  exit /b 1
)

echo  Diagnostic chala raha hoon ...
echo.
%PYEXE% -m vidlore.diagnose

echo.
echo  ============================================================
echo   Upar wala poora text + is folder ki "vidlore_diag.txt"
echo   file mujhe bhej dein (screenshot ya file). Usse exact
echo   wajah pata chal jayegi.
echo  ============================================================
echo.
pause
