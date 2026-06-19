@echo off
setlocal EnableExtensions
title Vidlore GPU Diagnostic
cd /d "%~dp0\.."

echo  ============================================================
echo   Vidlore - Windows GPU acceleration diagnostic
echo   (sirf safe info dikhata hai - koi API key / secret nahi)
echo  ============================================================
echo.

set "PYEXE="
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not defined PYEXE ( where py >nul 2>nul && set "PYEXE=py -3" )
if not defined PYEXE ( where python >nul 2>nul && set "PYEXE=python" )
if not defined PYEXE (
  echo  [X] Python nahi mila. Pehle run-windows.bat chala kar setup karein.
  pause
  exit /b 1
)

set "PYTHONPATH=%CD%"
%PYEXE% tools\windows_gpu_probe.py

echo.
echo  Upar ka poora text mujhe bhej dein (screenshot ya copy/paste).
pause
endlocal
