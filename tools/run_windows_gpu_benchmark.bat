@echo off
setlocal EnableExtensions
title Vidlore RTX Benchmark
cd /d "%~dp0\.."

echo  ============================================================
echo   Vidlore - Windows RTX benchmark
echo   CPU vs NVENC encode  +  CPU vs DirectML CLIP  (+ parity)
echo   ~1-2 min. Internet ki zaroorat NAHI (deterministic).
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
%PYEXE% tools\windows_gpu_benchmark.py

echo.
echo  Result file: research\final_release\windows_gpu\BENCHMARK_RESULT.json
echo  Yeh file (ya upar ki table) mujhe bhej dein.
pause
endlocal
