@echo off
REM ============================================================================
REM  Vidlore — Windows package runtime probe
REM  Run this ON THE ACTUAL WINDOWS MACHINE, from inside the Vidlore-Windows
REM  folder, to prove which physical files the app loads + that the removed
REM  motion graphic (process_flow_steps) is absent and registry == 70.
REM  Prints filenames only (no secrets).
REM ============================================================================
setlocal
cd /d "%~dp0\.."
echo Running Vidlore Windows runtime probe from: %CD%
echo.

REM Prefer a local venv if present, else the system python.
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

REM Make the bundled package importable, then run the probe.
set "PYTHONPATH=%CD%"
"%PY%" tools\windows_runtime_path_probe.py

echo.
echo === parity check (this Windows package vs authoritative source) ===
"%PY%" tools\verify_full_dist_parity.py "%CD%"

echo.
echo Done. If REGISTRY count = 70 and process_flow_steps = ABSENT, the Windows
echo package is current and clean. Press any key to close.
pause >nul
endlocal
