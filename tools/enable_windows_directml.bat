@echo off
setlocal EnableExtensions
title Vidlore - Enable DirectML (RTX) CLIP
cd /d "%~dp0\.."

echo  ============================================================
echo   Vidlore - GPU CLIP (DirectML) ENABLE
echo   onnxruntime (CPU) ko onnxruntime-directml se badal deta hai
echo   taa-ke visual-relevance gate aapke NVIDIA RTX par chale.
echo   (Agar GPU fail ho to gate khud CPU par safe chala jayega.)
echo  ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo  [X] .venv nahi mila. Pehle run-windows.bat chala kar setup karein.
  pause
  exit /b 1
)
set "VPY=.venv\Scripts\python.exe"

echo  [*] Purana onnxruntime (CPU build) hata raha hoon ...
"%VPY%" -m pip uninstall -y onnxruntime onnxruntime-directml >nul 2>nul

echo  [*] onnxruntime-directml==1.19.2 install (Python 3.9 ke liye, internet chahiye) ...
"%VPY%" -m pip install "onnxruntime-directml==1.19.2"
if errorlevel 1 (
  echo.
  echo  [X] install fail. Internet check karein. CPU CLIP phir bhi theek chalega.
  pause
  exit /b 1
)

echo.
echo  [*] Verify providers ...
set "PYTHONPATH=%CD%"
"%VPY%" -c "import onnxruntime as o; ps=o.get_available_providers(); print('   providers:', ps); print('   DirectML ready:', 'DmlExecutionProvider' in ps)"

echo.
echo  NOTE: agar aap dobara run-windows.bat se setup chalayein (jo requirements
echo  reinstall karta hai), to yeh script DUBARA chala lein - warna CPU onnxruntime
echo  wapas aa sakta hai. Phir tools\check_windows_gpu_acceleration.bat se tasdeeq.
pause
endlocal
