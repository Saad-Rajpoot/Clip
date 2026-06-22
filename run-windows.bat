@echo off
setlocal EnableExtensions EnableDelayedExpansion
title ClipStudio Portal (Windows)
cd /d "%~dp0"

echo ==================================================
echo   ClipStudio - Windows launcher
echo   (pehli baar setup karega, phir portal chalega)
echo ==================================================
echo.

REM --- 1) Python dhoondo (py launcher behtar, warna python) ---
set "PYLAUNCH="
where py >nul 2>nul && set "PYLAUNCH=py -3"
if not defined PYLAUNCH ( where python >nul 2>nul && set "PYLAUNCH=python" )
if not defined PYLAUNCH (
  echo  [X] Python nahi mila.
  echo      Python 3.10+ install karein:  https://www.python.org/downloads/
  echo      Install karte waqt "Add Python to PATH" zaroor tick karein, phir yeh file dobara chalayein.
  echo.
  pause
  exit /b 1
)

REM --- 2) .env (API keys) check — config.py ise khud load karta hai ---
if not exist ".env" (
  echo  [!] .env file nahi mili ^(ismein DeepSeek aur baaki API keys hoti hain^).
  echo      ".env.example" ko copy kar ke ".env" banayein aur apni keys daalein.
  echo      Bina keys ke brain/AI kaam nahi karega.
  echo.
)

REM --- 3) pehli baar: virtual environment bana lo ---
if not exist ".venv\Scripts\python.exe" (
  echo  [*] Pehli baar setup: .venv bana raha hoon ...
  %PYLAUNCH% -m venv .venv
  if errorlevel 1 ( echo  [X] venv banane mein masla aaya. & pause & exit /b 1 )
)
set "VENV_PY=.venv\Scripts\python.exe"

REM --- 4) dependencies install / update (pehli baar thoda waqt) ---
echo  [*] Dependencies check kar raha hoon ...
"%VENV_PY%" -m pip install --upgrade pip >nul 2>nul
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo  [X] Dependencies install nahi huin. Internet connection check kar ke dobara chalayein.
  pause
  exit /b 1
)

REM --- 5) environment ---
set "PYTHONPATH=%CD%;%PYTHONPATH%"
if not defined VIDLORE_CLIPSTUDIO_PORT set "VIDLORE_CLIPSTUDIO_PORT=5151"

REM --- 6) us port par pehle se chalu portal band karo (address-in-use se bachne ke liye) ---
REM     sirf usi port ki LISTENING line ka PID — kisi aur process ko haath nahi lagta.
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr "LISTENING" ^| findstr ":%VIDLORE_CLIPSTUDIO_PORT% "') do taskkill /F /PID %%P >nul 2>nul

echo.
echo ==================================================
echo   ClipStudio portal -^> http://127.0.0.1:%VIDLORE_CLIPSTUDIO_PORT%
echo   Browser khud khul jayega. Yeh window khula rehne dein.
echo   Band karne ke liye yeh window band karein (ya Ctrl+C).
echo ==================================================
echo.

REM --- 7) server up hote hi browser kholo (background; curl na ho to manually URL kholein) ---
start "" cmd /c "for /l %%i in (1,1,30) do (curl -s -m 1 http://127.0.0.1:%VIDLORE_CLIPSTUDIO_PORT%/ >nul 2>nul && (start "" http://127.0.0.1:%VIDLORE_CLIPSTUDIO_PORT% & exit) || (timeout /t 1 >nul))"

REM --- 8) portal chalao (Ctrl+C ya window band = stop) ---
"%VENV_PY%" -m vidlore.clipstudio.web

echo.
echo  Portal band ho gaya.
pause
