@echo off
setlocal EnableExtensions EnableDelayedExpansion
title ClipStudio Portal (Windows)
cd /d "%~dp0"

REM  CLIP visual model = the animated / video-game / toy footage filter. If the models ship next to
REM  this launcher (models\clip), point the tool at them so it never falls back to an EMPTY user cache
REM  (empty cache = filter OFF = cartoon/game footage leaks in; the pipeline now refuses in that case).
if exist "%~dp0models\clip\clip_vision.onnx" set "VIDLORE_CLIP_DIR=%~dp0models\clip"

echo ==================================================
echo   ClipStudio - Windows launcher
echo   (pehli baar setup karega, phir portal chalega)
echo ==================================================
echo.

REM --- 1) Python 3.10-3.12 dhoondo --------------------------------------------------------------
REM     ClipStudio ke ML deps (onnxruntime/numpy/opencv/av/ctranslate2) ki Windows wheels SIRF
REM     3.10-3.12 ke liye hain. 3.13/3.14 par pip source compile karne ki koshish kar ke fail
REM     hota hai. Detection ek subroutine (:trypy) mein hai - parenthesized FOR-block mein NAHI -
REM     kyunki version-check ke python code mein ( ) hote hain jo block ko jaldi band kar dete.
set "PYLAUNCH="
call :trypy py -3.12
call :trypy py -3.11
call :trypy py -3.10
call :trypy python
call :trypy py -3
if not defined PYLAUNCH (
  echo  [X] Sahi Python nahi mila. ClipStudio ko Python 3.10 - 3.12 chahiye.
  echo      Aapke PC par shayad 3.13 / 3.14 hai - us par onnxruntime/numpy/opencv install nahi hote.
  echo.
  echo      KARNA YEH HAI:
  echo        1^) Python 3.12 install karein:
  echo             https://www.python.org/downloads/release/python-3129/
  echo           ^(install karte waqt "Add Python to PATH" zaroor TICK karein^)
  echo        2^) Phir yeh file dobara double-click karein.
  echo.
  pause
  exit /b 1
)
echo  [*] Python mila: !PYLAUNCH!

REM --- 2) .env (API keys) check - config.py ise khud load karta hai -----------------------------
if not exist ".env" (
  echo  [!] .env file nahi mili ^(ismein DeepSeek aur baaki API keys hoti hain^).
  echo      ".env.example" ko copy kar ke ".env" banayein aur apni keys daalein.
  echo      Bina keys ke brain/AI kaam nahi karega.
  echo.
)

REM --- 3) virtual environment: galat-version ka purana .venv hata kar naya banao ----------------
set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" call :checkvenv
if not exist "%VENV_PY%" (
  echo  [*] Pehli baar setup: .venv bana raha hoon ^(!PYLAUNCH!^) ...
  !PYLAUNCH! -m venv .venv
  if errorlevel 1 ( echo  [X] venv banane mein masla aaya. & pause & exit /b 1 )
)

REM --- 4) dependencies install: dono files EK saath (pip ek hi baar consistent solve kare,
REM        warna base latest numpy/onnxruntime laata aur clipstudio reqs use downgrade karte) -----
echo  [*] Dependencies install kar raha hoon ^(pehli baar 5-15 min - ML libraries bhaari hain^) ...
"%VENV_PY%" -m pip install --upgrade pip >nul 2>nul
"%VENV_PY%" -m pip install -r requirements.txt -r requirements-clipstudio.txt
if errorlevel 1 (
  echo  [X] Dependencies install nahi huin.
  echo      Sabse aam wajah: Python 3.13/3.14 ^(ML wheels nahi hote^). Python 3.12 use karein - upar dekhein.
  echo      Warna internet connection check kar ke dobara chalayein.
  pause
  exit /b 1
)

REM --- 5) environment -------------------------------------------------------------------------
set "PYTHONPATH=%CD%;%PYTHONPATH%"
if not defined VIDLORE_CLIPSTUDIO_PORT set "VIDLORE_CLIPSTUDIO_PORT=5151"

REM --- 6) us port par pehle se chalu portal band karo (address-in-use se bachne ke liye) -------
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr "LISTENING" ^| findstr ":%VIDLORE_CLIPSTUDIO_PORT% "') do taskkill /F /PID %%P >nul 2>nul

echo.
echo ==================================================
echo   ClipStudio portal -^> http://127.0.0.1:%VIDLORE_CLIPSTUDIO_PORT%
echo   Browser khud khul jayega. Yeh window khula rehne dein.
echo   Band karne ke liye yeh window band karein (ya Ctrl+C).
echo ==================================================
echo.

REM --- 7) server up hote hi browser kholo ------------------------------------------------------
start "" cmd /c "for /l %%i in (1,1,30) do (curl -s -m 1 http://127.0.0.1:%VIDLORE_CLIPSTUDIO_PORT%/ >nul 2>nul && (start "" http://127.0.0.1:%VIDLORE_CLIPSTUDIO_PORT% & exit) || (timeout /t 1 >nul))"

REM --- 8) portal chalao (Ctrl+C ya window band = stop) -----------------------------------------
"%VENV_PY%" -m vidlore.clipstudio.web

echo.
echo  Portal band ho gaya.
pause
exit /b 0

REM ============================================================================================
REM  Subroutines (sirf CALL se chalte hain - main flow upar exit /b 0 par khatam ho jata hai)
REM ============================================================================================

REM  :trypy <python-command...>  -> agar woh python 3.10/3.11/3.12 hai aur abhi tak koi nahi mila,
REM                                 to PYLAUNCH set kar do. (%* = poora command, e.g. "py -3.12")
:trypy
if defined PYLAUNCH goto :eof
%* -c "import sys;sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11),(3,12)) else 1)" >nul 2>nul
if not errorlevel 1 set "PYLAUNCH=%*"
goto :eof

REM  :checkvenv  -> agar maujooda .venv ka python 3.10-3.12 NAHI hai to use hata do (naya banega)
:checkvenv
"%VENV_PY%" -c "import sys;sys.exit(0 if sys.version_info[:2] in ((3,10),(3,11),(3,12)) else 1)" >nul 2>nul
if not errorlevel 1 goto :eof
echo  [!] Purana .venv galat Python ^(3.13/3.14^) se bana tha - hata kar naya banata hoon...
rmdir /S /Q ".venv"
goto :eof
