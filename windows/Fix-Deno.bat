@echo off
setlocal EnableExtensions
title Deno install (HD downloads ke liye)
cd /d "%~dp0.."

REM  Ek hi kaam karti hai: Deno install karti hai, jiske baghair HD (720-1080p) downloads
REM  band rehte hain aur SAARI footage ~360p aati hai (koi error nahi aata - isliye yeh
REM  file banai gayi hai). Normally run-windows.bat khud yeh kar leta hai; yeh us surat ke
REM  liye hai jab wahan install fail ho jaye.

echo ==================================================
echo   Deno install - HD ^(720-1080p^) downloads ke liye
echo ==================================================
echo.

set "DENO_EXE=%USERPROFILE%\.deno\bin\deno.exe"

if exist "%DENO_EXE%" (
  echo  [*] Deno pehle se mojood hai:
  "%DENO_EXE%" --version
  echo.
  echo  Ab sirf ClipStudio dobara chalayein ^(Start-ClipStudio.bat^).
  echo.
  pause
  exit /b 0
)

echo  [*] Official release ZIP download kar raha hoon ...
echo      ^(sirf aapke user ke liye - admin/password ki zaroorat nahi^)
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { New-Item -ItemType Directory -Force -Path \"$env:USERPROFILE\.deno\bin\" | Out-Null; $z=Join-Path $env:TEMP 'deno-win-x64.zip'; Write-Host '     downloading ...'; Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip' -OutFile $z; Write-Host '     extracting ...'; Expand-Archive -Path $z -DestinationPath \"$env:USERPROFILE\.deno\bin\" -Force; Remove-Item $z -Force } catch { Write-Host ('     FAIL: ' + $_.Exception.Message) }"

if not exist "%DENO_EXE%" (
  echo.
  echo  [*] Doosra tareeqa aazma raha hoon ^(official installer^) ...
  echo.
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://deno.land/install.ps1 | iex"
)

echo.
if exist "%DENO_EXE%" (
  echo  ==================================================
  echo   [OK] Deno lag gaya:
  "%DENO_EXE%" --version
  echo.
  echo   AB YEH KAREIN:  Start-ClipStudio.bat dobara chalayein.
  echo   Pre-flight mein "[OK] HD download" aana chahiye.
  echo  ==================================================
) else (
  echo  ==================================================
  echo   [X] Deno install nahi hua.
  echo.
  echo   Sabse aam wajah: internet/proxy ne GitHub block kiya.
  echo   Upar jo FAIL wali line aayi hai, uska screenshot bhej dein.
  echo.
  echo   Iske baghair ClipStudio chalega, magar saari footage ~360p
  echo   aayegi ^(HD band rahega^).
  echo  ==================================================
)
echo.
pause
