@echo off
rem Windows counterpart of run_review_server.sh. Scans first when there is no
rem fits_light_data.json yet; --rescan / --full-rescan (anywhere) scan again before serving.
rem Every other argument goes to fits_review_server.py (e.g. -p 9000 --open).
setlocal
cd /d "%~dp0"
set "PY=python"
if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
set "RESCAN="
set "ARGS="
:loop
if "%~1"=="" goto done
if /i "%~1"=="--rescan" (set "RESCAN=--rescan") else if /i "%~1"=="--full-rescan" (set "RESCAN=--full-rescan") else set ARGS=%ARGS% %1
shift
goto loop
:done
if defined RESCAN (
    call run_light_stats.bat %RESCAN%
) else if not exist "fits_light_data.json" (
    call run_light_stats.bat
)
"%PY%" fits_review_server.py --json "%CD%\fits_light_data.json" --root . %ARGS%
