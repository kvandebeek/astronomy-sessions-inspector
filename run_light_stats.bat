@echo off
rem Windows counterpart of run_light_stats.sh: scans this telescope's folder (the one this
rem script sits in), calibrating against the masters in darks\ and the flats under each
rem session, and writes fits_light_data.json for run_review_server.bat to read.
rem Extra arguments (e.g. --rescan, --solve) are passed through.
setlocal
cd /d "%~dp0"
set "PY=python"
if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
"%PY%" fits_light_stats.py . --json-cache fits_light_data.json --output fits_light_report.html --darks darks %*
