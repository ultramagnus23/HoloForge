@echo off
REM One-click local runner: from a double-click or `run_local.bat` in a
REM terminal, runs M1/M2/S1/S2 to completion on the local GPU. Safe to
REM re-run after any interruption -- finished jobs are skipped.
cd /d "%~dp0"
python -m experiments.run_local
echo.
echo Done (or stopped -- see messages above). Press any key to close.
pause >nul
