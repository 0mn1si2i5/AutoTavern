@echo off
cd /d "%~dp0..\.."
set PYTHONUTF8=1
if not defined OMNIBOB_RUN set OMNIBOB_RUN=runs\long-stage-v03
".venv\Scripts\python.exe" -m omnibob.campaign status "%OMNIBOB_RUN%" %*
pause
