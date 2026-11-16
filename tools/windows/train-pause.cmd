@echo off
rem Switch the running training to pause mode (takes effect within a few seconds).
cd /d "%~dp0..\.."
set PYTHONUTF8=1
if not defined OMNIBOB_RUN set OMNIBOB_RUN=runs\long-stage-v03
".venv\Scripts\python.exe" -m omnibob.campaign mode "%OMNIBOB_RUN%" pause %*
timeout /t 5
