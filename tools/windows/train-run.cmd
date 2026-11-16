@echo off
rem Start or resume the long run at below-normal CPU priority. Extra arguments are passed on,
rem for example: train-run.cmd --priority idle
cd /d "%~dp0..\.."
set PYTHONUTF8=1
if not defined OMNIBOB_RUN set OMNIBOB_RUN=runs\long-stage-v03
".venv\Scripts\python.exe" -m omnibob.campaign run "%OMNIBOB_RUN%" %*
echo.
".venv\Scripts\python.exe" -m omnibob.campaign status "%OMNIBOB_RUN%"
pause
