@echo off
rem Open the small window that switches the training between full speed, lower levels and pause.
cd /d "%~dp0..\.."
set PYTHONUTF8=1
if not defined OMNIBOB_RUN set OMNIBOB_RUN=runs\long-stage-v03
start "" ".venv\Scripts\pythonw.exe" -m omnibob.panel "%OMNIBOB_RUN%"
