@echo off
rem After setup: the tests that cover training, resume, pause, low power and the run lock.
cd /d "%~dp0..\.."
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests\test_training.py tests\test_run_control.py tests\test_campaign.py
pause
