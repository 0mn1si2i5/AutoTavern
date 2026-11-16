@echo off
rem One time: freeze inputs for the long run in runs\long-stage-v03 (refuses if it exists).
cd /d "%~dp0..\.."
set PYTHONUTF8=1
if not defined OMNIBOB_RUN set OMNIBOB_RUN=runs\long-stage-v03
".venv\Scripts\python.exe" -m omnibob.campaign init "%OMNIBOB_RUN%" --updates 19000 --config configs\long-stage-v03.json --initialize runs\releases\omnibob-v0.3-20261101\policy\policy.pt --rules configs\arena-v2.json --device cpu --checkpoint-every 10 --archive-every 250 --max-rss-gib 4 --segment-hours 24
pause
