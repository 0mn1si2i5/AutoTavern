@echo off
rem One time on Windows: Python 3.12, Rust 1.98.1 and the TavernGym extension, all inside this checkout.
rem Needs uv and Visual Studio Build Tools with the "Desktop development with C++" workload.
setlocal
cd /d "%~dp0..\.."
set PYTHONUTF8=1
set "CARGO_HOME=%CD%\.local-tools\cargo"
set "RUSTUP_HOME=%CD%\.local-tools\rustup"
set "UV_PYTHON_INSTALL_DIR=%CD%\.local-tools\python"
set "PATH=%CARGO_HOME%\bin;%PATH%"
set CARGO_BUILD_JOBS=4
where uv >nul 2>nul || (echo uv is required: winget install astral-sh.uv & exit /b 1)
if not exist .local-tools mkdir .local-tools
if not exist "%CARGO_HOME%\bin\rustup.exe" (
  curl.exe --proto =https --tlsv1.2 -fsSL https://win.rustup.rs/x86_64 -o .local-tools\rustup-init.exe || exit /b 1
  .local-tools\rustup-init.exe -y --no-modify-path --profile minimal --default-toolchain 1.98.1 || exit /b 1
)
rustup toolchain install 1.98.1 --profile minimal || exit /b 1
rustc --version || exit /b 1
if not exist .venv\Scripts\python.exe (uv venv --python 3.12 .venv || exit /b 1)
.venv\Scripts\python.exe -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version" || exit /b 1
uv pip install --python .venv\Scripts\python.exe -r requirements-omnibob.lock || exit /b 1
uv pip install --python .venv\Scripts\python.exe -r requirements-omnibob-mps.lock || exit /b 1
uv pip install --python .venv\Scripts\python.exe --no-build-isolation -e ".[test]" || exit /b 1
.venv\Scripts\python.exe -m omnibob check || exit /b 1
echo Setup finished.
