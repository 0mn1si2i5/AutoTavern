#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'uv is required; install it separately, then rerun.' >&2; exit 1; }
mkdir -p .local-tools runs/setup
source tools/omnibob-env.sh
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
  curl --proto '=https' --tlsv1.2 -fsS https://sh.rustup.rs -o .local-tools/rustup-init.sh
  sh .local-tools/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain 1.98.1
fi
# Run serially: concurrent rustup downloads can race on the same cache files.
rustup toolchain install 1.98.1 --profile minimal
rustc --version
if [[ ! -x .venv/bin/python ]]; then
  UV_PYTHON_INSTALL_DIR="$PWD/.local-tools/python" uv venv --python 3.12 .venv
fi
.venv/bin/python -c 'import platform,sys; assert sys.version_info[:2] == (3,12), "use Python 3.12 for this lock"; print(platform.machine(),sys.version)'
uv pip install --python .venv/bin/python -r requirements-omnibob.lock
uv pip install --python .venv/bin/python --no-build-isolation -e '.[test]'
if [[ "${1:-}" == "--mps" ]]; then
  uv pip install --python .venv/bin/python -r requirements-omnibob-mps.lock
fi
.venv/bin/python -m omnibob check
.venv/bin/python -m pytest -q
