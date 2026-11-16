#!/bin/zsh
set -eu
cd "$(dirname "$0")/../.."
exec .venv/bin/python -m tools.real_client.advisor live --panel
