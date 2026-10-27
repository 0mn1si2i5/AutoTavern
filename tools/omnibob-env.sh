#!/usr/bin/env bash
# Source from the repository root. All Rust state stays inside this project.
export CARGO_HOME="$PWD/.local-tools/cargo"
export RUSTUP_HOME="$PWD/.local-tools/rustup"
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_BUILD_JOBS=4
export TAVERNGYM_GUARD_DIR="$PWD/runs/guards"
