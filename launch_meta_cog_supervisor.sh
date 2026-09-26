#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="./external/granulon_Codex"
RUNTIME_DIR="${CODE_ROOT}/runtime/meta_cog_supervisor"
mkdir -p "$RUNTIME_DIR"

pkill -f "${CODE_ROOT}/meta_cog_supervisor.sh" || true
nohup bash "${CODE_ROOT}/meta_cog_supervisor.sh" >> "${RUNTIME_DIR}/launcher.log" 2>&1 &
echo $! > "${RUNTIME_DIR}/supervisor.pid"
echo "supervisor_started pid=$(cat "${RUNTIME_DIR}/supervisor.pid")"
