#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="./external/granulon_Codex"
RUNTIME_DIR="${CODE_ROOT}/runtime/meta_cog_supervisor"
mkdir -p "$RUNTIME_DIR"

STAGE="${1:?stage required}"
MODE="${2:?mode required}"
TARGET_SCRIPT="${3:?target script required}"

STARTED_FILE="${RUNTIME_DIR}/${STAGE}.started_at"
FINISHED_FILE="${RUNTIME_DIR}/${STAGE}.finished_at"
EXITCODE_FILE="${RUNTIME_DIR}/${STAGE}.exitcode"
META_FILE="${RUNTIME_DIR}/${STAGE}.meta"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "$(timestamp)" > "$STARTED_FILE"
rm -f "$FINISHED_FILE" "$EXITCODE_FILE"
cat > "$META_FILE" <<EOF
stage=${STAGE}
mode=${MODE}
script=${TARGET_SCRIPT}
started_at=$(timestamp)
EOF

set +e
bash "$TARGET_SCRIPT"
EXIT_CODE=$?
set -e

echo "$EXIT_CODE" > "$EXITCODE_FILE"
echo "$(timestamp)" > "$FINISHED_FILE"
echo "[stage-runner] stage=${STAGE} mode=${MODE} exit_code=${EXIT_CODE} finished_at=$(timestamp)"
exit "$EXIT_CODE"
