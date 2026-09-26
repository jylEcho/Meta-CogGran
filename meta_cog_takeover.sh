#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="./external/granulon_Codex"
RUNTIME_DIR="${CODE_ROOT}/runtime/meta_cog_supervisor"
ARCHIVE_DIR="${RUNTIME_DIR}/archive"

mkdir -p "$ARCHIVE_DIR"

if [[ -f "${RUNTIME_DIR}/active_train.log" ]]; then
  mv -f "${RUNTIME_DIR}/active_train.log" "${ARCHIVE_DIR}/active_train_takeover_$(date +%Y%m%d_%H%M%S).log"
fi
if [[ -f "${RUNTIME_DIR}/supervisor.log" ]]; then
  mv -f "${RUNTIME_DIR}/supervisor.log" "${ARCHIVE_DIR}/supervisor_takeover_$(date +%Y%m%d_%H%M%S).log"
fi

pkill -f "${CODE_ROOT}/meta_cog_supervisor.sh" || true
pkill -f "${CODE_ROOT}/pretrain_10_reason_bankV5-fixedV2.py" || true
pkill -f "deepspeed.launcher.launch" || true
sleep 10

rm -f \
  "${RUNTIME_DIR}/primary.exitcode" \
  "${RUNTIME_DIR}/primary.finished_at" \
  "${RUNTIME_DIR}/primary.started_at" \
  "${RUNTIME_DIR}/primary.meta" \
  "${RUNTIME_DIR}/refine.exitcode" \
  "${RUNTIME_DIR}/refine.finished_at" \
  "${RUNTIME_DIR}/refine.started_at" \
  "${RUNTIME_DIR}/refine.meta" \
  "${RUNTIME_DIR}/stage.txt" \
  "${RUNTIME_DIR}/mode.txt" \
  "${RUNTIME_DIR}/trainer.pid"

cd "$CODE_ROOT"
bash launch_meta_cog_supervisor.sh
sleep 8
tail -n 20 "${RUNTIME_DIR}/supervisor.log"
