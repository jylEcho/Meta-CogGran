#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="./external/granulon_Codex"
RUNTIME_DIR="${CODE_ROOT}/runtime/meta_cog_supervisor"
PRIMARY_SCRIPT="${CODE_ROOT}/train_meta_cog_primary.sh"
FALLBACK_SCRIPT="${CODE_ROOT}/train_meta_cog_fallback.sh"
REFINE_SCRIPT="${CODE_ROOT}/train_meta_cog_refine.sh"
RUNNER_SCRIPT="${CODE_ROOT}/meta_cog_stage_runner.sh"
SUMMARIZER_SCRIPT="${CODE_ROOT}/meta_cog_summarize.py"
CHECK_INTERVAL=60
ZOMBIE_CHECK_INTERVAL=1800
HOURLY_CHECK_INTERVAL=3600
mkdir -p "$RUNTIME_DIR"

PID_FILE="${RUNTIME_DIR}/trainer.pid"
MODE_FILE="${RUNTIME_DIR}/mode.txt"
STAGE_FILE="${RUNTIME_DIR}/stage.txt"
LOG_FILE="${RUNTIME_DIR}/supervisor.log"
ACTIVE_LOG="${RUNTIME_DIR}/active_train.log"
LATEST_REPORT="${RUNTIME_DIR}/latest_report.md"
ARCHIVE_DIR="${RUNTIME_DIR}/archive"
REPORT_DIR="${RUNTIME_DIR}/reports"
mkdir -p "$ARCHIVE_DIR" "$REPORT_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$LOG_FILE"
}

get_mode() {
  if [[ -f "$MODE_FILE" ]]; then
    cat "$MODE_FILE"
  else
    echo "primary"
  fi
}

get_stage() {
  if [[ -f "$STAGE_FILE" ]]; then
    cat "$STAGE_FILE"
  else
    echo "primary"
  fi
}

write_stage() {
  echo "$1" > "$STAGE_FILE"
}

stage_output_dir() {
  local stage="$1"
  case "$stage" in
    primary) echo "${CODE_ROOT}/outputs/reason_trained_meta_cog_v2_actioneq" ;;
    refine) echo "${CODE_ROOT}/outputs/reason_trained_meta_cog_v2_refine" ;;
    *) echo "${CODE_ROOT}/outputs/unknown_stage" ;;
  esac
}

next_stage() {
  local stage="$1"
  case "$stage" in
    primary) echo "refine" ;;
    refine) echo "done" ;;
    *) echo "done" ;;
  esac
}

script_for_stage_mode() {
  local stage="$1"
  local mode="$2"
  if [[ "$stage" == "primary" && "$mode" == "fallback" ]]; then
    echo "$FALLBACK_SCRIPT"
  elif [[ "$stage" == "refine" ]]; then
    echo "$REFINE_SCRIPT"
  else
    echo "$PRIMARY_SCRIPT"
  fi
}

stage_exit_code_file() {
  local stage="$1"
  echo "${RUNTIME_DIR}/${stage}.exitcode"
}

stage_done_file() {
  local stage="$1"
  echo "${RUNTIME_DIR}/${stage}.finished_at"
}

stage_started_file() {
  local stage="$1"
  echo "${RUNTIME_DIR}/${stage}.started_at"
}

write_mode() {
  echo "$1" > "$MODE_FILE"
}

archive_active_log() {
  local suffix="$1"
  if [[ -f "$ACTIVE_LOG" ]] && [[ -s "$ACTIVE_LOG" ]]; then
    mv "$ACTIVE_LOG" "${ARCHIVE_DIR}/active_train_${suffix}.log"
  fi
}

start_mode() {
  local stage="$1"
  local mode="$1"
  local script
  mode="$2"
  script="$(script_for_stage_mode "$stage" "$mode")"
  : > "$ACTIVE_LOG"
  rm -f "$(stage_exit_code_file "$stage")" "$(stage_done_file "$stage")"
  write_stage "$stage"
  write_mode "$mode"
  log "starting stage=${stage} mode=${mode} script=${script}"
  nohup bash "$RUNNER_SCRIPT" "$stage" "$mode" "$script" >> "$ACTIVE_LOG" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 5
  log "started pid=$(cat "$PID_FILE")"
}

stop_active() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
      log "stopping pid=${pid}"
      pkill -TERM -P "$pid" || true
      kill -TERM "$pid" || true
      sleep 10
      pkill -KILL -P "$pid" || true
      kill -KILL "$pid" || true
    fi
  fi
}

training_alive() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null
}

has_zombie_descendant() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -z "${pid}" ]] && return 1
  ps -eo pid=,ppid=,stat=,cmd= | awk -v root="$pid" '
    {
      pid[$1] = 1
      ppid[$1] = $2
      stat[$1] = $3
    }
    END {
      for (id in pid) {
        cur = id
        while (cur in ppid && ppid[cur] != "" && ppid[cur] != cur) {
          if (ppid[cur] == root) {
            if (stat[id] ~ /Z/) {
              found = 1
            }
            break
          }
          cur = ppid[cur]
        }
      }
      exit(found ? 0 : 1)
    }'
}

ensure_started() {
  local stage
  stage="$(get_stage)"
  if [[ "$stage" == "done" ]]; then
    return 0
  fi
  if ! training_alive; then
    start_mode "$stage" "$(get_mode)"
  fi
}

switch_to_fallback() {
  local stage
  stage="$(get_stage)"
  log "zombie detected, switching to fallback strategy for stage=${stage}"
  stop_active
  start_mode "$stage" "fallback"
}

run_summary() {
  local stage="$1"
  local mode="$2"
  local status="$3"
  python "$SUMMARIZER_SCRIPT" \
    --stage "$stage" \
    --mode "$mode" \
    --status "$status" \
    --log-file "$ACTIVE_LOG" \
    --output-dir "$(stage_output_dir "$stage")" \
    --report-dir "$REPORT_DIR" \
    --latest-report "$LATEST_REPORT" || true
}

handle_stage_completion() {
  local stage="$1"
  local mode="$2"
  local exit_code_file
  exit_code_file="$(stage_exit_code_file "$stage")"
  [[ -f "$exit_code_file" ]] || return 1
  local exit_code
  exit_code="$(cat "$exit_code_file" 2>/dev/null || echo 1)"

  if [[ "$exit_code" == "0" ]]; then
    log "stage completed successfully stage=${stage} mode=${mode}"
    run_summary "$stage" "$mode" "completed"
    archive_active_log "${stage}_completed_$(date +%Y%m%d_%H%M%S)"
    local upcoming
    upcoming="$(next_stage "$stage")"
    if [[ "$upcoming" == "done" ]]; then
      write_stage "done"
      log "pipeline completed; latest report=${LATEST_REPORT}"
      return 0
    fi
    log "advancing pipeline from stage=${stage} to stage=${upcoming}"
    start_mode "$upcoming" "primary"
    return 0
  fi

  log "stage exited with failure stage=${stage} mode=${mode} exit_code=${exit_code}"
  run_summary "$stage" "$mode" "failed"
  archive_active_log "${stage}_failed_$(date +%Y%m%d_%H%M%S)"
  if [[ "$stage" == "primary" && "$mode" != "fallback" ]]; then
    log "switching to fallback after primary failure"
    start_mode "$stage" "fallback"
  else
    log "restarting current stage after failure"
    start_mode "$stage" "$mode"
  fi
}

hourly_status_check() {
  local stage mode
  stage="$(get_stage)"
  mode="$(get_mode)"
  [[ "$stage" == "done" ]] && return 0
  run_summary "$stage" "$mode" "running"
  log "hourly-check ok stage=${stage} mode=${mode} report=${LATEST_REPORT}"
}

main() {
  log "supervisor boot"
  ensure_started
  local elapsed=0
  local hourly_elapsed=0
  while true; do
    sleep "$CHECK_INTERVAL"
    elapsed=$((elapsed + CHECK_INTERVAL))
    hourly_elapsed=$((hourly_elapsed + CHECK_INTERVAL))
    if [[ "$(get_stage)" == "done" ]]; then
      log "pipeline already completed; supervisor idle"
      sleep "$HOURLY_CHECK_INTERVAL"
      continue
    fi
    if ! training_alive; then
      if handle_stage_completion "$(get_stage)" "$(get_mode)"; then
        elapsed=0
        hourly_elapsed=0
        continue
      fi
      log "training process missing without exit marker, restarting current stage"
      start_mode "$(get_stage)" "$(get_mode)"
      elapsed=0
      hourly_elapsed=0
      continue
    fi
    if (( elapsed >= ZOMBIE_CHECK_INTERVAL )); then
      elapsed=0
      if has_zombie_descendant; then
        switch_to_fallback
        hourly_elapsed=0
        continue
      fi
      log "zombie-check ok stage=$(get_stage) mode=$(get_mode) pid=$(cat "$PID_FILE")"
    fi
    if (( hourly_elapsed >= HOURLY_CHECK_INTERVAL )); then
      hourly_elapsed=0
      hourly_status_check
    else
      log "heartbeat ok stage=$(get_stage) mode=$(get_mode) pid=$(cat "$PID_FILE")"
    fi
  done
}

main "$@"
