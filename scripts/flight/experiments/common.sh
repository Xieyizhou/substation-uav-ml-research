#!/usr/bin/env bash
# Shared helpers for the formal staged experiment runners.
#
# Each runner sources this file after PX4 SITL has been started separately.
# The helpers activate the project virtual environment, clean stale MAVSDK
# processes, analyze the latest flight log, mark the corresponding staged
# output folder, and refresh per-stage reports.
set -euo pipefail

if [[ -z "${PROJECT_ROOT+x}" ]]; then
  COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd -- "$COMMON_DIR/../../.." && pwd)"
fi
RUNTIME_DIR="$PROJECT_ROOT/.runtime"
FLIGHT_PID_FILE="$RUNTIME_DIR/flight.pid"

cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv not found at $PROJECT_ROOT/.venv"
  exit 1
fi

if [[ -z "${OBSTACLE_CONFIG:-}" ]]; then
  MAP_SWITCHER="$PROJECT_ROOT/scripts/maps/switch_map.py"
  if [[ -f "$MAP_SWITCHER" ]]; then
    OBSTACLE_CONFIG="$(python "$MAP_SWITCHER" current --field obstacle_config)"
  else
    OBSTACLE_CONFIG="config/substation_obstacles.json"
  fi
fi
echo "Experiment obstacle map: $OBSTACLE_CONFIG"

cleanup_mavsdk() {
  mkdir -p "$RUNTIME_DIR"
  if [[ ! -f "$FLIGHT_PID_FILE" ]]; then
    echo "No project-managed stale flight process found."
    return 0
  fi

  local pid
  pid="$(cat "$FLIGHT_PID_FILE")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "WARNING: removing invalid project flight PID file."
    rm -f "$FLIGHT_PID_FILE"
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Removing stale project flight PID file for exited PID $pid."
    rm -f "$FLIGHT_PID_FILE"
    return 0
  fi

  local command_text
  command_text="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [[ "$command_text" != *"main.py astar fly"* && "$command_text" != *"fly_astar_path.py"* && "$command_text" != *"scripts/flight/run_task.py"* ]]; then
    echo "WARNING: PID $pid no longer looks like this project's flight process; not terminating it."
    rm -f "$FLIGHT_PID_FILE"
    return 0
  fi

  echo "Stopping project-managed stale flight PID $pid..."
  kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: project-managed flight PID $pid did not stop; refusing to kill it forcefully."
    return 1
  fi
  rm -f "$FLIGHT_PID_FILE"
}

run_managed_flight() {
  mkdir -p "$RUNTIME_DIR"
  "$@" &
  local flight_pid=$!
  printf '%s\n' "$flight_pid" > "$FLIGHT_PID_FILE"
  echo "Project-managed flight PID: $flight_pid"

  local status
  if wait "$flight_pid"; then
    status=0
  else
    status=$?
  fi
  rm -f "$FLIGHT_PID_FILE"
  return "$status"
}

analyze_latest() {
  echo
  echo "Analyzing latest A* log..."
  python main.py report analyze --obstacle-config "$OBSTACLE_CONFIG"
}

update_stage_reports() {
  echo
  echo "Updating per-stage experiment summaries..."
  python main.py report summarize
}

update_cross_stage_comparison() {
  echo
  echo "Updating landmark cross-stage comparison..."
  python main.py report compare
}

latest_output_dir() {
  # Stage names map to the canonical output folders documented in
  # docs/EXPERIMENT_PROTOCOL.md.
  local stage="$1"
  local stage_dir
  case "$stage" in
    static_astar) stage_dir="outputs/01_static_astar/runs" ;;
    perception_response) stage_dir="outputs/02_perception_response/runs" ;;
    replan_log_only) stage_dir="outputs/03_replan_log_only/runs" ;;
    active_replan) stage_dir="outputs/04_active_replan/runs" ;;
    *)
      echo "ERROR: unknown output stage: $stage" >&2
      return 1
      ;;
  esac
  find "$stage_dir" -maxdepth 1 -type d -name 'as_[0-9]*' | sort | tail -1
}

mark_latest_output() {
  local stage="$1"
  local marker_name="$2"
  local latest
  latest="$(latest_output_dir "$stage")"

  if [[ -z "$latest" ]]; then
    echo "WARNING: no $stage as_[timestamp] folder found to mark."
    return 0
  fi

  touch "$latest/$marker_name"
  echo
  echo "Marked output:"
  echo "  $latest/$marker_name"
}
