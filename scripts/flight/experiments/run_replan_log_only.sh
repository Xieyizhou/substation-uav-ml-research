#!/usr/bin/env bash
# Formal stage 03 runner: replan_log_only.
#
# Perception can trigger local A* replan attempts, but `replan-mode log_only`
# keeps the active route unchanged. The output shows whether replans would have
# been available without changing flight behavior.
set -euo pipefail

source "$(dirname "$0")/common.sh"

echo "===================================="
echo "Experiment 3: replan_log_only"
echo "Output stage: 03_replan_log_only"
echo "Perception: enabled"
echo "Risk action: log_only_local_replan"
echo "Replan: log_only / no active route replacement"
echo "===================================="
cleanup_mavsdk

run_managed_flight python scripts/flight/run_task.py run replan_log_only -- \
  --obstacle-config "$OBSTACLE_CONFIG"

analyze_latest
mark_latest_output "replan_log_only" "KEEP_CONTROL__log_only_local_replan_no_route_replacement.txt"
update_stage_reports
