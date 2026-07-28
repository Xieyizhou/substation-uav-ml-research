#!/usr/bin/env bash
# Formal stage 04 runner: active_replan.
#
# Perception can trigger local A* replans, and successful outbound replans may
# replace the remaining waypoint route. Current results should still be reviewed
# for target-switching behavior before claiming final active-replan validation.
set -euo pipefail

source "$(dirname "$0")/common.sh"

echo "===================================="
echo "Experiment 4: active_replan"
echo "Output stage: 04_active_replan"
echo "Perception: enabled"
echo "Risk action: active_local_replan"
echo "Replan: active / route replacement enabled"
echo "===================================="

cleanup_mavsdk

run_managed_flight python scripts/flight/run_task.py run fly_with_replan -- \
  --obstacle-config "$OBSTACLE_CONFIG"

analyze_latest
mark_latest_output "active_replan" "KEEP_LANDMARK__active_local_replan.txt"
update_stage_reports
