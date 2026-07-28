#!/usr/bin/env bash
# Formal stage 02 runner: perception_response.
#
# The route remains the original A* path; only speed is reduced during
# warning/danger risk. This isolates behavior change from replanning.
set -euo pipefail

source "$(dirname "$0")/common.sh"

echo "===================================="
echo "Experiment 2: perception_response"
echo "Output stage: 02_perception_response"
echo "Perception: enabled"
echo "Risk action: slow_down"
echo "Replan: disabled"
echo "===================================="

cleanup_mavsdk

run_managed_flight python scripts/flight/run_task.py run fly_with_perception -- \
  --obstacle-config "$OBSTACLE_CONFIG"

analyze_latest
mark_latest_output "perception_response" "KEEP_COMPARISON__perception_slow_down.txt"
update_stage_reports
