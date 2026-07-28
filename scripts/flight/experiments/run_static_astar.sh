#!/usr/bin/env bash
# Formal stage 01 runner: static_astar.
#
# Perception and local replanning are disabled. This run is the control case
# for comparing later perception and replanning behavior.
set -euo pipefail

source "$(dirname "$0")/common.sh"

echo "===================================="
echo "Experiment 1: static_astar"
echo "Output stage: 01_static_astar"
echo "Perception: disabled"
echo "Risk action: none"
echo "Replan: disabled"
echo "===================================="

cleanup_mavsdk

run_managed_flight python scripts/flight/run_task.py run fly_round_trip -- \
  --obstacle-config "$OBSTACLE_CONFIG"

analyze_latest
mark_latest_output "static_astar" "KEEP_BASELINE__astar_only_perception_disabled.txt"
update_stage_reports
