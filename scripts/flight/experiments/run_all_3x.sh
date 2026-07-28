#!/usr/bin/env bash
set -uo pipefail

# Run all four official experiment stages multiple times.
#
# Usage:
#   bash scripts/flight/experiments/run_all_3x.sh
#   bash scripts/flight/experiments/run_all_3x.sh 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRIALS="${1:-3}"

EXPERIMENTS=(
  "run_static_astar.sh"
  "run_perception_response.sh"
  "run_replan_log_only.sh"
  "run_active_replan.sh"
)

echo "====================================="
echo "Running all official experiment stages"
echo "Project root: ${PROJECT_ROOT}"
echo "Trials per experiment: ${TRIALS}"
echo "Total planned runs: $((TRIALS * ${#EXPERIMENTS[@]}))"
echo "====================================="

FAILED_RUNS=0

cd "${PROJECT_ROOT}" || exit 1

for trial in $(seq 1 "${TRIALS}"); do
  echo ""
  echo "====================================="
  echo "Trial ${trial}/${TRIALS}"
  echo "====================================="

  for experiment in "${EXPERIMENTS[@]}"; do
    EXPERIMENT_PATH="${SCRIPT_DIR}/${experiment}"

    echo ""
    echo "-------------------------------------"
    echo "Running ${experiment} | Trial ${trial}/${TRIALS}"
    echo "-------------------------------------"

    if [ ! -f "${EXPERIMENT_PATH}" ]; then
      echo "FAILED: ${experiment} does not exist at ${EXPERIMENT_PATH}"
      FAILED_RUNS=$((FAILED_RUNS + 1))
      continue
    fi

    bash "${EXPERIMENT_PATH}"
    STATUS=$?

    if [ "${STATUS}" -ne 0 ]; then
      echo "FAILED: ${experiment} | Trial ${trial}/${TRIALS} | exit code ${STATUS}"
      FAILED_RUNS=$((FAILED_RUNS + 1))
    else
      echo "PASSED: ${experiment} | Trial ${trial}/${TRIALS}"
    fi

    sleep 3
  done
done

echo ""
echo "====================================="
echo "Finished all experiment runs"
echo "Failed runs: ${FAILED_RUNS}"
echo "====================================="

if [ "${FAILED_RUNS}" -ne 0 ]; then
  exit 1
fi

echo ""
echo "All experiment runs completed successfully."

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

echo ""
echo "Generating landmark and aggregate cross-stage comparisons..."
python main.py report compare --mode both --min-runs-per-stage "${TRIALS}"
COMPARE_STATUS=$?
if [ "${COMPARE_STATUS}" -ne 0 ]; then
  echo "FAILED: cross-stage comparison generation exited with code ${COMPARE_STATUS}"
  exit "${COMPARE_STATUS}"
fi
