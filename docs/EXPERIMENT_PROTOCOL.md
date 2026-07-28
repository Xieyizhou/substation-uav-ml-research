# Experiment Protocol

This document contains the detailed commands and conventions for the four staged PX4 SITL A* substation experiments.

## Experiment Stages

- `01_static_astar`: Experiment 1, baseline A* route planning and flight without perception response.
- `02_perception_response`: perception_response stage. This is the second experiment; the current risk action is `slow_down` and local replanning is disabled.
- `03_replan_log_only`: Experiment 3, local replanning evaluated in log-only mode; the active route should not be replaced.
- `04_active_replan`: Experiment 4, local replanning enabled and allowed to replace the remaining outbound route.
- `comparisons`: intentional cross-stage comparison outputs.
- `archive`: old, failed, or deprecated outputs that should not be mixed with active experiment results.

`scripts/flight/experiments/common.sh` is shared runner infrastructure, not an experiment stage.

Do not compare outputs from different stages by default. Use the comparison tool for intentional cross-stage evaluation.

## Output Folder Conventions

```text
outputs/
  README.md
  01_static_astar/
    runs/
    previews/
    summaries/
  02_perception_response/
    runs/
    previews/
    summaries/
  03_replan_log_only/
    runs/
    previews/
    summaries/
  04_active_replan/
    runs/
    previews/
    summaries/
  comparisons/
  archive/
```

Per-run analysis outputs should be written as:

```text
outputs/<stage>/runs/as_YYYYMMDD_HHMMSS/
```

Summary outputs are written as:

```text
outputs/<stage>/summaries/experiment_summary.csv
outputs/<stage>/summaries/experiment_summary.md
outputs/<stage>/summaries/experiment_evaluation.csv
outputs/<stage>/summaries/experiment_evaluation.md
```

Comparison outputs are written under:

```text
outputs/comparisons/landmark/
outputs/comparisons/aggregate/
```

## Setup

Run the following commands from the repository root.

Install dependencies once:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Start PX4 SITL with the substation world in a separate terminal:

```bash
python main.py map start
```

Activate the project virtual environment before running experiment commands:

```bash
source .venv/bin/activate
```

## Runner Commands

Use the unified experiment commands:

```bash
python main.py experiment run static
python main.py experiment run perception
python main.py experiment run replan-log
python main.py experiment run active-replan
```

Expected four-experiment runner mapping:

- Experiment 1: `run_static_astar.sh` -> `outputs/01_static_astar/`
- Experiment 2: `run_perception_response.sh` -> `outputs/02_perception_response/`
- Experiment 3: `run_replan_log_only.sh` -> `outputs/03_replan_log_only/`
- Experiment 4: `run_active_replan.sh` -> `outputs/04_active_replan/`

Each runner flies, analyzes the latest log, marks the latest staged output, and regenerates per-stage summaries. Individual runners do not refresh landmark or aggregate cross-stage comparisons.

## Manual Preview, Fly, Analyze

Static A* preview:

```bash
python main.py astar preview \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5
```

Static A* flight:

```bash
python main.py astar fly \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 0.8 \
  --return-speed-scale 0.7 \
  --waypoint-acceptance 0.3
```

Analyze the latest A* log:

```bash
python main.py report analyze \
  --obstacle-config config/substation_obstacles.json
```

Generate deeper diagnostic plots only when needed:

```bash
python main.py report analyze \
  --obstacle-config config/substation_obstacles.json \
  --debug-plots
```

## Manual Perception-Response Runs

Diagnostic perception `log_only` run, not a separate formal experiment:

```bash
python main.py astar fly \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 1.0 \
  --return-speed-scale 0.8 \
  --waypoint-acceptance 0.4 \
  --enable-perception \
  --risk-action log_only
```

Manual `perception_response` run using the current `slow_down` action:

```bash
python main.py astar fly \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 1.0 \
  --return-speed-scale 0.8 \
  --waypoint-acceptance 0.4 \
  --enable-perception \
  --risk-action slow_down
```

## Manual Local Replan Runs

Offline local replan preview:

```bash
python main.py check replan
```

Expected preview output:

```text
outputs/03_replan_log_only/previews/replan_preview/
```

Experiment 3 log-only local replan flight:

```bash
python main.py astar fly \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 1.0 \
  --return-speed-scale 0.8 \
  --waypoint-acceptance 0.4 \
  --enable-perception \
  --risk-action log_only \
  --enable-local-replan \
  --replan-mode log_only \
  --replan-risk-level danger
```

Experiment 4 active local replan flight:

```bash
python main.py astar fly \
  --obstacle-config config/substation_obstacles.json \
  --return-home \
  --altitude 1.5 \
  --max-speed 1.0 \
  --return-speed-scale 0.8 \
  --waypoint-acceptance 0.4 \
  --enable-perception \
  --risk-action log_only \
  --enable-local-replan \
  --replan-mode active \
  --replan-risk-level danger
```

## Summary Commands

Regenerate all stage summaries and evaluation tables:

```bash
python main.py report summarize
```

This scans only staged run folders:

```text
outputs/<stage>/runs/as_*/
```

## Comparison Commands

Generate both cross-stage comparison layers:

```bash
python main.py report compare --mode both --min-runs-per-stage 1
```

Collect real PX4/Gazebo telemetry coverage by map:

```bash
python main.py report collect-maps --min-runs-per-map 1
```

Use `--strict` in automation when missing map coverage should fail the
command. The collector counts only analyzed telemetry runs; route previews do
not satisfy flight-data coverage.

Generate only the landmark README/demo comparison:

```bash
python main.py report compare --mode landmark
```

Generate only the repeated-trial aggregate comparison:

```bash
python main.py report compare --mode aggregate --min-runs-per-stage 3
```

The landmark comparison looks for selected marker files inside staged run folders and writes outputs only when all four required stages have at least one selected analyzed run. If any required stage is missing, the command prints a skip message, lists missing stages, writes `comparison_status.md`, and leaves existing landmark summaries unchanged.

The aggregate comparison uses all valid analyzed runs inside the four staged run folders. If a stage has fewer than the requested run count, the command prints a warning and records the incomplete stage set in `aggregate_summary.md`.

```text
outputs/comparisons/landmark/comparison_summary.csv
outputs/comparisons/landmark/comparison_summary.md
outputs/comparisons/landmark/selected_runs.json
outputs/comparisons/landmark/comparison_status.md
outputs/comparisons/aggregate/aggregate_summary.csv
outputs/comparisons/aggregate/aggregate_summary.md
outputs/comparisons/aggregate/included_runs.csv
outputs/comparisons/maps/map_run_inventory.csv
outputs/comparisons/maps/map_run_inventory.md
```

## Complex And Extreme Map Collection

For each map, start PX4/Gazebo in one terminal and run at least a static
baseline plus an active-replan trial in another terminal:

```bash
python main.py map start complex
python main.py experiment run static
python main.py experiment run active-replan
```

Repeat with `extreme`. Stop the PX4 launcher before switching maps. After all
runs, regenerate the reports:

```bash
python main.py report summarize
python main.py report compare --mode both --min-runs-per-stage 3
python main.py report collect-maps --min-runs-per-map 1 --strict
```

Every counted run must have a generated analysis folder containing a manifest
or summary. A successful route preview proves planner feasibility but is not
reported as PX4 flight evidence.

## Interpretation Guide

- Baseline vs `slow_down`: check whether slow-down reduces safety-buffer violations or improves minimum obstacle distance, and compare the added flight time.
- Diagnostic perception `log_only` runs can still be useful controls, but they are not one of the four formal experiment stages.
- Log-only replan vs active replan: log-only shows whether a local replan would have been available; active replan shows whether the route was actually replaced.
- Active replan success requires more than a successful A* attempt. Check active route replacement count, final status, safety-buffer violations, and obstacle clearance.
- Missing metrics should remain blank or `unavailable`; do not invent values when logs or manifest fields are absent.

## Active Replan Target Validation

Each analyzed active-replan run records a target-switching status of `PASS`,
`FAIL`, `UNAVAILABLE`, or `NOT_APPLICABLE` in `summary.md` and `manifest.json`.
`PASS` means an outbound route replacement was recorded, the pre-replan target
and first RWP target were observable, no original `WP<number>` target returned
during the rest of outbound flight, the distinct RWP numbers were contiguous,
the original outbound goal was reached, and the mission completed without an
error/danger landing or incomplete log. Return-home targets are excluded.

Repeated telemetry samples are collapsed before checking the sequence because
the logger normally records the same active target several times. The first
observed replacement target may be greater than `RWP01`: the runtime
intentionally skips initial replanned waypoints already inside waypoint
tolerance. From that first observed RWP onward, every distinct RWP number must
increase by exactly one.

Original-goal validation requires the runtime to enter `goal_hover`, the final
outbound RWP target coordinates to match the original goal target coordinates
logged in that phase, and a goal-hover sample to have horizontal and vertical
errors strictly below the current 0.4 m waypoint acceptance thresholds. A
successful local A* attempt or route replacement does not prove goal arrival.

After analyzing the trials, validate the latest three eligible runs with:

```bash
python main.py report validate-active --latest 3
```

The latest three eligible PX4/Gazebo runs currently pass this validation:
`as_20260713_065534`, `as_20260713_070149`, and `as_20260713_070842`.
Their replacement sequences are contiguous (`RWP01` through `RWP06`), no old
outbound waypoint target reappears, and every run reaches the original goal.
Run the command again after collecting new trials so the claim always follows
the latest evidence.
