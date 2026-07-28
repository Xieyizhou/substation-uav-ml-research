# Experiment Results

This page summarizes the current curated landmark comparison. The source files are:

- `data/sample_outputs/comparison_summary.csv`
- `data/sample_outputs/comparison_summary.md`
- `data/sample_outputs/selected_runs.json`

The full generated output tree remains local under `outputs/`.

Cross-stage comparison outputs are generated intentionally, not after every individual experiment. Running a single experiment updates that stage's summaries but does not overwrite cross-stage comparisons.

There are now two comparison layers:

- **Landmark comparison**: selects one representative run per stage and writes `outputs/comparisons/landmark/comparison_summary.csv`. This is useful for README/demo presentation.
- **Aggregate comparison**: summarizes all valid analyzed runs per stage and writes `outputs/comparisons/aggregate/aggregate_summary.csv` plus `included_runs.csv`. This is the correct repeated-trial output after `run_all_3x.sh`.

A four-row `comparison_summary.csv` is not wrong; it is the landmark comparison, not the repeated-trial statistical summary.

## Four-Stage Experiment Design

The formal experiment pipeline has four stages, matching `outputs/01_*` through `outputs/04_*`:

| Experiment | Output Stage | Launcher | Meaning |
|---:|---|---|---|
| 1 | `01_static_astar` | `run_static_astar.sh` | Baseline A* path following |
| 2 | `02_perception_response` | `run_perception_response.sh` | Perception-enabled risk response using `slow_down` |
| 3 | `03_replan_log_only` | `run_replan_log_only.sh` | Generate and log local replan candidates without route replacement |
| 4 | `04_active_replan` | `run_active_replan.sh` | Active local route replacement |

Older or diagnostic perception `log_only` runs may exist in local outputs, but they are not a separate formal experiment in the current four-stage design.

## Landmark Comparison

| experiment | output stage | mode | run id | flight time (s) | planned path (m) | actual distance (m) | min obstacle distance (m) | risk detections | slow_down events | replan attempts | successful replans | active replacements | status |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `01_static_astar` | baseline A* | `as_20260707_081327` | 149.316 | 68.000 | 67.403 | n/a | 0 | 0 | 0 | 0 | 0 | PASS |
| 2 | `02_perception_response` | perception_response (`slow_down` action) | `as_20260707_082821` | 174.563 | 68.000 | 67.744 | 0.854 | 671 | 296 | 0 | 0 | 0 | PASS |
| 3 | `03_replan_log_only` | replan log-only | `as_20260707_083125` | 149.850 | 68.000 | 67.401 | 0.894 | 547 | 0 | 4 | 4 | 0 | PASS |
| 4 | `04_active_replan` | active replan | `as_20260713_070842` | 222.392 | 82.000 | 67.678 | 0.870 | 867 | 0 | 3 | 3 | 1 | PASS |

## Aggregate Comparison

Generate the aggregate repeated-trial summary after staged runs are available:

```bash
python main.py report compare --mode aggregate --min-runs-per-stage 3
```

This creates:

- `outputs/comparisons/aggregate/aggregate_summary.csv`
- `outputs/comparisons/aggregate/aggregate_summary.md`
- `outputs/comparisons/aggregate/included_runs.csv`

The current aggregate includes 4 static, 3 perception-response, 3 log-only
replan, and 6 active-replan runs. All 16 runs are completed and marked `PASS`;
the comparison records zero safety-buffer violations.

If a stage has fewer than the requested run count, the command still writes the aggregate files and records a warning so the incomplete stage set is visible.

## Interpretation

- Baseline A* completed the route without perception risk logging.
- The selected `perception_response` run produced 296 slow-down events; the three-run aggregate mean is 300.
- Replan log-only found successful local replan candidates four times while leaving the active route unchanged.
- Active replan recorded one route replacement per analyzed run. The latest three eligible runs also pass strict outbound target-sequence validation.
- All selected landmark runs are marked `PASS`, with zero safety-buffer violations in the comparison table.

## Known Issues and Next Steps

- Formal experiment evidence currently covers only `substation_simple_v3`; representative PX4/Gazebo runs are still needed on the other four maps.
- Several runs include near-boundary clearance warnings even when they do not enter raw obstacle footprints or inflated safety buffers.
- Active replanning still needs cross-map and dynamic-obstacle validation before being treated as a general solution.
- Add a system architecture diagram or short demo GIF for GitHub readers.
