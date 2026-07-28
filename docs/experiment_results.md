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

## Live LiDAR v0.1 Evidence

The map-oracle comparison below is retained as a deterministic control. A
separate v0.1 validation used live Gazebo 2D LiDAR with geometric risk and a
rolling local costmap:

| Map | Runs | Targets | Total flight time | Mean flight time | Minimum LiDAR return | Lowest sensor health | Worst P95 age | Worst P95 inference |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex` | 3 | left, center, top_right | 699.932 s | 233.311 s | 1.940 m | 99.879% | 31.581 ms | 2.752 ms |
| `extreme` | 3 | left, center, top_right | 588.433 s | 196.144 s | 2.366 m | 99.821% | 31.638 ms | 3.106 ms |
| **Combined** | **6** | **two maps / three target types** | **1,288.365 s** | **214.727 s** | **1.940 m** | **99.821%** | **31.638 ms** | **3.106 ms** |

All six missions completed with confirmed landing. Analysis found zero physical
collisions, zero inflated-buffer entries, and zero reported LiDAR drops. The
600-second stability capture recorded 18,165 frames at 30.295 Hz with 11.56 ms
P95 frame age; a 300-frame slice passed deterministic replay.

See the [full validation report](results/v0.1_lidar_validation_20260728.md) and
[machine-readable evidence](../data/sample_outputs/v0.1_lidar_validation_20260728.json).
The normalized [per-run CSV](../data/sample_outputs/v0.1_lidar_closed_loop_runs.csv)
is suitable for plotting and statistical analysis.

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

- The legacy four-stage comparison remains a `substation_simple_v3` map-oracle
  control; it must not be combined statistically with the live-LiDAR runs.
- Complex/extreme live-LiDAR coverage is route-diversity evidence from six
  runs, not the planned 30-seed ML benchmark.
- Several runs include near-boundary clearance warnings even when they do not
  enter raw obstacle footprints or inflated safety buffers.
- Active replanning still needs unknown/dynamic-obstacle fault injection and
  cross-map sensor-driven validation before being treated as a general solution.
- No trained LiDAR ML or YOLO result is claimed yet.
