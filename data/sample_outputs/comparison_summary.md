# Cross-Experiment Evaluation

Generated: `2026-07-20T05:17:09.684178+00:00`

Selected landmark runs are grouped by canonical stage. Missing metrics are shown as `n/a`.

## 01_static_astar

| run_id | experiment type | completed/failed | flight time (s) | planned path (m) | actual distance (m) | min obstacle distance (m) | buffer violations | risk detections | slow_down events | replan attempts | successful replans | active replacements | final status | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| as_20260707_081327 | astar_only_perception_disabled | completed | 149.316 | 68.000 | 67.403 | n/a | 0 | 0 | 0 | 0 | 0 | 0 | PASS | Near-boundary clearance warning: actual trajectory came within about 0.86 m of an obstacle cell center. |

## 02_perception_response

| run_id | experiment type | completed/failed | flight time (s) | planned path (m) | actual distance (m) | min obstacle distance (m) | buffer violations | risk detections | slow_down events | replan attempts | successful replans | active replacements | final status | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| as_20260707_082821 | perception_slow_down | completed | 174.563 | 68.000 | 67.744 | 0.854 | 0 | 671 | 296 | 0 | 0 | 0 | PASS | Near-boundary clearance warning: actual trajectory came within about 0.82 m of an obstacle cell center. |

## 03_replan_log_only

| run_id | experiment type | completed/failed | flight time (s) | planned path (m) | actual distance (m) | min obstacle distance (m) | buffer violations | risk detections | slow_down events | replan attempts | successful replans | active replacements | final status | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| as_20260707_083125 | local_replan_log_only | completed | 149.850 | 68.000 | 67.401 | 0.894 | 0 | 547 | 0 | 4 | 4 | 0 | PASS | Near-boundary clearance warning: actual trajectory came within about 0.84 m of an obstacle cell center. |

## 04_active_replan

| run_id | experiment type | completed/failed | flight time (s) | planned path (m) | actual distance (m) | min obstacle distance (m) | buffer violations | risk detections | slow_down events | replan attempts | successful replans | active replacements | final status | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| as_20260713_070842 | active_local_replan | completed | 222.392 | 82.000 | 67.678 | 0.870 | 0 | 867 | 0 | 3 | 3 | 1 | PASS | Near-boundary clearance warning: actual trajectory came within about 0.78 m of an obstacle cell center. |

## Interpretation

- This is an intentional cross-stage comparison; ordinary stage summaries remain separate.
- The active replan landmark recorded route replacement, so it changed the outbound route rather than only logging a possible replan.

## Selected Runs

- `KEEP_BASELINE__astar_only_perception_disabled.txt` -> `outputs/01_static_astar/runs/as_20260707_081327`
- `KEEP_COMPARISON__perception_slow_down.txt` -> `outputs/02_perception_response/runs/as_20260707_082821`
- `KEEP_CONTROL__log_only_local_replan_no_route_replacement.txt` -> `outputs/03_replan_log_only/runs/as_20260707_083125`
- `KEEP_LANDMARK__active_local_replan.txt` -> `outputs/04_active_replan/runs/as_20260713_070842`

CSV output: `outputs/comparisons/landmark/comparison_summary.csv`
