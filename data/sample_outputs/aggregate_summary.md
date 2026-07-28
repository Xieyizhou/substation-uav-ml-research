# Aggregate Cross-Stage Comparison

Generated: `2026-07-20T05:17:09.685929+00:00`

This summary uses all valid analyzed runs found under the four official stage folders.

Minimum requested runs per stage: `3`

| stage | run count | completed | pass | mean flight time (s) | std flight time (s) | min flight time (s) | max flight time (s) | mean planned path (m) | mean actual distance (m) | mean min obstacle distance (m) | total buffer violations | mean risk detections | mean slow_down events | mean replan attempts | mean successful replans | mean active replacements |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_static_astar | 4 | 4 | 4 | 149.918 | 0.493 | 149.316 | 150.450 | 68.000 | 68.048 | n/a | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 02_perception_response | 3 | 3 | 3 | 175.393 | 1.960 | 173.985 | 177.631 | 68.000 | 67.500 | 0.872 | 0 | 674.333 | 300.000 | 0.000 | 0.000 | 0.000 |
| 03_replan_log_only | 3 | 3 | 3 | 149.383 | 0.728 | 148.545 | 149.850 | 68.000 | 67.505 | 0.897 | 0 | 545.333 | 0.000 | 4.000 | 4.000 | 0.000 |
| 04_active_replan | 6 | 6 | 6 | 223.078 | 2.091 | 220.785 | 225.878 | 82.667 | 68.414 | 0.839 | 0 | 868.333 | 0.000 | 3.000 | 3.000 | 1.000 |

CSV output: `outputs/comparisons/aggregate/aggregate_summary.csv`
Included runs: `outputs/comparisons/aggregate/included_runs.csv`
