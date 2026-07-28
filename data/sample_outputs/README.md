# Sample Outputs

This folder contains small, curated result artifacts that are safe to commit
and useful for reviewing evidence without downloading raw logs.

Current sample set:

- `comparison_summary.csv`: landmark cross-stage comparison.
- `comparison_summary.md`: markdown rendering of the same comparison.
- `selected_runs.json`: metadata for the selected landmark runs.
- `aggregate_summary.csv`: aggregate metrics for all valid runs by stage.
- `aggregate_summary.md`: human-readable aggregate comparison.
- `included_runs.csv`: normalized rows included in aggregate statistics.
- `v0.1_lidar_validation_20260728.json`: machine-readable stability, replay,
  and six-run complex/extreme live-LiDAR evidence.
- `v0.1_lidar_closed_loop_runs.csv`: flat per-run LiDAR metrics for plotting
  and statistical analysis.

The current landmark uses active-replan run `as_20260713_070842`, and the
aggregate covers 16 completed PASS map-oracle runs. The LiDAR JSON is a
separate sensor-driven evidence set and is not pooled into that comparison.
Optional future additions should remain small summaries. Full generated trees remain under
`outputs/` during local work and are ignored by git. Raw telemetry CSV logs
remain under `data/logs/` or `data/raw_logs/` and are also ignored.
