# Sample Outputs

This folder contains small, curated result artifacts that are safe to commit to GitHub and useful for quick portfolio review.

Current sample set:

- `comparison_summary.csv`: landmark cross-stage comparison.
- `comparison_summary.md`: markdown rendering of the same comparison.
- `selected_runs.json`: metadata for the selected landmark runs.
- `aggregate_summary.csv`: aggregate metrics for all valid runs by stage.
- `aggregate_summary.md`: human-readable aggregate comparison.
- `included_runs.csv`: normalized rows included in aggregate statistics.

The current landmark uses active-replan run `as_20260713_070842`, and the
aggregate covers 16 completed PASS runs. Optional future additions should be
small preview images only. Full generated experiment trees remain under
`outputs/` during local work and are ignored by git. Raw telemetry CSV logs
remain under `data/logs/` or `data/raw_logs/` and are also ignored.
