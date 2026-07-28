# GitHub Release Preparation

This repository is maintained as a resume-oriented stable demo. Releases should contain small, reviewable artifacts that let a recruiter or engineer understand the project quickly without downloading raw experiment logs.

## Current Release

Use `v0.2.0` for the unified multi-map resume demo. `v0.1-demo` remains the
original public demo release.

## Recommended Release Assets

Include these files in a GitHub release:

- `comparison_summary.csv`: current landmark comparison table.
- `comparison_summary.md`: markdown rendering of the same comparison.
- `selected_runs.json`: metadata for the selected landmark runs.
- `aggregate_summary.csv`: all valid analyzed runs summarized by stage.
- `aggregate_summary.md`: human-readable aggregate results.
- `included_runs.csv`: runs included in aggregate statistics.
- Optional preview image: small route plot or comparison screenshot suitable for the README/release page.
- Optional demo video: compact black minimalist 35-45 second MP4 assembled with `scripts/media/make_demo_video.py`.

Do not include raw telemetry logs, full generated `outputs/` run trees, simulator logs, cache folders, or large unedited videos.

## Suggested Package Layout

```text
sample_outputs.zip
|-- comparison_summary.csv
|-- comparison_summary.md
|-- selected_runs.json
|-- aggregate_summary.csv
|-- aggregate_summary.md
|-- included_runs.csv
`-- preview.png
```

The preview image and demo video are optional until release-safe media are prepared.

## Release Checklist

- Release tag is `v0.2.0`.
- README opens with the portfolio/demo summary and current key results.
- `PROJECT_STATUS.md` states that this is the stable resume-oriented demo version.
- `data/sample_outputs/` contains only small curated artifacts.
- `outputs/`, raw logs, simulator logs, videos, and temporary files remain ignored by git.
- Any demo GIF or video is attached as a release asset unless it is intentionally small enough to keep in the repository.
- Generated release videos stay under `release_assets/` and remain ignored by git.
- The latest three active-replan runs pass target-switching validation.
- Cross-map and dynamic-obstacle research follow-up is documented as future work.

## Current Release Notes Template

```text
Unified multi-map portfolio demo release.

Includes:
- Four-stage PX4/Gazebo UAV planning experiment pipeline.
- Curated landmark comparison across static A*, perception response, log-only replanning, and active replanning.
- Small sample outputs for quick review: comparison_summary.csv, comparison_summary.md, and selected_runs.json.
- Documentation for running the demo and regenerating comparisons.
- Five coordinated Gazebo/A* maps and 25 validated target presets.
- Repeated active-replan target-switching validation.

Not included:
- Raw telemetry logs.
- Full generated output trees.
- Large videos or simulator logs.
- Future active-replanning research work.
```
