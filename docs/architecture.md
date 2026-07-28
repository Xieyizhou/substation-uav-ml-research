# Architecture

This project is a simulation-first UAV autonomy pipeline. The core source code
lives in `src/`; scripts under `scripts/` are runnable wrappers and experiment
launchers.

## System Flow

```text
PX4/Gazebo simulation
  -> Unified command layer (`main.py` / `src/cli/`)
  -> Flight entry (`src/flight/fly_astar_path.py`)
  -> Mission lifecycle (`src/flight/mission_lifecycle.py`)
  -> A* global planner (`src/planner/`)
  -> Local NED waypoint execution
  -> Telemetry CSV logging (`src/logging/flight_logger.py`)
  -> Simulated perception/risk state (`src/perception/`)
  -> Risk action
       - Experiment 2 perception_response: reduce speed during warning/danger risk
       - Experiment 3 replan log-only: test replan availability without replacing route
       - Experiment 4 active route replacement: replace remaining outbound waypoints
  -> Per-run analysis and staged summaries (`src/logging/`)
  -> Curated sample comparison (`data/sample_outputs/comparison_summary.csv`)
```

## Major Modules

- `src/planner/`: grid A* search, path simplification, obstacle-map conversion.
- `src/perception/`: map-based simulated obstacle detector and structured risk state.
- `src/flight/fly_astar_path.py`: thin CLI and backward-compatible exports.
- `src/flight/mavsdk_preflight.py`: connection and position readiness.
- `src/flight/waypoint_executor.py`: Offboard waypoint and route execution.
- `src/flight/landing_manager.py`: normal and failsafe landing confirmation.
- `src/flight/perception_response.py`: simulated detection state and detector setup.
- `src/flight/replanning_controller.py`: local A* and active route replacement.
- `src/flight/telemetry_runtime.py`: MAVSDK subscriptions and CSV logging.
- `src/flight/mission_lifecycle.py`: connection, supervision, status, and cleanup.
- `src/flight/flight_config.py`: backward-compatible configuration exports.
- `src/flight/flight_cli.py`, `flight_defaults.py`,
  `flight_planner_config.py`, `flight_runtime_config.py`: argument parsing,
  shared defaults, map loading, safety checks, perception, and replan settings.
- `src/logging/analyze_astar_log.py`: thin per-run analysis entry and report orchestration.
- `src/logging/analysis_*.py`: run classification, warnings, perception, and replan summaries.
- `src/logging/summarize_experiments.py`: thin per-stage summary entry.
- `src/logging/summary_*.py`: normalized value collection and CSV/Markdown output.
- `src/logging/plotting.py`: backward-compatible plotting exports.
- `src/logging/plot_timeseries.py`, `plot_trajectory.py`,
  `plot_diagnostics.py`: focused time-series, trajectory, and diagnostic plots.
- `src/logging/report_writer.py`: backward-compatible report exports.
- `src/logging/report_sections.py`, `report_summary.py`, `report_files.py`:
  Markdown sections, human-readable summaries, CSV metadata, and manifests.
- `src/logging/compare_experiment_sets.py`: thin comparison CLI and
  backward-compatible exports.
- `src/logging/comparison_*.py`: run discovery, landmark output, aggregation,
  and shared comparison schemas.
- `scripts/flight/experiments/`: repeatable staged experiment runners.
- `config/substation_obstacles.json`: substation obstacle map used by planning and simulated perception.

## Formal Experiment Runners

| Experiment | Output Stage | Runner |
|---:|---|---|
| 1 | `01_static_astar` | `scripts/flight/experiments/run_static_astar.sh` |
| 2 | `02_perception_response` | `scripts/flight/experiments/run_perception_response.sh` |
| 3 | `03_replan_log_only` | `scripts/flight/experiments/run_replan_log_only.sh` |
| 4 | `04_active_replan` | `scripts/flight/experiments/run_active_replan.sh` |

`scripts/flight/experiments/common.sh` is shared runner infrastructure, not an experiment.

## Runtime Boundaries

PX4 SITL is started separately by `scripts/flight/start_px4_substation.sh`.
Flight commands connect through MAVSDK and do not modify PX4 flight code.

Generated raw logs and full experiment outputs remain local:

- `data/logs/`
- `data/px4_console_logs/`
- `outputs/`

Small GitHub-ready sample outputs are copied to `data/sample_outputs/`.
