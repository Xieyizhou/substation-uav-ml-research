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
  -> Sensor source (`src/sensors/`)
       - map oracle baseline
       - Gazebo 2D LiDAR
       - deterministic replay
  -> Local costmap + geometric/optional ONNX risk (`src/perception/`, `src/ml/`)
  -> Risk action
       - Experiment 2 perception_response: reduce speed during warning/danger risk
       - Experiment 3 replan log-only: test replan availability without replacing route
       - Experiment 4 active route replacement: replace remaining outbound waypoints
  -> Telemetry CSV logging (`src/logging/flight_logger.py`)
  -> Per-run analysis and staged summaries (`src/logging/`)
  -> Curated evidence (`data/sample_outputs/`)
```

## Major Modules

- `src/planner/`: grid A* search, path simplification, obstacle-map conversion.
- `src/sensors/`: Gazebo/replay sources, scan parsing, health, and stable data contracts.
- `src/perception/`: map-oracle baseline, LiDAR detector, rolling costmap, and
  safety state.
- `src/ml/`: shared dataset schemas, split isolation, ONNX risk, LiDAR
  training, and domain randomization.
- `src/vision/`: deterministic camera collection, visual contracts, YOLO
  training, evaluation, model packaging, and static replay.
- `src/flight/fly_astar_path.py`: thin CLI and backward-compatible exports.
- `src/flight/mavsdk_preflight.py`: connection and position readiness.
- `src/flight/waypoint_executor.py`: Offboard waypoint and route execution.
- `src/flight/landing_manager.py`: normal and failsafe landing confirmation.
- `src/flight/perception_response.py`: sensor-driven detection state and detector setup.
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
- `config/maps/`: synchronized map-specific planner configurations.
- `config/perception/`: research protocol, equipment classes, and domain randomization.

## Visual Workflow Architecture

Visual research code follows one-way domain layers. JSON dictionaries enter
and leave at serialization boundaries; workflow code passes validated records
and identity objects internally.

| Layer | Input | Output | Invariant |
|---|---|---|---|
| `src/vision/contracts/` | Serialized records and explicit fields | Validated camera, annotation, identity, protocol, and benchmark records | Content identity and schema validation contain no workflow decisions |
| `src/vision/collection/` | Protocol, layout, route, and simulator streams | Recordings, audit reports, and dataset identities | Truth status remains distinct from synchronization failure |
| `src/vision/training/` | Development identity and sampling configuration | Ordered YOLO view and training artifacts | Selected membership and labels are reproducible and hash-bound |
| `src/vision/evaluation/` | Frozen model, validation or held-out receipt | Metrics, ONNX gates, and package manifest | Held-out settings cannot be selected from held-out predictions |
| `src/vision/replay/` | Frozen package, dataset identity, and replay condition | Ordered timing and accuracy results | Conditions share source order unless the condition explicitly changes frame cadence |
| `src/cli/visual*.py` | Command-line arguments | Application-service calls and formatted output | No experiment policy or identity rule lives in the parser |

The principal data flow is:

```text
collection plan
  -> recording + annotation manifests
  -> dataset identity
  -> deterministic training view
  -> trained weights
  -> validation threshold + ONNX equivalence gates
  -> frozen model package
  -> held-out receipt and saved predictions
  -> static replay conditions and results
```

Public workflow entry points are grouped by responsibility:

- Collection: `collection/plan.py`, `collection/batch.py`, and
  `collection/dataset.py`. Version-specific materialization is isolated in
  `collection/v1_recording.py` and `collection/v2_recording.py`; route timing
  and per-recording coverage gates remain independent services. One-scenario
  process ownership lives in `collection/scenario_runner.py`, while
  `collection/batch.py` only controls ordering and retries.
- Training: `training/view.py` and `training/yolo_training.py`.
- Package freeze and held-out evaluation: `evaluation/yolo_package.py`,
  `evaluation/heldout_view.py`, and `evaluation/yolo_evaluation.py`.
- Static replay: `replay/static_replay.py`.

Versioned protocols may share validated contracts, but a newer protocol does
not change the meaning or hashes of existing artifacts. Protocol selection is
persisted in the collection plan and consumed from that plan by later stages.

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

## Validated Runtime Evidence

The v0.1 simulation gate covers a 600-second Gazebo LiDAR capture, deterministic
replay, and six live-LiDAR round trips across complex/extreme targets. All six
completed with confirmed landing, zero physical collisions, zero inflated
buffer entries, and zero reported LiDAR drops. See
[`docs/results/v0.1_lidar_validation_20260728.md`](results/v0.1_lidar_validation_20260728.md).
