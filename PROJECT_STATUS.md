# Substation UAV ML Research Status

## Current Product Goal

Move the locally Beta-ready Sandbox v1 into a repeatable public GitHub Beta.
The release must install cleanly, make its external-runtime boundary clear,
run one App-managed Development flight through confirmed landing and cleanup,
and expose the completed visual model workflow without requiring command-line
experimentation for routine inspection.

The product release is no longer blocked by unfinished paper-oriented formal
studies. Research evidence continues on a separate track and must retain its
own identities and gates.

## Research Goal

Develop a simulation-first platform for studying ML-assisted UAV risk
assessment, traversability, route planning, and substation equipment
perception. The map-aware detector remains an oracle baseline rather than a
claim of real perception.

The completed research milestone is `v0.1`: ten-minute 2D LiDAR stability
evidence plus repeatable complex/extreme closed-loop runs across multiple
targets. Research milestone `v0.2` has completed the visual workflow while its
LiDAR formal evidence remains optional product-independent work.

The LiDAR stability gate and the complex/extreme multi-target closed-loop gates
were completed on 2026-07-28. Six live-LiDAR round trips completed with
confirmed landing, no physical collisions, and no inflated-buffer entries. A
300-frame slice of the stability capture also passed deterministic replay.

Sandbox v1 is now locally Beta-ready. Its clean-install gate builds a
Git-tracked source copy, creates a fresh virtual environment, runs bootstrap
and the deterministic Demo, inspects both receipts, and verifies the loopback
App shell plus its profile, storage, and operator APIs. The gate passed on
Darwin arm64 with Python 3.11 and 3.14; GitHub Actions runs the same gate on
Python 3.11 and 3.13 without project dependencies or research artifacts.

The Development Profile includes an offline visual model workbench. It can
register native visual training identities or import audited YOLO Detect data,
then execute a development recipe through YOLO11n training, validation-only
threshold selection, static ONNX export, PT/ONNX equivalence, ordered replay,
and a same-membership baseline comparison. A 256/64, 1-epoch, 320-pixel smoke
run and a 15,585/6,941-frame, 100-epoch, 640-pixel Full run completed the full
chain locally on the M2 Pro. The Full run reached 81.93% mAP50-95, 93.14%
macro-F1, 80.51% small-object recall, and 1.90% no-target false-positive rate;
its 6,941-frame ONNX replay completed without failed frames. These outputs
remain development experiments rather than frozen deployment packages.

## Product Milestone Status

- **Complete:** Sandbox v1 local Beta core, native App shell, Demo and
  Development profiles, runtime compatibility management, guarded jobs,
  recording and storage inspection, unsigned packaging, and the visual model
  workbench.
- **Active:** rebuild the latest unsigned Beta, repeat clean-install validation,
  complete one release-commit Development flight acceptance, and publish the
  GitHub Beta with first-run guidance.
- **Next:** local single-image inference using a selected verified Workbench
  model, followed by recording and live-camera inference.
- **Later:** detection-assisted scene and planning-map generation. Flight-ready
  maps require calibrated multi-view pose plus depth or LiDAR; a single RGB
  image may only create a draft observation, never an automatically trusted
  occupancy map.

## Active Workflow

Run public commands through `main.py` from the repository root:

```bash
source .venv/bin/activate
python main.py map
python main.py point
python main.py map start
```

In a second terminal, choose a compact task or official experiment:

```bash
python main.py task list
python main.py task run fly_round_trip
python main.py experiment list
python main.py experiment run static
```

Reports and checks are also integrated:

```bash
python main.py report summarize
python main.py report compare --mode both --min-runs-per-stage 3
python main.py report validate-active --latest 3
python main.py check all
```

The v0.2 data/model/study workflow is integrated:

```bash
python main.py data --help
python main.py model --help
python main.py study --help
```

See [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) for the complete command set.

## Repository Structure

- `main.py` and `src/cli/`: unified user command layer.
- `src/planner/`: A* search and obstacle-map conversion.
- `src/perception/`: simulated obstacle detection and risk states.
- `src/sensors/`: Gazebo/replay sources and timestamped sensor contracts.
- `src/ml/`: randomized scenarios, truth labels, datasets, training, ONNX
  packages, metrics, and protocols.
- `src/study/`: local SQLite registry, resumable run queues, comparison gates,
  and paired confidence intervals.
- `src/vision/`: deterministic camera collection, visual contracts, YOLO
  training/evaluation, model packaging, and static replay.
- `src/flight/`: MAVSDK/PX4 flight execution, tasks, and replanning.
- `src/maps/`: map catalog, destination persistence, and goal-marker sync.
- `src/logging/`: telemetry, analysis, summaries, and comparisons.
- `scripts/flight/experiments/`: internal four-stage experiment launchers.
- `scripts/maps/`: map generation and manager implementations.
- `simulation/worlds/`: five Gazebo test environments.
- `config/maps/`: matching A* configurations and map catalog.
- `data/sample_outputs/`: small curated landmark and aggregate results.
- `outputs/`: local generated runs and reports; ignored except its README.

## Safety and Reliability

- Connection, position, telemetry, waypoint, landing, and logger timeouts.
- Explicit non-zero failure propagation through the CLI.
- Confirmed landing state and atomic run-status records.
- Cleanup limited to project-managed flight and PX4 PIDs.
- Map switching blocked while a managed flight or PX4 session is active.
- Parameter, map, destination, and A* reachability validation.
- Offline tests plus shell and five-map preview checks. The suite also covers
  sensor parsing/replay, costmaps, split leakage, visual identity gates,
  deterministic collection, safety policy, and flight lifecycle behavior.

## Experiment Status

The four official stages currently have these analyzed runs:

| Stage | Runs | Completed | PASS |
| --- | ---: | ---: | ---: |
| Static A* | 5 | 5 | 5 |
| Perception response | 3 | 3 | 3 |
| Replan log-only | 3 | 3 | 3 |
| Active replan | 6 | 6 | 6 |

The latest three eligible active-replan runs all pass strict target-switching
validation. Each records the contiguous outbound sequence
`RWP01 → RWP02 → RWP03 → RWP04 → RWP05 → RWP06`, no old `WP` target after
replacement, original-goal arrival, and completed landing.

The legacy four-stage sample contains 16 completed PASS runs with zero
safety-buffer violations. The separate live-LiDAR v0.1 evidence adds six
complex/extreme closed-loop runs, for 1,288.365 seconds of flight, without
mixing sensor-driven results into the map-oracle comparison.

## Remaining Evidence Gaps

- Formal experiment manifests currently come from `substation_simple_v3`.
- Complex and extreme each have three representative live-LiDAR PX4/Gazebo
  round trips; training and medium still need representative sensor-driven runs.
- Published landmark runs still use the map oracle; they are baseline data.
- Live/replayed 2D Gazebo LiDAR and geometric local costmaps are implemented.
  The 600-second stability capture produced 18,165 valid frames at 30.295 Hz,
  with 0 dropped frames and 11.56 ms P95 frame age.
- Extreme `left`, `center`, and `top_right` live-LiDAR round trips completed
  with confirmed landing, no physical collisions, and no inflated-buffer
  entries.
- Complex `left`, `center`, and `top_right` replicated that result with
  confirmed landing, no physical collisions, no inflated-buffer entries, and
  no reported LiDAR drops.
- The v0.2 dataset/model/study infrastructure is implemented, including
  deterministic SDF mutation, sensor faults, true direction labels,
  validation-selected training, model packages, 15-run closed-loop and 120-run
  formal matrices, idempotent resume, and paired bootstrap intervals.
- Trained weights and closed-loop statistical ML results are not yet evidence.
- The project has no real-airframe validation or dynamic-obstacle benchmark.
- Flight execution and configuration, per-run analysis and report writing,
  stage summaries, plotting, and cross-stage comparison now use bounded
  modules with compatibility entries. No Python implementation file exceeds
  500 lines.

## Next Product Priorities

1. Build and verify the versioned unsigned Beta from the latest clean commit.
2. Repeat the clean-install gate and one App-managed Development PX4/Gazebo
   flight acceptance on that commit.
3. Publish the GitHub Beta and collect first-run installation, runtime, and
   workflow feedback.
4. Add verified-model selection and local image inference to Workbench.
5. Extend structured multi-frame detections toward reviewed scene layouts and,
   only after geometry and safety validation, draft planning maps.

## Optional Research Priorities

1. Complete LiDAR replay, five-scenario, and 120-run formal evidence when new
   research conclusions are required; these runs do not block Sandbox Beta.
2. Iterate map complexity, flight speed, and safety trade-offs using controlled
   exploratory recipes before defining a new evidence gate.
3. Keep real-airframe, dynamic-obstacle, and 3D/2.5D claims explicitly outside
   the validated scope until their own tests exist.

## Release State

- Current product milestone: Sandbox v1 has passed its local Beta installation
  gate; the latest release-commit install/Development flight acceptance and
  public GitHub Beta publication are next.
- No research release tag is implied until the reviewed summary is committed
  and explicitly published.
- The predecessor resume demo and its releases remain in
  `uav-path-planning-demo`; they are not release targets for this repository.
- License: MIT.
- Raw telemetry, simulator logs, datasets, weights, and full output trees
  remain outside Git history.

## Detailed Documentation

- [README.md](README.md): portfolio overview and measured results.
- [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md): experiment protocol.
- [docs/MAP_TESTING.md](docs/MAP_TESTING.md): map and destination workflow.
- [docs/experiment_results.md](docs/experiment_results.md): current evidence.
- [docs/results/v0.1_lidar_validation_20260728.md](docs/results/v0.1_lidar_validation_20260728.md):
  live-LiDAR stability and extreme multi-target evidence.
