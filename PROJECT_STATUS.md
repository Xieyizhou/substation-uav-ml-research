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
targets. Research milestone `v0.2` has completed and audited the visual
workflow. Its existing 120-run LiDAR output is historical static diagnostics
rather than formal evidence, so dynamic-replanning evidence remains open.

The LiDAR stability gate and the complex/extreme multi-target closed-loop gates
were completed on 2026-07-28. Six live-LiDAR round trips completed with
confirmed landing, no physical collisions, and no inflated-buffer entries. A
300-frame slice of the stability capture also passed deterministic replay.

Sandbox v1 is available as the integrity-verifiable unsigned 0.6.4 Beta.
The 0.7.0 candidate reorganizes the App as a desktop research toolbox and adds
Map Studio, immutable custom-map revisions, controlled map flight, live
trajectory inspection, and audited development-dataset registration without
changing formal research identities.
Its clean-install gate builds a
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

Workbench now also exposes receipt-verified local image inference. The native
App stages a selected PNG/JPEG in a managed inbox; the browser can select only
verified experiment IDs, never a model path, source path, threshold, or shell
command. A run rechecks the completion receipt and ONNX hashes, applies each
model's frozen validation threshold, renders annotated images, and stores an
identity-bound development result. A dual-model acceptance run completed on a
1920×1080 validation image with both the 640 Full and 416 Quick candidates.

The frozen visual YOLO11n v2 package is the current verified simulation
baseline. On 68,511 paired blind frames it reached 82.07% mAP50-95, 93.25%
macro-F1, 94.14% precision, 92.38% recall, 81.96% small-object recall, and a
2.34% no-target false-positive rate. Its 320/416/640 ONNX equivalence receipts
and all nine static replay conditions close against package identity
`e2df0d854b2f...`.

## Product Milestone Status

- **Complete:** Sandbox v1 local Beta core, native App shell, Demo and
  Development profiles, runtime compatibility management, guarded jobs,
  recording and storage inspection, unsigned packaging, and the visual model
  workbench.
- **Complete:** 0.6.4 unsigned Beta publication, clean-install validation,
  App-managed Development flight acceptance, four-step first-run guidance,
  verified single-image inference, and same-image candidate comparison.
- **Active:** stabilize public-Beta feedback and run the full-Xcode Swift/UI
  gates in CI; Command Line Tools-only machines now fail with a direct setup
  instruction instead of using an incompatible SDK/module cache.
- **Complete in 0.7.0 candidate:** workflow navigation, bounded run and
  inference history, contextual flight controls, canonical routes, and a
  single web workspace in the native shell.
- **Complete in 0.7.0 candidate:** deterministic custom substation drafts and
  revisions, 2D route preview, Headless/Visual Preview execution, 5 Hz
  trajectory display, optional PNG/truth recording, and development dataset
  registration after audit.
- **Next:** stabilize the Map Studio Beta through clean-install and external
  user feedback, then add low-rate camera preview without weakening the
  current trajectory and recording contracts.
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
- The visual v2 paired-blind result is verified simulation evidence. The local
  120-run LiDAR tree contains 120 results and a matching arithmetic report, but
  spans 11 study identities/commits and contains zero truth-danger samples.
  It is retained as historical static diagnostics and cannot support a formal
  dynamic-replanning or learned-LiDAR claim.
- The project has no real-airframe validation or dynamic-obstacle benchmark.
- Flight execution and configuration, per-run analysis and report writing,
  stage summaries, plotting, and cross-stage comparison now use bounded
  modules with compatibility entries. No Python implementation file exceeds
  500 lines.

## Next Product Priorities

1. Collect first-run installation, runtime, and Map Studio feedback for 0.7.0.
2. Extend verified inference to existing recording frames and live Gazebo
   camera frames without changing model or threshold selection rules.
3. Distinguish cold model-load latency from warm inference latency in App
   diagnostics and replay reports.
4. Extend structured multi-frame detections toward reviewed scene layouts and,
   only after geometry and safety validation, draft planning maps.

## Optional Research Priorities

1. Add route-quality acceptance gates before allowing custom revisions to fly.
2. Build a receipt-bound deterministic blocker benchmark; do not repeat the
   existing low-dynamic 120-run matrix.
3. Measure map complexity, flight speed, and safety trade-offs only after the
   blocker event chain is verified.
4. Keep real-airframe, dynamic-obstacle, and 3D/2.5D claims explicitly outside
   the validated scope until their own tests exist.

## Release State

- Current product milestone: Sandbox v1 unsigned Beta 0.6.4 is published and
  has passed clean installation plus Development flight acceptance. The local
  image inference loop is implemented for receipt-verified Workbench models.
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
