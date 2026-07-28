# Substation UAV ML Research Status

## Current Goal

Develop a simulation-first platform for studying ML-assisted UAV risk
assessment, traversability, route planning, and substation equipment
perception. The map-aware detector remains an oracle baseline rather than a
claim of real perception.

The first release target is `v0.1`: ten-minute 2D LiDAR stability evidence plus
repeatable complex/extreme closed-loop runs across multiple targets.

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

See [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) for the complete command set.

## Repository Structure

- `main.py` and `src/cli/`: unified user command layer.
- `src/planner/`: A* search and obstacle-map conversion.
- `src/perception/`: simulated obstacle detection and risk states.
- `src/sensors/`: Gazebo/replay sources and timestamped sensor contracts.
- `src/ml/`: research datasets, metrics, training, ONNX, YOLO, and protocols.
- `src/backends/`: high-level flight abstraction and MAVSDK/DJI boundaries.
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
- Offline tests plus shell and five-map preview checks. The suite now also
  covers sensor parsing/replay, costmaps, split leakage, semantic fusion, BEV,
  2.5D planning, safety policy, and backend contracts.

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

Local stage summaries now include 17 valid analyzed runs. The refreshed public
landmark uses active run `as_20260713_070842`; the committed public aggregate
remains the curated 16-run release sample with zero safety-buffer violations.

## Remaining Evidence Gaps

- Formal experiment manifests currently come from `substation_simple_v3`.
- The other four maps are validated offline but still need representative
  PX4/Gazebo flight runs.
- Published landmark runs still use the map oracle; they are baseline data.
- Live/replayed 2D Gazebo LiDAR and geometric local costmaps are implemented,
  but the 10-minute stability gate needs a captured run artifact.
- ML, 3D/BEV, four-class YOLO, semantic fusion, and DJI PSDK interfaces exist,
  but trained weights and closed-loop statistical results are not yet evidence.
- The project has no real-airframe validation or dynamic-obstacle benchmark.
- Flight execution and configuration, per-run analysis and report writing,
  stage summaries, plotting, and cross-stage comparison now use bounded
  modules with compatibility entries. No Python implementation file exceeds
  500 lines.

## Next Priorities

1. Capture and validate a 10-minute x500 research LiDAR recording.
2. Generate split-isolated randomized LiDAR datasets on complex/extreme maps.
3. Train and evaluate geometric, ML, and fused conditions, then run the checked
   30-seed protocol without suppressing negative results.
4. Generate four-class equipment labels and train the locked 640-input YOLO
   model while keeping extreme layouts unseen.
5. Implement and bench the DJI C++ PSDK service only after simulation and HIL
   gates pass.

## Release State

- `v0.1-demo`: original public demo release.
- `v0.2.0`: current resume-demo release target.
- License: MIT.
- Generated videos, raw telemetry, simulator logs, and full output trees remain
  outside Git history.

## Detailed Documentation

- [README.md](README.md): portfolio overview and measured results.
- [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md): experiment protocol.
- [docs/MAP_TESTING.md](docs/MAP_TESTING.md): map and destination workflow.
- [docs/experiment_results.md](docs/experiment_results.md): current evidence.
- [docs/RELEASE_PREP.md](docs/RELEASE_PREP.md): release checklist.
