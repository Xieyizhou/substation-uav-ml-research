# Substation UAV ML Research Status

## Current Goal

Develop a simulation-first platform for studying ML-assisted UAV risk
assessment, traversability, route planning, and substation equipment
perception. The map-aware detector remains an oracle baseline rather than a
claim of real perception.

The first release target is `v0.1`: ten-minute 2D LiDAR stability evidence plus
repeatable complex/extreme closed-loop runs across multiple targets.

The LiDAR stability gate and the complex/extreme multi-target closed-loop gates
were completed on 2026-07-28. Six live-LiDAR round trips completed with
confirmed landing, no physical collisions, and no inflated-buffer entries. A
300-frame slice of the stability capture also passed deterministic replay.

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
- ML, 3D/BEV, four-class YOLO, semantic fusion, and DJI PSDK interfaces exist,
  but trained weights and closed-loop statistical results are not yet evidence.
- The project has no real-airframe validation or dynamic-obstacle benchmark.
- Flight execution and configuration, per-run analysis and report writing,
  stage summaries, plotting, and cross-stage comparison now use bounded
  modules with compatibility entries. No Python implementation file exceeds
  500 lines.

## Next Priorities

1. Generate split-isolated randomized LiDAR datasets on complex/extreme maps.
2. Train and evaluate geometric, ML, and fused conditions, then run the checked
   30-seed protocol without suppressing negative results.
3. Generate four-class equipment labels and train the locked 640-input YOLO
   model while keeping extreme layouts unseen.
4. Implement and bench the DJI C++ PSDK service only after simulation and HIL
   gates pass.

## Release State

- Current research milestone: `v0.1` evidence gates complete locally.
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
