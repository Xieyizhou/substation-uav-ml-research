# Substation UAV ML Research Platform

> **Status: Experimental Research Platform**
>
> This repository develops sensor-driven risk, traversability learning, and
> semantic inspection planning in simulation. It is not production software,
> does not include trained model weights, and has not been validated on a real
> airframe or energized substation.

The project extends the stable
[uav-path-planning-demo](https://github.com/Xieyizhou/uav-path-planning-demo)
baseline with Gazebo LiDAR, record/replay, local costmaps, ML/ONNX research
interfaces, semantic perception components, and future DJI integration
boundaries.

![A* route preview](docs/assets/grid_path.png)

[Demo video](https://github.com/Xieyizhou/uav-path-planning-demo/releases/tag/v0.1-demo)
· [Command reference](docs/CLI_REFERENCE.md)
· [Architecture](docs/architecture.md)
· [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
· [ML research platform](docs/ML_RESEARCH_PLATFORM.md)
· [Research roadmap](ROADMAP.md)
· [Project provenance](PROVENANCE.md)

## Project Snapshot

| Area | Implementation |
| --- | --- |
| Autonomous planning | Height-aware A* routing, obstacle inflation, path simplification, and return-route generation |
| Flight execution | MAVSDK local-NED waypoint control against PX4 SITL and Gazebo |
| Risk response | Map-oracle baseline plus live/replayed 2D LiDAR costmaps, geometric risk, safety actions, and optional ONNX fusion |
| ML research | Versioned LiDAR datasets, split-leakage checks, 1D CNN/ONNX tooling, four-class YOLO boundary, semantic fusion, BEV, and 2.5D A* |
| Hardware boundary | Vendor-neutral high-level flight protocol with MAVSDK implementation and a future DJI M30/M30T PSDK gRPC interface |
| Local replanning | Candidate-only evaluation and active replacement of remaining outbound waypoints |
| Test environments | 5 coordinated Gazebo/A* maps, 5 safe destination presets per map, and map/target switching |
| Evaluation | Structured telemetry, run manifests, plots, stage summaries, and cross-stage comparisons |
| Reliability | Explicit failure codes, timeout-bounded runtime tasks, landing confirmation, PID-scoped cleanup, and parameter validation |
| Developer experience | One modular `main.py` command center with offline regression, replay, safety, and research-contract tests |

## Selected Engineering Contributions

- Built an end-to-end autonomy pipeline from grid planning to simulated flight,
  telemetry collection, risk response, replanning, and experiment analysis.
- Kept the Gazebo world, PX4 spawn pose, A* obstacle grid, and selected
  destination synchronized through a validated map catalog.
- Designed five test environments ranging from a 16 × 16 m training yard to a
  dense 32 × 32 m multi-voltage station, with 25 validated destination presets.
- Added four reproducible experiment stages: static A*, perception response,
  log-only local replanning, and active route replacement.
- Hardened execution with connection, telemetry, waypoint, landing, and logger
  timeouts; atomic run-status records; confirmed landing state; and cleanup
  limited to project-managed processes.
- Refactored project execution into a small modular CLI while preserving
  advanced flight parameters and correct subprocess exit codes.

## Measured Simulation Results

The committed sample artifacts contain one selected landmark run from each
official stage:

| Stage | Flight time | Key result | Safety-buffer violations | Status |
| --- | ---: | --- | ---: | --- |
| Static A* | 149.316 s | Completed a 68 m planned round trip | 0 | PASS |
| Perception response | 174.563 s | 671 risk detections and 296 slow-down events | 0 | PASS |
| Replan log-only | 149.850 s | 4 successful candidates from 4 replan attempts | 0 | PASS |
| Active replan | 222.392 s | 3 successful replans and 1 active route replacement | 0 | PASS |

Source data: [comparison summary](data/sample_outputs/comparison_summary.md) and
[selected run metadata](data/sample_outputs/selected_runs.json).

The refreshed landmark uses active run `as_20260713_070842`. The latest three
eligible active-replan runs all pass strict target-switching validation with a
contiguous `RWP01` through `RWP06` sequence, no old outbound `WP` target after
replacement, original-goal arrival, and completed landing.

The aggregate comparison now includes 4 static, 3 perception-response, 3
log-only replan, and 6 active-replan runs. All 16 are completed and marked
`PASS`, with zero recorded safety-buffer violations. See the committed
[aggregate summary](data/sample_outputs/aggregate_summary.md).

## System Architecture

```text
Map + destination catalog
          ↓
Height-aware obstacle grid → A* global route → simplified waypoints
                                                ↓
Gazebo world ← PX4 SITL ← MAVSDK local-NED flight controller
                                                ↓
                         telemetry + map oracle / LiDAR perception
                                                ↓
                      risk action / local route replacement
                                                ↓
                    CSV logs → metrics → plots → comparisons
```

The project separates user commands, flight execution, planning, perception,
map management, and reporting into small modules under `src/`.

## Technology Stack

- Python 3.11, `asyncio`, `unittest`
- PX4 SITL, Gazebo Sim, MAVSDK
- A* search and grid-based obstacle modeling
- pandas and Matplotlib for telemetry analysis
- Bash experiment launchers and GitHub Actions offline validation
- JSON/SDF configuration for synchronized planning and simulation maps

## Quick Start

Prerequisites: Python 3.9+, PX4 SITL/Gazebo, and a local
`~/PX4-Autopilot` checkout.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py check environment
```

Select a map and destination:

```bash
python main.py map
python main.py point
```

Start PX4/Gazebo in terminal A:

```bash
python main.py map start
```

Preview or fly in terminal B:

```bash
source .venv/bin/activate
python main.py task run preview_route
python main.py task run fly_round_trip
```

Flight commands control PX4 SITL. Use the preview command first when testing a
new map or destination.

## Unified Command Center

```bash
python main.py --help

python main.py map list
python main.py point list
python main.py task list

python main.py astar preview --return-home
python main.py astar fly --return-home --max-speed 0.8

python main.py experiment run static
python main.py experiment run-all --trials 3

python main.py report summarize
python main.py report compare --mode both --min-runs-per-stage 3

python main.py check all

python main.py map start complex --vehicle-model x500_research
python main.py sensor check --source gazebo_lidar_2d
python main.py model protocol --config config/perception/research_protocol.json
```

See [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) for every command and advanced
parameter-forwarding example.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `main.py` | Unified user entry point |
| `src/cli/` | Modular command routing |
| `src/planner/` | A* search, obstacle conversion, and path simplification |
| `src/flight/` | MAVSDK flight runtime, task presets, and replanning orchestration |
| `src/perception/` | Simulated obstacle detector and risk-state logic |
| `src/sensors/` | Unified live/replay sensor sources and stable data contracts |
| `src/ml/` | Dataset, metrics, ONNX, LiDAR training, YOLO, and research protocols |
| `src/backends/` | Vendor-neutral flight backend contract and PX4/DJI adapters |
| `src/maps/` | Map catalog, target selection, and Gazebo marker synchronization |
| `src/logging/` | Telemetry, metrics, plots, reports, and comparisons |
| `scripts/flight/experiments/` | Reproducible four-stage experiment launchers |
| `config/maps/` | Generated map-specific A* configurations |
| `simulation/worlds/` | Gazebo SDF test environments |
| `tests/` | Offline regression and safety tests |

## Verification

```bash
python main.py check perception
python main.py check replan
python main.py check maps
python main.py check tests
python main.py check all
```

The offline suite covers CLI routing, map/target
alignment, A* reachability, parameter safety, exit-code propagation, timeout
behavior, landing confirmation, sensor parsing/replay, costmaps, dataset
isolation, semantic fusion, 2.5D planning, backend contracts, task presets,
goal-marker synchronization, and active-replan validation. PX4/Gazebo stability,
closed-loop flight, and model benchmarks remain separate research runs and are
not implied by a passing offline CI run.

## Scope and Limitations

- Simulation only; the system has not been validated on real UAV hardware.
- Existing committed experiment results use the map oracle. They are baseline
  evidence, not real-sensor or learned-perception results.
- Live/replayed 2D LiDAR and optional ONNX risk inference are implemented, but
  no trained model or statistically complete 30-seed benchmark is committed.
- YOLO, 3D/BEV, 2.5D, and DJI PSDK boundaries are research components; they
  still require generated data, trained weights, and staged integration runs.
- Obstacles are static in the current portfolio demo.
- Active route replacement has passed repeated target-switching validation on
  the simple map, but still needs cross-map and dynamic-obstacle validation.
- The committed results are selected demonstration runs, not a statistical
  performance claim.

## Resume-Ready Summary

- Engineered a PX4/Gazebo UAV autonomy pipeline integrating A* route planning,
  MAVSDK waypoint control, simulated perception, telemetry analysis, and local
  route replanning across five substation test environments.
- Developed a four-stage experimental framework with structured logs,
  reproducible reports, and safety-aware runtime controls; measured an average
  of 300 perception-triggered slow-down events across three response runs and
  validated active route replacement in simulation with zero recorded
  safety-buffer violations across 16 analyzed runs.
- Improved maintainability and reliability through a unified modular CLI, 25
  validated target presets, bounded asynchronous cleanup, explicit failure
  propagation, and automated offline safety and research-contract tests.
