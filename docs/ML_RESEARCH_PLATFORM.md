# Real-Time ML Perception and Planning Research Platform

This document defines the simulation-first research path from the existing
map-aware oracle to sensor-driven autonomy. The oracle remains useful for
labels and controlled comparisons, but it is not described as real perception.

## Safety boundary

Learned models produce risk, traversability, local cost, and semantic target
estimates. They never produce motor commands. A separate safety supervisor can
reduce speed, hover, reject a route, request replanning, or land when sensor
health, data age, confidence, clearance, or collision-time limits are violated.

## Runtime architecture

```text
Gazebo LiDAR / replay / future DJI payload
                    |
            timestamped SensorSource
                    |
          LaserScanFrame / PointCloudFrame
                    |
       geometric + optional ONNX inference
                    |
      LocalCostmap + RiskEstimate + health
                    |
          independent safety supervisor
                    |
          A* / active local replanning
                    |
         FlightBackend high-level commands
              /                    \
       MAVSDK + PX4          DJI PSDK gRPC bridge
```

The stable internal contracts live in `src/sensors/types.py`. Coordinates use
metres, seconds, and local NED. Every sensor frame includes a timestamp, frame
identifier, source, sequence, and receive time for age checks.

## What is implemented

### Sensor and geometric baseline

- `map_baseline`, live `gazebo_lidar_2d`, and deterministic `replay` sources.
- Gazebo topic discovery, JSON scan parsing, startup timeout, health, frame age,
  effective frequency, dropped-frame count, and subprocess cleanup.
- A repository-owned `x500_research` model containing x500, 2D LiDAR, forward
  RGB, and 2D bounding-box cameras.
- Ray clearing, occupied-cell inflation, unknown-cell penalty, forward-corridor
  risk, stopping distance, collision time, and recommended free direction.
- Inflated local costmap cells are projected into the global grid consumed by
  active replanning. Runtime LiDAR does not read map obstacle names or cells.
- Sensor loss and stale data are fail-safe danger conditions.

### ML and future-sensor research components

- Versioned LiDAR research sample schema and streaming JSONL writer.
- Scenario and map/seed split-isolation checks that reject leakage.
- Lightweight PyTorch 1D CNN for risk classification, sector traversability,
  direction, and uncertainty, with ONNX export.
- ONNX runtime adapter and geometry-plus-ML fusion. ML can increase a geometric
  risk level, but cannot downgrade it.
- Four-class YOLO wrapper and fixed class table: `transformer`, `switchgear`,
  `capacitor_bank`, and `reactor`.
- Simulator bounding-box to YOLO label conversion and deterministic domain
  randomization manifests.
- LiDAR/YOLO projection association; semantic positions are only created when
  LiDAR range evidence exists.
- Point-cloud to BEV height, occupancy, and clearance conversion, plus a
  height-layer A* planner with explicit climb cost.
- High-level `FlightBackend` contract, a MAVSDK implementation, and a Python
  client boundary for a future DJI PSDK C++ gRPC service.

These components make M1-M3 offline development and M4-M7 interface development
possible. They do not constitute trained production models, a completed C++
DJI application, or real-airframe validation.

## Commands

Start the research vehicle and inspect LiDAR:

```bash
python main.py map start complex --vehicle-model x500_research
python main.py sensor list
python main.py sensor check --source gazebo_lidar_2d
python main.py sensor record --output data/research/scans/complex_seed_1001.jsonl
```

Replay a recording without PX4:

```bash
python main.py sensor replay \
  --input data/research/scans/complex_seed_1001.jsonl
python main.py astar fly \
  --perception-source replay \
  --sensor-replay data/research/scans/complex_seed_1001.jsonl
```

Train and evaluate:

```bash
pip install -r requirements-ml.txt
python main.py model dataset --input data/research/lidar_samples.jsonl
python main.py model train --kind lidar \
  --dataset data/research/lidar_samples.jsonl \
  --output models/lidar/risk_v1.onnx
python main.py model evaluate --predictions outputs/research/predictions.jsonl
python main.py model benchmark --predictions outputs/research/predictions.jsonl
```

Validate the formal comparison matrix and generate a reproducible randomized
scenario:

```bash
python main.py model protocol \
  --config config/perception/research_protocol.json
python main.py model randomize \
  --config config/perception/domain_randomization.json \
  --map complex --seed 1001
```

Use an ONNX model during flight only after replay evaluation:

```bash
python main.py astar fly \
  --perception-source gazebo_lidar_2d \
  --risk-model models/lidar/risk_v1.onnx
```

Weights are loaded once before flight; runtime weight replacement is not
supported.

## Dataset rules

Each LiDAR sample records scenario ID, split, map, seed, scan, local pose,
velocity, goal context, risk class, sector traversability, and optional
equipment labels. A map/seed pair and scenario ID may appear in only one split.

The held-out `extreme` layouts are reserved for final visual evaluation.
Training randomization covers equipment pose and scale, lighting, material age,
weather, LiDAR noise and dropout, attitude jitter, and camera noise.

Raw scans, images, full logs, generated datasets, checkpoints, and ONNX weights
stay outside Git. The repository contains schemas, configuration, tests, and
small reproducible examples only.

## Formal evaluation

The checked protocol defines four conditions:

1. map oracle
2. geometric LiDAR
3. ML LiDAR
4. geometric + ML safety fusion

Every formal condition uses at least 30 unique seeds. Reports must include task
and landing success, collision and near-miss counts, buffer entries, path
length, flight time, speed changes, replan count and latency, sensor frequency,
drops and age, inference latency, risk F1/ECE, and traversability IoU. Model
results are reported even when they do not outperform the geometric baseline.

CI runs unit and short replay checks. Ten-minute sensor stability runs, complete
PX4/Gazebo closed-loop trials, model training, and 30-seed comparisons are
manual research pipelines whose logs and manifests provide the evidence.

## DJI boundary

`integrations/dji_psdk/flight_bridge.proto` defines the local gRPC boundary.
The future C++ companion process owns PSDK initialization, control authority,
flight-control subscriptions, joystick/FlyTo calls, HMS/watchdog handling,
camera streams, link-loss behavior, and hardware limits. Python remains
vendor-neutral.

M30/M30T integration assumes E-Port plus PSDK and an independently integrated
LiDAR payload. The aircraft's built-in obstacle sensing and single-point laser
range finder are not treated as a scanning LiDAR. Payload mass, compute,
mounting, power, and cooling must be recalculated before hardware work.

Hardware validation order is fixed: software-in-the-loop, hardware-in-the-loop,
propellers removed, tethered low altitude, open field, then controlled
substation exterior testing.

References:

- [DJI PSDK product capabilities](https://developer.dji.com/doc/payload-sdk-tutorial/en/product-introduction/product-capabilities.html)
- [DJI PSDK flight control](https://developer.dji.com/doc/payload-sdk-tutorial/en/function-overview/advanced-function/flight-control.html)
- [DJI Matrice 30 specifications](https://enterprise.dji.com/matrice-30/specs)
