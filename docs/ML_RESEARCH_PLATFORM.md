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

- Versioned LiDAR research sample schema, synchronized replay collector,
  geometry-derived truth labels, and hashed dataset manifests.
- Fixed seed ranges and scenario/map isolation checks that reject leakage and
  prevent formal evaluation seeds from entering training data.
- Deterministic SDF materialization for equipment pose/scale, unknown
  obstacles, lighting, scan noise/dropout, stream outage, and attitude-label
  jitter.
- Lightweight PyTorch 1D CNN for risk classification, sector traversability,
  direction, and uncertainty. Training uses validation selection, class
  weighting, early stopping, a real direction target, and a best checkpoint.
- ONNX export is rejected unless PyTorch and ONNX Runtime outputs agree within
  tolerance. Model packages include hashes, contracts, history, and metrics.
- ONNX runtime supports an isolated `ml_only` research condition and
  `safety_max` fusion. The latter never permits ML to downgrade geometric risk.
- A local SQLite registry schedules idempotent replay, five-scenario
  closed-loop, and 30-scenario paired formal studies with resume, comparison,
  and promotion gates.
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

Create a randomized world and collect an automatically labelled replay:

```bash
python main.py data world \
  --source simulation/worlds/substation_complex.sdf \
  --output outputs/research/worlds/complex_2001.sdf \
  --map complex --seed 2001
python main.py data collect \
  --input data/research/scans/complex_2001.jsonl \
  --telemetry data/research/telemetry/complex_2001.jsonl \
  --output data/research/lidar_v2 \
  --map complex --target center --seed 2001
python main.py data validate --dataset data/research/lidar_v2
python main.py data summarize --dataset data/research/lidar_v2
```

The world command also writes a matching `.planner.json` oracle map. Oracle
trials use this truth config; geometric and ML trials retain the original
static planner map and must discover injected unknown obstacles from LiDAR.

Train, package, predict, and evaluate:

```bash
pip install -r requirements-ml.txt
python main.py model train \
  --dataset data/research/lidar_v2/samples.jsonl \
  --output models/lidar/risk_v2 \
  --epochs 20 --seed 7
python main.py model inspect --package models/lidar/risk_v2
python main.py model predict \
  --model models/lidar/risk_v2 \
  --dataset data/research/lidar_v2/samples.jsonl \
  --output outputs/research/predictions/risk_v2.jsonl
python main.py model evaluate \
  --predictions outputs/research/predictions/risk_v2.jsonl
python main.py model benchmark \
  --predictions outputs/research/predictions/risk_v2.jsonl
```

Create and run a resumable study:

```bash
python main.py model protocol \
  --config config/perception/research_protocol.json
python main.py study create --name risk-cnn-v2 \
  --candidate models/lidar/risk_v2
python main.py study run STUDY_ID --tier replay
python main.py study run STUDY_ID --tier closed-loop
python main.py study run STUDY_ID --tier formal
python main.py study resume STUDY_ID
python main.py study status STUDY_ID
python main.py study compare STUDY_ID
python main.py study promote STUDY_ID
```

`study run` creates an auditable queue and ingests available result JSON. It
does not invent metrics for simulator trials that have not run. Failed or
interrupted entries remain resumable in the ignored local registry.

Use an ONNX model during flight only after replay evaluation:

```bash
python main.py astar fly \
  --perception-source gazebo_lidar_2d \
  --risk-model models/lidar/risk_v2/model.onnx \
  --risk-fusion safety_max
```

Weights are loaded once before flight; runtime weight replacement is not
supported.

## Dataset rules

Each LiDAR sample records scenario/config identity, split, map, seed, scan,
sensor health/age, local pose, velocity, goal/future path, risk, TTC, safety,
72-bin traversability, recommended direction, occupancy, and optional equipment
labels. A map/seed pair and scenario ID may appear in only one split.

- train: training/simple/medium/complex, seeds `2001–2040`
- validation: new layouts on the same map classes, seeds `2041–2050`
- test: held-out extreme only, seeds `2051–2060`
- formal: seeds `1001–1030`, never accepted by the dataset collector

Raw scans, images, full logs, generated datasets, checkpoints, and ONNX weights
stay outside Git. The repository contains schemas, configuration, tests, and
small reproducible examples only.

## Formal evaluation

The checked protocol defines four conditions:

1. map oracle
2. geometric LiDAR
3. ML LiDAR
4. geometric + ML safety fusion

Thirty balanced scenarios are expanded across the four conditions for exactly
120 paired runs. Reports must include task
and landing success, collision and near-miss counts, buffer entries, path
length, flight time, speed changes, replan count and latency, sensor frequency,
drops and age, inference latency, risk F1/ECE, and traversability IoU. Model
results are reported even when they do not outperform the geometric baseline.

CI runs unit and short replay checks. The v0.1 manual gate has completed a
600-second sensor capture plus six complex/extreme closed-loop trials; see the
[validation report](results/v0.1_lidar_validation_20260728.md). Fault injection
and the experiment registry are implemented; model training and 30-scenario
results remain manual evidence work.

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
