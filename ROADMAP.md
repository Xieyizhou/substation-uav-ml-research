# Research Roadmap

The roadmap is evidence-gated. Interfaces or untrained model wrappers do not
count as completed capabilities.

## v0.1 — LiDAR stability and cross-map evidence — Evidence complete

- [x] Record a continuous 600-second 2D LiDAR run.
- [x] Validate frequency, drops, P95 frame age, shutdown, and replay.
- [x] Complete complex and extreme flights across three target presets each.
- [x] Preserve the map-aware detector as an oracle comparison only.

Evidence: [v0.1 LiDAR validation](docs/results/v0.1_lidar_validation_20260728.md).
This milestone is simulation evidence, not a real-airframe or statistical ML
claim.

## v0.2 — Unknown-obstacle replanning

- Add reproducible unknown-obstacle and sensor-fault scenarios.
- Validate slowdown, hover, landing, and active route replacement.
- Report collision, near-miss, clearance, success, and replan latency.

## v0.3 — LiDAR ML risk and traversability

- Generate split-isolated datasets using map layout and seed boundaries.
- Train and export the 1D CNN/ONNX baseline.
- Compare oracle, geometric LiDAR, ML LiDAR, and safety fusion over at least
  thirty independent seeds per formal condition.

## v0.4 — Four-class equipment perception

- Generate labeled transformer, switchgear, capacitor-bank, and reactor data.
- Train and evaluate a locked 640-input lightweight YOLO model.
- Add LiDAR-supported semantic position estimates and inspection viewpoints.

## v0.5 — 3D/2.5D planning and DJI preparation

- Add a Gazebo 3D point-cloud source and BEV traversability benchmark.
- Validate height-layer planning around suspended substation structures.
- Implement the DJI PSDK C++ bridge, then progress through SIL and HIL gates.

Raw datasets, model weights, complete logs, and generated run trees remain
outside Git. Negative and neutral experimental results must be reported.
