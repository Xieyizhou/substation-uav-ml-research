# Research Roadmap

The roadmap is evidence-gated. Interfaces or untrained model wrappers do not
count as completed capabilities.

## v0.1 — LiDAR stability and cross-map evidence

- Record a continuous ten-minute 2D LiDAR run.
- Validate frequency, drops, P95 frame age, shutdown, and replay.
- Complete complex and extreme flights across multiple target presets.
- Preserve the map-aware detector as an oracle comparison only.

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
