# Sandbox Roadmap

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

## v0.2 — ML experiment sandbox — Visual workflow complete, LiDAR evidence pending

- [x] Materialize deterministic Gazebo worlds with equipment variation,
  unknown obstacles, and sensor-fault manifests.
- [x] Generate geometry-derived risk, TTC, occupancy, 72-bin traversability,
  and recommended-direction labels from synchronized replay.
- [x] Enforce train/validation/test seed boundaries and reserve formal seeds.
- [x] Train with validation selection, class weighting, early stopping, a best
  checkpoint, real direction targets, and PyTorch/ONNX consistency checks.
- [x] Package dataset/model identities, hashes, contracts, history, and metrics.
- [x] Schedule replay, 5-scenario closed-loop, and 30-scenario paired formal
  studies in a resumable SQLite registry.
- [x] Freeze PNG visual replay identity, annotation, preprocessing, static
  condition/result contracts, nine unmaterialized templates, and a pilot
  recording protocol.
- [x] Implement the stable PNG camera/truth adapters, deterministic
  synchronization, and pilot recording/identity workflow.
- [x] Pass the live visual source, synchronization, route-coverage, and pilot
  identity/annotation/determinism gate.
- [x] Collect the 50-recording split-isolated visual dataset and preserve its
  aggregate identities.
- [x] Train and freeze the first visual candidate, run one paired blind
  comparison, and complete the nine-condition static ONNX replay.
- [x] Provide identity-bound non-blind experiment recipes and bounded App
  execution for repeatable visual diagnostics.
- [ ] Complete the LiDAR replay, 5-scenario, and 120-run formal evidence gates.

The runner uses 30 paired scenarios × four conditions, not the previous
1,800-run Cartesian expansion.

## v0.3 — First LiDAR ML evidence

- Publish the first candidate model card and fixed replay results.
- Compare oracle, geometric LiDAR, ML LiDAR, and safety fusion without
  suppressing neutral or negative results.
- Promote a model only when safety does not regress and at least one quality or
  latency metric improves.

## Sandbox v1 — Reliable local operation

- [x] Provide a dependency-free Demo profile and local browser App.
- [x] Enforce single-job execution, bounded timeouts, safe process-group stop,
  persistent history, and failure diagnostics.
- [x] Add a controlled three-flight LiDAR capability challenge with an
  integrity-bound receipt.
- [x] Block large LiDAR gates when the challenge receipt is missing, stale,
  changed, or belongs to another model.
- [ ] Recover or explicitly adopt a still-running owned job after App restart.
- [ ] Add output budgets, retention policy, and structured failure classes.
- [ ] Complete a GitHub Beta installation gate on a clean supported machine.

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
