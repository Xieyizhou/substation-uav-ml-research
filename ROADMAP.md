# Product and Research Roadmap

The repository has two independent tracks. The Sandbox product track is gated
by installability, safe local operation, repeatable workflows, and useful
diagnostics. The research track is gated by experimental evidence. An
unfinished research study does not block a Sandbox Beta, and a polished App
does not count as research evidence.

## Sandbox v1 — Local Beta core complete

- [x] Provide a dependency-free Demo profile and local browser App.
- [x] Enforce single-job execution, bounded timeouts, safe process-group stop,
  persistent history, failure diagnostics, and controlled recovery.
- [x] Add output budgets, retention policy, runtime compatibility checks, and
  explicit Development/Formal dependency boundaries.
- [x] Package a reproducible, integrity-verifiable unsigned macOS Beta.
- [x] Add an offline visual model workbench with audited YOLO import, bounded
  recipes, checkpoint resume, ONNX equivalence, replay, and comparison.
- [x] Complete a full 640-pixel Workbench run through training, validation,
  ONNX export, replay, and receipt verification.
- [x] Rebuild the latest unsigned Beta and repeat clean-install plus one
  App-managed Development flight acceptance on the release commit.
- [x] Publish the GitHub Beta with a first-run guide and verified checksums.

## Sandbox v1.1 — Public Beta stabilization

- Validate Demo installation on the supported GitHub macOS matrix.
- [x] Validate one documented Development workflow from runtime discovery through
  PX4/Gazebo flight, confirmed landing, cleanup, and receipt inspection.
- [x] Improve first-run guidance, artifact discovery, storage reporting, and
  actionable failure summaries from early-user feedback.
- [x] Make macOS test jobs require one full Xcode toolchain and fail clearly
  when only mismatched Command Line Tools/SDK modules are available.
- [x] Add executable UI-state regression checks for tab persistence, polling,
  onboarding navigation, and controlled inference actions.
- Keep advanced workflows repository-backed; do not bundle PX4, Gazebo,
  datasets, model weights, or the ML Python environment in the App.

## Sandbox v1.2 — Visual inference workbench

- [x] Select a completed, receipt-verified Workbench candidate for local inference.
- [x] Accept a local PNG/JPEG through the native file picker, run its fixed ONNX
  preprocessing and threshold, and render boxes, classes, confidence, and
  latency without exposing arbitrary commands or model paths.
- [x] Save identity-bound inference results and support side-by-side candidate
  comparison on the same image.
- [ ] Extend the same boundary to recording replay and live Gazebo camera frames
  after single-image inference is stable.

## Sandbox v1.3 — Detection-assisted scene and planning maps

- Convert multi-frame detections into observations with camera identity,
  timestamp, pose, scale, and uncertainty; a single RGB image is not treated
  as a flight-ready map.
- Fuse calibrated multi-view images with depth or LiDAR to estimate equipment
  positions and produce an editable scene layout.
- Validate coordinate frames, free space, obstacle inflation, reachability,
  and provenance before exporting an A* occupancy map or Gazebo scene.
- Keep automatic output in draft status until a user reviews scale, geometry,
  unknown obstacles, and safe-flight constraints.

# Research Evidence Track

This track remains evidence-gated. Interfaces or untrained model wrappers do
not count as completed capabilities.

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
