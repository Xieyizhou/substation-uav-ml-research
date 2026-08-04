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
Gazebo LiDAR / deterministic replay
                 |
         timestamped SensorSource
                 |
            LaserScanFrame
                 |
    geometric + optional ONNX inference
                 |
   LocalCostmap + RiskEstimate + health
                 |
       independent safety supervisor
                 |
       A* / active local replanning
                 |
            MAVSDK + PX4
```

The stable internal contracts live in `src/sensors/types.py`. Coordinates use
metres, seconds, and local NED. Every sensor frame includes a timestamp, frame
identifier, source, sequence, and receive time for age checks.

### Camera data and clock contracts

`CameraFrame` is the simulator- and detector-independent visual input
contract. It records:

- `frame_id`, `source_id`, and `sequence_number`;
- `capture_timestamp` and its explicit `capture_clock_domain`;
- `receive_monotonic_timestamp`;
- `width` and `height`;
- `payload_format`: `png`, `jpeg`, or `raw`;
- `pixel_format`: `rgb8`, `bgr8`, `rgba8`, `bgra8`, or `mono8`;
- portable `payload_relative_path` and `payload_sha256`;
- optional JSON-serializable metadata.

Image bytes never appear in JSONL. Dimensions must be positive, sequence
numbers non-negative, timestamps finite, formats supported, payload paths
relative and portable, and SHA256 values well formed.

Storage and decoded pixel layout are deliberately separate. `payload_format`
describes the stored bytes. `pixel_format` describes the consumer-visible
layout that a replay decoder must expose; it is never inferred from a PNG or
JPEG codec's internal representation. In particular, RGB and BGR are not
interchangeable defaults. A future adapter may retain its native source layout
as optional `source_pixel_format` metadata, but that provenance does not
override the declared replay output layout.

Schema v1 raw payloads are restricted to tightly packed unsigned 8-bit
channels with no row padding. Their exact byte length is therefore
`width × height × channel_count`, where the channel count comes from the
declared `pixel_format`. Recording and replay both enforce this rule. Formal
visual replay uses PNG as its canonical payload because PNG is lossless,
portable, inspectable, and smaller than raw for these images. Raw remains
available for provenance and storage/transport experiments. JPEG remains
supported for controlled experiments, but it is lossy; converting a JPEG
source to PNG does not restore lost information, so source provenance remains
explicit.

Capture and receive timestamps are not assumed comparable. Simulator, source,
and wall-clock capture times remain useful for ordering and provenance, but
`capture_to_receive_ms` is available only when the capture clock is explicitly
`local_monotonic`. Runtime durations use the local high-resolution monotonic
clock. No latency is produced by subtracting incompatible clocks.

### Deterministic camera recording and replay

`src/sensors/camera_recording.py` extends the existing sensor JSONL approach
with external, hash-addressed payload validation:

```text
camera_recording/
  metadata.json
  frames.jsonl
  summary.json
  frames/
    000000000.png
    000000001.png
```

The recorder accepts synchronous or asynchronous iterables. Accepted frames
are written in input order with stable payload names. Invalid inputs are not
converted into valid-looking frames: their input indexes, error types, and
messages are retained in `metadata.json` and counted in `summary.json`.

The summary contains observed accepted and invalid counts, missing and
duplicate sequences, non-monotonic capture timestamps, valid timestamp span
and effective frequency when computable, payload bytes, endpoint sequence and
frame IDs, clock domains, and schema version.

Replay treats `frames.jsonl` order as authoritative; file names never reorder
frames. Before yielding anything it validates the schema, every manifest row,
payload existence, and every SHA256. Missing, corrupted, malformed, and
unsupported-schema artifacts raise distinct recording errors. `no_sleep`
replay is deterministic. `paced` replay additionally requires one clock
domain and non-decreasing capture timestamps.

Inspect, validate, or replay a recorded directory without starting PX4 or
Gazebo:

```bash
python main.py sensor camera-inspect --input data/research/camera/example
python main.py sensor camera-validate --input data/research/camera/example
python main.py sensor camera-replay \
  --input data/research/camera/example --mode no_sleep
python main.py sensor camera-replay \
  --input data/research/camera/example --mode paced --rate 1.0
```

### Gazebo RGB and simulator-truth adapters

The live visual path reuses Gazebo Transport through the existing subprocess
boundary; it does not add ROS or control PX4:

```text
gz.msgs.Image                         gz.msgs.AnnotatedAxisAligned2DBox_V
      |                                             |
GazeboCameraSource                         GazeboTruthSource
      |                                             |
canonical PNG CameraFrame                    SimulatorTruthFrame
      +---------------- simulation time -------------+
                            |
              deterministic timestamp matching
                            |
       VisualFrameAnnotation + SynchronizationRecord
```

The repository-owned SDF configures `research_camera/image` at 30 Hz and
`research_camera/boxes` at 15 Hz. Runtime topics may be scoped by Gazebo, so
the CLI discovers the full topic names and verifies their message types before
subscribing. It never invents a scoped topic. Both sources require
`header.stamp` and identify its clock as `gazebo_sim_time`; local monotonic
receive time is recorded separately and is not used for RGB/truth matching.

Gazebo image messages declare their row stride and pixel enum. Supported
uint8 RGB, BGR, RGBA, BGRA, and mono layouts are converted explicitly to
canonical RGB PNG. The source raw-message hash, source layout, stride, topic,
message type, and sequence provenance remain metadata. Invalid messages
produce failure events rather than disappearing.

Generated transformer, switchgear, capacitor-bank, and reactor models carry
Gazebo labels 1 through 4, mapped to locked dataset class IDs 0 through 3.
Cabinets, poles, buildings, and other scene objects remain background.
Bounding boxes use `xyxy_pixels_half_open`; partial out-of-bounds boxes are
clipped and marked truncated, while unknown labels, zero-area boxes, and
incompatible dimensions invalidate the truth message. The sensor does not
provide a reliable occlusion flag, so visibility remains `unknown`.

Synchronization first uses exact simulation timestamps, then the nearest
timestamp within a configurable tolerance. The default 33.334 ms is half the
configured 15 Hz truth period and is an engineering starting point, not a
scientifically validated threshold. Equidistant candidates are ordered by
earlier timestamp and message identity for reproducibility but remain
`ambiguous` and do not receive annotations. `unmatched`, `ambiguous`,
`invalid_truth`, and a valid synchronized `verified_no_target` frame are
different states.

Inspect and probe without launching external processes:

```bash
python main.py visual gazebo-inspect --timeout 5
python main.py visual gazebo-probe --timeout 5
```

Gazebo Sim and the `x500_research` entity must already be running. PX4 may run
separately, but these commands do not start it or send flight commands.

### Labelled visual pilot recording

The pilot recorder retains every valid source frame and writes:

```text
pilot_recording_ID/
  metadata.json
  frames.jsonl
  annotations.jsonl
  synchronization.jsonl
  truth_events.jsonl
  mission_events.jsonl
  summary.json
  identity/
    recording_identity.json
    dataset_membership.jsonl  # only after acceptance validation
    dataset_identity.json  # only after acceptance validation
  frames/
    *.png
```

An expected source rate of 10 Hz is recorded as an expectation and is not
forced onto the configured 30 Hz camera. A material difference is reported
as an observation. The summary measures actual frame counts, valid timestamp
duration and rate, P50/P95/P99 intervals, gaps, duplicate and non-monotonic
sequences/timestamps, receive stalls, timeouts, PNG bytes, match states,
synchronization offsets, no-target and labelled frames, boxes, and per-class
counts. Unavailable rate or percentile values remain null. Rate and jitter
flags are observations unless a reviewed protocol later defines a hard gate.

The protocol is `visual-pilot-png-v3`. Its 100–300-frame range is a manual
inspection target, not an automatic stop or validity threshold. Recording
does not downsample; every-second and every-third policies apply only after
the ordered dataset is frozen. A partial, timed-out, camera-only, unmatched,
or phase-incomplete recording cannot materialize a `DatasetIdentity`.
Successful materialization always uses dataset role `pilot`, never `formal`.
Invalid simulator truth remains explicitly invalid and is preserved in
`truth_events.jsonl`; it is never rewritten as a no-target annotation. V3
allows at most 0.1% of RGB frames to reference invalid truth and at most two
such frames consecutively. Those RGB frames remain in the source recording
but are excluded from `dataset_membership.jsonl`. Unmatched and ambiguous
frames remain disallowed. Each required mission phase must contain at least
one second of valid synchronized simulation time; its share of the total
recording duration is not an acceptance field.

```bash
python main.py visual pilot-record \
  --output data/research/visual_pilot/pilot_recording_001 \
  --duration 20 --source-timeout 5
python main.py visual pilot-phase \
  --output data/research/visual_pilot/pilot_recording_001 \
  --phase cruise_distant
python main.py visual pilot-phase \
  --output data/research/visual_pilot/pilot_recording_001 \
  --phase approach
python main.py visual pilot-phase \
  --output data/research/visual_pilot/pilot_recording_001 \
  --phase close_inspection
python main.py visual pilot-phase \
  --output data/research/visual_pilot/pilot_recording_001 \
  --phase target_transition
python main.py visual pilot-inspect \
  --input data/research/visual_pilot/pilot_recording_001
python main.py visual pilot-validate \
  --input data/research/visual_pilot/pilot_recording_001
python main.py visual pilot-materialize \
  --input data/research/visual_pilot/pilot_recording_001
```

Run `pilot-record` while the existing flight task is active, and issue
`pilot-phase` from another terminal when each reviewed route phase begins.
The marker reads the last accepted RGB simulation timestamp from the live
recording status; it does not infer phases from detector output. An externally
prepared JSONL event log may instead be supplied with `--mission-events`.
Exactly one stopping mode is required: `--duration`, `--frame-limit`, or the
explicitly unbounded `--until-interrupt`. At the configured 30 Hz source rate,
300 retained frames represent about 10 seconds of simulation time. PNG
compression reduces bytes per frame but does not control recording length.

### Multi-scenario visual collection

The accepted pilot unlocks `visual-multiscenario-png-v1`, a frozen 60-recording
plan for model development and held-out evaluation. It uses seeds 2001–2040
for train, 2041–2050 for validation, and 2051–2060 for extreme-map held-out
test. Formal evaluation seeds 1001–1030 cannot enter any dataset identity.
Each map/target/seed recording remains an indivisible split unit.

`visual collection-prepare` materializes a reachable randomized SDF, matching
planner config, and scenario report. Only equipment geometry, equipment
position, light intensity, and unknown obstacles currently contribute to the
visual scenario identity. Randomization fields that are not rendered are
listed explicitly as not applied. The final applied configuration, world, and
planner hashes become recording metadata.

`visual collection-record` reuses the validated Gazebo transport, PNG,
synchronization, truth, and phase pipeline while recording collection rather
than pilot identity. Flight lifecycle events supply semantic route state but
no fabricated simulation time; the recorder assigns the latest accepted RGB
Gazebo timestamp. The final outbound waypoint, goal hover, and return route
generate the reviewed phases automatically. Confirmed landing stops recording
after a short drain, triggers validation, and hashes the raw flight-event
manifest into recording identity. `visual collection-record-validate` remains
available as an explicit retry. After every planned recording passes,
`visual collection-materialize` creates separate development and held-out-test
DatasetIdentity artifacts with globally unique `recording_id:frame_id`
membership. Aggregate class and no-target coverage are mandatory per split.

Protocol v2 adds ten split-isolated Gazebo layouts and five routes per layout:
four equipment-centered routes plus a verified-no-target background transit.
Its tracked plan and layout manifests live under
`benchmarks/visual_static_v2/`. Labelled equipment models are world-level
entities and each label plugin is attached to the equipment's own visual;
unlabelled infrastructure remains background. Equipment routes retain two
views separated by 90 degrees and use the measured PX4-to-Gazebo yaw mapping
for explicit target-facing yaw, including the close-range -30/0/+30 degree
sweep. The protocol binds the x500 camera installation: a
5-degree downward pitch, a measured 341-degree heading offset from PX4 body
heading,
and centered square-pixel intrinsics derived from the configured horizontal
field of view for both RGB and truth cameras. A visual hold starts only after
yaw is within tolerance and roll/pitch are level for the required settling
interval; the flight event log records that boundary. Straight A* segments
are represented by their turning cells rather than every grid cell. Flight
timeouts are derived from route distance, holds, and waypoint settling,
bounded to 180--360 seconds; an explicit CLI override remains available for
diagnosis.

Every v2 recording must contain at least 1,000 target frames, at least 150
small, medium, and large target frames, two seconds of valid target visibility
in every required phase, and no more than 50 percent truncated target frames.
Background routes must contain only verified-no-target frames. Blind layouts
may be collected and integrity-checked, but predictions remain locked until a
model package is frozen.

`visual training-view-materialize` derives a path-independent, hash-bound
training view from the development identity. It performs deterministic
recording-proportional temporal thinning, retains minority classes, limits
negative frames, emits explicit empty YOLO labels for verified no-target
images, and never reads the held-out identity. The training identity binds
the selected memberships and generated label hashes rather than claiming the
entire development dataset was fitted.

The first visual baseline uses COCO-pretrained YOLO11n at 640 pixels. Training
and validation use Apple MPS locally; model selection uses the balanced
validation view, followed by one full-validation pass. A frozen package
exports static batch-1 FP32 ONNX graphs at 320, 416, and 640, each with its
own preprocessing and model identity. The package manifest is written only
after all three 200-frame PT/ONNX equivalence gates pass. Held-out
materialization is gated on that finalized package; its canonical 640 model
must use the full-validation threshold and cannot fit or select a new one.

### Canonical camera decoding

`src/sensors/camera_decoder.py` decodes a recorded frame on the CPU into the
detector-neutral `DecodedImage` contract from
`src/sensors/camera_decoded.py`. The canonical in-memory representation is
RGB8, unsigned 8-bit, HWC, three-channel, C-contiguous, read-only, and exactly
the declared width and height. Full arrays are never written to JSON
manifests. `CameraReplaySource.iter_decoded()` preserves manifest order and
yields the canonical result plus measured decoder-stage timing.

PNG, JPEG, and raw are supported. The declared payload format, portable file
extension, and encoded format must agree. PNG/JPEG modes are limited to
grayscale (`L`), RGB, and RGBA. Grayscale luma is repeated into all three RGB
channels. Alpha is dropped without compositing. Declared BGR/BGRA channels are
reordered explicitly to RGB; RGB is never silently treated as BGR. Palette,
CMYK, and other modes fail with a typed error. EXIF orientation and ICC colour
profiles are ignored by the canonical policy. JPEG is supported but remains
lossy. Formal visual inputs use PNG.

Raw schema v1 has one mandatory interpretation:
`tightly_packed_uint8_no_row_padding`. Its dtype is `uint8`, byte order is
`not_applicable`, row stride is `width × channel_count`, and the byte count is
`width × height × channel_count`. These values may be restated in frame
metadata, but any conflicting dtype, stride, byte order, or recording-level
raw-layout contract is rejected. The decoder never guesses an alternative raw
layout.

Decoder identity contains the implementation name and version, Pillow backend
and version, canonical target format/dtype/layout, grayscale policy, alpha
policy, raw-layout policy, colour-conversion policy, and relevant fixed
options. Canonical sorted JSON of those fields is SHA256-hashed as
`decoder_configuration_id`; this identity must accompany future dataset and
benchmark results. It deliberately excludes paths, timestamps, process IDs,
and random identifiers.

The source identity is SHA256 of the payload bytes and is checked before
decoding. Decoded identity is SHA256 over canonical sorted JSON containing the
decoded schema version, width, height, pixel format, dtype, and layout,
followed by one newline byte and the contiguous C-order pixel bytes. Repeated
CPU decoding with one recording and configuration must preserve manifest
order, dimensions, dtype, layout, pixel format, source hash, decoded hash, and
decoder configuration ID. Timing values are deliberately excluded from the
determinism comparison.

### Visual latency contract

`VisualTiming` separates the following optional stages:

| Field | Definition and provenance |
| --- | --- |
| `capture_to_receive_ms` | Capture to local receive; only for explicitly comparable local-monotonic timestamps |
| `queue_wait_ms` | Queue entry to backend-call start, measured locally when queue context is supplied |
| `payload_load_ms` | Local monotonic elapsed time spent reading the payload bytes from storage; excludes hashing and decoding |
| `decode_ms` | Local monotonic elapsed time from payload-read completion through source-hash verification, codec/raw decode, explicit channel conversion, canonical hashing, and validated canonical image construction |
| `preprocess_ms` | Backend-reported preprocessing, only when present |
| `backend_call_ms` | Entire outer detector call, measured locally with a monotonic clock |
| `inference_ms` | Backend-reported inference, only when present |
| `postprocess_ms` | Backend-reported postprocessing, only when present |
| `decision_finalize_ms` | Local conversion of backend results into stable detections |
| `end_to_end_ms` | Capture to finalized detections, only with a compatible local-monotonic capture clock |

The record also carries queue depth, deadline, deadline outcome, model ID,
runtime backend, device, input dimensions, detection count, and per-stage
timing provenance. A deadline outcome is calculated only when the deadline
and a comparable elapsed duration both exist. Missing stages remain `null`;
they are never represented by zero. Summary helpers exclude missing values and
report count, P50, P95, P99, minimum, maximum, and mean for each stage.

`EquipmentDetector.detect(...)` retains its tuple-of-`EquipmentDetection`
behavior, locked class order, and default input size of 640.
`detect_with_metrics(...)` returns equivalent detections plus `VisualTiming`.
The outer backend call and detection finalization are locally measured;
Ultralytics preprocess/inference/postprocess durations are copied only when
the returned result actually provides them.

Decoder `decode_ms` does not include model resize, letterboxing, normalization,
HWC-to-CHW conversion, float conversion, inference, or detection
postprocessing. Those operations belong to `preprocess_ms` or later stages.
The current canonical path is CPU Pillow only; there is no GPU decoder
identity, orientation transform, colour-management transform, palette-mode
conversion, or padded/typed raw support.

### Frozen visual research identities

The versioned contracts in `src/vision/contracts/` and the static matrix
validation in `src/vision/replay/benchmark_matrix.py` keep six different
identities separate:

1. `DatasetIdentity` binds ordered recording membership, payload hashes through
   the recording manifest, annotations, scenarios, splits, decoder
   configuration, class order, maps, seeds, and source recording IDs.
2. `VisualFrameAnnotation` links ground truth to the exact source payload and
   canonical decoded-content hash. Object labels use the locked four-class
   order and `xyxy_pixels_half_open` boxes. Pseudo-labels are not formal truth.
3. `PreprocessingIdentity` binds resize, letterbox, interpolation, padding,
   channel/tensor conversion, normalization, batch, dtype, contiguity, and
   implementation version.
4. `ModelIdentity` binds weights and model-file hashes, architecture, task,
   class order, dimensions, preprocessing, runtime, precision, export, and
   available training provenance. Unknown external provenance remains null.
5. `VisualBenchmarkCondition` binds one dataset, decoder, preprocessing,
   model, static inference policy, runtime, device, precision, deadline, and
   software commit.
6. `VisualBenchmarkResult` binds the exact condition and records frame
   outcomes, available timing summaries, scheduling outcomes, visual/resource
   metrics, unavailable metrics, failures, and the raw-artifact manifest.

Each identity hash is SHA256 of canonical sorted JSON excluding its own hash.
Absolute paths, usernames, timestamps that do not affect content, process IDs,
and random UUIDs are not identity fields. Changing ordered frame membership,
annotations, split assignment, scenario identity, decoder, preprocessing,
class order, model contents, precision, or runtime-relevant settings changes
the corresponding identity.

The boundaries are deliberate:

```text
stored PNG bytes
  -> source payload SHA256
  -> canonical decoded RGB8 SHA256 + decoder_configuration_id
  -> model tensor + preprocessing_configuration_id
  -> inference + model_identity_sha256
  -> condition_identity_sha256
  -> result_identity_sha256
```

The static v1 definition lives in `benchmarks/visual_static_v1/`. Its nine
unmaterialized templates cross input sizes 320, 416, and 640 with every-frame,
every-second-frame, and every-third-frame inference. Dataset and model
identities stay null until real validated artifacts exist. ROI is disabled,
and no adaptive policy is present. The matrix may be narrowed after pilot
validation if a size is unsupported; it must not be silently expanded.

Inspect the frozen contracts offline:

```bash
python main.py visual matrix-validate
python main.py visual condition-list
python main.py visual dataset-validate --input DATASET_IDENTITY.json
python main.py visual dataset-inspect --input DATASET_IDENTITY.json
python main.py visual model-validate --input MODEL_IDENTITY.json
python main.py visual model-inspect --input MODEL_IDENTITY.json
python main.py visual preprocessing-validate --input PREPROCESSING_IDENTITY.json
python main.py visual result-inspect --input RESULT.json
```

These commands validate files and templates only. They do not access the
network, start a simulator, materialize a template, or execute a benchmark.

## What is implemented

### Sensor and geometric baseline

- `map_baseline`, live `gazebo_lidar_2d`, and deterministic `replay` sources.
- Gazebo topic discovery, JSON scan parsing, startup timeout, health, frame age,
  effective frequency, dropped-frame count, and subprocess cleanup.
- A repository-owned `x500_research` model containing x500, 2D LiDAR, forward
  RGB, and 2D bounding-box cameras.
- Stable Gazebo RGB and labelled bounding-box sources, explicit PNG
  materialization, timestamp synchronization, pilot manifests, health
  summaries, and pilot-only dataset identity materialization.
- Ray clearing, occupied-cell inflation, unknown-cell penalty, forward-corridor
  risk, stopping distance, collision time, and recommended free direction.
- Inflated local costmap cells are projected into the global grid consumed by
  active replanning. Runtime LiDAR does not read map obstacle names or cells.
- Sensor loss and stale data are fail-safe danger conditions.

### ML research components

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

These components support offline development and simulation evaluation. They
do not constitute trained production models or real-airframe validation.

The camera record/replay, deterministic decoder, identity contracts, pilot
protocol, static templates, and timing contracts are experimental
infrastructure. They do not constitute a visual dataset, trained-model
evidence, a static visual benchmark result, adaptive-scheduler evidence, or
real-airframe evidence. They make no claim about visual accuracy, throughput,
latency, scheduling quality, or flight safety.

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
