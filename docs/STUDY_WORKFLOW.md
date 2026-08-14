# ML Study Workflow

The study registry records experiment identity and progress locally without
putting raw runs, weights, or SQLite state into Git.

## Tiers

| Tier | Matrix | Purpose |
| --- | ---: | --- |
| replay | 5 fixed cases × candidate and optional champion | Offline risk, calibration, traversability, and CPU latency gate |
| closed-loop | 5 scenarios × geometric, ML, and safety fusion | Collision and fail-safe gate before formal work |
| formal | 30 paired scenarios × four conditions | Final paired effect sizes and bootstrap 95% intervals |

Formal seeds `1001–1030` are reserved. The same scenario is repeated across
conditions so comparisons are paired rather than independent.

## Future visual replay inputs

Static visual-model and scheduling studies will use one validated camera
recording per scenario. Every candidate resolution, model, skip policy, and
future adaptive scheduler must consume the same `frames.jsonl` manifest order
and hash-verified payloads. This prevents input drift between conditions.
Formal visual inputs use PNG. Raw may be retained as provenance or as an
explicit storage/transport experiment, but it is not the canonical formal
payload. JPEG cannot be canonical; converting an already lossy JPEG to PNG
does not make the source lossless. Each frame declares storage
`payload_format` separately from consumer-visible `pixel_format`, so benchmark
code must not infer RGB or BGR from a codec.

Camera recordings remain under ignored raw-data/output paths and are referenced
by identity rather than copied into the repository. Failed, missing, corrupt,
duplicate, gapped, and non-monotonic outcomes remain visible in recording
summaries. A visual study result may report a latency stage only when
`VisualTiming` marks it as measured and records its provenance; unavailable
stages remain null.

Future visual study identity must also include the canonical decoder
configuration ID, source-payload SHA256, and decoded-content SHA256. This
locks the CPU decoding backend, version, colour conversion, grayscale, alpha,
and raw-layout policies alongside the ordered input frames. Timing values are
measurements, not identity fields.

### Visual pilot gate

The versioned pilot protocol is
`benchmarks/visual_static_v1/pilot_protocol.json`. Before a fixed visual
dataset is collected, the project must:

1. implement a stable PNG camera-source adapter;
2. confirm the actual source rate, route visibility, and ground-truth source;
3. record one training-map, center-target, seed-2001 approach/inspection run;
4. retain every valid source frame; use 100–300 frames only as a practical
   manual inspection target;
5. validate hashes, ordered identity, decoder determinism, annotation linkage,
   storage, phase coverage, no-target frames, occlusion where available, and
   the offline CLI workflow.

The range is a manual-inspection target, not a scientific sample-size claim.
The pilot is pipeline validation and cannot be included as formal performance
evidence merely because it passes.

Recording and benchmark skipping are separate. Pilot protocol v3 preserves
every valid source frame and measures the actual rate. It does not force or
claim the expected 10 Hz rate and does not reduce the pilot to 5 Hz. Benchmark
skipping operates later on the same frozen ordered membership. Duplicate
timestamps, source gaps, invalid frames, unmatched frames, no-target frames,
and near-duplicate concentration remain explicit.

Gazebo can emit a zero-visible-area box for one truth period while a target
crosses the image boundary. V3 keeps that truth message explicitly invalid,
retains the corresponding RGB source frames, and excludes them from pilot
dataset membership. Acceptance permits at most 0.1% affected RGB frames and
at most two consecutively; unmatched and ambiguous synchronization remain
disallowed. Required mission phases use a minimum of one second of valid
synchronized simulation-time coverage instead of a fraction of the complete
recording.

Before recording, `visual gazebo-inspect` must resolve the configured RGB and
truth topics and verify `gz.msgs.Image` and
`gz.msgs.AnnotatedAxisAligned2DBox_V`. `visual gazebo-probe` must then receive
one valid message from each source within an explicit timeout. A missing
topic, incompatible message type, invalid clock, or truth-source failure
blocks the pilot.

The synchronization-health gate reviews exact, nearest, unmatched, ambiguous,
and invalid-truth counts plus the measured offset distribution. The default
33.334 ms tolerance is provisional. An expected/observed rate difference is
reported but does not itself change the protocol. Changes to collection or
sampling policy require a later pilot protocol version.

The pilot review gate requires complete manifests, hash-valid PNGs,
deterministic decoding, one synchronization row per RGB frame, explicit
mission-phase events, labelled and verified no-target frames, and a successful
pilot-role identity. Passing this gate validates the pipeline only. No formal
visual dataset has been collected, no trained visual model evidence exists,
no static visual benchmark has run, and no adaptive scheduler exists.

During the existing flight task, mark `cruise_distant`, `approach`,
`close_inspection`, and `target_transition` with `visual pilot-phase`. Each
event uses the latest accepted RGB simulation timestamp exposed by the
recorder. A phase listed in an event file but not represented by a
synchronized frame does not satisfy coverage.

Training, validation, held-out, and formal partitions occur at scenario, map,
route, recording, or seed boundaries. Adjacent frames from one recording must
never be randomly divided between train and evaluation splits.

### Multi-scenario visual dataset collection

After the PNG v3 pilot passes, materialize the frozen collection plan:

```bash
python main.py visual collection-plan \
  --output data/research/visual_collection_v1/collection_plan.json
python main.py visual collection-plan-validate \
  --input data/research/visual_collection_v1/collection_plan.json
python main.py visual collection-status \
  --plan data/research/visual_collection_v1/collection_plan.json
```

The plan contains 40 train scenarios using seeds 2001–2040, 10 validation
scenarios using 2041–2050, and 10 extreme-map held-out test scenarios using
2051–2060. Map-target combinations are balanced within each split allocation.
Formal evaluation seeds 1001–1030 are rejected. One complete recording is the
minimum split unit.

Prepare one randomized, reachable scenario before launching PX4/Gazebo:

```bash
python main.py visual collection-prepare \
  --plan data/research/visual_collection_v1/collection_plan.json \
  --scenario-id training-top_right-2001
```

The result prints the launcher environment, world, planner, scenario report,
flight command, and recording directory. Start the returned world with the
`x500_research` model, probe RGB/truth, then record:

```bash
python main.py visual collection-record \
  --plan data/research/visual_collection_v1/collection_plan.json \
  --scenario-id training-top_right-2001 \
  --scenario-report \
    data/research/visual_collection_v1/scenarios/training-top_right-2001/scenario.json
```

Start the returned flight command only after the recorder is receiving RGB
frames. The flight command writes `flight_events.jsonl`; the recorder maps the
first outbound route, final outbound waypoint, goal hover, and return route to
`cruise_distant`, `approach`, `close_inspection`, and `target_transition`
using the latest RGB Gazebo timestamp. Manual `collection-phase` commands
remain available for debugging but are not required in the normal workflow.
Manual runs must pass `--manual-lifecycle` plus exactly one explicit stopping
mode and omit `--visual-mission-events` from the flight command.

After the flight reports `completed`, `landed`, and `landing_confirmed=true`,
the recorder drains the visual streams for one second, stops, validates the
recording, and writes an acceptance receipt. A failed flight produces a
partial recording and a nonzero command result. `collection-status` reads
receipts instead of repeatedly hashing and decoding every PNG. An active
recorder with `live_status.json` is reported as `recording`; `partial` is
reserved for an interrupted output without complete manifests. The command
also identifies the first invalid-receipt, unvalidated, or missing scenario.
Final materialization always revalidates every recording.

After one scenario has passed interactively, the remaining scenarios can be
run sequentially with the resumable local orchestrator:

```bash
python main.py visual collection-run \
  --plan data/research/visual_collection_v1/collection_plan.json
```

The command prepares each missing scenario, starts its PX4/Gazebo world,
waits for a successful RGB/truth probe, starts the recorder before the flight,
and requires a current validation receipt before advancing. It never
overwrites a partial or invalid recording. Per-scenario simulator, probe,
recorder, and flight logs are retained under
`data/research/visual_collection_v1/batch_logs/`. Simulator standard output is
discarded because the non-interactive PXH console output is unbounded; simulator
standard error and all probe/recorder/flight output remain available. Use
`--max-scenarios 1` for a bounded smoke run or `--dry-run` to inspect the
pending order.

Only after all 60 recordings pass:

```bash
python main.py visual collection-materialize \
  --plan data/research/visual_collection_v1/collection_plan.json
```

Materialization fails if any recording is missing, any split lacks transformer,
switchgear, capacitor-bank, reactor, or verified no-target coverage, decoder
identities differ, or scenario/recording/seed identity overlaps. It writes
separate development and held-out-test identities; the held-out identity must
not be used for fitting or model selection.

### Visual collection v2

Protocol v2 uses tracked frozen layouts and equipment-centered yaw routes.
The planner derives obstacle footprints from the same yaw-rotated geometry
used by Gazebo; fixed infrastructure does not receive equipment randomization.
Materialize or verify the frozen 50-recording plan with:

```bash
python main.py visual collection-plan \
  --protocol v2 \
  --output benchmarks/visual_static_v2/collection_plan.json
python main.py visual collection-audit \
  --plan benchmarks/visual_static_v2/collection_plan.json
```

For experiment outputs, copy the frozen plan to the collection root, then run
the resumable collector:

```bash
mkdir -p data/research/visual_collection_v2
cp benchmarks/visual_static_v2/collection_plan.json \
  data/research/visual_collection_v2/collection_plan.json
python main.py visual collection-run \
  --plan data/research/visual_collection_v2/collection_plan.json \
  --output-root data/research/visual_collection_v2
```

`collection-status` defaults to the `recordings/` directory beside the plan.
The v2 visual runner computes a route-specific 180--360 second flight timeout.
The LiDAR closed-loop worker independently computes a 240--480 second budget
from the A* path length and its outbound and return speeds. Pass
`--flight-timeout` only as an explicit diagnostic override. Each target route
contains an identity-bound A* return path generated from the final observation
cell to the start cell; it is not a reversal of accumulated outbound segments.
It must pass target-count, phase-visibility, size-bin, truncation,
synchronization, and landing gates before the next scenario starts. The blind
split is eligible for collection integrity checks only; do not create
predictions or a training view from it before package freeze.

### Visual baseline training

Create the deterministic YOLO view only from the development identity:

```bash
python main.py visual training-view-materialize \
  --collection-root data/research/visual_collection_v1 \
  --output data/research/visual_yolo_v1
```

The view caps each training class at 5,000 frames, retains up to 8,000
verified no-target frames, and creates a smaller class-balanced validation
view. Quotas are proportional by recording and selected frames are evenly
spaced by source sequence. A separate full-validation partition is generated
for the one-time post-training evaluation. Images are hard-linked, with a
relative symlink fallback when hard links are unavailable; PNG bytes are not
duplicated.

Run the one-epoch gate before the full baseline:

```bash
python main.py visual train-yolo --smoke
python main.py visual train-yolo
```

Training requires a clean tracked worktree and records the commit, resolved
configuration, package versions, training-view identity, pretrained-weight
hash, and checkpoint hashes. The frozen baseline is YOLO11n at 640 pixels on
Apple MPS, batch 8 with an explicit batch-4 restart policy for out-of-memory
failures. Resume uses the run's `last.pt`; it does not silently change
hyperparameters.

After full validation and threshold selection, export the identity-bound
package:

```bash
python main.py visual export-yolo \
  --weights models/equipment/visual-yolo11n-baseline-v1/weights/best.pt \
  --training-view-identity \
    data/research/visual_yolo_v1/identity/training_view_identity.json \
  --training-provenance \
    models/equipment/visual-yolo11n-baseline-v1/training_provenance.json \
  --validation-results \
    outputs/research/visual_yolo11n/full_validation_results.json \
  --equivalence-dataset data/research/visual_yolo_v1 \
  --output models/equipment/visual-yolo11n-baseline-v1-package
```

The package contains one weights identity and fixed FP32 ONNX exports at 320,
416, and 640. Export runs the fixed 200-frame equivalence gate for every size
and writes `manifest.json` only after all three pass. A failed export remains a
non-executable staging directory. `heldout-view-materialize` requires that
finalized package and writes a receipt containing the canonical 640 model and
the full-validation confidence threshold. Held-out evaluation cannot search
or override that threshold.

For the v2 comparison, keep the new blind split sealed until both packages
are finalized. Materialize one shared view and an identity-bound access
receipt, then run both canonical 640 ONNX models in the fixed order:

```bash
python main.py visual paired-heldout-materialize \
  --collection-root data/research/visual_collection_v2 \
  --v1-package models/equipment/visual-yolo11n-baseline-v1-package \
  --v2-package models/equipment/visual-yolo11n-baseline-v2-package \
  --output data/research/visual_yolo_v2_blind
python main.py visual paired-heldout-evaluate \
  --dataset data/research/visual_yolo_v2_blind \
  --output outputs/visual_yolo_v2/paired_blind \
  --device cpu
```

The evaluator rejects package or model hash changes, uses each package's
frozen validation threshold, saves predictions for later analysis, and
computes a 2,000-repeat paired bootstrap over recording units. A completed
result cannot be overwritten or rerun.

### Static replay gate before adaptive scheduling

LiDAR candidates must pass a readiness audit before any ML-controlled flight.
The audit rejects dirty training commits, incomplete risk-label coverage,
unbound ONNX external data, identity mismatches, and missing per-class metrics.
A deterministic development-only overlay view may bootstrap sandbox testing
from existing Gazebo scans; it is provenance-labelled and is not formal data.

```bash
python main.py data overlay \
  --source outputs/research/pilot/raw/training_2001.jsonl:training:2001 \
  --source outputs/research/pilot/raw/complex_2041.jsonl:complex:2041 \
  --source outputs/research/pilot/raw/extreme_2051.jsonl:extreme:2051 \
  --output outputs/research/lidar_overlay_v1
python main.py model readiness \
  --package models/lidar/risk-v1 \
  --dataset outputs/research/lidar_overlay_v1
python main.py model replay-gate \
  --package models/lidar/risk-v1 \
  --dataset outputs/research/lidar_overlay_v1 \
  --output outputs/research/lidar_replay_gate_v1
```

The replay gate reads only the validation partition. It requires validation
macro-F1 of at least 0.50, danger recall of at least 0.60, and ONNX CPU P95
latency no greater than 50 ms. These are flight-entry safety gates, not a
formal performance claim.

`benchmarks/visual_static_v1/conditions.json` freezes nine unmaterialized
static templates: sizes 320/416/640 crossed with every frame/every second/every
third frame. A template is not executable until exact dataset, decoder,
preprocessing, model, runtime, device, precision, deadline, and commit
identities are supplied. Unrun templates never produce result rows.

Materialize and run the frozen matrix after the held-out receipt exists:

```bash
python main.py visual static-replay-materialize \
  --package models/equipment/visual-yolo11n-baseline-v1-package \
  --dataset data/research/visual_yolo_v1_heldout \
  --output outputs/research/visual_static_v1
python main.py visual static-replay-run \
  --input outputs/research/visual_static_v1 \
  --package models/equipment/visual-yolo11n-baseline-v1-package \
  --dataset data/research/visual_yolo_v1_heldout
```

Adaptive scheduling starts only after:

- the pilot gate passes;
- a fixed split-isolated dataset identity exists;
- a real hashed model package and preprocessing identity exist;
- supported static conditions complete successfully;
- result artifacts pass identity and availability validation.

Paper tables must be generated from validated result artifacts. Missing
metrics remain null, failed frames remain failures rather than zero-latency
successes, and results without raw-artifact manifest hashes cannot claim
completion.

The existing study registry does not yet schedule visual replay conditions.
The frozen visual contracts and templates do not create a second registry and
do not imply that a dataset, model, static result, or adaptive scheduler
exists.

## Local state

```text
outputs/research/
  registry.sqlite
  study_results/
    STUDY_ID/
      TIER/
        run_queue.json
        scenarios/
        worlds/
```

`study run` materializes each non-replay scenario as a randomized SDF, matching
oracle planner truth, and sensor-fault manifest. Only the map-oracle condition
receives the randomized truth config. `run_queue.json` contains stable run IDs,
map/target/seed identity, launcher environment, setup commands, flight
arguments, and expected result paths. It is the hand-off contract for local
simulator workers.

Scenario materialization verifies A* reachability for the selected target.
When a sampled unknown obstacle disconnects the route, obstacles are removed in
deterministic reverse order; if necessary, equipment geometry falls back to
the baseline layout. Every adjustment is stored in the scenario manifest
instead of being hidden.

A worker writes one result per scenario and condition:

```text
outputs/research/study_results/
  SCENARIO_ID__CONDITION.json
```

The JSON must follow
[`config/schemas/study_result.schema.json`](../config/schemas/study_result.schema.json).
At minimum it contains:

```json
{
  "schema_version": 1,
  "run_id": "stable UUID from run_queue.json",
  "scenario_id": "complex-center-1003",
  "condition": "ml_lidar",
  "metrics": {
    "mission_success": 1,
    "landing_success": 1,
    "collision_count": 0,
    "safety_failure_count": 0,
    "risk_f1": 0.81,
    "traversability_iou": 0.69,
    "inference_p95_ms": 18.2
  }
}
```

## Commands

Before a five-scenario or larger flight matrix, run the bounded challenge
tier. It materializes one simulator-only blocker on the original A* route
while retaining a verified detour. The same grounded scenario is exercised
once by geometric LiDAR, ML LiDAR, and safety-max fusion. Each condition must
detect the threat, attempt and complete a replan, replace the active route,
finish the mission, and land without collision.

```bash
python main.py study create --name risk-cnn-v2 \
  --candidate models/lidar/risk_v2

python main.py study run STUDY_ID --tier replay
python main.py study execute-challenge STUDY_ID
python main.py study capabilities STUDY_ID --tier challenge
python main.py study run STUDY_ID --tier closed-loop
python main.py study execute-closed-loop STUDY_ID --max-runs 1
python main.py study capabilities STUDY_ID --tier closed-loop
python main.py study run STUDY_ID --tier formal
python main.py study capabilities STUDY_ID --tier formal

python main.py study resume STUDY_ID
python main.py study status STUDY_ID
python main.py study compare STUDY_ID
python main.py study promote STUDY_ID
```

Commands are idempotent. Completed runs are preserved; resume resets only
running, failed, or blocked records. Missing simulator results remain pending
and are never converted into zero-valued metrics.

The closed-loop worker is sequential and fail-fast. Start with `--max-runs 1`;
only remove the limit after the first run has a confirmed landing, healthy
LiDAR, no collision, and a valid result receipt. Each study stores results
under its own `STUDY_ID/TIER/results` directory so separate candidates cannot
overwrite one another.

Capability reports always read the explicitly selected tier. A formal report
is observational and does not replace the preflight challenge. It also exposes
both predicted and truth danger sample counts so a prediction cannot be
mistaken for ground-truth danger coverage. Safe early replanning may correctly
produce zero truth-danger samples; the challenge is grounded by the
route-intersecting blocker recorded in its scenario manifest.

Promotion requires all three tiers to be complete. Replay enforces ≤50 ms ONNX
CPU P95 latency and no danger-recall regression greater than two percentage
points. Closed-loop and formal gates reject safety regressions. Formal
promotion also requires a preset quality gain or relative latency improvement.
