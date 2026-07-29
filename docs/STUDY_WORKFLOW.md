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

### Static replay gate before adaptive scheduling

`benchmarks/visual_static_v1/conditions.json` freezes nine unmaterialized
static templates: sizes 320/416/640 crossed with every frame/every second/every
third frame. A template is not executable until exact dataset, decoder,
preprocessing, model, runtime, device, precision, deadline, and commit
identities are supplied. Unrun templates never produce result rows.

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

```bash
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

Commands are idempotent. Completed runs are preserved; resume resets only
running, failed, or blocked records. Missing simulator results remain pending
and are never converted into zero-valued metrics.

Promotion requires all three tiers to be complete. Replay enforces ≤50 ms ONNX
CPU P95 latency and no danger-recall regression greater than two percentage
points. Closed-loop and formal gates reject safety regressions. Formal
promotion also requires a preset quality gain or relative latency improvement.
