# Static Visual Replay Benchmark v1

This directory freezes the paper-enabling contracts for static visual replay.
It contains specifications and unmaterialized condition templates only. It
does not contain a visual dataset, model weights, benchmark results, or
adaptive scheduling evidence.

## Frozen decisions

- Formal replay payloads use lossless PNG.
- Canonical decoded pixels are RGB8, HWC, uint8, and C-contiguous.
- Source payload, decoded pixels, preprocessing, model, condition, and result
  identities remain separate.
- The class order is transformer, switchgear, capacitor_bank, reactor.
- Static inference policies are `every_frame` and `every_nth_frame`.
- ROI is disabled in v1. Adaptive scheduling is outside this benchmark.

JPEG may appear only as explicit source provenance. Re-encoding JPEG as PNG
does not recover information lost by JPEG compression. Raw remains available
for provenance and storage/transport experiments, but it is not the canonical
formal payload.

## Template status

`conditions.json` contains nine templates: input sizes 320, 416, and 640,
crossed with every frame, every second frame, and every third frame. A
template has null dataset and model identities and cannot be executed or
reported as a completed condition. Materialization requires a validated
dataset identity, model identity, preprocessing identity, decoder identity,
runtime identity, and software commit.

No result rows are stored here. Future result tables must be generated from
validated `VisualBenchmarkResult` artifacts rather than typed manually.

## Pilot gate

`pilot_protocol.json` defines the version-3 small recording gate. The recorder
retains every valid source frame as PNG and measures the actual source rate;
the expected 10 Hz value is not forced or claimed as observed. The practical
100–300-frame range is not an automatic stop. Every-second and every-third
frame policies apply only to later replay of the frozen ordered dataset.

Invalid simulator truth remains an explicit recorded failure. V3 permits at
most 0.1% of RGB frames to reference invalid truth, with no more than two such
frames consecutively, and excludes those frames from dataset membership.
Unmatched and ambiguous synchronization remain disallowed. Required mission
phases use minimum synchronized simulation-time coverage rather than a share
of the recording's total duration.

The pilot checks identity continuity, annotation linkage, decoding
determinism, synchronization health, storage, scene coverage, and CLI
inspection. It is not formal benchmark evidence. Runtime topics, shared
Gazebo clock behavior, route visibility, observed rate, and synchronization
offsets still require a live probe and human review.

## Multi-scenario collection gate

`collection_protocol.json` freezes the model-development dataset plan after
the pilot gate. It expands to 60 recording-isolated scenarios: 40 training,
10 validation, and 10 extreme-map held-out test recordings. Dataset seeds
2001–2060 are disjoint from formal evaluation seeds 1001–1030. Adjacent frames
never cross split boundaries.

The v1 visual randomization identity includes only effects currently applied
to the rendered SDF: equipment scale, equipment position, light intensity,
and unknown obstacles. Weather, material age, camera noise, attitude jitter,
LiDAR noise, dropout, and outages remain explicit non-applied fields; they do
not create fictitious visual-domain variation. Every prepared scenario hashes
the final world, planner configuration, and applied visual configuration.

After all recordings pass per-recording synchronization and phase gates,
materialization creates separate `development` and `held_out_test`
DatasetIdentity artifacts. Aggregate validation requires every equipment
class and verified no-target frames in train, validation, and test. These
identities enable later training and static replay; they are not themselves
formal benchmark results.

Collection phases are normally generated from the flight lifecycle rather
than entered manually. The final outbound waypoint starts `approach`,
`goal_hover` starts `close_inspection`, and `return_to_start` starts
`target_transition`. Recording stops only after the flight reports confirmed
landing, then performs a short stream drain and per-recording validation.

## Schemas

Canonical JSON schemas live in `config/schemas/`. The local `schemas/README.md`
maps each benchmark artifact to its schema without duplicating definitions.
