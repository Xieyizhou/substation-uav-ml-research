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

## Schemas

Canonical JSON schemas live in `config/schemas/`. The local `schemas/README.md`
maps each benchmark artifact to its schema without duplicating definitions.
