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

`pilot_protocol.json` defines the first small recording gate. The pilot checks
identity continuity, annotation linkage, decoding determinism, storage,
sampling, scene coverage, and CLI inspection. It is not formal benchmark
evidence. The capture adapter must exist and the remaining operator fields
must be confirmed before recording.

## Schemas

Canonical JSON schemas live in `config/schemas/`. The local `schemas/README.md`
maps each benchmark artifact to its schema without duplicating definitions.
