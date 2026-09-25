# Canonical view collection

Offline-only collection for the unchanged Simple, Medium and Complex scenes.
No PX4 is started and no vehicle is armed. Existing v2.11 stays official.

## Entrypoints

Use the existing visual CLI dispatcher, for example:

```sh
.venv/bin/python -c 'from src.cli.visual import main; raise SystemExit(main())' canonical-view-plan --map simple --output data/research/new-canonical-plan
```

- `canonical-view-plan --map simple|medium|complex --output DIR`: snapshot world, sensor source, obstacle config and label mapping; generate calibration and 40-view pilot plans.
- `canonical-view-collect --plan PLAN --output DIR --mode calibration|pilot [--review FILE]`: isolated Gazebo server, pose confirmation and synchronized RGB-D/truth. Pilot requires accepted, identity-bound calibration review.
- `canonical-view-audit --input RECEIPT --output FILE [--review FILE] [--development-reference FILE ...] [--protected-reference FILE ...]`: image integrity, semantic-review gating, exact/near duplicates and incremental candidate coverage.

Reference files use a `selected` array containing `image_sha256`, `perceptual_hash` and `split`. Protected references must not contain development members. Complete historical reference pools must be supplied; absence blocks release. The pilot audit never releases training by itself.

Review JSON uses canonical `identity`, `plan_identity`, `collection_identity`, and `views`. Each view decision binds `view_id` and `image_sha256`; accepted training candidates additionally require `all_visible_targets_correct: true` and, for empty target truth, `no_target_confirmed: true`. Calibration acceptance also includes `status: accepted` and a relative `collection_receipt` path. Do not auto-approve reviews.

## Current status

The September 5 recovery audit inventories 38 collection receipts and 22 plans
across the canonical and recovery directories. Pilot receipts contain 491 captured
records, including recovery aliases: 379 distinct pixel images remain, of which
226 pass the existing whole-frame framing policy and joint reference screen.
The geometric candidate pool is not a set of independently validated images.

The bound visual review retains 49 frames at contact-sheet resolution. Only 26
come from complete collections and join the provisional candidate manifest;
23 remain held for incomplete collections, and 177 unresolved frames are excluded
from this round with all source annotations preserved. Training is not admitted.
See [the increment audit](results/ml_canonical_increment_audit_20260905.md).

The source label placement is preserved by default. `--label-mode visual-instance`
creates an explicit annotation-only revision; do not use it to silently bypass a
failed source calibration or change device taxonomy. Visible pixel extrema need
the explicit versioned adapter; compatibility with the existing annotation scope,
protected world/trajectory isolation and final quotas remain training gates.
