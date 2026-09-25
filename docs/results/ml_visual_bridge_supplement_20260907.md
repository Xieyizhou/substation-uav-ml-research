# Visual bridge supplement v2 — 72-frame candidate freeze

Date: 2026-09-07  
Scope: development diagnosis only

## Outcome

The supplement is captured, explicitly reviewed and frozen as 72 non-admitted
candidate frames: 48 positive bridge frames and 24 difficult negatives. The
positive subset contains 16 three-variant lineage groups. The negative subset
contains 12 two-light lineage groups across the Simple, Medium and Complex
canonical layouts. All decisions are marked `AI-assisted`; no frame is
training-admitted or promotable. The unseen-scene test remains
`sealed_not_evaluated`.

Frozen ledger:
`data/research/ml_training_recovery_v1/visual-bridge-supplement-v2/frozen-positive-ledger.json`

Ledger identity:
`53848f4c3f96084cf8b9154c603f58fd864c3f1bf1418c998056d3ee59afaf1a`

Negative ledger:
`data/research/ml_training_recovery_v1/visual-bridge-supplement-v2/negative-redesign-v2/frozen-negative-ledger.json`

Negative ledger identity:
`a431c61dfd953d7736b91d6ebf14b256265d582cc47eddf3a018960b6cfeadfb`

Completion receipt:
`data/research/ml_training_recovery_v1/visual-bridge-supplement-v2/supplement-completion.json`

Completion identity:
`dad9ffad870e496131732e2666d68bb46710f708dd485fe0f6d0b528105795bf`

## Gate and retry history

The first matrix was not silently reused. Its pilot was rejected before Gazebo
launch because the plan declared `full_2d` while the persisted world still used
`visible_2d`. No v1 frame was captured. Matrix identity:
`bad49eb984a577b31b84c264e44dabed6be37d2bd170f85196d593511004b533`.

The corrected v2 builder changes `box_type` on the persisted XML tree and
rereads the saved world before freezing the matrix. Every collection receipt
binds the actual `full_2d` mode, `visual-instance` labels,
`top-level-equipment` hierarchy, `canonical-collection-gates-v2`, and its world
hash. Matrix identity:
`c70a84be7b372daa82397ea881bd3f80af507e1bd746b7cda09be5f5360cb945`.

The first v2 pilot process reached preflight but the sandbox denied Gazebo
Transport socket creation. Its zero-frame blocked receipt is retained at
`pilot-v1/runs/original/capture/collection-receipt.json`. The narrowly scoped
retry accepted only that exact zero-frame failure and wrote a separate
`capture-retry-1` lineage. It did not overwrite or count the blocked attempt.

## Capture and review evidence

| Phase | Captured | Accepted | Held | Review identity | Dedup identity |
| --- | ---: | ---: | ---: | --- | --- |
| Pilot | 6 | 6 | 0 | `2a5ed2c7674a396d1e8a5d0c645780b4d8edea0667ffd6b325a1750b14e24b58` | `1912f3d44750b23040a58d5ba5b422f6094aac80117a75c3de7204a8606f3768` |
| Remaining | 42 | 42 | 0 | `822fa4d626aafbd0d7c7bead832cc79145786858de006797912bef5b22f79627` | `6e7acc977e1637d953a1f8c3258795d5c3130d4f33647288f4128ed31a45b541` |

The remaining 42 frames were inspected through seven hash-bound contact sheets,
each showing the complete annotated image and a planned-target crop. Planned
targets were visible, untruncated and consistently boxed. Across each of the 14
remaining three-variant groups, the maximum target-box coordinate difference was
0 pixels. Target area occupied about 1.35% to 3.24% of the full image, confirming
that the batch exercises the intended far scale.

Exact and perceptual duplicate checks compare each frame with retained history,
the pilot, and unrelated peers in the same batch. Designed siblings sharing a
`pair_id` are recorded as one lineage and excluded only from mutual independence
checks. All 48 frames passed; the minimum independent dHash distance was 10.

## Negative scope correction and completion

The first negative design referenced three isolated background worlds. Before
collection, the repository's later `PAUSED.json` was discovered with
`collection_allowed=false` because those worlds are outside canonical product
scope. The plan was therefore superseded rather than silently executed. None of
the paused fence, masonry or pipe-rack sources was used.

The replacement uses target-isolated derivatives of the Simple, Medium and
Complex canonical worlds. It covers six layout-bound families, four frames each:
Simple cabinet array, Simple utility pole, Medium ordinary cabinet, Medium
utility pole, Complex control building and Complex ordinary cabinet. Each
family has two directionally separated positions rendered under normal and
cool-low light, producing 12 independent poses and 24 frames.

The world transformation is limited to removing configured four-class target
models, persisting `full_2d`, and applying the declared cool-low ambient/sun
values. Obstacle configurations remain conservative. Every receipt reports
empty synchronized truth, no observed instance labels and
`canonical-collection-gates-v2`.

The 12-frame gate pilot and 12-frame continuation were separately inspected in
hash-bound contact sheets. All 24 expected non-target subjects were visible and
all 24 frames were accepted. The pilot minimum independent dHash distance was 7;
the continuation minimum was 11. No exact or threshold-near independent
duplicate was admitted.

## Verification

- 24 focused canonical, shutdown, paired-factor and supplement tests passed.
- `git diff --check` passed.
- The v2.11 baseline verifier passed all 40 pinned-file checks and model-package
  validation; it still reports `training_data_ready=false`.
- The final status is
  `visual_bridge_supplement_frozen_pending_training_admission_decision`.
- No training, inference evaluation, threshold adjustment, protected-label
  access or model promotion was performed in this stage.
