# Visual augmentation A/B/C/D development comparison

Date: 2026-09-07  
Scope: development-only incremental-data comparison. No protected-label access, unseen-scene evaluation, threshold tuning, or promotion.

## Frozen inputs and execution

- Common base: 66 frames, re-reviewed frame by frame and frozen as `trusted-training-base-v1`.
- New candidates: `visual-augmentation-240-v2`, containing 144 appearance/lighting positives, 48 regular positives, and 48 isolated hard negatives.
- A: base + regular (114 unique frames).
- B: A + appearance/lighting positives (258).
- C: A + hard negatives (162).
- D: A + both additions (306).
- Seeds: 7, 17, 27. Every cell used the same frozen v2.11 initialization, 640 input, AdamW, 10 epochs, batch 6, 600 scheduled draws, and 100 optimizer steps. Augmentation and validation-driven selection were disabled.
- All 12 cells completed; actual exposure exactly matched the frozen schedule.

## Fixed development results

Values are three-seed means. Hit is planned-instance hit rate; FPR is the fraction of isolated no-target frames with at least one prediction at confidence 0.37.

| Arm | Original hit | Material hit | Background hit | Lighting hit | No-target FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.917 | 0.083 | 0.694 | 0.472 | 0.417 |
| B | 0.667 | 0.444 | 0.472 | 0.639 | 0.514 |
| C | 0.944 | 0.056 | 0.750 | 0.417 | 0.035 |
| D | 0.778 | 0.667 | 0.694 | 0.694 | 0.160 |

The frozen v2.11 reference scored 0.250/0.000/0.167/0.250 on original/material/background/lighting planned-instance hit, and 0.250 no-target FPR. These values are context only; A is the intended data-increment control.

## Interpretation

- Appearance examples materially improve material sensitivity: D gains +0.584 material hit over A. B also improves material and lighting, but loses original-condition performance and does not control false positives.
- Hard negatives are effective for this isolated development set: C reduces mean no-target FPR from 0.417 to 0.035 while preserving A-like original hit, but does not solve material sensitivity.
- D combines the strongest material and lighting results with substantially lower FPR than A/B. However, its mean original hit is 0.778 versus A's 0.917, and seed behavior is unstable: D original hit spans 0.583–0.917 and no-target FPR spans 0.021–0.354.
- Therefore the intended success condition is not cleanly met. The result is a real tradeoff, not a promotion candidate.

No seed or arm is selected. Opening the sealed unseen-scene test now would require a post-hoc acceptance rule. The next development step should pre-freeze an acceptable original-hit/FPR retention rule and test a composition between C and D that lowers appearance exposure or strengthens preservation of the base/regular subsets.

All outputs remain `training_admitted=false` and `promotable=false`.
