# Visual bridge A/B/C/D development comparison

Date: 2026-09-07  
Scope: development-only data-increment comparison. No protected-label access, unseen-scene evaluation, threshold tuning, formal admission, or promotion.

## Frozen design and execution

- The common A pool contains the prior 66-frame trusted base plus 48 regular-condition positives (114 unique frames).
- B adds 48 independently captured bridge positives; C adds 24 canonical target-isolated hard negatives; D adds both.
- All source-related variants remain in one lineage group. The 48 previously viewed paired images are development regression only and never enter training.
- Seeds 7, 17, and 27 use the same v2.11 initialization, 640 input, AdamW, 10 epochs, batch 6, 600 scheduled draws, and 100 optimizer steps. Augmentation and validation-driven selection are disabled.
- All 12 training cells completed with exact scheduled exposure. Every artifact remains `training_admitted=false` and `promotable=false`.

## Fixed development results

Values are three-seed means. Hit means planned-instance hit rate. FPR is the fraction of the fixed 48-frame no-target regression with at least one prediction at confidence 0.37.

| Arm | Original hit | Material hit | Background hit | Lighting hit | No-target FPR | Frozen policy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 0.917 | 0.083 | 0.694 | 0.472 | 0.417 | Fail |
| B | 0.889 | 0.472 | 0.778 | 0.556 | 0.410 | Fail |
| C | 0.972 | 0.083 | 0.694 | 0.472 | 0.222 | Fail |
| D | 0.917 | 0.472 | 0.806 | 0.583 | 0.368 | Fail |

The pre-frozen policy requires original mean/min-seed hit at least 0.833/0.750, material mean/min-seed at least 0.500/0.333, background and lighting means at least 0.600, and no-target FPR mean/max-seed at most 0.100/0.200. No arm passes every rule, so no arm or seed is selected.

## Diagnosis

- D preserves the A-level original-condition mean and improves background hit from 0.694 to 0.806. Its material and lighting gains are directional but stop just below their frozen thresholds (0.472 versus 0.500 and 0.583 versus 0.600).
- The decisive failure is false-positive control. C lowers FPR from A's 0.417 to 0.222, showing that the isolated hard negatives help, but it still misses the 0.100 requirement. D regresses to 0.368 when positives and negatives are combined.
- Across D's three seeds, the fixed no-target regression contains 50 transformer, 7 reactor, 6 capacitor-bank, and 4 switchgear predictions. Transformer-like false positives therefore dominate the remaining error and are the highest-value next collection target.
- By class, D's material planned hits total 1/9 capacitor-bank, 4/9 reactor, 3/9 switchgear, and 9/9 transformer. Lighting totals 2/9, 4/9, 7/9, and 8/9 respectively. The average gain does not establish uniform robustness across classes.
- The 24 new negatives are canonical target-isolated scenes, while the fixed no-target regression represents the older viewed failure distribution. Their incomplete transfer is evidence of source/composition mismatch, not grounds to lower the threshold or replace evaluation frames.

## Gate decision and next action

This stage closes as `development_complete_no_candidate`. The new-layout/unseen-asset test remains sealed. Selecting D, choosing a favorable seed, changing confidence, or opening the test would violate the frozen protocol.

The next bounded development step is to review the fixed no-target predictions by source and object morphology without using protected labels, then collect source-matched transformer-like hard negatives (ordinary cabinets, buildings, walls, and related structures) while retaining the positive bridge set. A new composition and its acceptance rules must be frozen before retraining.

This result supports only a directional diagnosis in the current viewed development scenes. It does not demonstrate cross-site generalization or identify a unique root cause.
