# E21: frozen synthetic evaluation with visible weaknesses

Historical experiment: 2026-09-28. [Selected results, count tables and source identities](../../data/sample_outputs/case_studies/e21-model-experiment.json).

E21 trained matched seeds **7, 17 and 27** at **704-pixel input** for two epochs, then used the existing development criteria to choose a candidate and confidence threshold before opening the sealed Q3 test. The data covers four canonical synthetic equipment families with source-bound annotations reviewed by AI. These results do not establish accuracy on real sites, photographs or unseen equipment assets.

| Development result | Recorded value |
| --- | ---: |
| Paired composite-score deltas, seeds 7 / 17 / 27 | +0.015946 / +0.033785 / +0.005477 |
| Mean paired delta | +0.018403, required ≥ +0.01 |
| Positive paired deltas | 3, required ≥ 2 |
| Individual gates passed, seeds 7 / 17 / 27 | true / true / **false** |
| Selected candidate / threshold | candidate-17 / 0.43 |

A positive composite score did not make every seed eligible: seed 27 failed individual development gates. The selected seed 17 was frozen with ONNX SHA-256 `0f76dd97623943f092522da950c574207a72ee0562be976b580a30ae1c9bf9a2` before the 56-frame, two-layout Q3 evaluation.

| Q3 pooled result | Current baseline | Parent baseline | Candidate 17 |
| --- | ---: | ---: | ---: |
| Confidence threshold | 0.11 | 0.15 | 0.43 |
| Macro F1 | 0.219719 | 0.371358 | **0.868032** |
| Small-object recall | 0.272727 | 0.454545 | **0.545455** |

Thresholds were selected under the historical development protocol, not equalized after viewing Q3. The candidate passed the recorded Q3 gates, including absolute macro F1 ≥ 0.85, but its class detail shows a meaningful remaining weakness:

| Candidate class | TP | FP | FN | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 66 | 5 | 5 | 0.929577 | 0.929577 |
| Switchgear | 42 | 9 | 22 | **0.656250** | **0.730435** |
| Capacitor bank | 55 | 3 | 10 | 0.846154 | 0.894309 |
| Reactor | 67 | 7 | 5 | 0.930556 | 0.917808 |

Small-object recall was **12/22**, not evidence of robust small-target detection. The historical evaluator recomputed “small” as `source_box_area * (640/1920)^2 < 32^2`; it kept the 640-reference definition even for the 704-input candidate. Reading an unrelated cached `small` flag would change that metric's meaning.

Q3 has been consumed and cannot serve as a fresh holdout for future model selection. Weights, source images, full training receipts and prediction caches remain outside this bundle. The supplied checker can recompute precision, recall and F1 from the excerpt counts, reconcile layout sums and confirm frozen identities. It cannot independently rerun detection matching, review labels, train the model or infer on the original frames. The useful reusable part is the selection and reporting discipline: retain failed seeds, freeze selection before test access, and report weak classes alongside the pooled score.
