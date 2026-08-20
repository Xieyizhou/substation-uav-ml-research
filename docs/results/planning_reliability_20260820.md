# Planning Reliability and Temporal Perception Results

## Scope

This report records the first deterministic dynamic-blocker benchmark, the
speed-envelope study, temporal visual observations, a real-image stress test,
semantic inspection planning, and height-layer planning. All flight evidence
is from PX4 SITL and Gazebo on one macOS development machine. It is not
real-airframe evidence.

## Dynamic replanning

The frozen benchmark contains 12 runs: three map structures crossed with
early, mid-route, near-target, and return-leg blocker injection phases. Every
run required a simulator-spawned blocker and the complete event chain from
spawn through detection, decision, replanning, route acceptance, resumption,
mission completion, and landing.

| Metric | Result |
| --- | ---: |
| Runs | 12 / 12 |
| Successful replan | 100% |
| Route-switch correctness | 100% |
| Mission completion | 100% |
| Landing success | 100% |
| Complete event chain | 100% |
| Collisions | 0 |
| Safety failures | 0 |
| False replans | 0% |

The benchmark passes its frozen acceptance criteria. Missing blocker or event
evidence cannot be counted as success.

## Speed envelope

The speed matrix contains four representative dynamic scenarios, five speeds
(0.50, 0.75, 1.00, 1.25, and 1.50 m/s), and three repeats per combination.
The final recommendation is the highest contiguous speed whose 12 runs pass
the fixed collision, safety, route-switch, completion, landing, event-chain,
and detection-to-resume latency criteria.

All 60 registered runs completed. Metrics were then uniformly regenerated
from the original telemetry with the altitude-aware collision map: low assets
that do not intersect the configured flight envelope are not physical
collisions. The raw telemetry and mission events were not changed.

| Speed | Runs | Mission | Route switch | Event chain | Landing | Collisions | P95 detect-to-resume | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.50 m/s | 12 | 100% | 100% | 100% | 100% | 0 | 2.49 ms | yes |
| 0.75 m/s | 12 | 91.67% | 91.67% | 91.67% | 100% | 0 | incomplete | no |
| 1.00 m/s | 12 | 100% | 100% | 100% | 100% | 0 | 2.90 ms | yes |
| 1.25 m/s | 12 | 100% | 91.67% | 91.67% | 100% | 1 | incomplete | no |
| 1.50 m/s | 12 | 66.67% | 66.67% | 66.67% | 100% | 0 | 2.37 ms | no |

The conservative recommended maximum is **0.50 m/s**. Although the isolated
1.00 m/s tier passed, the 0.75 m/s tier between them did not; the frozen rule
therefore does not treat 1.00 m/s as a contiguous safe operating envelope.
The non-monotonic result also shows why one successful high-speed tier is not
enough to justify increasing the Sandbox default.

## Temporal visual observations

The temporal filter uses class-aware association, confidence smoothing,
three-of-five confirmation, a 200 ms detection hold, and explicit expiration.
It reports inference-frame recall separately from full-timeline coverage.

| Input and policy | Inference frames | Inference-frame recall | Timeline coverage recall |
| --- | ---: | ---: | ---: |
| 320, every frame | 68,511 | 91.99% | 88.83% |
| 320, every second frame | 34,256 | 91.88% | 81.80% |
| 320, every third frame | 22,837 | 92.06% | 75.91% |
| 640, every frame | 68,511 | 92.46% | 89.15% |

Skipping inference frames barely changes recall on frames that are processed,
but it materially reduces complete-timeline coverage. The 320 input remains
the real-time candidate; the 640 input is the quality reference.

For the 416-pixel ONNX model, a cold-process probe measured 947.27 ms for
model construction plus the first prediction. Reusing the loaded model for
200 predictions produced 28.07 ms P50, 29.07 ms P95, 29.36 ms P99, and
29.72 ms maximum latency. This probe includes the Ultralytics prediction
boundary; it is not interchangeable with the lower core-runtime figures in
the static replay table.

## Real-image stress test

The stress view contains 752 external images under an MIT license. Source
annotations map only to switchgear and capacitor-bank classes, so transformer
and reactor metrics are intentionally unsupported rather than inferred.

At the frozen 0.37 threshold, neither resolution produced a matched true
positive for the two represented classes. The 320 model produced 19
capacitor-bank and 5 switchgear false positives, with a 57.34% no-target
frame false-positive rate. The 640 model produced 25 and 7 respectively, with
a 70.20% no-target false-positive rate. Small-object recall was 0% in both
cases. These are negative domain-transfer results, not a reason to alter the
already-frozen model. They show that the strong synthetic blind result does
not transfer to this real-image source without new representative training
coverage and a reviewed semantic class mapping.

## Semantic inspection planning

A stable temporal observation, calibrated camera intrinsics, vehicle pose,
and depth estimate are converted into a map-relative equipment estimate. A
held observation cannot initialize a location. The estimate must match a
compatible mapped target within tolerance, and its generated inspection route
must pass the same Route Quality gate as an edited map.

The reference transformer case localized the target with 0.0 m simulated
error. Its inspection route passed with 2.0 m minimum clearance, 0% boundary
exposure, 82.61% interior coverage, a 2.527 detour ratio, and a 126 s estimated
duration. A dense-map attempt with excessive detour was correctly rejected.

## Height-layer planning

The deterministic 2.5D planner represents altitude as explicit ordered layers
and permits vertical transitions only between free cells. Obstacle height,
vehicle half-height, vertical clearance, and horizontal inflation determine
blocked nodes.

The dense-map reference plan used 1.5, 3.0, and 5.0 m layers. It found a
61.0 m route with 58.0 m horizontal travel, 3.0 m vertical travel, and two
layer changes. This is a planning artifact and test result, not a PX4 flight
validation of vertical clearance.

## Interpretation

The strongest current evidence is deterministic simulator replanning at the
0.50 m/s conservative envelope and the visual detector on matched synthetic
domains. Real-image transfer failed on the two supported classes, the other
two classes remain unrepresented, and height-layer planning has not yet been
flown. These boundaries must remain visible when the Sandbox exposes the
results.
