# Verified Visual and Flight Baselines — 2026-08-20

This note separates evidence that closes its identity chain from historical
diagnostics that remain useful but cannot support a formal claim.

## Visual YOLO11n v2

The visual v2 chain is internally consistent across its training-view
identity, frozen package, paired blind evaluation, and static replay manifest.

| Item | Value |
| --- | --- |
| Training-view identity | `2491b645cfb14eda913a510f91da9b1058242c795e0472b159cf8dc1f1e8d1c1` |
| Package identity | `e2df0d854b2f126f94687790ebe19fc42ebb17ed0dfdd39b2f3d51f6ec064d08` |
| Frozen confidence threshold | 0.37 |
| Blind frames | 68,511 |
| Blind mAP50-95 | 82.07% |
| Blind macro-F1 | 93.25% |
| Blind precision / recall | 94.14% / 92.38% |
| Blind small-object recall | 81.96% |
| Blind no-target false-positive rate | 2.34% |
| Per-class recall | capacitor bank 93.90%; reactor 92.20%; switchgear 92.96%; transformer 90.67% |
| Static replay | 9/9 conditions materialized for 320/416/640 and every/every-2nd/every-3rd policies |

All three exported ONNX models are FP32, batch 1, static shape, opset 19, and
have package-bound equivalence receipts. These results are simulation evidence;
they do not establish real-camera or real-airframe performance.

## Historical 120-run LiDAR matrix

The local tree contains 120 JSON results covering 30 scenarios and four
conditions, with 30 results per condition. The generated comparison report
matches the arithmetic aggregates of those files. It records 119 mission
successes, 120 confirmed landings, one collision, and zero safety failures.

This matrix is **not a verified formal baseline**:

- the result files span 11 different formal-study identities and 11 execution
  commits;
- only three results match the identity and commit in the directory's current
  formal receipt;
- the complete matrix contains zero truth-danger samples;
- consequently, reported replans do not validate detection and avoidance of a
  deterministic dynamic blocker.

The report remains a historical static closed-loop diagnostic. Re-running the
same low-dynamic matrix would not close the evidence gap. The next planning
evidence must use accepted route-quality gates followed by deterministic
blockers whose spawn, detection, decision, replan, route switch, completion,
and landing events are all verified.

## Evidence boundary

- Use visual v2 for the current simulation visual baseline and static replay
  comparisons.
- Use the July v0.1 live-LiDAR results for sensor stability and representative
  static closed-loop flights.
- Do not cite the mixed-identity 120-run tree as formal dynamic-replanning or
  learned-LiDAR evidence.
- Establish a new receipt-bound dynamic benchmark before making route-switch
  reliability or safe-speed claims.
