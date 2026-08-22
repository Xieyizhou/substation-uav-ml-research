# Sandbox active semantic inspection

`active_semantic_inspection` uses the frozen YOLO11n v2 artifact. Runtime code
may read map bounds and obstacle occupancy, but not equipment classes, target
identifiers, completion truth, or the true target count. Truth matching exists
only in the offline report.

```text
python -m src.cli.sandbox active-inspection-plan --map MAP.json \
  --observations OBSERVATIONS.jsonl \
  --policy config/perception/active_inspection_policy.json \
  --output active-inspection-plan.json
```

The command writes the plan, event JSONL, SVG route preview, and evaluation
report. Real-domain v3 is deferred research; its archives, audits, and curated
manifests remain preserved and are not consumed by this runtime.

Live Gazebo input is provided by `ActiveInspectionLiveBridge`: RGB frames pass
through the frozen equipment detector and temporal tracker, while range must
come from a synchronized depth provider. Missing or invalid depth never creates
a target; map equipment truth is not a supported runtime input.
