# ML Study Workflow

The study registry records experiment identity and progress locally without
putting raw runs, weights, or SQLite state into Git.

## Tiers

| Tier | Matrix | Purpose |
| --- | ---: | --- |
| replay | 5 fixed cases × candidate and optional champion | Offline risk, calibration, traversability, and CPU latency gate |
| closed-loop | 5 scenarios × geometric, ML, and safety fusion | Collision and fail-safe gate before formal work |
| formal | 30 paired scenarios × four conditions | Final paired effect sizes and bootstrap 95% intervals |

Formal seeds `1001–1030` are reserved. The same scenario is repeated across
conditions so comparisons are paired rather than independent.

## Local state

```text
outputs/research/
  registry.sqlite
  study_results/
    STUDY_ID/
      TIER/
        run_queue.json
        scenarios/
        worlds/
```

`study run` materializes each non-replay scenario as a randomized SDF, matching
oracle planner truth, and sensor-fault manifest. Only the map-oracle condition
receives the randomized truth config. `run_queue.json` contains stable run IDs,
map/target/seed identity, launcher environment, setup commands, flight
arguments, and expected result paths. It is the hand-off contract for local
simulator workers.

Scenario materialization verifies A* reachability for the selected target.
When a sampled unknown obstacle disconnects the route, obstacles are removed in
deterministic reverse order; if necessary, equipment geometry falls back to
the baseline layout. Every adjustment is stored in the scenario manifest
instead of being hidden.

A worker writes one result per scenario and condition:

```text
outputs/research/study_results/
  SCENARIO_ID__CONDITION.json
```

The JSON must follow
[`config/schemas/study_result.schema.json`](../config/schemas/study_result.schema.json).
At minimum it contains:

```json
{
  "schema_version": 1,
  "run_id": "stable UUID from run_queue.json",
  "scenario_id": "complex-center-1003",
  "condition": "ml_lidar",
  "metrics": {
    "mission_success": 1,
    "landing_success": 1,
    "collision_count": 0,
    "safety_failure_count": 0,
    "risk_f1": 0.81,
    "traversability_iou": 0.69,
    "inference_p95_ms": 18.2
  }
}
```

## Commands

```bash
python main.py study create --name risk-cnn-v2 \
  --candidate models/lidar/risk_v2

python main.py study run STUDY_ID --tier replay
python main.py study run STUDY_ID --tier closed-loop
python main.py study run STUDY_ID --tier formal

python main.py study resume STUDY_ID
python main.py study status STUDY_ID
python main.py study compare STUDY_ID
python main.py study promote STUDY_ID
```

Commands are idempotent. Completed runs are preserved; resume resets only
running, failed, or blocked records. Missing simulator results remain pending
and are never converted into zero-valued metrics.

Promotion requires all three tiers to be complete. Replay enforces ≤50 ms ONNX
CPU P95 latency and no danger-recall regression greater than two percentage
points. Closed-loop and formal gates reject safety regressions. Formal
promotion also requires a preset quality gain or relative latency improvement.
