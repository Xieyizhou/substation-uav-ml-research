# Local research sandbox app

The local App combines a read-only experiment inspector with a controlled job
operator. It shows collection progress, environment checks, process health,
bounded logs, route phases, recorded PNG frames, and the complete visual ML
lifecycle from training-view identity through frozen-package, paired blind,
and static replay results. It can also run a small set of fixed workflows
without exposing an arbitrary command shell.

The LiDAR Gates tab reads the latest identity-bound validation replay receipt
and its matching closed-loop study. It reports gate state, model identity,
replay quality and latency, completed flights, landings, collisions, safety
buffer entries, sensor health, and live inference latency.

Start it from the repository root:

```bash
./scripts/run_sandbox_app.sh
```

Then open `http://127.0.0.1:8765`. The equivalent Demo Profile command is:

```bash
python3 main.py sandbox --profile demo serve
```

The launcher defaults to `demo`, which needs no datasets, weights, PX4,
Gazebo, or optional Python packages. Use `./scripts/run_sandbox_app.sh
--profile development` for the local `visual_collection_v2` plan and full
simulator actions. The port may be changed. The host is restricted to a
loopback name or address; the operator is deliberately unavailable over the
network.

## Managed workflows

The Operator tab exposes fixed actions:

- environment doctor;
- one non-blind flight smoke run;
- one collection scenario;
- a five-scenario collection gate;
- deterministic v2 training-view materialization;
- a one-epoch v2 training, checkpoint reload, ONNX export, and inference smoke
  gate.
- offline integrity verification of the frozen v2 model package.
- identity-bound visual evaluation recipes on selection or full validation;
- bounded 320/416/640 ONNX evaluation with every-frame, every-second-frame,
  or every-third-frame scheduling.
- offline revalidation of the latest accepted LiDAR candidate;
- one pending LiDAR closed-loop flight at a time.

The ML Results tab is read-only. It exposes aggregate identities and metrics,
including the controlled 416-pixel latency replicate, but does not expose
blind images, per-frame predictions, labels, or scenario details. Formal blind
evaluation remains unavailable as an App action and cannot be rerun from the
browser.

LiDAR replay revalidation resolves the package and validation dataset from
their recorded identities; the browser cannot provide paths or substitute a
model. Closed-loop continuation resolves the matching study from the local
registry, refuses completed studies, and always adds `--max-runs 1`. It uses
the same single-job lock, runtime conflict checks, timeout, stop sequence, and
job history as the other managed workflows.

The Experiments tab provides a dependency-free Demo classifier plus selectors for visual ONNX replay,
LiDAR replay, one pending LiDAR closed-loop flight, and Sandbox v1 acceptance.
Only workflow-specific safe fields are shown; paths, thresholds, models, and
blind partitions cannot be supplied by the browser.

Profile capabilities are enforced in command construction as well as the UI.
Demo Profile exposes only the doctor and Demo classifier; direct HTTP requests
cannot unlock simulator or formal workflows.

Every managed job now writes `workflow_recipe.json` before process launch and
`workflow_receipt.json` after termination. The recipe binds the command,
source commit, artifact identities, timeout, scenario, and expected outputs.
The receipt binds final job state, exit status, diagnostics, bounded log hash,
and hashes of produced files. Existing specialized visual and LiDAR identities
remain unchanged and are referenced rather than replaced.

A visual recipe
binds the training view, exact membership, frozen package, selected ONNX
identity, preprocessing identity, frozen confidence threshold, frame budget,
uniform full-partition sampling algorithm, schedule, CPU runtime, and clean
source commit. Completed cards report fixed-
threshold precision, recall, per-size behavior, latency, throughput, small-
object recall, and no-target false-positive rate. These are diagnostic
development results; AP and formal held-out claims are intentionally absent.

Recipe and result commands are also available without the browser:

```bash
python main.py sandbox recipe-create \
  --name validation-416-every-frame \
  --partition validation --input-size 416 --frame-skip 1 --frame-limit 256
python main.py sandbox experiment-run \
  --recipe outputs/sandbox/experiments/validation-416-every-frame/recipe.json
python main.py sandbox experiment-inspect \
  --input outputs/sandbox/experiments/validation-416-every-frame/result.json
```

Each experiment directory contains `recipe.json`, `status.json`, a hashed raw
prediction artifact, and `result.json`. The App exposes only the recipe state
and aggregate result.

Sandbox v1 acceptance uses the latest valid 64-frame-or-larger visual replay,
the latest passed LiDAR replay, and the complete matching closed-loop study.
It also runs two local supervisor checks: graceful process-group stop and
capture of a deliberate non-zero exit with its diagnostic. The result binds
all five evidence identities. Visual and LiDAR evidence is paired in this
gate; it is not represented as synchronized sensor-fusion inference.

```bash
python main.py sandbox acceptance-run \
  --output outputs/sandbox/acceptance/manual-v1
python main.py sandbox acceptance-inspect \
  --input outputs/sandbox/acceptance/manual-v1/acceptance.json
```

Jobs follow `preparing → running → stopping → complete/failed`. Metadata,
diagnostics, and logs are stored under `outputs/sandbox/operator/jobs`. A file
lock permits only one managed job across app processes. Each action has a
bounded timeout, and stopping uses interrupt, terminate, then kill only for
the process groups descended from that job. After an App restart, a completed
exit sidecar is finalized automatically. A live job is adopted only when its
persisted ownership token remains actively locked; a reused PID is never
stopped.

## Safety boundary

- The browser chooses only fixed actions and approved non-blind scenarios.
- State-changing requests require an unguessable token generated by the local
  server and are confirmed in the browser.
- Approved collection, recording, log, and PNG paths are resolved centrally.
- Logs are escaped and bounded to at most 1,000 lines per request.
- Frames are loaded from ordered manifests on demand; arbitrary paths and
  non-PNG payloads are rejected.
- Blind scenario details, recordings, logs, frames, predictions, and training
  access remain sealed. Aggregate blind counts may be displayed.
- Experiment parameters cannot provide file paths, model paths, thresholds,
  devices, class orders, or arbitrary commands.
- Process status returns roles and PIDs, never command lines or environment
  values.
- Interrupted jobs are preserved as failed history. The app does not silently
  delete datasets, retries, model artifacts, or logs.

The command line remains the source of truth for formal collection audit,
dataset identities, model freezing, held-out evaluation, and formal studies.

## Recommended gate sequence

1. Run the doctor and resolve failures.
2. Pass one flight smoke route.
3. Pass one recorded scenario.
4. Pass a five-scenario gate.
5. Materialize the v2 training view.
6. Pass the v2 smoke training gate.
7. Run a 64-frame validation recipe, then increase the frame budget only when
   the diagnostic result and runtime are healthy.
8. Run Sandbox v1 acceptance and preserve its identity-bound result.

Formal training should begin only from a clean tracked commit after these
checks pass. Generated datasets, weights, job histories, and training outputs
remain local and are excluded from Git.
