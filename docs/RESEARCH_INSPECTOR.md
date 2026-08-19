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

The Workbench tab provides a dependency-free Demo classifier plus a complete
Development Profile YOLO11n experiment path. Native training identities can be
registered without mutation, while standard YOLO Detect directories are
audited and copied into an immutable managed view. Source class IDs must map to
transformer, switchgear, capacitor bank, and reactor before import.

A workbench recipe binds its dataset identity, YOLO11n weight hash, preset,
allow-listed parameters, seed, environment, output directory, and optional
baseline package. Smoke, Quick, and Full presets are available; the browser can
only adjust epochs, patience, input size, batch, device, and worker count.
Training records its current phase, epoch, ETA, checkpoint, and stable failure
code. Completion automatically runs fixed validation, validation-only threshold
selection, one static FP32 ONNX export, a bounded class-stratified equivalence
gate, ordered replay, and an optional same-membership baseline comparison.

The App uses a native folder picker for imports. Browser requests never carry a
filesystem path, model path, threshold, blind partition, or command string.
Closing the App window leaves a job running. Explicitly quitting while a job is
active offers keep-running, safe-stop, and cancel choices; stopped training
retains `last.pt` for an explicit resume.

Completed Workbench receipts unlock local image inference. The native picker
accepts PNG/JPEG, copies the selected file into the managed Workbench inbox,
and submits only the staged filename plus one or two verified experiment IDs.
Before inference, the service revalidates the completion receipt, best/ONNX
hashes, validation result, equivalence gate, and replay identity. Each model
uses its recorded input size and frozen validation threshold. Annotated images,
detections, per-model latency, input/model hashes, and an inference identity are
stored under `outputs/sandbox/workbench/inference/`; the receipt explicitly
marks these results as development diagnostics rather than formal evidence.

The same controlled actions are available from the CLI:

```bash
python main.py sandbox --profile development workbench-dataset-import \
  --source /path/to/yolo --dataset-id imported-substation-v1 \
  --class-map 0=transformer --class-map 1=switchgear \
  --class-map 2=capacitor_bank --class-map 3=reactor
python main.py sandbox --profile development workbench-recipe-create \
  --experiment-id visual-smoke-01 --dataset-id imported-substation-v1 \
  --preset smoke
python main.py sandbox --profile development workbench-run \
  --recipe outputs/sandbox/workbench/runs/visual-smoke-01/recipe.json
python main.py sandbox --profile development workbench-inspect \
  --input outputs/sandbox/workbench/runs/visual-smoke-01
```

The same tab retains selectors for visual ONNX replay, LiDAR replay, one pending
LiDAR closed-loop flight, and Sandbox v1 acceptance.
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

Every managed action has a conservative output budget. A job starts only when
the filesystem has enough free space for that allowance and the configured
reserve. The same allowance and reserve are checked during execution so a
runaway output is stopped and classified before it consumes the remaining
disk. Failed jobs expose a stable failure code (`user_cancelled`,
`deadline_exceeded`, `resource_exhausted`, `dependency_unavailable`,
`artifact_invalid`, `runtime_unavailable`, `recovery_unsafe`,
`command_failed`, or `internal_error`) and whether retry is reasonable.

Storage inspection and pruning remain separate operations:

```bash
python main.py sandbox storage
python main.py sandbox retention-plan \
  --output outputs/sandbox/retention/manual.json
python main.py sandbox retention-inspect \
  --input outputs/sandbox/retention/manual.json
python main.py sandbox retention-apply \
  --input outputs/sandbox/retention/manual.json \
  --confirm-identity PLAN_IDENTITY_FROM_INSPECT
```

The plan keeps recent entries, excludes anything newer than 24 hours, and is
revalidated immediately before removal. Applying is refused while a managed
job is active or when using the formal profile. The policy never includes
`data/research`, model weights, or arbitrary paths.

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
