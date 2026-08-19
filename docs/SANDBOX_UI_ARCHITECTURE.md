# Sandbox research toolbox

The desktop interface is organized as one experiment workspace with a fixed
five-step research workflow. The top bar carries application and environment
state; the left sidebar carries navigation; the main canvas carries the current
task. This avoids stacking global and local tab bars above every page.

| Workflow step | Purpose | Existing routes |
| --- | --- | --- |
| Flight Console | Run simulator checks, collect a scenario, and inspect recordings | `#fly/run`, `#fly/recordings` |
| Dataset Manager | Select, audit, import, and version training data | `#model/datasets` |
| Training Studio | Create a bounded YOLO recipe and start or resume training | `#model/train` |
| Model Tester | Run verified local image inference and model comparison | `#model/inference` |
| Report Viewer | Read visual, LiDAR, and acceptance evidence | `#results/*` |

Models, datasets, and result counts are available as project resources rather
than workflow steps. Jobs, logs, environment checks, and managed storage are
grouped under System. Existing APIs and operator safety boundaries are not
changed by this presentation structure.

The interface is designed for the macOS desktop window at 1024–1440 px. Mobile
layouts are not a product target. The compact top toolbar keeps application
identity, profile, health, and refresh actions in one row. The persistent
sidebar keeps the experiment and its five-step workflow visible without
requiring a long page header.

## First-use path

A new user should be able to follow one visible sequence without understanding
the underlying commands:

1. **System:** confirm environment readiness when the selected profile reports
   a missing requirement.
2. **Flight Console:** run one flight smoke before collecting a scenario or
   attempting a five-scenario gate.
3. **Dataset Manager:** register a native training view or import an
   audited YOLO dataset.
4. **Training Studio:** start a bounded recipe, monitor it, and
   resume a checkpoint if necessary.
5. **Model Tester:** optionally inspect a candidate on one local
   image before relying on aggregate metrics.
6. **Report Viewer:** review visual, LiDAR, and acceptance summaries, expanding
   technical provenance only when needed.
7. **System:** diagnose jobs, logs, environment, or storage when a task is
   blocked; this is a support destination, not a required workflow step.

## Route compatibility

New links use task routes such as `#model/train` and `#fly/recordings`.
Existing bookmarks remain valid and are rewritten to their canonical route:

- `#home`, `#setup`, and `#overview` → `#model/datasets`
- `#experiments` → `#model/train`
- `#research` → `#results/visual`
- `#lidar` and `#preflight` → `#results/lidar`
- `#frames` → `#fly/recordings`
- `#operator`, `#logs`, and `#doctor` → the matching Activity subpage

The route is the only navigation state. Refresh, browser history, and periodic
status polling must preserve it.

## Interaction boundaries

Task pages call the existing fixed operator actions. They do not accept a
command string, filesystem model path, blind partition, or confidence
threshold. Confirmation, single-job locking, output budgets, bounded timeouts,
and safe process-group cleanup remain enforced below the UI.

Run and inference history is bounded by default. Users can filter or expand it
without changing the underlying receipts. Identity hashes, commits, and the
complete replay matrix are available under technical details instead of being
the first information shown.
