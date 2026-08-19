# Sandbox task navigation

The Sandbox interface is organized around five user tasks rather than the
underlying command and reporting subsystems.

| Main page | Purpose | Subpages |
| --- | --- | --- |
| Home | Show health, current work, storage, and one recommended next action | — |
| Fly & Collect | Run simulator checks and collection tasks, then inspect recordings | Run, Recordings |
| Model Lab | Prepare data, train a model, inspect runs, and test a local image | Datasets, Train, Runs, Test Image |
| Results | Read visual, LiDAR, and acceptance evidence | Visual, LiDAR, Acceptance |
| Activity | Diagnose jobs, logs, environment, and managed storage | Jobs, Logs, Environment, Storage |

The native macOS shell has no second section selector while the local service
is online. The web Home page is the single entry point for both the packaged
App and the repository launcher.

## Route compatibility

New links use task routes such as `#model/train` and `#fly/recordings`.
Existing bookmarks remain valid and are rewritten to their canonical route:

- `#setup` and `#overview` → `#home`
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
