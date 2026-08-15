# Sandbox v0.1 Quick Start

The local App has three explicit profiles. This keeps the first-run experience
small while preserving the stricter research boundaries used by the full
simulator.

![Sandbox Demo Profile](assets/sandbox_demo.jpg)

| Profile | Local data and weights | PX4/Gazebo | Intended use |
| --- | --- | --- | --- |
| `demo` | Not required | Not required | App tour and workflow-contract example |
| `development` | Required for related actions | Required for flight | Non-blind collection, replay, and model iteration |
| `formal` | Frozen artifacts required | Required for flight | Evidence-preserving evaluation with blind-data gates |

The Demo Profile does not download assets, simulate performance, or expose
formal evaluation controls. It creates only ignored runtime receipts under
`outputs/sandbox/demo/`.

## Start the App

Requirements:

- Git
- Python 3.11 or newer
- A modern browser

No virtual environment is required for the Demo Profile.

```bash
git clone https://github.com/Xieyizhou/substation-uav-ml-research.git
cd substation-uav-ml-research
./scripts/run_sandbox_app.sh
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The launcher performs an
idempotent bootstrap before starting the loopback-only server. Press `Ctrl-C`
in the terminal to stop it.

Open **Get started** after launch. The page evaluates requirements for the
selected profile, separates required items from optional simulator features,
and provides copyable commands or official PX4/Gazebo documentation. It never
executes installation commands. Select **Check again** after making changes.

To use another port:

```bash
./scripts/run_sandbox_app.sh --port 8876
```

## First Reproducible Experiment

In the App:

1. Open **Experiments**.
2. Select **Demo classifier**.
3. Select **Create recipe and run** and confirm the bounded local action.
4. Open **Operator** to inspect the job and its bounded log.
5. Return to **Experiments** to inspect the workflow receipt.

The workflow trains a nearest-centroid classifier from eight deterministic
synthetic range/density samples and evaluates it on eight separate samples.
It demonstrates:

- fixed feature order and algorithm identity;
- an immutable recipe identity;
- a result identity that detects modification;
- deterministic metrics and predictions;
- explicit `dataset_role: demo` and `formal_evidence: false` boundaries.

It does not claim real model accuracy, simulator performance, or airframe
performance.

The same workflow is available from the terminal:

```bash
python3 main.py sandbox --profile demo demo-run \
  --output outputs/sandbox/demo/runs/first

python3 main.py sandbox --profile demo demo-inspect \
  --input outputs/sandbox/demo/runs/first
```

## Environment and Release Checks

The doctor distinguishes required Demo components from optional full-simulator
components:

```bash
python3 main.py sandbox --profile demo doctor
```

Missing PX4, Gazebo transport, MAVSDK, plotting, or table-analysis tools appear
as warnings in Demo Profile. A missing tracked plan or world definition is a
failure.

The v0.1 release gate checks bootstrap identity, the deterministic demo,
doctor failures, and required web assets:

```bash
python3 main.py sandbox --profile demo release-gate \
  --output outputs/sandbox/demo/release-gate/local

python3 main.py sandbox --profile demo release-gate-inspect \
  --input outputs/sandbox/demo/release-gate/local/release_gate.json
```

The clean-install Beta gate goes one step further: it copies only Git-tracked
files, creates a new virtual environment with an isolated home directory, and
repeats bootstrap, Demo execution, receipt inspection, and a loopback App
smoke check:

```bash
python3 main.py sandbox --profile demo beta-install-gate \
  --output outputs/sandbox/demo/beta-install/local

python3 main.py sandbox --profile demo beta-install-gate-inspect \
  --input outputs/sandbox/demo/beta-install/local/beta_install_gate.json
```

No package download, dataset, model weight, PX4, or Gazebo is used. Both the
release gate and clean-install gate run in GitHub Actions on Python 3.11 and
3.13.

## Enable the Full Simulator

Create the project environment and install the core runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install PX4 and Gazebo separately, then start the development profile:

```bash
./scripts/run_sandbox_app.sh --profile development
```

The setup page checks the active project virtual environment, PX4 checkout and
SITL build, Gazebo command-line tools, MAVSDK, and tracked SDF worlds. A ready
result means the local prerequisites are present; the existing Preflight and
flight-smoke gates still decide whether a real managed flight may start.

The development profile discovers the local v2 collection plan when present
and enables managed flight, collection, replay, training-view, and validation
actions. The App still enforces one active job, bounded timeouts, safe process
cleanup, non-blind browsing, and fixed command construction.

For a macOS preview release, follow the offline and clean-install gates with
one App-managed Development flight smoke on a simulator host. The
`development-app-gate` command binds its workflow receipt and flight summary
to the current clean commit and verifies that PX4, Gazebo, the flight task, and
recording processes have all stopped.

## Preflight Before Large LiDAR Jobs

Open **Preflight** before a multi-flight LiDAR gate. The page combines the
environment doctor, disk capacity, runtime ownership, and the latest controlled
capability challenge. A large LiDAR gate remains blocked until all four checks
pass.

The capability challenge runs three flights against one controlled route
blocker: geometric LiDAR, ML LiDAR, and safety fusion. Each condition must
detect danger, attempt and complete a local replan, replace the active route,
finish the mission, land, remain collision-free, and maintain sensor health.
Its receipt binds the model hash, challenge specification, result hashes, and
the clean source commit that produced the flights. Changing the model,
challenge contract, results, or tracked code makes the receipt stale.

Use **Run three-flight challenge** in the App. The equivalent command for a
registered model is:

```bash
python main.py sandbox --profile development challenge-run \
  --model-id sandbox-lidar-risk-v2
```

Existing completed challenge runs can be inspected without flying:

```bash
python main.py study challenge-receipt-inspect \
  --input outputs/research/study_results/STUDY_ID/challenge/challenge_receipt.json \
  --require-current
```

## Runtime Files

Inspect generated output use without changing files:

```bash
python main.py sandbox storage
```

Use `retention-plan` followed by `retention-inspect` to preview old generated
Sandbox outputs. Nothing is removed until `retention-apply` is supplied the
exact plan identity; formal profile output and research datasets are never
eligible.

Demo runtime files are kept outside version control:

```text
outputs/sandbox/demo/
├── bootstrap/
├── collection/
├── experiments/
├── operator/
├── release-gate/
└── runs/
```

Deleting this directory resets only local Demo receipts. It does not affect
tracked source files or formal research data.

## Troubleshooting

- **Port already in use:** start with `--port 8876` or stop the existing App.
- **Python is too old:** install Python 3.11+ and retry.
- **Bootstrap identity mismatch:** restore
  `config/sandbox/demo_collection_plan.json` from Git.
- **Demo result identity mismatch:** create a new output directory or remove
  only the modified local Demo run.
- **Flight controls are disabled:** this is expected in Demo Profile; start the
  development profile after installing the simulator stack.
