# Research preview: setup and distribution boundaries

This guide describes dependency and asset boundaries for the research-toolbox-20261102 source snapshot. Start with the [offline examples](../examples/README.md); the browser workspace and macOS shell are optional. The September desktop and October integration records retain their original dates and identities.

## Choose the smallest setup for your task

| Task | Python / dependencies | External inputs |
| --- | --- | --- |
| Offline dataset audit / cached comparison | Python 3.11+; `requirements-offline.txt` | Tracked generated format fixtures |
| Receipt verification example | Python 3.11+; standard library only | Tracked source; no model |
| Browser Demo | Python 3.11+; standard library only | None |
| Portable core tests | Python 3.12+; `requirements-test.txt` (includes core runtime); Node.js on PATH | None; tests use fixtures and simulated interfaces |
| Planning and analysis | `requirements.txt` | Small tracked map definitions; your logs for analysis |
| Visual training and ONNX | Python 3.12+; core + `requirements-ml.txt` | A reviewed YOLO dataset; parent weights where applicable |
| Historical research tests | `requirements-research.txt` | Original datasets, models and receipts |
| PX4/Gazebo flight | Compatible simulator installation plus relevant Python dependencies | Validated map, vehicle/sensor resources and action-specific inputs |
| Selected-model fixed-scene flight | Full local simulation + verified Workbench model | Registered historical archive to restore six fixed assets, then new current-runtime qualification |
| macOS shell build | Full Xcode with Swift 6 support | No Python for native Demo; external runtime for research profiles |

The workflow configures Demo on Linux with Python 3.11/3.13, lightweight examples on Linux/macOS with Python 3.11/3.13, portable core on Linux with Python 3.12/3.13/3.14, and the existing native build on macOS 15. These are configured jobs, not a claim that the unpublished candidate has run remotely. See [validation](VALIDATION.md) and the [workflow](../.github/workflows/ci.yml).

Core JavaScript contract checks use Node 26.4.0 locally; the workflow explicitly installs that version with [setup-node](https://github.com/actions/setup-node). There are no npm dependencies. Missing Node produces required skips and fails core acceptance. Demo and the three examples do not require it.

The pinned ML/research stack was exercised on macOS arm64 with Python 3.14.
`scipy==1.18.1` in the research requirements requires Python 3.12 or newer.
Do not use the Demo's minimum Python version as a promise that every optional
research dependency supports it. The full core test set still exercises frozen geometry adapters that import
research helpers with Python 3.12+ syntax. It therefore requires Python 3.12+,
even without the ML stack. No tests are dropped for Python 3.11: that version is
supported for Demo and the small offline examples; the complete core requires 3.12–3.14. CI also
compiles the full historical tree on Python 3.13.
Core pins MAVSDK 3.17.2 because this project
uses its gRPC interface; MAVSDK 4 changes that interface.

## First run

```sh
./scripts/run_sandbox_app.sh
```

Open `http://127.0.0.1:8765/#results/acceptance` and run **Demo classifier**.
A specific Python can be selected with `UAV_SANDBOX_PYTHON=/path/to/python`.
The launcher tries that runtime, the repository environment, then available
system installations. It never installs packages automatically.

The synthetic demo proves that managed jobs, recipes and receipts work. It
is deliberately separate from real YOLO training, flight and formal results.

## Bring your own development data

Create a virtual environment and install core plus ML requirements before
starting `./scripts/run_sandbox_app.sh --profile development`. Use the dataset
import/registration commands in the [workbench guide](RESEARCH_INSPECTOR.md)
or inspect `python main.py sandbox --help`.

Imported datasets must satisfy the expected YOLO Detect structure and audit
checks. Dataset acceptance does not establish third-party redistribution
rights, physical-site independence or generalization quality. Source eligibility
is governed separately by [the intake policy](REAL_DOMAIN_V3_SOURCES.md).

Training may require upstream pretrained weights. Those weights are not part
of the source distribution; retain their applicable upstream terms and provide
network access or an existing verified local parent as the workflow requires.

## What a source clone cannot reproduce by itself

- Author-local trained candidates and datasets under `models/`, `datasets/`
  and `data/research/` are not distributed.
- The 404 MB historical evidence archive and current flight receipts under
  `outputs/` are not distributed. The tracked archive registry is an identity
  declaration, not a download service. Without the archive, the current
  selected-model fixed-scene entry cannot restore its six scenario assets.
- PX4, Gazebo, vehicle resources, native plugins and the full ML environment
  are external installations. The source ZIP does not vendor them. The full October local flight-control worktree is not included; its selected findings are published as case excerpts.
- Historical reports include successful and failed runs from the author's
  environment. Reading a report is not reproducing that experiment.

Once the corresponding inputs are available, follow [validation](VALIDATION.md).
Historical record hashes must remain unchanged; do not edit receipts to make a
new environment appear qualified. Current code/model/runtime changes require
current positive and controlled-stop checks.

## Source and macOS distribution

Use `python3 scripts/export_source.py --output dist/source.zip` to export a
source ZIP with `SOURCE_MANIFEST.json` and `SOURCE_PROVENANCE.json`. The exporter defaults to selected tracked files; the clean-install gate uses the same inventory. The export includes preserved
research scripts, tests, guides, notices and curated samples, and excludes
local Git objects, datasets, environments and runtime outputs.

A normal Git push transfers reachable commits, not an entire filesystem copy
of the local `.git` directory. Do not upload the local project directory as a
single archive: its research data is much larger than the source checkout.

The optional [macOS build](MACOS_APP.md) is an ad-hoc-signed research preview.
A successful build or checksum verification does not imply Apple notarization.
Older Releases describe their own source commits; use the current checkout
when evaluating this preview's model feedback features.

## Reading the evidence

The [September 25 acceptance report](results/sandbox_core_completion_20260925.md)
records the historical desktop milestone. The earlier
[slimming report](PROJECT_SLIMMING_20260925.md) records an intermediate stage and
is not the latest test result. Older reports keep their dated observations and
limitations; no selected run should be interpreted as a universal success rate.
