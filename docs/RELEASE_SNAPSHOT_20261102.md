# Research toolbox source snapshot 20261102

This source candidate reorganizes the public project around offline visual research workflows and measured engineering cases. It starts from public commit `81b53b0141626369c95cd885eebd0060de6472f2`, with selected publication changes. It is separate from the old v0.6.4 desktop binary and does not publish the full October local flight-control worktree.

## Changes

- One selected, tracked-file source inventory for ZIP export and clean-install validation, with per-file hashes and unpacked-bundle provenance checks. Modified unpacked source cannot report the original clean source identity.
- A pytest-based runner that collects unittest classes, functions and parameterized cases. Empty selected modules, required skips and test errors fail acceptance; research checks retain process isolation.
- Three self-contained examples: audited YOLO format import, comparison of explicitly constructed prediction caches, and result integrity with rejection of an altered copy. Generated image inputs and their reproducible generator are included.
- English and Chinese entry pages, an example guide, code navigation, a history index and explicit setup boundaries.
- Three historical integration cases and an E21 model summary with small, source-identified excerpts and a standard-library arithmetic checker.

## Validation and distribution evidence

The source ZIP is accompanied by `RELEASE_REPORT.json` and `SHA256SUMS`. That report identifies the exact candidate commit, source manifest identity, environment, commands and actual results. Use it to distinguish executed local checks from configured GitHub jobs. The archive itself carries `SOURCE_MANIFEST.json` and `SOURCE_PROVENANCE.json`; a checksum is an integrity aid, not a digital signature.

The local acceptance matrix targets macOS arm64: core Python 3.12, 3.13 and 3.14; the standard-library Demo and the lightweight examples on Python 3.11; examples and source identity checks from an independently unpacked archive. The dependency record is retained with the release evidence. Core tests use fixtures and simulated interfaces, with local loopback binding permitted where required. Core also requires Git, Bash/POSIX shell tools and `/usr/bin/python3` for a retained packaging-refusal contract. Three executable JavaScript contracts also need Node.js on PATH; local checks and the workflow use Node 26.4.0, without npm dependencies.

The [CI workflow](../.github/workflows/ci.yml) additionally configures Ubuntu core checks, Ubuntu/macOS lightweight examples and the existing native macOS build. Consult the publication PR and release report for actual Ubuntu and native macOS results on the bound commit. Local source acceptance and remote Actions are separate records; a pre-existing green main-branch run belongs to its own commit.

Reproduce the portable checks after installing `requirements-test.txt` with Python 3.12–3.14:

```sh
python scripts/run_tests.py --suite core --report outputs/core-tests.json
python examples/generate_fixtures.py --check
python scripts/verify_case_studies.py
```

The [quick start](SANDBOX_QUICKSTART.md) gives the three example commands and dependency-free Demo. Full local checks, including independent installation, are described in [validation](VALIDATION.md).

## Remaining limits

The generated shapes and constructed prediction caches demonstrate software contracts, not detector accuracy. Historical E21 results concern consumed synthetic layouts and AI-reviewed annotations, with weak switchgear and small-object recall retained. The case checker does not rerun training, inference or native flights.

Raw data, trained weights, full ULog, frozen simulation archives and platform binaries are external. Reading the October case fixes does not mean their full native implementation was migrated here. No new SITL qualification, real-image generalization, real-aircraft readiness, installer, Apple notarization or renewed autonomous-flight delivery commitment is part of this source snapshot. See [publication scope](PUBLICATION_SCOPE.md).
