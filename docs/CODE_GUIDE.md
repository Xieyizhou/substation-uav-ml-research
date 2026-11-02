# Code reading guide

Start with the [offline examples](../examples/README.md). They run a small path
through the existing data and metrics code without requiring the historical
models, simulator or a desktop shell. The current example scope is four
canonical equipment classes, not a general robotics framework.

| Question | Start here | Follow the call into |
| --- | --- | --- |
| What exactly does an offline command do? | [`examples/*/run.py`](../examples/README.md) | [`offline_examples.py`](../src/sandbox/offline_examples.py): `example_main`, the three `run_*` functions and `verify_example_receipt` |
| How is a dataset accepted? | [`workbench_datasets.py`](../src/sandbox/workbench_datasets.py): `import_yolo_dataset` | Label and image checks, private copies, membership hashes and [`WorkbenchDataset`](../src/sandbox/workbench_models.py) |
| How are predictions matched and scored? | [`detection_metrics.py`](../src/vision/evaluation/detection_metrics.py) | `match_detections`, `threshold_metrics`, `select_confidence_threshold` |
| How is a managed experiment frozen? | [`workbench_recipe.py`](../src/sandbox/workbench_recipe.py): `materialize_workbench_recipe` | [`workbench_membership.py`](../src/sandbox/workbench_membership.py), [`workbench_weights.py`](../src/sandbox/workbench_weights.py), then [`workbench_runner.py`](../src/sandbox/workbench_runner.py) |
| Where is the actual model-to-baseline comparison? | [`workbench_comparison.py`](../src/sandbox/workbench_comparison.py): `compare_with_baseline` | Recipe membership, baseline identity and replay metrics |
| What does a completed model receipt bind? | [`workbench_receipt.py`](../src/sandbox/workbench_receipt.py): `materialize_workbench_receipt` | [`workbench_equivalence.py`](../src/sandbox/workbench_equivalence.py), artifact identities and completion state |
| How are reviewed feedback images registered? | [`feedback_review.py`](../src/sandbox/feedback_review.py) | [`feedback_dataset.py`](../src/sandbox/feedback_dataset.py) and dataset provenance |
| Where does real image inference happen? | [`workbench_inference.py`](../src/sandbox/workbench_inference.py): `run_image_inference` | Verified model selection and `_ultralytics_predict`; this needs optional ML dependencies and model artifacts |
| How do export and install choose the same source? | [`source_bundle.py`](../src/sandbox/source_bundle.py): `source_paths`, `describe_source` | [`export_source.py`](../scripts/export_source.py) and [`beta_install.py`](../src/sandbox/beta_install.py) |

The example comparison uses the existing matcher and annotation helper. Its
cache schema is intentionally a demonstration contract; it does not run the
managed training/comparison pipeline. Its receipt schema is also separate from
a completed Workbench model receipt. Read the validators before treating one
kind of record as interchangeable with another.

## CLI and browser entry points

[`main.py`](../main.py) delegates to [`src/cli/root.py`](../src/cli/root.py).
The root router imports subcommands lazily, which lets offline help and Demo
remain usable without loading the optional training or simulation stack.
`python main.py sandbox --help` enters [`src/cli/sandbox.py`](../src/cli/sandbox.py);
Workbench subcommands are registered and dispatched by
[`workbench_cli.py`](../src/sandbox/workbench_cli.py).

The standard-library browser Demo starts through
[`run_sandbox_app.sh`](../scripts/run_sandbox_app.sh) and
[`src/inspection/app.py`](../src/inspection/app.py). Follow
[`src/inspection/service.py`](../src/inspection/service.py) for application
services, [`workbench.py`](../src/inspection/workbench.py) for Workbench requests,
and [`static/`](../src/inspection/static/) for the browser implementation.
The [UI architecture](SANDBOX_UI_ARCHITECTURE.md) and
[Workbench guide](RESEARCH_INSPECTOR.md) explain the larger application.
The optional Swift shell lives in [`apps/macos/`](../apps/macos/); it has a
separate Xcode build and distribution lifecycle.

## Read checks with the implementation

[`tests/test_offline_examples.py`](../tests/test_offline_examples.py) exercises
the small workflows and rejection paths. The source distribution contract is in
[`test_source_bundle.py`](../tests/test_source_bundle.py) and
[`test_source_export.py`](../tests/test_source_export.py). The runner's own
regressions are in [`test_publication_runner.py`](../tests/test_publication_runner.py).
Run the selected collection through [`scripts/run_tests.py`](../scripts/run_tests.py);
see [validation](VALIDATION.md) for dependencies and suite boundaries.

The `core` collection uses fixtures and simulated interfaces. Passing it proves
the checked contracts in that collection, not a current PX4 flight. Some local
HTTP tests bind loopback ports, so an environment that denies local socket
binding cannot complete those checks. A refusal or skip must remain visible in
the report.

## Optional simulation and historical research

[`src/flight/`](../src/flight/), [`scripts/flight/`](../scripts/flight/),
[`simulation/`](../simulation/README.md) and the optional model commands are
preserved for readers who need the larger research system. For example,
[`semantic_mission.py`](../src/flight/semantic_mission.py) and
[`sitl_lifecycle.py`](../src/flight/sitl_lifecycle.py) describe the selected-model
fixed-scene runtime. Their external binaries, model assets and historical
evidence are not supplied by the offline examples. Current positive and
controlled-stop qualification is required before treating a changed runtime as
qualified; a source export or old receipt cannot supply it.

The November source candidate starts from the published September 25 baseline
and adds the selected offline publication work. The October telemetry,
shutdown and magnetic-model investigations are provided as
[curated case studies](case-studies/README.md). Their later native flight
implementations were not wholesale migrated into this source candidate.

[`scripts/vision/__init__.py`](../scripts/vision/__init__.py) extends the package
search path into [`archive/research_scripts/vision/`](../archive/research_scripts/vision/).
This keeps historical `python -m scripts.vision.<module>` imports available.
It does not restore old direct filesystem paths or guarantee frozen-source
hashes after a move. Use the [history index](HISTORY_INDEX.md) and archive
manifest when following a dated protocol; do not rewrite receipts to accept
different source bytes.
