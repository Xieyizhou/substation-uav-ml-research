# Offline visual workflows

These three examples demonstrate data checks, detection comparison and artifact
integrity without PX4, Gazebo, trained weights or a network connection at runtime.
The images are generated geometric rectangles. The cached predictions were
constructed by hand; their scores describe these fixtures, not equipment
recognition quality.

Run the following commands from the repository root with Python 3.11 or newer.
The first installation needs access to the package index, or an existing local
package cache. On Windows, use `.venv-offline\Scripts\python.exe` in place of
`.venv-offline/bin/python`.

```sh
python3 -m venv .venv-offline
.venv-offline/bin/python -m pip install -r requirements-offline.txt
.venv-offline/bin/python examples/dataset_audit/run.py --output outputs/examples/audit-1
.venv-offline/bin/python examples/prediction_comparison/run.py --output outputs/examples/comparison-1
.venv-offline/bin/python examples/receipt_check/run.py --output outputs/examples/receipt-1
```

Every `--output` must name a directory that does not exist. Use a new name for
each run. Existing results are refused rather than overwritten; generated
results belong under the ignored `outputs/` directory.

| Workflow | Input and implementation | Result to inspect |
| --- | --- | --- |
| [Dataset audit](dataset_audit/README.md) | Twelve generated images and YOLO labels; the existing Workbench importer | `audit.json`: eight training and four validation images; a private imported copy and a receipt |
| [Prediction comparison](prediction_comparison/README.md) | Two constructed caches over the same four validation images; the existing detection metrics | `comparison.html`, `comparison.json`, `per_class.csv`, eight annotated PNGs and a receipt |
| [Receipt check](receipt_check/README.md) | A tiny result and a deliberately altered copy | `tamper_check.json`: original accepted, altered output rejected |

The lightweight dependency file contains only Pillow and PyYAML. The receipt
example and fixture generator use the Python standard library and can also run
without installing that file. The complete portable test suite has a separate
Python minimum and dependency set; see [validation](../docs/VALIDATION.md).

Check an audit or comparison receipt with:

```sh
.venv-offline/bin/python examples/receipt_check/run.py --receipt outputs/examples/audit-1/receipt.json
.venv-offline/bin/python examples/receipt_check/run.py --receipt outputs/examples/comparison-1/receipt.json
```

A successful check covers the receipt identity and its listed output bytes.
It does not reopen the original inputs, authenticate a signer, establish label
correctness or qualify a model for flight. These demonstration receipts use
their own schema and cannot promote a model through Workbench or flight gates.

The four canonical class names are `transformer`, `switchgear`, `capacitor_bank`
and `reactor`. The generated drawings are format samples, not independently
reviewed examples of those devices. Fixtures and their SHA-256 records are in
[`fixtures/PROVENANCE.json`](fixtures/PROVENANCE.json); they are project-generated
and distributed under the repository's MIT license. Verify their deterministic
generation without modifying files:

```sh
python3 examples/generate_fixtures.py --check
```

Continue with the [code guide](../docs/CODE_GUIDE.md) to find the reused modules,
or the [case studies](../docs/case-studies/README.md) for measured historical
engineering and model experiments. Those historical excerpts have different
provenance from the constructed demonstration fixtures.
