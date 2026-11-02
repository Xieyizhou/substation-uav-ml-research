# Audit a small YOLO dataset

From the repository root, install the [lightweight dependencies](../README.md)
and run:

```sh
.venv-offline/bin/python examples/dataset_audit/run.py --output outputs/examples/audit-1
```

The command imports [`fixtures/dataset/`](../fixtures/dataset/) with the same
`import_yolo_dataset` function used by Workbench. It checks readable images,
required labels, the canonical class mapping, finite normalized boxes inside
image bounds and duplicate image bytes across the imported dataset. A failed
check does not produce a successful receipt. The output directory must be new.

Inspect `outputs/examples/audit-1/audit.json`. It reports eight training and four
validation images, object counts, a dataset identity and the explicit demo
scope. `datasets/format-demo/` contains a private copy of the twelve images,
normalized labels, dataset configuration and membership records. The importer
writes an absolute root for this generated working copy; regenerate or import
it again after moving it. The shipped fixture configuration uses relative paths.

The importer expects a file named `dataset.yaml`, `train` and `val` image
directories below an `images` directory, and corresponding `.txt` files below
`labels`. Its current four-class contract is not a promise to accept every YOLO
variant. See [`workbench_datasets.py`](../../src/sandbox/workbench_datasets.py)
for the exact checks and optional class mapping.

Verify the recorded outputs:

```sh
.venv-offline/bin/python examples/receipt_check/run.py --receipt outputs/examples/audit-1/receipt.json
```

This receipt binds the audit and imported files by size and SHA-256. Structural
acceptance does not establish correct equipment labels, independent physical
sites, redistribution rights or model quality. The twelve geometric drawings
and their labels were generated solely to exercise the format.
