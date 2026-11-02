# Check a receipt and reject an altered output

This example needs only Python 3.11 or newer and the standard library. From the
repository root:

```sh
python3 examples/receipt_check/run.py --output outputs/examples/receipt-1
```

Use a directory that does not exist. The example writes a tiny result under
`original/`, records its hash in `original/receipt.json`, then makes its own copy
under `altered/` and changes only the copied result. It leaves the original
artifact intact. `tamper_check.json` records `original.valid: true` and
`altered_output_rejected: true`.

The demo itself exits successfully because rejection is the expected outcome.
To inspect either receipt directly, run a read-only check:

```sh
python3 examples/receipt_check/run.py --receipt outputs/examples/receipt-1/original/receipt.json
python3 examples/receipt_check/run.py --receipt outputs/examples/receipt-1/altered/receipt.json
```

The first command exits `0`. The second exits `1` with a message such as
`Recorded output changed: result.json`. The same `--receipt` command checks
receipts from the [dataset audit](../dataset_audit/README.md) and
[prediction comparison](../prediction_comparison/README.md). Choose either
`--receipt` or `--output`, never both.

The verifier recomputes the receipt identity and checks each listed output's
relative path, size and SHA-256. Absolute paths, traversal, duplicate paths,
symlink output files and missing or altered listed files are refused. It does
not reject unrelated extra files or reopen the recorded source inputs.

An attacker who can replace a result and its entire receipt can recompute the
hashes. This is an integrity and traceability demonstration, not a digital
signature or a trusted timestamp. It does not establish scientific validity or
label quality. The separate `example_receipt_schema_version` and demo flags do
not satisfy the existing model or flight qualification contracts.
