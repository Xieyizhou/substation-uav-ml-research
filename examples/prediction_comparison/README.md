# Compare cached predictions

From the repository root, install the [lightweight dependencies](../README.md)
and run:

```sh
.venv-offline/bin/python examples/prediction_comparison/run.py --output outputs/examples/comparison-1
```

Open `outputs/examples/comparison-1/comparison.html` in a browser. The report
contains eight annotated images: baseline and candidate predictions for the
same four validation frames. All report images sit beside the HTML file, so the
directory can be copied and viewed offline. `comparison.json` stores the full
metrics and deltas; `per_class.csv` stores TP, FP, FN, precision, recall and F1.
The output directory must be new.

Both inputs are explicitly marked manually constructed demonstration caches:
[`baseline.json`](../fixtures/baseline.json) and
[`candidate.json`](../fixtures/candidate.json). No model is downloaded or run.
The function refuses differences in the ordered image membership, image hashes,
dimensions or ground truth. It checks finite, in-frame boxes and valid
confidence values before comparing them.

The default shared confidence threshold is `0.5`. To explore another value,
use a new output directory:

```sh
.venv-offline/bin/python examples/prediction_comparison/run.py --threshold 0.7 --output outputs/examples/comparison-07
```

[`threshold_metrics`](../../src/vision/evaluation/detection_metrics.py) matches
predictions in descending confidence order to one unused ground-truth box of
the same class at IoU at least `0.5`. Macro F1 averages the four canonical
classes. The fixture's small-object convention is box area below `32 * 32`
pixels in the 320 by 240 image. Small-object recall is `null` when no small
objects exist; no-target false-positive rate is `null` when no no-target frames
exist. Class precision and recall use zero for empty class denominators, as
defined by the shared metrics implementation.

Verify the output artifacts with:

```sh
.venv-offline/bin/python examples/receipt_check/run.py --receipt outputs/examples/comparison-1/receipt.json
```

The constructed score difference demonstrates comparison and reporting code.
It is not an experimental accuracy gain, a result of training, a benchmark on
real equipment or qualification for flight. The [E21 historical model case](../../docs/case-studies/README.md)
describes a separate measured experiment and its own limited evaluation scope.
