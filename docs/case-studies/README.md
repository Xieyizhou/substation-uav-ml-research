# Engineering and model experiment cases

These articles preserve selected findings from the author's September–October 2026 research archive. They explain measurements, failures and bounded changes. They do not certify this publication snapshot for flight or reproduce the complete historical environment.

| Case | Question | What the excerpt supports |
| --- | --- | --- |
| [Telemetry event-loop blocking](telemetry-event-loop.md) | Why did a pre-ARM freshness gate reject telemetry after a long pause? | A synchronous provenance check blocked progress; moving the same check off the event loop restored logger cadence in the dated comparison |
| [Bounded shutdown diagnostics](bounded-shutdown-diagnostics.md) | Why did simulator stop arrive late even though the SDK child had exited? | A slow port diagnostic delayed stop notification and coincided with a real logger queue overflow |
| [Shared magnetic model](shared-magnetic-model.md) | Why did simulator and estimator heading disagree on the ground? | Different magnetic tables produced a systematic offset; an explicit shared-table candidate reduced it in unarmed checks |
| [E21 model experiment](e21-model-experiment.md) | What improved in a frozen synthetic benchmark, and what remained weak? | Three-seed development selection and a consumed 56-frame synthetic test, with class and small-object limitations retained |

Each article links a compact JSON excerpt in [data/sample_outputs/case_studies](../../data/sample_outputs/case_studies/). Every selected value records its original archive-relative path and JSON pointer. Source file sizes and SHA-256 hashes identify the author-held originals; those originals are **not included**. The excerpt has its own identity hash. Hashes check consistency and do not establish authorship, label correctness or flight safety.

From the repository root, Python 3.11 or later can check the bundled excerpts without third-party packages:

```bash
python scripts/verify_case_studies.py
```

The command verifies excerpt identities and displayed arithmetic, including queue accounting, magnetic differences and F1 calculations. Its output says `raw_sources_reopened: false`: a fresh checkout cannot independently audit unbundled ULog, images, binaries or complete training runs. Archive holders can additionally compare every source hash and selected JSON value:

```bash
python scripts/verify_case_studies.py --source-root /path/to/research-archive
```

The three runnable [offline examples](../../examples/README.md) have a different role. Their geometric images and hand-constructed predictions teach the workflow; the historical case excerpts report dated experiment records. Neither is a real-site accuracy benchmark.
