# Historical records and experiments

Use the [offline examples](../examples/README.md) for the current small runnable
entry points and the [code guide](CODE_GUIDE.md) for implementation navigation.
This index preserves a route into the larger research history. Dated reports
describe their own source, data and environment; their completion counts and
qualification claims do not automatically apply to a later checkout.

| Area | Reading entry | Boundary |
| --- | --- | --- |
| October integration investigations | [Curated case studies](case-studies/README.md) and their small public excerpts | Selected measurements and limitations are shipped; raw native runs, full ULog and later native source changes are not bundled |
| E21 model comparison | [Model case in the case index](case-studies/README.md) | Measured synthetic-layout evaluation; no trained weights or complete training-data reconstruction in this source package |
| September 25 desktop milestone | [Desktop acceptance report](results/sandbox_core_completion_20260925.md) | Historical application acceptance on its recorded environment |
| September source organization | [Slimming report](PROJECT_SLIMMING_20260925.md) | Explains archive moves and their hashes; intermediate test results remain dated |
| Model development and failure analysis | [`docs/results/`](results/) and [ML research platform](ML_RESEARCH_PLATFORM.md) | Many workflows require original data, models and review records outside the source checkout |
| Experiment design | [Experiment protocol](EXPERIMENT_PROTOCOL.md) and [study workflow](STUDY_WORKFLOW.md) | Protocols have different frozen inputs and evaluation boundaries; a common repository does not make their scores interchangeable |
| Visual source intake | [Source policy](REAL_DOMAIN_V3_SOURCES.md), [canonical view collection](canonical-view-collection.md) | Format acceptance, review, provenance and redistribution eligibility are separate checks |
| Historical flight evidence | [Validation and evidence archives](VALIDATION.md) | The registered external archive identifies old runs; it is not a bundled download or current flight approval |
| Desktop packaging | [macOS app guide](MACOS_APP.md) and release-specific install guides | The old v0.6.4 binary is a separate historical release, not a build of this source candidate |

## Archived scripts

[`archive/research_scripts/README.md`](../archive/research_scripts/README.md)
explains why the historical vision scripts were moved.
[`manifest.json`](../archive/research_scripts/manifest.json) records original
and archived paths with unchanged SHA-256 values. Python module imports can use
the compatibility package path, but a protocol that hashes an old file path or
derives resources from `__file__` may need the verified files restored together.

The archive is source preservation, not a set of independently supported
examples. The opt-in `research` suite retains per-module process isolation
because historical adapters configure shared module globals. Missing original
artifacts are a reproducibility boundary and should be reported as such.

## How to interpret a historical result

Read the run's date, evaluation membership, threshold policy, source identity
and remaining limitations before using its headline score. Keep failed and
unqualified records alongside successful observations. A completed route can
coexist with a logging failure, and a lower static magnetic bias does not by
itself explain a later airborne yaw reset.

The small case-study JSON files record the original relative artifact names
and source hashes. Those names identify the local historical archive; they
are not broken download links or files promised to be in the source ZIP. The
public arithmetic checker verifies the curated excerpt's internal calculations,
not the absent raw run. Preserve historical record bytes when moving data, and
create a new qualification for changed models, runtime inputs or code.
