# Source publication scope

The November 2, 2026 candidate presents this repository as a visual research
toolbox with engineering experiment cases. Its source baseline is the
September 25 published commit
[`81b53b014162`](https://github.com/Xieyizhou/substation-uav-ml-research/commit/81b53b0141626369c95cd885eebd0060de6472f2).
Selected additions provide three offline visual examples, correct mixed pytest
collection, shared source inventories and curated historical case excerpts.
It does not claim that the original end-to-end aircraft research objective is
complete. The project is a research snapshot; the optional flight paths require
their own inputs and current qualification.

| Included in the source candidate | Use |
| --- | --- |
| Existing data audit, recipe, metrics, comparison, receipt, CLI and browser modules | Read and reuse the existing research workflows within their contracts |
| Twelve generated format images, YOLO labels and constructed prediction caches | Run the [three offline examples](../examples/README.md) without a model or simulator |
| Portable tests and explicit opt-in research/native/SITL collection fields | Distinguish fixture-based checks from author-environment research |
| Three engineering articles, an E21 model article and small provenance-labelled excerpts | Inspect measured observations and remaining uncertainty without bundling full experiment trees |
| Existing optional simulation sources, historical scripts, notices and configuration | Preserve research context and module paths |

The later October native integration source is not wholesale imported into
this candidate. Its selected measurements appear in the case studies. Do not
assume a fix described in a historical case is implemented by every retained
September flight module.

The source package excludes the local Git object database, environments,
builds, caches, trained weights, full research datasets, complete native logs
and generated run directories. The exported tree includes selected curated
files under `data/sample_outputs/`; those are different from ignored raw data.
PX4, Gazebo, native plugins and optional training dependencies remain external.
The [research preview](RESEARCH_PREVIEW.md) describes these setup boundaries.
The repository license and [third-party notices](../THIRD_PARTY_NOTICES.md)
apply to their respective content; they do not grant rights to separately
obtained datasets or pretrained weights.

## Source identity and distribution

[`source_bundle.py`](../src/sandbox/source_bundle.py) is the common source
inventory used by export and install checks. In a Git checkout, its default
inventory contains allowlisted tracked files. Untracked research files are not
silently added. `--include-untracked` is an explicit development export option
and should not be used to represent a clean committed release.

```sh
python3 scripts/export_source.py --output dist/research-toolbox-source.zip
```

The ZIP contains `SOURCE_MANIFEST.json` with paths, sizes and SHA-256 values,
plus `SOURCE_PROVENANCE.json` binding the manifest identity and source commit.
An unpacked bundle verifies its recorded bytes and uses its own provenance;
it does not borrow the identity of an enclosing Git repository. Source copying
for the beta install gate uses the same records. A dirty checkout is marked as
dirty and cannot stand in for a clean candidate commit.

The byte manifest supports traceability and detects alterations to recorded
files. It is not a signature or proof that the code is safe or scientifically
valid. Keep the separately distributed ZIP checksum and candidate commit when
reviewing a release. Preserve executable permissions when extracting, for
example with `unzip`, before running shell launchers.

The source candidate is separate from the historical v0.6.4 macOS installer.
No new DMG, Apple notarization, GitHub merge, remote CI success or aircraft
qualification follows merely from preparing this source package. See the
candidate's release evidence for the tests actually run on its exact commit;
workflow definitions describe configured checks, not results of unexecuted CI.

## Reading and maintenance boundaries

The [code guide](CODE_GUIDE.md) points to the reusable modules, while the
[history index](HISTORY_INDEX.md) keeps dated reports accessible. Existing
module paths and frozen historical records are preserved. Future archive
moves need dependency and identity checks before changing paths; rewriting old
receipt hashes is not an acceptable migration.

The current class scope is `transformer`, `switchgear`, `capacitor_bank` and
`reactor`. The offline comparison accepts an explicitly constructed demo cache
schema. Real inference currently enters through Ultralytics and needs the
optional ML stack and a verified model. Adding other classes, a broader YOLO
import format or a standalone ONNX inference adapter is separate implementation
work that needs its own data and numerical checks.
