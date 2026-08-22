# Real-domain v3 source intake

The real-domain pipeline separates source feasibility from dataset audit.
Public availability alone does not make an image eligible for training or
redistribution.

## Workflow

```text
source registry
  -> provenance and license review
  -> quarantine sample review
  -> canonical four-class manifest
  -> duplicate and split audit
  -> immutable dataset
  -> mixed v3 training view
```

Run the read-only feasibility audit before downloading a full source:

```bash
python main.py visual real-source-audit \
  --registry config/perception/real_domain_sources.json \
  --output outputs/research/real_domain_source_audit
```

The output contains JSON and CSV reports. Only `approved` sources with
verified upstream provenance can appear in a training manifest. `quarantine`
sources may be sampled for provenance, semantic, grouping, and duplicate
review but cannot be materialized. `evaluation_only` source families remain
excluded from training and threshold selection.

The single currently approved public artifact can be downloaded through its
registry-pinned Zenodo URL and verified before conversion:

```bash
python main.py visual real-source-download \
  --source-id gomes-substation-equipment-zenodo-7884270
```

The downloader accepts no caller-provided URL. It checks the official byte
count and upstream checksum, computes SHA256 locally, and leaves interrupted
downloads with a `.partial` suffix. The 1.84 GB download is not started by an
audit command.

Public previews can be copied into a bounded quarantine review directory:

```bash
python main.py visual real-source-intake \
--source-id roboflow-substation-equipment-34
```

This command only follows preview URLs pinned in the registry, limits file
count and size, verifies each image, and writes a review queue. It never marks
a source as training eligible. Roboflow's complete dataset export requires an
authenticated account-generated download; export credentials and links must
not be stored in the repository. Import that ZIP into:

`data/external/real_domain_v3/quarantine/<source_id>/raw`

before any semantic or split decision.
semantic or split decision.

Audit a manually downloaded YOLOv11 ZIP without extracting it:

```bash
python main.py visual real-source-export-audit \
  --source-id space-weather-reactors \
  --archive ~/Downloads/reactors.zip
```

The audit rejects traversal paths, symbolic links, oversized archives,
missing train/validation pairs, duplicate sample stems, invalid normalized
boxes, class IDs outside `data.yaml`, and registry labels absent from the
export. A structurally valid ZIP is still quarantined until provenance,
site/sequence grouping, duplicate, domain-fit, and whole-equipment reviews
are complete.

## Current decision

The Gomes/Zenodo dataset is the only approved source. It is licensed under
CC BY 4.0 and may supply reviewed development examples, but all images come
from one substation and therefore remain one atomic site group. Complete power
transformers are a verified semantic match. Breakers, reclosers, and disconnect
switches require whole-equipment review before any switchgear mapping.

The Roboflow Substation Equipment 34-class dataset has been downgraded to
evaluation-only and is not a quarantine training candidate. Remaining
Roboflow sources (Space Weather, fin_switch gear, det-2, and Electrical Devices)
remain quarantined. Their landing pages declare
CC BY 4.0, but upstream rights, site grouping, semantic scope, and cross-source
duplicates are not yet resolved. The existing sxiong source family remains a
frozen stress set.

A four-image public-preview probe of Substation Equipment found both field
photography and isolated product imagery. This is useful for bootstrapping a
review queue, but it confirms that the dataset must be filtered by domain and
whole-equipment semantics rather than accepted as one training block. Its
landing page also reports 636 preview images while dataset version 1 reports
1,584 images; the pinned export must therefore be audited rather than inferred
from the model overview.

Review priority is:

1. Space Weather Reactors for the hardest reactor gap.
2. fin_switch gear only for complete capacitor-bank candidates; most other
   classes are indoor components and cannot define outdoor switchgear.
3. Substation Equipment det-2 is introduced as the quarantine polygon supplement
   candidate for transformer/switchgear/capacitor/reactor.
4. Roboflow Substation Equipment 34-class remains excluded except for manual
   historical comparison.
5. Space Weather Merged Dataset for redundancy after cross-family deduplication.
6. Electrical Devices only as a semantic supplement or stress source.

The preferred integration is therefore source-specific rather than one large
merged download: Gomes supplies transformer development examples, Space
Weather Reactors is audited next for reactor, and fin_switch gear is audited
only for complete capacitor-bank frames. Switchgear plus all validation and
blind partitions still require independent outdoor sources. No Roboflow
export is accepted wholesale merely because its landing page declares a
license.

Training remains blocked until every class has independent development,
validation, and blind source groups. New collection should target only the
class/partition cells still missing after quarantine review.

## Canonical semantics

- `transformer`: complete power transformer equipment.
- `switchgear`: complete breaker, switchgear, or GIS unit.
- `capacitor_bank`: complete capacitor bank.
- `reactor`: complete shunt or series reactor.

PT, CT, bushings, insulators, busbars, relays, contactors, individual
capacitors, and ambiguous components are never mapped automatically.

## External tools

Datumaro is the preferred conversion boundary. Image hashes and perceptual
hashes enforce duplicate and split checks; imagededup can produce additional
near-duplicate candidates. CleanVision and FiftyOne support quality and
distribution review. CVAT is the manual semantic and bounding-box review
surface. Third-party models may propose annotations but are never ground
truth or v3 initialization weights.
