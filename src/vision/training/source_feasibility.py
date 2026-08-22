"""Audit public real-domain sources before any image enters a dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256, write_json
from src.vision.contracts.real_domain import ALLOWED_LICENSES, MINIMUM_COVERAGE, PARTITIONS


SCHEMA_VERSION = 2
PROVENANCE_STATES = frozenset({"verified", "pending", "rejected"})
INGESTION_STATES = frozenset({"approved", "quarantine", "rejected"})
USAGE_ROLES = frozenset({
    "development_candidate", "supplemental_candidate", "evaluation_only"
})
SEMANTIC_STATES = frozenset({"verified", "review", "unsupported"})


def load_source_registry(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("real_domain_source_registry_schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported real-domain source registry schema")
    sources = {}
    for raw in record.get("sources", []):
        source = _validate_source(raw)
        source_id = source["source_id"]
        if source_id in sources:
            raise ValueError(f"duplicate source id: {source_id}")
        sources[source_id] = source
    if not sources:
        raise ValueError("source registry is empty")
    return record, sources


def _required_text(row, field):
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"source is missing {field}")
    return value


def _validate_source(raw):
    row = dict(raw)
    source_id = _required_text(row, "source_id")
    for field in (
        "name", "version", "landing_url", "declared_license_url",
        "citation", "source_family",
    ):
        _required_text(row, field)
    license_id = row.get("declared_license_id")
    if license_id not in ALLOWED_LICENSES:
        raise ValueError(f"source {source_id} has unsupported declared license")
    if row.get("provenance_status") not in PROVENANCE_STATES:
        raise ValueError(f"source {source_id} has invalid provenance status")
    if row.get("ingestion_status") not in INGESTION_STATES:
        raise ValueError(f"source {source_id} has invalid ingestion status")
    if row.get("usage_role") not in USAGE_ROLES:
        raise ValueError(f"source {source_id} has invalid usage role")
    if row["ingestion_status"] == "approved":
        if row["provenance_status"] != "verified":
            raise ValueError(f"approved source {source_id} lacks verified provenance")
        if not row.get("upstream_url") or not row.get("upstream_license_id"):
            raise ValueError(f"approved source {source_id} lacks upstream rights")
        if row["upstream_license_id"] not in ALLOWED_LICENSES:
            raise ValueError(f"approved source {source_id} has unsupported upstream license")
        artifact = row.get("public_artifact")
        if artifact is not None and not all(
            artifact.get(field) for field in
            ("url", "filename", "bytes", "upstream_checksum")
        ):
            raise ValueError(f"approved source {source_id} has incomplete artifact metadata")
    mappings = row.get("class_candidates", {})
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError(f"source {source_id} has no class candidates")
    for class_name, mapping in mappings.items():
        if class_name not in EQUIPMENT_CLASSES:
            raise ValueError(f"source {source_id} maps unknown class {class_name}")
        if mapping.get("semantic_status") not in SEMANTIC_STATES:
            raise ValueError(f"source {source_id} has invalid semantic status")
        if not isinstance(mapping.get("source_labels"), list):
            raise ValueError(f"source {source_id} labels must be a list")
    eligible = row.get("eligible_partitions", [])
    if not isinstance(eligible, list) or not set(eligible).issubset(
        {"development", "validation", "blind", "stress"}
    ):
        raise ValueError(f"source {source_id} has invalid partitions")
    access = row.get("access")
    if access is not None:
        if access.get("mode") not in {"authenticated_manual_export"}:
            raise ValueError(f"source {source_id} has invalid access mode")
        if not str(access.get("dataset_locator", "")).strip():
            raise ValueError(f"source {source_id} has no dataset locator")
        urls = access.get("review_sample_urls", [])
        if not isinstance(urls, list) or len(urls) > 16:
            raise ValueError(f"source {source_id} has invalid review samples")
    return row


def source_is_training_eligible(source):
    return (
        source["ingestion_status"] == "approved"
        and source["provenance_status"] == "verified"
        and source["usage_role"] != "evaluation_only"
    )


def _source_row(source):
    classes = {
        name: source.get("class_candidates", {}).get(name, {}).get(
            "semantic_status", "unsupported"
        )
        for name in EQUIPMENT_CLASSES
    }
    return {
        "source_id": source["source_id"],
        "name": source["name"],
        "declared_license_id": source["declared_license_id"],
        "provenance_status": source["provenance_status"],
        "ingestion_status": source["ingestion_status"],
        "usage_role": source["usage_role"],
        "source_family": source["source_family"],
        "review_priority": int(source.get("review_priority", 999)),
        "reported_image_count": source.get("reported_image_count"),
        "eligible_partitions": source.get("eligible_partitions", []),
        "class_semantics": classes,
        "training_eligible": source_is_training_eligible(source),
        "blocking_issues": list(source.get("blocking_issues", [])),
    }


def audit_real_sources(registry_path, output_root=None, content_audit_root=None):
    registry, sources = load_source_registry(registry_path)
    rows = [_source_row(source) for source in sources.values()]
    approved = [row for row in rows if row["training_eligible"]]
    quarantine = [row for row in rows if row["ingestion_status"] == "quarantine"]
    class_sources = _class_sources(approved)
    missing = [name for name, source_ids in class_sources.items() if not source_ids]
    partition_sources = {
        partition: _class_sources([
            row for row in approved if partition in row["eligible_partitions"]
        ])
        for partition in ("development", "validation", "blind")
    }
    partition_gaps = [
        {"partition": partition, "class_name": class_name}
        for partition, classes in partition_sources.items()
        for class_name, source_ids in classes.items() if not source_ids
    ]
    content = _content_coverage(content_audit_root, sources)
    report = {
        "real_source_feasibility_schema_version": 1,
        "source_registry_sha256": object_sha256(registry),
        "source_count": len(rows),
        "approved_source_count": len(approved),
        "quarantine_source_count": len(quarantine),
        "rows": rows,
        "verified_training_sources_by_class": class_sources,
        "verified_sources_by_partition_and_class": partition_sources,
        "missing_verified_classes": missing,
        "source_partition_gaps": partition_gaps,
        "verified_coverage": content["verified"],
        "quarantine_candidate_coverage": content["candidate"],
        "shortfall": content["shortfall"],
        "ready_to_materialize_training_dataset": not partition_gaps,
        "recommended_strategy": _recommend(rows, class_sources, partition_gaps),
    }
    report["feasibility_identity_sha256"] = object_sha256(report)
    if output_root is not None:
        _write_report(Path(output_root), report)
    return report


def _content_coverage(root, sources):
    verified = {partition: {name: 0 for name in EQUIPMENT_CLASSES} for partition in PARTITIONS}
    candidate = {name: 0 for name in EQUIPMENT_CLASSES}
    coverage_by_source = {
        source_id: {name: 0 for name in EQUIPMENT_CLASSES} for source_id in sources
    }
    family_overrides = {}
    if root is not None:
        paths = (list(Path(root).rglob("content-audit.json"))
                 + list(Path(root).rglob("audit.json"))
                 + list(Path(root).rglob("family-audit.json"))
                 + list(Path(root).rglob("semantic-audit.json")))
        for path in sorted(set(paths)):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            source = sources.get(receipt.get("source_id"))
            if source is None:
                continue
            if "gomes_source_family_audit_schema_version" in receipt:
                family_overrides[receipt["source_id"]] = {
                    "transformer": receipt.get(
                        "accepted_transformer_image_count", 0
                    )
                }
                continue
            elif "source_semantic_review_schema_version" in receipt:
                family_overrides[receipt["source_id"]] = receipt.get(
                    "accepted_unique_image_count", {}
                )
                continue
            elif "gomes_yolo_component_audit_schema_version" in receipt:
                counts = {"transformer": receipt.get("transformer_image_count", 0)}
            else:
                counts = receipt.get("candidate_unique_image_count", {})
            for name in EQUIPMENT_CLASSES:
                coverage_by_source[receipt["source_id"]][name] += int(counts.get(name, 0))
        for source_id, override in family_overrides.items():
            coverage_by_source[source_id] = {
                name: int(override.get(name, 0)) for name in EQUIPMENT_CLASSES
            }
        for source_id, counts in coverage_by_source.items():
            source = sources[source_id]
            if source["ingestion_status"] == "quarantine":
                for name in EQUIPMENT_CLASSES:
                    candidate[name] += counts[name]
            if source_is_training_eligible(source):
                for partition in source.get("eligible_partitions", []):
                    if partition in verified:
                        for name in EQUIPMENT_CLASSES:
                            mapping = source["class_candidates"].get(name, {})
                            if mapping.get("semantic_status") == "verified":
                                verified[partition][name] += counts[name]
    shortfall = []
    for partition in PARTITIONS:
        for name in EQUIPMENT_CLASSES:
            required = MINIMUM_COVERAGE[partition]["positive_per_class"]
            actual = verified[partition][name]
            shortfall.append({"partition": partition, "class_name": name,
                              "actual": actual, "required": required,
                              "missing": max(0, required - actual)})
        shortfall.append({"partition": partition, "class_name": "no_target",
                          "actual": 0, "required": MINIMUM_COVERAGE[partition]["no_target"],
                          "missing": MINIMUM_COVERAGE[partition]["no_target"]})
    return {"verified": verified, "candidate": candidate, "shortfall": shortfall}


def _class_sources(rows):
    return {
        class_name: [
            row["source_id"] for row in rows
            if row["class_semantics"][class_name] == "verified"
        ]
        for class_name in EQUIPMENT_CLASSES
    }


def _recommend(rows, class_sources, partition_gaps):
    approved = [row["source_id"] for row in rows if row["training_eligible"]]
    quarantine = [
        row["source_id"] for row in sorted(
            rows, key=lambda item: (item["review_priority"], item["source_id"])
        ) if row["ingestion_status"] == "quarantine"
    ]
    missing = [name for name, values in class_sources.items() if not values]
    return {
        "approved_development_sources": approved,
        "audit_next": quarantine,
        "unresolved_classes_before_quarantine_review": missing,
        "unresolved_partition_cells": partition_gaps,
        "targeted_collection_policy": (
            "Only collect remaining class/partition cells after quarantine review"
        ),
        "decision": (
            "materialize_after_manifest_review" if not partition_gaps
            else "hold_training_and_audit_quarantine_sources"
        ),
    }


def _write_report(root, report):
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "real_source_feasibility.json", report)
    fields = [
        "source_id", "name", "declared_license_id", "provenance_status",
        "ingestion_status", "usage_role", "source_family", "training_eligible",
        "review_priority", "reported_image_count", "eligible_partitions",
        "blocking_issues",
    ]
    with (root / "real_source_feasibility.csv").open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({
                key: ";".join(value) if isinstance(value, list) else value
                for key, value in row.items() if key in fields
            })
