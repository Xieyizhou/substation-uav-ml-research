"""Aggregate accepted visual recordings into split-isolated identities."""

from __future__ import annotations

from pathlib import Path

from src.ml.artifacts import git_commit, object_sha256
from src.vision.collection.plan import load_collection_protocol
from src.vision.collection.recording import validate_collection_recording
from src.vision.collection.dataset_coverage import (
    aggregate_split_counts,
    dataset_groups,
    validate_aggregate_coverage,
)
from src.vision.contracts.identity import DatasetIdentity, class_order_identity
from src.vision.contracts.protocol import V2_PROTOCOL_ID, protocol_for_plan
from src.vision.collection.pilot import _read_json, _read_jsonl, _sha256, _write_json, _write_jsonl


def _load_recording(root, row, plan, protocol_path):
    validation = validate_collection_recording(
        root,
        plan,
        protocol_path=protocol_path,
    )
    metadata = _read_json(root / "metadata.json")
    summary = _read_json(root / "summary.json")
    frames = {
        item["frame_id"]: item for item in _read_jsonl(root / "frames.jsonl")
    }
    annotations = _read_jsonl(root / "annotations.jsonl")
    return {
        "row": row,
        "root": root,
        "validation": validation,
        "metadata": metadata,
        "summary": summary,
        "frames": frames,
        "annotations": annotations,
        "annotation_manifest_sha256": _sha256(root / "annotations.jsonl"),
    }


def _aggregate_recordings(plan, recordings_root, protocol):
    recordings = []
    missing = []
    for row in plan["scenarios"]:
        root = Path(recordings_root) / row["recording_id"]
        if not root.is_dir():
            missing.append(row["recording_id"])
            continue
        try:
            recordings.append(_load_recording(root, row, plan, protocol.path))
        except ValueError as error:
            raise ValueError(
                f"collection recording {row['recording_id']} is invalid: {error}"
            ) from error
    if missing:
        raise ValueError(
            f"visual collection is missing {len(missing)} recordings; "
            f"first missing: {missing[0]}"
        )
    decoder_ids = {
        item["summary"]["decoder_configuration_id"] for item in recordings
    }
    if len(decoder_ids) != 1:
        raise ValueError("visual collection recordings use multiple decoders")
    split_counts = aggregate_split_counts(recordings, protocol)
    return recordings, decoder_ids.pop(), split_counts


def _materialize_group(
    name,
    splits,
    recordings,
    decoder_id,
    output_root,
    protocol,
    creation_commit_sha,
):
    selected = [item for item in recordings if item["row"]["split"] in splits]
    membership = []
    aggregate_annotations = []
    for item in selected:
        row = item["row"]
        for annotation in item["annotations"]:
            frame = item["frames"][annotation["frame_id"]]
            sample_id = f"{row['recording_id']}:{annotation['frame_id']}"
            membership.append(
                {
                    "sample_id": sample_id,
                    "recording_id": row["recording_id"],
                    "scenario_id": row["scenario_id"],
                    "split": row["split"],
                    "frame_id": annotation["frame_id"],
                    "sequence_number": annotation["sequence_number"],
                    "payload_sha256": annotation["payload_sha256"],
                    "payload_relative_path": frame["payload_relative_path"],
                }
            )
            aggregate_annotations.append(
                {
                    "sample_id": sample_id,
                    "recording_id": row["recording_id"],
                    "annotation": annotation,
                }
            )
    identity_root = Path(output_root) / "identity"
    membership_path = identity_root / f"{name}_membership.jsonl"
    annotations_path = identity_root / f"{name}_annotations.jsonl"
    _write_jsonl(membership_path, membership)
    _write_jsonl(annotations_path, aggregate_annotations)
    rows = [item["row"] for item in selected]
    split_manifest = [
        {
            "recording_id": row["recording_id"],
            "scenario_id": row["scenario_id"],
            "split": row["split"],
        }
        for row in rows
    ]
    collection_version = (
        "v2" if protocol["protocol_id"] == V2_PROTOCOL_ID else "v1"
    )
    identity = DatasetIdentity(
        dataset_name=f"visual_collection_{collection_version}_{name}",
        dataset_version=protocol["protocol_id"],
        dataset_role=name,
        recording_schema_version=2,
        annotation_schema_version=1,
        decoder_configuration_id=decoder_id,
        recording_manifest_sha256=_sha256(membership_path),
        scenario_manifest_sha256=object_sha256(rows),
        split_manifest_sha256=object_sha256(split_manifest),
        ordered_frame_count=len(membership),
        labelled_frame_count=sum(
            item["annotation"]["annotation_status"] == "labelled"
            for item in aggregate_annotations
        ),
        source_recording_ids=tuple(row["recording_id"] for row in rows),
        scenario_ids=tuple(row["scenario_id"] for row in rows),
        map_ids=tuple(sorted({row["map_id"] for row in rows})),
        seed_ids=tuple(row["seed"] for row in rows),
        class_order_identity=class_order_identity(),
        annotation_manifest_sha256=_sha256(annotations_path),
        source_payload_formats=("png",),
        creation_commit_sha=creation_commit_sha,
    )
    path = identity_root / f"{name}_dataset_identity.json"
    _write_json(path, identity.to_record())
    return identity, path


def materialize_collection_datasets(
    plan,
    recordings_root,
    output_root,
    *,
    protocol_path=None,
    creation_commit_sha=None,
):
    protocol = (
        protocol_for_plan(plan)
        if protocol_path is None
        else load_collection_protocol(protocol_path)
    )
    if creation_commit_sha is None:
        observed_commit = git_commit()
        creation_commit_sha = (
            None
            if observed_commit == "unknown" or observed_commit.endswith("-dirty")
            else observed_commit
        )
    recordings, decoder_id, split_counts = _aggregate_recordings(
        plan,
        recordings_root,
        protocol,
    )
    validate_aggregate_coverage(split_counts, protocol)
    identities = {}
    for name, splits in dataset_groups(protocol).items():
        identity, path = _materialize_group(
            name,
            splits,
            recordings,
            decoder_id,
            output_root,
            protocol,
            creation_commit_sha,
        )
        identities[name] = {
            "path": str(path),
            "dataset_identity_sha256": identity.dataset_identity_sha256,
            "ordered_frame_count": identity.ordered_frame_count,
            "labelled_frame_count": identity.labelled_frame_count,
        }
    collection = {
        "visual_collection_identity_schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "collection_plan_identity_sha256": plan[
            "collection_plan_identity_sha256"
        ],
        "split_counts": split_counts,
        "datasets": identities,
    }
    collection["visual_collection_identity_sha256"] = object_sha256(collection)
    path = Path(output_root) / "identity/collection_identity.json"
    _write_json(path, collection)
    return {**collection, "path": str(path)}
