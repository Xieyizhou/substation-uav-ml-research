"""Explicit review of bounded feedback collections; predictions are suggestions."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import re

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import file_sha256, object_sha256, write_json

HEX = re.compile(r"[0-9a-f]{64}\Z")
COLLECTION = re.compile(r"[0-9a-f]{24}\Z")


def checked_json(path, identity_field):
    value = json.loads(Path(path).read_text())
    identity = value.pop(identity_field, None)
    if not identity or object_sha256(value) != identity:
        raise ValueError("Record identity changed: " + str(path))
    return dict(value, **{identity_field: identity})


def _inside(root, path):
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Feedback path escapes its approved root")
    return path


class FeedbackReviewStore:
    def __init__(self, config):
        self.config = config
        self.root = Path(config.project_root).resolve()
        self.reviews = _inside(self.root, config.workbench_root / "reviews")

    def collections(self):
        result = []
        patterns = ("outputs/sandbox/semantic_runs/semantic-*/vision/feedback/collection.json",
                    "outputs/sandbox/feedback/*/collection.json",
                    "outputs/sandbox/feedback/*/feedback/collection.json",
                    "data/research/material-shadow-v1/autonomy-avoidance-v1/sandbox-replan-v1-*/vision/feedback/collection.json")
        for pattern in patterns:
            for path in self.root.glob(pattern):
                try:
                    directory = _inside(self.root, path.parent)
                    summary = checked_json(directory / "summary.json", "collection_identity_sha256")
                    relative = directory.relative_to(self.root).as_posix()
                    result.append(dict(collection_id=object_sha256(relative)[:24], path=relative,
                        name=directory.parent.parent.name if "/vision/feedback" in relative else relative.removeprefix("outputs/sandbox/feedback/"),
                        state=summary["state"], sample_count=summary["sample_count"],
                        source_group=object_sha256(summary["source_identity"]),
                        identity=summary["collection_identity_sha256"]))
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return sorted(result, key=lambda row: row["path"])

    def resolve(self, collection_id):
        if not COLLECTION.fullmatch(str(collection_id)):
            raise ValueError("Invalid feedback collection identifier")
        matches = [row for row in self.collections() if row["collection_id"] == collection_id]
        if len(matches) != 1:
            raise ValueError("Feedback collection is missing or invalid")
        return _inside(self.root, self.root / matches[0]["path"])

    @contextmanager
    def completed(self, collection_id):
        directory = self.resolve(collection_id)
        with _inside(directory, directory / ".writer.lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Collection is still recording; wait for completion") from error
            summary = checked_json(directory / "summary.json", "collection_identity_sha256")
            config = json.loads((directory / "collection.json").read_text())
            if summary["state"] != "complete" or summary.get("error"):
                raise ValueError("Only completed feedback collections can be reviewed")
            if any(summary.get(key) != value for key, value in config.items()):
                raise ValueError("Feedback collection configuration changed")
            members = []
            for path in sorted((directory / "samples").glob("*.json")):
                row = checked_json(_inside(directory, path), "sample_identity_sha256")
                if (path.stem != row["rgb_sha256"] or not HEX.fullmatch(path.stem)
                        or row["source_identity"] != summary["source_identity"]):
                    raise ValueError("Feedback sample source changed")
                members.append((path.stem, row["sample_identity_sha256"]))
            if len(members) != summary["sample_count"] or object_sha256(members) != summary["membership_sha256"]:
                raise ValueError("Feedback collection membership changed")
            yield directory, summary, members

    def sample(self, collection_id, sample_id, *, directory=None):
        if not HEX.fullmatch(str(sample_id)):
            raise ValueError("Invalid feedback sample identifier")
        directory = directory or self.resolve(collection_id)
        row = checked_json(_inside(directory, directory / "samples" / (sample_id + ".json")), "sample_identity_sha256")
        frame = row["frame"]
        if row["rgb_sha256"] != sample_id or frame["payload_relative_path"] != "frames/" + sample_id + ".png":
            raise ValueError("Feedback image membership changed")
        image = _inside(directory, directory / frame["payload_relative_path"])
        if file_sha256(image) != frame["payload_sha256"]:
            raise ValueError("Feedback image changed")
        return row, image

    def review(self, collection_id, sample_id, sample):
        path = self.reviews / collection_id / sample_id / "latest.json"
        if not path.exists():
            return None
        value = checked_json(_inside(self.reviews, path), "review_identity_sha256")
        if (value["sample_identity_sha256"] != sample["sample_identity_sha256"]
                or value["image_sha256"] != sample["frame"]["payload_sha256"]):
            raise ValueError("Review no longer matches the original sample")
        return value

    def detail(self, collection_id):
        with self.completed(collection_id) as (directory, summary, members):
            rows = []
            for sample_id, _ in members:
                sample = checked_json(directory / "samples" / (sample_id + ".json"), "sample_identity_sha256")
                review = self.review(collection_id, sample_id, sample)
                rows.append(dict(sample_id=sample_id, status=review["decision"] if review else "unreviewed",
                                 boxes=len(review["boxes"]) if review else None))
            return dict(collection_id=collection_id, summary=summary, samples=rows)

    def sample_detail(self, collection_id, sample_id):
        with self.completed(collection_id) as (directory, summary, _):
            sample, _ = self.sample(collection_id, sample_id, directory=directory)
            return dict(sample=sample, review=self.review(collection_id, sample_id, sample),
                        source_group=object_sha256(summary["source_identity"]))

    def save(self, collection_id, sample_id, decision, expected_identity=None):
        if self.config.profile != "development":
            raise ValueError("Feedback review requires Development profile")
        with self.completed(collection_id) as (directory, summary, _):
            sample, _ = self.sample(collection_id, sample_id, directory=directory)
            value = validate_decision(decision, sample["frame"])
            destination = _inside(self.reviews, self.reviews / collection_id / sample_id)
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / ".lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                old = self.review(collection_id, sample_id, sample)
                if (old or {}).get("review_identity_sha256") != expected_identity:
                    raise ValueError("Review changed in another window; reload before saving")
                value.update(collection_id=collection_id, sample_id=sample_id,
                    source_group=object_sha256(summary["source_identity"]),
                    sample_identity_sha256=sample["sample_identity_sha256"],
                    image_sha256=sample["frame"]["payload_sha256"],
                    previous_review_identity=expected_identity,
                    reviewed_at=datetime.now(timezone.utc).isoformat(),
                    annotation_source="explicit_review", training_admitted=False)
                value["review_identity_sha256"] = object_sha256(value)
                write_json(destination / (value["review_identity_sha256"] + ".json"), value)
                write_json(destination / "latest.json", value)
                return value


def validate_decision(value, frame):
    if not isinstance(value, dict) or set(value) != {"decision", "boxes", "no_target", "complete_annotation", "reviewer", "note"}:
        raise ValueError("Review must contain an explicit decision, boxes and reviewer")
    if value["decision"] not in ("accepted", "rejected"):
        raise ValueError("Choose accepted or rejected")
    if not isinstance(value["reviewer"], str) or not 1 <= len(value["reviewer"].strip()) <= 100:
        raise ValueError("Reviewer name is required (1–100 characters)")
    if not isinstance(value["note"], str) or len(value["note"]) > 2000:
        raise ValueError("Review note is too long")
    if type(value["no_target"]) is not bool or type(value["complete_annotation"]) is not bool:
        raise ValueError("Review confirmations must be explicit booleans")
    if not isinstance(value["boxes"], list) or len(value["boxes"]) > 100:
        raise ValueError("Review accepts at most 100 boxes")
    boxes = []
    for box in value["boxes"]:
        if not isinstance(box, dict) or set(box) != {"class_name", "xyxy"} or box["class_name"] not in EQUIPMENT_CLASSES:
            raise ValueError("Unsupported reviewed class")
        if not isinstance(box["xyxy"], list) or len(box["xyxy"]) != 4:
            raise ValueError("Reviewed box requires four coordinates")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in box["xyxy"]):
            raise ValueError("Reviewed box coordinates must be finite numbers")
        x1, y1, x2, y2 = box["xyxy"]
        if not (0 <= x1 < x2 <= frame["width"] and 0 <= y1 < y2 <= frame["height"]):
            raise ValueError("Reviewed box is outside the image or has zero area")
        boxes.append(dict(class_name=box["class_name"], xyxy=list(box["xyxy"])))
    if value["decision"] == "accepted" and (value["complete_annotation"] is not True or value["no_target"] != (not boxes)):
        raise ValueError("Confirm complete labels, or explicitly confirm no target")
    return dict(value, boxes=boxes, reviewer=value["reviewer"].strip())
