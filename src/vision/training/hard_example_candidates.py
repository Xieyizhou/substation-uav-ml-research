"""Input records, image identities and split-safe duplicate clusters."""

from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass
import json
from pathlib import Path
from PIL import Image


@dataclass(frozen=True)
class Candidate:
    collection: str
    collection_identity: str
    frame_id: str
    map_id: str
    split: str
    seed: int
    rgb_path: str
    rgb_timestamp: float
    image_sha256: str
    perceptual_hash: str
    objects: tuple[dict, ...]

    @property
    def kind(self):
        return "target" if self.objects else "no_target"

    @property
    def key(self):
        return self.collection, self.frame_id

    def record(self):
        return {
            "collection": self.collection,
            "collection_identity": self.collection_identity,
            "frame_id": self.frame_id,
            "image_sha256": self.image_sha256,
            "map_id": self.map_id,
            "objects": list(self.objects),
            "perceptual_hash": self.perceptual_hash,
            "rgb_path": self.rgb_path,
            "rgb_timestamp": self.rgb_timestamp,
            "seed": self.seed,
            "split": self.split,
        }


def load_collection(collection, allowed_classes, maximum_truth_skew_ms=33.334, perceptual_hash_algorithm="receipt", hash_cache=None):
    collection = Path(collection).resolve()
    receipt = json.loads((collection / "collection-receipt.json").read_text())
    truth = [json.loads(line) for line in (collection / receipt["truth"]["relative_path"]).read_text().splitlines() if line.strip()]
    truth = sorted((row for row in truth if row.get("validation_status") == "valid"), key=lambda row: row["simulation_timestamp"])
    timestamps = [row["simulation_timestamp"] for row in truth]
    accepted, rejected = [], []
    hash_cache = {} if hash_cache is None else hash_cache
    for member in receipt["members"]:
        index = bisect_left(timestamps, member["rgb_timestamp"])
        indexes = [item for item in (index - 1, index) if 0 <= item < len(truth)]
        if not indexes:
            rejected.append({"frame_id": member["frame_id"], "reason": "missing_truth"})
            continue
        matched = min((truth[item] for item in indexes), key=lambda row: abs(row["simulation_timestamp"] - member["rgb_timestamp"]))
        skew_ms = abs(matched["simulation_timestamp"] - member["rgb_timestamp"]) * 1000.0
        if skew_ms > maximum_truth_skew_ms:
            rejected.append({"frame_id": member["frame_id"], "reason": "stale_truth", "skew_ms": skew_ms})
            continue
        objects = tuple(sorted((dict(item) for item in matched.get("objects", []) if item.get("validation_status") == "validated"), key=lambda item: item["annotation_id"]))
        unknown = sorted({item.get("class_name") for item in objects} - set(allowed_classes))
        if unknown:
            rejected.append({"frame_id": member["frame_id"], "reason": "unknown_truth_class", "classes": unknown})
            continue
        perceptual_hash = member["perceptual_hash"]
        if perceptual_hash_algorithm == "dhash64-v1":
            if member["image_sha256"] not in hash_cache:
                hash_cache[member["image_sha256"]] = dhash64(collection / member["rgb_path"])
            perceptual_hash = hash_cache[member["image_sha256"]]
        elif perceptual_hash_algorithm != "receipt":
            raise ValueError("unsupported perceptual hash algorithm")
        accepted.append(Candidate(
            str(collection), receipt["identity"], member["frame_id"], receipt["map_id"], receipt["split"],
            int(receipt["seed"]), member["rgb_path"], float(member["rgb_timestamp"]), member["image_sha256"],
            perceptual_hash, objects,
        ))
    return accepted, rejected, receipt["identity"]


def dhash64(path):
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:016x}"


def hamming_hex(left, right):
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _bucket(candidate):
    return ("all" if candidate.kind == "no_target" else candidate.map_id, candidate.kind, candidate.split)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _clusters(candidates, threshold):
    clusters = []
    representatives = []
    for candidate in sorted(candidates, key=lambda row: (row.perceptual_hash, row.key)):
        cluster_index = next(
            (index for index, value in enumerate(representatives) if hamming_hex(candidate.perceptual_hash, value) <= threshold),
            None,
        )
        if cluster_index is None:
            representatives.append(candidate.perceptual_hash)
            clusters.append([candidate])
        else:
            clusters[cluster_index].append(candidate)
    return [sorted(rows, key=lambda row: row.key) for rows in clusters]


