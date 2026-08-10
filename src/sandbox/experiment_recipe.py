"""Identity-bound recipes for non-blind visual sandbox experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from src.ml.artifacts import file_sha256, git_commit, object_sha256, write_json
from src.vision.contracts.identity import ModelIdentity
from src.vision.contracts.identity_validation import positive_int, required_text, sha256
from src.vision.contracts.training_identity import TrainingViewIdentity
from src.vision.evaluation.yolo_package import validate_yolo_package


RECIPE_SCHEMA_VERSION = 1
ALLOWED_PARTITIONS = frozenset({"validation", "full_validation"})
ALLOWED_INPUT_SIZES = frozenset({320, 416, 640})
ALLOWED_FRAME_SKIP_INTERVALS = frozenset({1, 2, 3})
DEFAULT_DATASET_ROOT = Path("data/research/visual_yolo_v2")
DEFAULT_PACKAGE_ROOT = Path("models/equipment/visual-yolo11n-baseline-v2-package")
EXPERIMENT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


@dataclass(frozen=True)
class ExperimentRecipe:
    experiment_id: str
    partition: str
    input_size: int
    frame_skip_interval: int
    frame_limit: int
    source_frame_count: int
    inference_frame_count: int
    dataset_root: str
    package_root: str
    training_view_identity_sha256: str
    membership_sha256: str
    package_identity_sha256: str
    model_identity_sha256: str
    model_file_sha256: str
    preprocessing_configuration_id: str
    confidence_threshold: float
    runtime_backend: str
    device: str
    software_commit_sha: str
    recipe_schema_version: int = RECIPE_SCHEMA_VERSION

    def __post_init__(self):
        if self.recipe_schema_version != RECIPE_SCHEMA_VERSION:
            raise ValueError("unsupported sandbox experiment recipe schema")
        if not EXPERIMENT_ID.fullmatch(str(self.experiment_id)):
            raise ValueError("invalid experiment_id")
        if self.partition not in ALLOWED_PARTITIONS:
            raise ValueError("sandbox recipes only allow non-blind validation partitions")
        if self.input_size not in ALLOWED_INPUT_SIZES:
            raise ValueError("unsupported sandbox input size")
        if self.frame_skip_interval not in ALLOWED_FRAME_SKIP_INTERVALS:
            raise ValueError("unsupported sandbox frame-skip interval")
        for name in ("frame_limit", "source_frame_count", "inference_frame_count"):
            positive_int(getattr(self, name), name)
        if self.frame_limit != self.source_frame_count:
            raise ValueError("frame_limit must equal the bounded source frame count")
        expected = (self.source_frame_count + self.frame_skip_interval - 1) // self.frame_skip_interval
        if self.inference_frame_count != expected:
            raise ValueError("inference frame count does not match frame-skip policy")
        for name in ("dataset_root", "package_root", "runtime_backend", "device", "software_commit_sha"):
            object.__setattr__(self, name, required_text(getattr(self, name), name))
        for name in (
            "training_view_identity_sha256", "membership_sha256",
            "package_identity_sha256", "model_identity_sha256",
            "model_file_sha256", "preprocessing_configuration_id",
        ):
            sha256(getattr(self, name), name)
        threshold = float(self.confidence_threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("confidence_threshold must be between zero and one")
        object.__setattr__(self, "confidence_threshold", threshold)
        if self.device != "cpu":
            raise ValueError("sandbox ONNX recipes currently require device=cpu")

    def identity_record(self):
        return asdict(self)

    @property
    def recipe_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {**self.identity_record(), "recipe_identity_sha256": self.recipe_identity_sha256}

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        supplied = values.pop("recipe_identity_sha256", None)
        recipe = cls(**values)
        if supplied != recipe.recipe_identity_sha256:
            raise ValueError("sandbox experiment recipe identity mismatch")
        return recipe


def _clean_commit(project_root):
    commit = git_commit(project_root)
    if commit == "unknown" or commit.endswith("-dirty"):
        raise ValueError("sandbox experiment recipes require a clean tracked commit")
    return commit


def _load_membership(dataset_root, partition, identity):
    path = dataset_root / "identity" / f"{partition}_membership.jsonl"
    expected_hash = getattr(identity, f"{partition}_membership_sha256")
    if file_sha256(path) != expected_hash:
        raise ValueError("sandbox experiment membership hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected_count = getattr(identity, f"{partition}_frame_count")
    if len(rows) != expected_count:
        raise ValueError("sandbox experiment membership count mismatch")
    prefixes = (f"images/{partition}/", f"labels/{partition}/")
    for row in rows:
        if not str(row.get("image_relative_path", "")).startswith(prefixes[0]):
            raise ValueError("sandbox recipe contains an invalid image partition")
        if not str(row.get("label_relative_path", "")).startswith(prefixes[1]):
            raise ValueError("sandbox recipe contains an invalid label partition")
    return path, rows


def materialize_recipe(
    project_root,
    experiment_id,
    *,
    partition="validation",
    input_size=416,
    frame_skip_interval=1,
    frame_limit=None,
):
    if not EXPERIMENT_ID.fullmatch(str(experiment_id)):
        raise ValueError("invalid experiment_id")
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError("sandbox recipes only allow non-blind validation partitions")
    if input_size not in ALLOWED_INPUT_SIZES:
        raise ValueError("unsupported sandbox input size")
    if frame_skip_interval not in ALLOWED_FRAME_SKIP_INTERVALS:
        raise ValueError("unsupported sandbox frame-skip interval")
    project_root = Path(project_root).resolve()
    dataset_root = project_root / DEFAULT_DATASET_ROOT
    package_root = project_root / DEFAULT_PACKAGE_ROOT
    identity = TrainingViewIdentity.from_record(
        json.loads((dataset_root / "identity/training_view_identity.json").read_text())
    )
    package = validate_yolo_package(package_root)
    if package["training_view_identity_sha256"] != identity.training_view_identity_sha256:
        raise ValueError("model package and training view identities differ")
    membership_path, rows = _load_membership(dataset_root, partition, identity)
    total = len(rows)
    limit = total if frame_limit is None else int(frame_limit)
    if limit <= 0 or limit > total:
        raise ValueError(f"frame_limit must be between 1 and {total}")
    models = json.loads((package_root / "model_identities.json").read_text())
    model = ModelIdentity.from_record(models[str(input_size)])
    export = package["exports"][str(input_size)]
    recipe = ExperimentRecipe(
        experiment_id=experiment_id,
        partition=partition,
        input_size=input_size,
        frame_skip_interval=frame_skip_interval,
        frame_limit=limit,
        source_frame_count=limit,
        inference_frame_count=(limit + frame_skip_interval - 1) // frame_skip_interval,
        dataset_root=DEFAULT_DATASET_ROOT.as_posix(),
        package_root=DEFAULT_PACKAGE_ROOT.as_posix(),
        training_view_identity_sha256=identity.training_view_identity_sha256,
        membership_sha256=file_sha256(membership_path),
        package_identity_sha256=package["package_identity_sha256"],
        model_identity_sha256=model.model_identity_sha256,
        model_file_sha256=export["sha256"],
        preprocessing_configuration_id=model.preprocessing_configuration_id,
        confidence_threshold=package["frozen_confidence_threshold"],
        runtime_backend=model.runtime_backend,
        device="cpu",
        software_commit_sha=_clean_commit(project_root),
    )
    output = project_root / "outputs/sandbox/experiments" / experiment_id
    if output.exists():
        raise ValueError("sandbox experiment directory already exists")
    write_json(output / "recipe.json", recipe.to_record())
    write_json(output / "status.json", {
        "sandbox_experiment_status_schema_version": 1,
        "state": "ready",
        "recipe_identity_sha256": recipe.recipe_identity_sha256,
    })
    return recipe, output


def inspect_recipe(path):
    path = Path(path)
    recipe = ExperimentRecipe.from_record(json.loads(path.read_text()))
    return {"valid": True, "path": str(path), "recipe": recipe.to_record()}
