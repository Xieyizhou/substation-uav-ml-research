"""Stable records shared by the visual model workbench."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.ml import EQUIPMENT_CLASSES
from src.ml.artifacts import object_sha256


DATASET_SCHEMA_VERSION = 1
RECIPE_SCHEMA_VERSION = 1
PRESETS = {
    "smoke": {"epochs": 1, "patience": 1, "imgsz": 320, "batch": 4,
              "device": "auto",
              "workers": 2, "train_limit": 256, "validation_limit": 64},
    "quick": {"epochs": 10, "patience": 5, "imgsz": 416, "batch": 8,
              "device": "auto",
              "workers": 4, "train_limit": 5_000, "validation_limit": 1_000},
    "full": {"epochs": 100, "patience": 15, "imgsz": 640, "batch": 8,
             "device": "auto",
             "workers": 4, "train_limit": None, "validation_limit": None},
}
ALLOWED = {
    "epochs": range(1, 101), "patience": range(0, 31),
    "imgsz": (320, 416, 640), "batch": (1, 2, 4, 8, 16),
    "workers": range(0, 9), "device": ("auto", "mps", "cpu"),
}


@dataclass(frozen=True)
class WorkbenchDataset:
    dataset_id: str
    source_type: str
    dataset_root: str
    dataset_identity_sha256: str
    class_names: tuple[str, ...] = EQUIPMENT_CLASSES
    split_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    no_target_counts: dict[str, int] = field(default_factory=dict)
    source_identity_sha256: str | None = None
    status: str = "ready"
    issues: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["workbench_dataset_schema_version"] = DATASET_SCHEMA_VERSION
        value["class_names"] = list(self.class_names)
        value["issues"] = list(self.issues)
        return value

    @classmethod
    def from_record(cls, value: dict[str, Any]) -> "WorkbenchDataset":
        if value.get("workbench_dataset_schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError("unsupported workbench dataset schema")
        fields = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
        fields["class_names"] = tuple(fields.get("class_names", ()))
        fields["issues"] = tuple(fields.get("issues", ()))
        result = cls(**fields)
        if result.class_names != EQUIPMENT_CLASSES:
            raise ValueError("workbench dataset class order mismatch")
        return result


@dataclass(frozen=True)
class WorkbenchExperimentRecipe:
    experiment_id: str
    dataset_id: str
    dataset_identity_sha256: str
    preset: str
    parameters: dict[str, Any]
    pretrained_weights_sha256: str
    seed: int = 7
    dataset_role: str = "development"
    baseline_package_identity_sha256: str | None = None
    recipe_identity_sha256: str = ""

    def to_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["workbench_recipe_schema_version"] = RECIPE_SCHEMA_VERSION
        identity = value.pop("recipe_identity_sha256", "")
        value["recipe_identity_sha256"] = identity or object_sha256(value)
        return value

    @classmethod
    def create(cls, **values) -> "WorkbenchExperimentRecipe":
        preset = values["preset"]
        if preset not in PRESETS:
            raise ValueError("unknown workbench preset")
        parameters = {**PRESETS[preset], **values.pop("overrides", {})}
        for name, allowed in ALLOWED.items():
            if parameters[name] not in allowed:
                raise ValueError(f"unsupported workbench parameter: {name}")
        result = cls(parameters=parameters, **values)
        return cls(**{**asdict(result), "recipe_identity_sha256": object_sha256(
            {key: value for key, value in result.to_record().items()
             if key != "recipe_identity_sha256"}
        )})

    @classmethod
    def from_record(cls, value: dict[str, Any]) -> "WorkbenchExperimentRecipe":
        if value.get("workbench_recipe_schema_version") != RECIPE_SCHEMA_VERSION:
            raise ValueError("unsupported workbench recipe schema")
        supplied = value.get("recipe_identity_sha256")
        payload = {key: item for key, item in value.items()
                   if key not in ("recipe_identity_sha256", "workbench_recipe_schema_version")}
        if supplied != object_sha256({**payload, "workbench_recipe_schema_version": 1}):
            raise ValueError("workbench recipe identity mismatch")
        return cls(**payload, recipe_identity_sha256=supplied)
