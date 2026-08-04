"""Minimum ground-truth contracts for static visual replay benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.ml import EQUIPMENT_CLASSES
from src.vision.contracts.identity import _required_text, _sha256
from src.sensors.camera_decoded import DecodedImage
from src.sensors.types import CameraFrame


VISUAL_ANNOTATION_SCHEMA_VERSION = 1
BOUNDING_BOX_CONVENTION = "xyxy_pixels_half_open"
VISIBILITY_STATES = frozenset(
    {"visible", "partially_occluded", "heavily_occluded", "unknown"}
)
TRUNCATION_STATES = frozenset({"not_truncated", "truncated", "unknown"})
TRUTH_SOURCES = frozenset(
    {
        "simulator_geometry",
        "gazebo_bounding_box_sensor",
        "manual_annotation",
        "verified_imported_annotation",
    }
)
VALIDATION_STATES = frozenset({"unverified", "validated", "rejected"})
ANNOTATION_STATES = frozenset(
    {"labelled", "verified_no_target", "pending", "rejected"}
)
MISSION_PHASES = frozenset(
    {
        "cruise_distant",
        "approach",
        "close_inspection",
        "target_transition",
        "other",
    }
)


@dataclass(frozen=True)
class VisualObjectAnnotation:
    annotation_id: str
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    visibility_status: str
    truncation_status: str
    truth_source: str
    validation_status: str
    bbox_coordinate_convention: str = BOUNDING_BOX_CONVENTION

    def __post_init__(self):
        object.__setattr__(
            self,
            "annotation_id",
            _required_text(self.annotation_id, "annotation_id"),
        )
        if self.class_name not in EQUIPMENT_CLASSES:
            raise ValueError(f"unsupported equipment class: {self.class_name!r}")
        if (
            isinstance(self.class_id, bool)
            or not isinstance(self.class_id, int)
            or self.class_id != EQUIPMENT_CLASSES.index(self.class_name)
        ):
            raise ValueError("class_id does not match the locked class order")
        bbox = tuple(float(value) for value in self.bbox_xyxy)
        if len(bbox) != 4:
            raise ValueError("bbox_xyxy must contain four values")
        x_min, y_min, x_max, y_max = bbox
        if min(bbox) < 0 or x_max <= x_min or y_max <= y_min:
            raise ValueError("bbox_xyxy must have positive visible area")
        object.__setattr__(self, "bbox_xyxy", bbox)
        if self.bbox_coordinate_convention != BOUNDING_BOX_CONVENTION:
            raise ValueError("unsupported bounding-box coordinate convention")
        if self.visibility_status not in VISIBILITY_STATES:
            raise ValueError("unsupported visibility status")
        if self.truncation_status not in TRUNCATION_STATES:
            raise ValueError("unsupported truncation status")
        if self.truth_source not in TRUTH_SOURCES:
            raise ValueError("unsupported visual truth source")
        if self.validation_status not in VALIDATION_STATES:
            raise ValueError("unsupported annotation validation status")

    def validate_dimensions(self, width, height):
        _, _, x_max, y_max = self.bbox_xyxy
        if x_max > width or y_max > height:
            raise ValueError(
                f"annotation {self.annotation_id!r} exceeds image dimensions"
            )

    def to_record(self):
        return asdict(self)

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        values["bbox_xyxy"] = tuple(values["bbox_xyxy"])
        return cls(**values)


@dataclass(frozen=True)
class VisualFrameAnnotation:
    frame_id: str
    source_id: str
    sequence_number: int
    payload_sha256: str
    decoded_content_sha256: str
    image_width: int
    image_height: int
    scenario_id: str
    map_id: str
    seed: int
    mission_phase: str
    frame_order_reference: int
    annotation_status: str
    objects: tuple[VisualObjectAnnotation, ...]
    dataset_identity_sha256: str | None = None
    recording_id: str | None = None
    annotation_schema_version: int = VISUAL_ANNOTATION_SCHEMA_VERSION

    def __post_init__(self):
        if self.annotation_schema_version != VISUAL_ANNOTATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported visual annotation schema "
                f"{self.annotation_schema_version}"
            )
        for name in ("frame_id", "source_id", "scenario_id", "map_id"):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        if self.recording_id is not None:
            object.__setattr__(
                self,
                "recording_id",
                _required_text(self.recording_id, "recording_id"),
            )
        if self.dataset_identity_sha256 is not None:
            object.__setattr__(
                self,
                "dataset_identity_sha256",
                _sha256(
                    self.dataset_identity_sha256, "dataset_identity_sha256"
                ),
            )
        if self.dataset_identity_sha256 is None and self.recording_id is None:
            raise ValueError(
                "annotation must reference a dataset identity or recording ID"
            )
        for name in ("payload_sha256", "decoded_content_sha256"):
            _sha256(getattr(self, name), name)
        for name in (
            "sequence_number",
            "seed",
            "frame_order_reference",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("image_width", "image_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.mission_phase not in MISSION_PHASES:
            raise ValueError("unsupported mission phase")
        if self.annotation_status not in ANNOTATION_STATES:
            raise ValueError("unsupported frame annotation status")
        objects = tuple(
            value
            if isinstance(value, VisualObjectAnnotation)
            else VisualObjectAnnotation.from_record(value)
            for value in self.objects
        )
        object.__setattr__(self, "objects", objects)
        annotation_ids = [item.annotation_id for item in objects]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("annotation IDs must be unique within a frame")
        for item in objects:
            item.validate_dimensions(self.image_width, self.image_height)
        if self.annotation_status == "verified_no_target" and objects:
            raise ValueError("verified_no_target frames cannot contain objects")
        if self.annotation_status == "labelled" and not objects:
            raise ValueError(
                "labelled frames require objects; use verified_no_target otherwise"
            )

    def validate_linkage(self, frame, decoded_image):
        if not isinstance(frame, CameraFrame):
            raise TypeError("frame must be a CameraFrame")
        if not isinstance(decoded_image, DecodedImage):
            raise TypeError("decoded_image must be a DecodedImage")
        expected = {
            "frame_id": frame.frame_id,
            "source_id": frame.source_id,
            "sequence_number": frame.sequence_number,
            "payload_sha256": frame.payload_sha256,
            "decoded_content_sha256": decoded_image.decoded_content_sha256,
            "image_width": decoded_image.width,
            "image_height": decoded_image.height,
        }
        mismatches = [
            name for name, value in expected.items() if getattr(self, name) != value
        ]
        if mismatches:
            raise ValueError(
                "visual annotation linkage mismatch: " + ", ".join(mismatches)
            )
        return True

    def to_record(self):
        record = asdict(self)
        record["objects"] = [item.to_record() for item in self.objects]
        return record

    @classmethod
    def from_record(cls, record):
        values = dict(record)
        values["objects"] = tuple(
            VisualObjectAnnotation.from_record(item)
            for item in values.get("objects", [])
        )
        return cls(**values)
