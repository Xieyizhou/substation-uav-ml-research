"""Confirmation-first seam for future aerial-image map generation."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json


@dataclass(frozen=True)
class MapDraftCandidate:
    candidate_id: str
    class_name: str
    confidence: float
    east_m: float
    north_m: float
    width_m: float
    depth_m: float
    height_m: float
    yaw_deg: float
    image_region_xyxy: tuple[float, float, float, float]
    user_confirmation_status: str = "pending"

    def __post_init__(self):
        if not 0 <= self.confidence <= 1 or min(self.width_m, self.depth_m, self.height_m) <= 0:
            raise ValueError("invalid MapDraft candidate confidence or dimensions")
        if self.user_confirmation_status not in {"pending", "accepted", "rejected"}:
            raise ValueError("invalid user confirmation status")


@dataclass(frozen=True)
class MapDraft:
    source_image_sha256: str
    image_width_px: int
    image_height_px: int
    candidates: tuple[MapDraftCandidate, ...] = field(default_factory=tuple)
    schema_version: int = 1

    @property
    def artifact_identity(self):
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
