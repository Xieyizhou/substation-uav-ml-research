"""Typed contracts for editable and immutable sandbox maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from src.maps.sandbox_assets import TARGET_ASSET_IDS, asset_for
from src.ml.artifacts import object_sha256


MAP_ID = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
OBJECT_ID = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
MISSION_TYPES = frozenset({"point_to_point", "round_trip", "equipment_inspection"})


def _finite(value, name):
    value = float(value)
    if value != value or value in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class SandboxMapObject:
    object_id: str
    asset_id: str
    east_m: float
    north_m: float
    width_m: float
    depth_m: float
    height_m: float
    yaw_deg: float = 0.0
    label_role: str = "background"

    def __post_init__(self):
        if not OBJECT_ID.fullmatch(self.object_id):
            raise ValueError(f"invalid sandbox object id: {self.object_id!r}")
        asset = asset_for(self.asset_id)
        for name in ("east_m", "north_m", "width_m", "depth_m", "height_m", "yaw_deg"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if min(self.width_m, self.depth_m, self.height_m) <= 0.0:
            raise ValueError(f"sandbox object {self.object_id} has a non-positive size")
        if max(self.width_m, self.depth_m) > 20.0 or self.height_m > 12.0:
            raise ValueError(f"sandbox object {self.object_id} exceeds the size limit")
        expected = "target" if asset.asset_id in TARGET_ASSET_IDS else "background"
        if self.label_role != expected:
            raise ValueError(
                f"sandbox object {self.object_id} must use label_role {expected!r}"
            )
        object.__setattr__(self, "yaw_deg", self.yaw_deg % 360.0)


@dataclass(frozen=True)
class SandboxMission:
    mission_id: str
    mission_type: str
    goal_east_m: float | None = None
    goal_north_m: float | None = None
    target_object_id: str | None = None
    altitude_m: float = 1.5
    speed_m_s: float = 1.0
    horizontal_inflation_cells: int = 1

    def __post_init__(self):
        if not OBJECT_ID.fullmatch(self.mission_id):
            raise ValueError(f"invalid sandbox mission id: {self.mission_id!r}")
        if self.mission_type not in MISSION_TYPES:
            raise ValueError(f"unsupported sandbox mission type: {self.mission_type}")
        object.__setattr__(self, "altitude_m", _finite(self.altitude_m, "altitude_m"))
        object.__setattr__(self, "speed_m_s", _finite(self.speed_m_s, "speed_m_s"))
        if not 0.5 <= self.altitude_m <= 10.0:
            raise ValueError("sandbox altitude must be between 0.5 and 10 m")
        if not 0.1 <= self.speed_m_s <= 5.0:
            raise ValueError("sandbox speed must be between 0.1 and 5 m/s")
        if not 0 <= int(self.horizontal_inflation_cells) <= 5:
            raise ValueError("sandbox obstacle inflation must be from 0 to 5 cells")
        if self.mission_type == "equipment_inspection":
            if not self.target_object_id or self.goal_east_m is not None or self.goal_north_m is not None:
                raise ValueError("equipment inspection requires only target_object_id")
        elif self.target_object_id is not None or self.goal_east_m is None or self.goal_north_m is None:
            raise ValueError("point missions require only goal_east_m and goal_north_m")
        if self.goal_east_m is not None:
            object.__setattr__(self, "goal_east_m", _finite(self.goal_east_m, "goal_east_m"))
            object.__setattr__(self, "goal_north_m", _finite(self.goal_north_m, "goal_north_m"))


@dataclass(frozen=True)
class SandboxMap:
    map_id: str
    display_name: str
    width_m: float
    height_m: float
    start_east_m: float
    start_north_m: float
    start_yaw_deg: float
    objects: tuple[SandboxMapObject, ...]
    missions: tuple[SandboxMission, ...]
    resolution_m: float = 1.0
    editor_snap_m: float = 0.5
    weather: str = "clear"
    light_level: float = 1.0
    camera_noise_stddev: float = 0.0
    parent_revision_identity: str | None = None
    sandbox_map_schema_version: int = 1

    def __post_init__(self):
        if not MAP_ID.fullmatch(self.map_id):
            raise ValueError(f"invalid sandbox map id: {self.map_id!r}")
        if not self.display_name.strip() or len(self.display_name) > 80:
            raise ValueError("sandbox map display name is invalid")
        for name in ("width_m", "height_m", "start_east_m", "start_north_m", "start_yaw_deg"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not 16.0 <= self.width_m <= 60.0 or not 16.0 <= self.height_m <= 60.0:
            raise ValueError("sandbox map dimensions must be between 16 and 60 m")
        if self.resolution_m != 1.0 or self.editor_snap_m != 0.5:
            raise ValueError("sandbox map v1 requires 1 m resolution and 0.5 m snapping")
        if self.weather not in {"clear", "overcast", "mist"}:
            raise ValueError(f"unsupported sandbox weather: {self.weather}")
        if not 0.2 <= float(self.light_level) <= 1.5:
            raise ValueError("sandbox light level must be between 0.2 and 1.5")
        if not 0.0 <= float(self.camera_noise_stddev) <= 0.2:
            raise ValueError("sandbox camera noise must be between 0 and 0.2")
        if len({item.object_id for item in self.objects}) != len(self.objects):
            raise ValueError("sandbox map object ids must be unique")
        if len({item.mission_id for item in self.missions}) != len(self.missions):
            raise ValueError("sandbox map mission ids must be unique")
        object_ids = {item.object_id for item in self.objects}
        for mission in self.missions:
            if mission.target_object_id and mission.target_object_id not in object_ids:
                raise ValueError(f"mission target does not exist: {mission.target_object_id}")
        object.__setattr__(self, "start_yaw_deg", self.start_yaw_deg % 360.0)

    def identity_record(self):
        return asdict(self)

    @property
    def map_identity_sha256(self):
        return object_sha256(self.identity_record())

    def to_record(self):
        return {**self.identity_record(), "map_identity_sha256": self.map_identity_sha256}

    @classmethod
    def from_record(cls, value):
        record = dict(value)
        supplied = record.pop("map_identity_sha256", None)
        record["objects"] = tuple(SandboxMapObject(**item) for item in record["objects"])
        record["missions"] = tuple(SandboxMission(**item) for item in record["missions"])
        result = cls(**record)
        if supplied is not None and supplied != result.map_identity_sha256:
            raise ValueError("sandbox map identity mismatch")
        return result
