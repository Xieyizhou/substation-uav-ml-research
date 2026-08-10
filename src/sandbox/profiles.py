"""Central sandbox profile registry and capability declarations."""

from __future__ import annotations

from dataclasses import dataclass


PROFILE_NAMES = ("demo", "development", "formal")


@dataclass(frozen=True)
class SandboxProfile:
    profile_id: str
    title: str
    description: str
    available_workflows: tuple[str, ...]
    flight_enabled: bool
    formal_evidence: bool

    def to_record(self):
        return {
            "profile_id": self.profile_id,
            "title": self.title,
            "description": self.description,
            "available_workflows": list(self.available_workflows),
            "flight_enabled": self.flight_enabled,
            "formal_evidence": self.formal_evidence,
        }


_DEMO = SandboxProfile(
    "demo",
    "Demo profile",
    "Runs locally without recorded datasets, model weights, PX4, or Gazebo.",
    ("demo_contract",),
    False,
    False,
)

_FULL_WORKFLOWS = (
    "demo_contract",
    "visual_replay",
    "lidar_replay",
    "lidar_closed_loop",
    "multimodal_acceptance",
)

_PROFILES = {
    "demo": _DEMO,
    "development": SandboxProfile(
        "development",
        "Development profile",
        "Uses local non-blind datasets and enables managed simulator workflows.",
        _FULL_WORKFLOWS,
        True,
        False,
    ),
    "formal": SandboxProfile(
        "formal",
        "Formal evidence profile",
        "Uses frozen local artifacts and preserves blind-data access gates.",
        _FULL_WORKFLOWS,
        True,
        True,
    ),
}


def sandbox_profile(profile_id: str) -> SandboxProfile:
    try:
        return _PROFILES[str(profile_id)]
    except KeyError as error:
        raise ValueError(f"unsupported sandbox profile: {profile_id}") from error
