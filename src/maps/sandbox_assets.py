"""Allow-listed assets for user-authored sandbox maps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxAsset:
    asset_id: str
    display_name: str
    default_size_m: tuple[float, float, float]
    shape: str
    color: str
    visual_label: int | None = None


ASSETS = (
    SandboxAsset("transformer", "Transformer", (3.0, 3.0, 2.0), "box", "0.07 0.16 0.19 1", 1),
    SandboxAsset("switchgear", "Switchgear", (2.0, 1.5, 1.5), "box", "0.08 0.38 0.48 1", 2),
    SandboxAsset("capacitor_bank", "Capacitor bank", (2.5, 2.5, 1.8), "box", "0.16 0.35 0.40 1", 3),
    SandboxAsset("reactor", "Reactor", (2.0, 2.0, 2.2), "cylinder", "0.24 0.29 0.34 1", 4),
    SandboxAsset("cabinet", "Cabinet", (1.2, 1.0, 1.5), "box", "0.02 0.32 0.43 1"),
    SandboxAsset("pole", "Pole", (0.5, 0.5, 4.5), "cylinder", "0.08 0.09 0.09 1"),
    SandboxAsset("control_building", "Control building", (5.0, 4.0, 3.0), "box", "0.32 0.34 0.35 1"),
    SandboxAsset("generic_obstacle", "Obstacle", (2.0, 2.0, 2.0), "box", "0.42 0.28 0.20 1"),
)

ASSET_BY_ID = {asset.asset_id: asset for asset in ASSETS}
TARGET_ASSET_IDS = frozenset(
    asset.asset_id for asset in ASSETS if asset.visual_label is not None
)


def asset_for(asset_id: str) -> SandboxAsset:
    try:
        return ASSET_BY_ID[str(asset_id)]
    except KeyError as error:
        raise ValueError(f"unsupported sandbox asset: {asset_id}") from error


def asset_records() -> list[dict[str, object]]:
    return [
        {
            "asset_id": asset.asset_id,
            "display_name": asset.display_name,
            "default_size_m": list(asset.default_size_m),
            "shape": asset.shape,
            "label_role": "target" if asset.visual_label is not None else "background",
        }
        for asset in ASSETS
    ]
