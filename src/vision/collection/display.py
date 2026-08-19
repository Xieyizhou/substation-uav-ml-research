"""Gazebo display-mode policy shared by flight and collection workflows."""

from __future__ import annotations


DISPLAY_MODES = ("headless", "visual_preview")


def launcher_environment(prepared, display_mode):
    if display_mode not in DISPLAY_MODES:
        raise ValueError(f"unsupported simulator display mode: {display_mode}")
    environment = dict(prepared["launcher_environment"])
    if display_mode == "headless":
        environment["HEADLESS"] = "1"
    else:
        environment.pop("HEADLESS", None)
    return environment
