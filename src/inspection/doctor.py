"""Read-only environment diagnostics; no process is started or installed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

from src.inspection.config import InspectionConfig
from src.inspection.models import DoctorCheck


def _check(name, passed, explanation, action="", warning=False):
    status = "pass" if passed else "warning" if warning else "failure"
    return DoctorCheck(name, status, explanation, action if not passed else "")


def run_doctor(config: InspectionConfig) -> tuple[DoctorCheck, ...]:
    version_ok = sys.version_info >= (3, 11)
    executable = Path(sys.executable).resolve()
    expected = (config.project_root / ".venv/bin/python").resolve()
    expected_present = expected.is_file()
    project_environment = executable == expected
    checks = [
        _check(
            "Python", version_ok,
            f"Python {sys.version.split()[0]} at {executable}",
            "Use Python 3.11 or newer.",
        ),
        _check(
            "Python environment", project_environment or not expected_present,
            "Using the project virtual environment" if project_environment else (
                f"Using {executable}; project environment is {expected}"
                if expected_present else "No project .venv was found"
            ),
            f"Run commands with {expected}.", warning=True,
        ),
    ]
    for executable in ("git", "gz", "make"):
        found = shutil.which(executable)
        checks.append(_check(
            executable, bool(found), found or f"{executable} not found on PATH",
            f"Install {executable} manually and add it to PATH.", warning=True,
        ))
    for module in ("mavsdk", "matplotlib", "pandas"):
        found = importlib.util.find_spec(module) is not None
        checks.append(_check(
            f"Python module: {module}", found,
            f"{module} {'is available' if found else 'is not available'}",
            "Install the declared project dependencies manually.", warning=True,
        ))
    checks.extend(_path_checks(config))
    usage = shutil.disk_usage(config.project_root)
    free_gib = usage.free / (1024 ** 3)
    checks.append(_check(
        "Disk space", free_gib >= config.minimum_free_gib,
        f"{free_gib:.1f} GiB free", f"Free at least {config.minimum_free_gib:g} GiB.",
        warning=True,
    ))
    return tuple(checks)


def _path_checks(config):
    paths = (
        ("Project", config.project_root, False),
        ("Collection plan", config.plan_path, False),
        ("Collection root", config.collection_root, True),
        ("Recordings", config.recordings_root, True),
        ("PX4", config.px4_root, True),
    )
    results = []
    for name, path, warning in paths:
        exists = path.exists()
        results.append(_check(
            name, exists, f"{path} {'exists' if exists else 'is missing'}",
            f"Verify the configured {name.lower()} path.", warning=warning,
        ))
    worlds = tuple((config.project_root / "simulation/worlds").glob("*.sdf"))
    results.append(_check(
        "Gazebo worlds", bool(worlds), f"{len(worlds)} SDF world(s) found",
        "Restore the tracked simulation world definitions.",
    ))
    return results
