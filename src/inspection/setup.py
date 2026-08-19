"""Profile-aware, read-only first-run dependency inspection."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
import sys

from src.inspection.config import InspectionConfig
from src.runtime_compatibility import inspect_runtime


PX4_SETUP_URL = "https://docs.px4.io/main/en/dev_setup/dev_env"
GAZEBO_SETUP_URL = "https://gazebosim.org/docs/harmonic/install_osx/"


@dataclass(frozen=True)
class SetupItem:
    item_id: str
    title: str
    status: str
    required: bool
    detail: str
    action: str = ""
    command: str = ""
    docs_url: str = ""
    compatibility_status: str = "compatible"

    def to_record(self):
        return self.__dict__


@dataclass(frozen=True)
class SetupReport:
    setup_schema_version: int
    profile: str
    ready: bool
    required_ready_count: int
    required_count: int
    items: tuple[SetupItem, ...]
    next_step: str
    journey: tuple[dict, ...]
    artifact_locations: dict

    def to_record(self):
        return {
            **{
                key: value
                for key, value in self.__dict__.items()
                if key not in {"items", "journey"}
            },
            "items": [item.to_record() for item in self.items],
            "journey": list(self.journey),
        }


class LocalSetupProbe:
    @staticmethod
    def executable(name: str) -> str | None:
        return shutil.which(name)

    @staticmethod
    def module(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    @staticmethod
    def python() -> tuple[tuple[int, int], str]:
        return sys.version_info[:2], str(Path(sys.executable).absolute())


def inspect_setup(config: InspectionConfig, probe=None) -> SetupReport:
    local_runtime = None
    if probe is None:
        try:
            local_runtime = inspect_runtime(config.project_root, config.profile)
        except (OSError, ValueError):
            # Older project roots and isolated callers may not have a runtime
            # manifest. Keep the read-only setup report available; App launch
            # still requires the verified manifest before advanced profiles run.
            local_runtime = None
    probe = probe or LocalSetupProbe()
    simulator_required = config.profile != "demo"
    runtime = ({item["component_id"]: item for item in local_runtime["components"]}
               if local_runtime else {})
    items = (
        _runtime_item(runtime["python"], True, config.profile) if runtime else _python_item(probe),
        _executable_item(probe, "git", "Git", True, "Install Git or Xcode Command Line Tools."),
        _project_environment_item(config, probe, simulator_required),
        _runtime_item(runtime["px4"], simulator_required, config.profile)
        if runtime else _px4_checkout_item(config, simulator_required),
        _px4_build_item(config, simulator_required),
        _runtime_item(runtime["gazebo"], simulator_required, config.profile)
        if runtime else _executable_item(
            probe, "gz", "Gazebo command-line tools", simulator_required,
            "Install Gazebo Harmonic, then reopen the App.",
            "brew tap osrf/simulation && brew install gz-harmonic",
            GAZEBO_SETUP_URL,
        ),
        _module_item(probe, "mavsdk", "MAVSDK Python", simulator_required),
        *(() if not runtime else (
            _runtime_item(runtime["opencv"], simulator_required, config.profile),
            _runtime_item(runtime["qt"], simulator_required, config.profile),
        )),
        _world_item(config, simulator_required),
    )
    required = tuple(item for item in items if item.required)
    ready_count = sum(item.status == "ready" for item in required)
    ready = ready_count == len(required)
    journey = _journey(config, ready)
    next_step = (
        "Open Experiments and run the Demo classifier."
        if ready and config.profile == "demo" else
        "Open Workbench to train or inspect a verified model."
        if ready and journey[1]["status"] == "complete" else
        "Open Operator and run one flight smoke."
        if ready else
        "Resolve the required items below, then select Check again."
    )
    return SetupReport(
        2, config.profile, ready, ready_count, len(required), items, next_step,
        journey, {
            "managed_jobs": config.sandbox_jobs_root.relative_to(
                config.project_root
            ).as_posix(),
            "flight_smoke": "outputs/sandbox/flight_smoke",
            "workbench": config.workbench_root.relative_to(config.project_root).as_posix(),
        },
    )


def _journey(config, environment_ready):
    from src.sandbox.job_models import SandboxJobStore

    jobs = SandboxJobStore(config.sandbox_jobs_root).list(200)
    flight_complete = any(
        job.action == "flight-smoke" and job.state == "complete" for job in jobs
    )
    model_ready = False
    if config.profile == "development":
        from src.sandbox.workbench_inference import list_verified_models

        model_ready = bool(list_verified_models(config.workbench_runs_root))
    return (
        {"step_id": "environment", "title": "Environment check",
         "status": "complete" if environment_ready else "blocked", "tab": "setup"},
        {"step_id": "flight", "title": "First managed flight",
         "status": "complete" if flight_complete else "ready" if environment_ready else "blocked",
         "tab": "operator"},
        {"step_id": "workbench", "title": "Verified model loop",
         "status": "complete" if model_ready else "ready" if environment_ready else "blocked",
         "tab": "experiments"},
        {"step_id": "results", "title": "Inspect results and artifacts",
         "status": "ready" if flight_complete or model_ready else "blocked", "tab": "overview"},
    )


def _item(item_id, title, found, required, detail, action="", command="", docs_url=""):
    status = "ready" if found else "missing" if required else "optional"
    return SetupItem(item_id, title, status, required, detail, action, command, docs_url)


def _runtime_item(value, required, profile):
    compatibility = value["status"]
    allowed = compatibility in {"compatible", "compatible_with_warning"}
    allowed = allowed or (compatibility == "untested" and profile != "formal")
    status = "ready" if allowed else "missing" if required else "optional"
    return SetupItem(
        value["component_id"], value["title"], status, required,
        value["detail"], compatibility_status=compatibility,
    )


def _python_item(probe):
    version, executable = probe.python()
    found = version >= (3, 11)
    return _item(
        "python", "Python 3.11+", found, True,
        f"Python {version[0]}.{version[1]} at {executable}",
        "Install Python 3.11 or newer.",
    )


def _executable_item(probe, item_id, title, required, action, command="", docs_url=""):
    path = probe.executable(item_id)
    return _item(
        item_id, title, bool(path), required, path or f"{title} was not found on PATH",
        action, command, docs_url,
    )


def _project_environment_item(config, probe, required):
    expected = config.project_root / ".venv/bin/python"
    _, executable = probe.python()
    found = expected.is_file() and Path(executable).absolute() == expected
    return _item(
        "project_environment", "Project Python environment", found, required,
        f"Active interpreter: {executable}",
        "Create the project environment and install runtime dependencies.",
        "python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt",
    )


def _px4_checkout_item(config, required):
    found = (config.px4_root / "CMakeLists.txt").is_file()
    return _item(
        "px4_checkout", "PX4 source checkout", found, required,
        str(config.px4_root), "Install PX4 using the official platform guide.",
        docs_url=PX4_SETUP_URL,
    )


def _px4_build_item(config, required):
    binary = config.px4_root / "build/px4_sitl_default/bin/px4"
    return _item(
        "px4_sitl", "PX4 SITL build", binary.is_file(), required, str(binary),
        "Build the PX4 SITL target once.",
        f"cd {config.px4_root} && make px4_sitl_default",
        PX4_SETUP_URL,
    )


def _module_item(probe, module, title, required):
    found = probe.module(module)
    return _item(
        module, title, found, required,
        f"Python module {module} is {'available' if found else 'not available'}",
        "Install the project runtime dependencies.",
        ".venv/bin/python -m pip install -r requirements.txt",
    )


def _world_item(config, required):
    count = len(tuple((config.project_root / "simulation/worlds").glob("*.sdf")))
    return _item(
        "simulation_worlds", "Sandbox simulation worlds", count > 0, required,
        f"{count} tracked SDF world(s) found",
        "Restore the tracked simulation assets from Git.", "git restore simulation/worlds",
    )
