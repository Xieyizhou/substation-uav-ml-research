"""Read-only compatibility inspection for externally managed runtimes."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


MANIFEST_PATH = Path("config/sandbox/runtime_compatibility.json")


def load_manifest(project_root: Path) -> dict:
    value = json.loads((Path(project_root) / MANIFEST_PATH).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported runtime compatibility manifest")
    return value


def inspect_runtime(project_root: Path, profile: str, environment=None) -> dict:
    root = Path(project_root).resolve()
    env = dict(os.environ if environment is None else environment)
    manifest = load_manifest(root)
    components = [
        _python(manifest),
        _px4(manifest, env),
        _gazebo(manifest, env),
        _formula("opencv", "OpenCV", manifest["opencv"], env),
        _formula("qt", "Qt", manifest["qt"], env),
    ]
    blocking = {"missing", "unsupported", "changed_since_validation"}
    if profile == "formal":
        blocking.add("untested")
    blockers = [item["detail"] for item in components if item["status"] in blocking]
    warning = any(item["status"] in {"untested", "compatible_with_warning"}
                  for item in components)
    overall = "unsupported" if blockers else "compatible_with_warning" if warning else "compatible"
    return {
        "runtime_compatibility_schema_version": 1,
        "profile": profile,
        "overall": overall,
        "ready": not blockers,
        "blockers": blockers,
        "components": components,
    }


def _component(component_id, title, status, detail, path=None, version=None, identity=None):
    return {
        "component_id": component_id,
        "title": title,
        "status": status,
        "detail": detail,
        "path": path,
        "version": version,
        "identity": identity,
    }


def _python(manifest):
    version = ".".join(str(value) for value in sys.version_info[:3])
    executable = str(Path(sys.executable).absolute())
    minimum = tuple(int(value) for value in manifest["python"]["minimum_version"].split("."))
    missing = [name for name in manifest["python"]["required_development_packages"]
               if importlib.util.find_spec(name) is None]
    if sys.version_info[:2] < minimum or missing:
        status = "unsupported"
    elif ".".join(version.split(".")[:2]) in manifest["python"]["tested_minor_versions"]:
        status = "compatible"
    else:
        status = "untested"
    suffix = f"; missing packages: {', '.join(missing)}" if missing else ""
    return _component(
        "python", "Python", status, f"Python {version} at {executable}{suffix}",
        executable, version, f"{executable}|{version}",
    )


def _px4(manifest, env):
    root = Path(env.get("PX4_ROOT", Path.home() / "PX4-Autopilot")).expanduser().resolve()
    required = [root / relative for relative in manifest["px4"]["required_paths"]]
    if not (root / ".git").exists() or not all(path.exists() for path in required):
        return _component("px4", "PX4", "missing", f"Invalid PX4 checkout: {root}", str(root))
    commit = _run(["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"])
    if not commit:
        return _component("px4", "PX4", "unsupported", "PX4 Git identity is unreadable", str(root))
    describe = _run(["/usr/bin/git", "-C", str(root), "describe", "--tags", "--always"])
    status = "compatible" if commit in manifest["px4"]["tested_commits"] else "untested"
    return _component(
        "px4", "PX4", status, f"PX4 {commit[:12]} at {root}",
        str(root), describe, commit,
    )


def _gazebo(manifest, env):
    explicit = env.get("UAV_SANDBOX_GZ_EXECUTABLE")
    candidates = ([Path(explicit)] if explicit else []) + _all_commands("gz", env)
    first_untested = None
    for executable in _unique(candidates):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            continue
        output = _run([str(executable), "sim", "--versions"])
        match = re.search(r"\b(\d+)\.(\d+)(?:\.(\d+))?\b", output or "")
        if not match:
            item = _component(
                "gazebo", "Gazebo", "untested",
                f"Gazebo at {executable}; version could not be verified", str(executable),
            )
            first_untested = first_untested or item
            continue
        version = match.group(0)
        status = ("compatible" if int(match.group(1))
                  in manifest["gazebo"]["tested_sim_major_versions"] else "unsupported")
        item = _component(
            "gazebo", "Gazebo", status,
            f"{manifest['gazebo']['expected_distribution']} / Gazebo Sim {version} at {executable}",
            str(executable.resolve()), version, f"{executable.resolve()}|{version}",
        )
        if status == "compatible":
            return item
    return first_untested or _component(
        "gazebo", "Gazebo", "missing", "No supported Gazebo Sim executable was found"
    )


def _formula(component_id, title, rule, env):
    variable = f"UAV_SANDBOX_{component_id.upper()}_PREFIX"
    prefix = env.get(variable)
    if prefix and not Path(prefix).exists():
        prefix = None
    brew = shutil.which("brew", path=env.get("PATH"))
    if not prefix and brew:
        prefix = _run([brew, "--prefix", rule["preferred_formula"]])
    if not prefix:
        return _component(component_id, title, "missing", f"{rule['preferred_formula']} not found")
    version = _run([brew, "list", "--versions", rule["preferred_formula"]]) if brew else None
    version = version.split(maxsplit=1)[1] if version and " " in version else None
    if not version:
        version_match = re.search(r"\b\d+\.\d+(?:\.\d+)?\b", Path(prefix).resolve().name)
        version = version_match.group(0) if version_match else None
    major = int(version.split(".")[0]) if version and version.split(".")[0].isdigit() else None
    status = "compatible" if major == rule["expected_major_version"] else "untested" if major is None else "unsupported"
    return _component(
        component_id, title, status, f"{title} {version or 'unknown'} at {prefix}",
        str(Path(prefix).resolve()), version, f"{prefix}|{version or 'unknown'}",
    )


def _all_commands(name, env):
    search = env.get("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
    return [Path(folder) / name for folder in search.split(os.pathsep)] + [
        Path("/opt/homebrew/bin") / name, Path("/usr/local/bin") / name,
    ]


def _unique(values):
    seen = set()
    return [value for value in values if not (str(value) in seen or seen.add(str(value)))]


def _run(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output if output else None
