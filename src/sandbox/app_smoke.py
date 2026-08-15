"""Loopback HTTP smoke check for the dependency-free Sandbox App."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

from src.inspection.app import create_server
from src.inspection.config import InspectionConfig
from src.ml.artifacts import object_sha256
from src.sandbox.gate_outcome import (
    ENVIRONMENT_UNAVAILABLE, PASSED, PRODUCT_FAILURE,
)
from src.sandbox.profiles import sandbox_profile
from src.sandbox.version import load_sandbox_version


APP_SMOKE_SCHEMA_VERSION = 2


def _read(url):
    with urlopen(url, timeout=5) as response:
        return response.status, response.headers.get_content_type(), response.read()


def _contract_checks(config):
    static = config.project_root / "src/inspection/static"
    try:
        version_valid = bool(load_sandbox_version(config.project_root).macos_app_version)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        version_valid = False
    return {
        "contract_profile": (
            config.profile == "demo"
            and sandbox_profile(config.profile).flight_enabled is False
        ),
        "contract_assets": all(
            (static / name).is_file() for name in (
                "index.html", "app.js", "style.css", "operator.css",
                "research.css", "experiments.css", "profile.css",
                "setup.css", "setup.js",
            )
        ),
        "contract_version": version_valid,
    }


def _result(checks, outcome, reason_code=None, detail=None):
    result = {
        "app_smoke_schema_version": APP_SMOKE_SCHEMA_VERSION,
        "profile": "demo",
        "checks": checks,
        "outcome": outcome,
        "reason_code": reason_code,
        "detail": detail,
        "passed": outcome == PASSED and all(checks.values()),
    }
    result["app_smoke_identity_sha256"] = object_sha256(result)
    return result


def run_app_smoke(project_root):
    config = InspectionConfig.for_profile(Path(project_root), "demo")
    checks = _contract_checks(config)
    if not all(checks.values()):
        return _result(checks, PRODUCT_FAILURE, "app_contract_failed")
    try:
        server = create_server(config, port=0)
    except PermissionError as error:
        return _result(
            checks, ENVIRONMENT_UNAVAILABLE, "loopback_bind_denied",
            f"{type(error).__name__}: {error}",
        )
    except OSError as error:
        return _result(
            checks, PRODUCT_FAILURE, "loopback_start_failed",
            f"{type(error).__name__}: {error}",
        )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    failure = None
    try:
        status, content_type, body = _read(f"{base}/")
        checks["app_shell"] = status == 200 and content_type == "text/html" and bool(body)
        for name in ("profile", "version", "setup", "storage", "operator"):
            status, content_type, body = _read(f"{base}/api/{name}")
            value = json.loads(body)
            checks[f"api_{name}"] = (
                status == 200 and content_type == "application/json" and bool(value)
            )
        profile_status, _, profile_body = _read(f"{base}/api/profile")
        profile = json.loads(profile_body)
        checks["demo_boundary"] = (
            profile_status == 200
            and profile.get("profile_id") == "demo"
            and profile.get("flight_enabled") is False
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    if failure is not None:
        return _result(
            checks, PRODUCT_FAILURE, "loopback_contract_failed", failure,
        )
    return _result(
        checks, PASSED if all(checks.values()) else PRODUCT_FAILURE,
        None if all(checks.values()) else "loopback_contract_failed",
    )


def main():
    result = run_app_smoke(Path.cwd())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["outcome"] == ENVIRONMENT_UNAVAILABLE:
        return 2
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
