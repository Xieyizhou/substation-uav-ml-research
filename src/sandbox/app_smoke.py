"""Loopback HTTP smoke check for the dependency-free Sandbox App."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

from src.inspection.app import create_server
from src.inspection.config import InspectionConfig
from src.ml.artifacts import object_sha256


APP_SMOKE_SCHEMA_VERSION = 1


def _read(url):
    with urlopen(url, timeout=5) as response:
        return response.status, response.headers.get_content_type(), response.read()


def run_app_smoke(project_root):
    config = InspectionConfig.for_profile(Path(project_root), "demo")
    server = create_server(config, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    checks = {}
    try:
        status, content_type, body = _read(f"{base}/")
        checks["app_shell"] = status == 200 and content_type == "text/html" and bool(body)
        for name in ("profile", "storage", "operator"):
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
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    result = {
        "app_smoke_schema_version": APP_SMOKE_SCHEMA_VERSION,
        "profile": "demo",
        "checks": checks,
        "passed": all(checks.values()),
    }
    result["app_smoke_identity_sha256"] = object_sha256(result)
    return result


def main():
    print(json.dumps(run_app_smoke(Path.cwd()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
