"""Dependency-free local HTTP application for safe sandbox operation."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
from urllib.parse import parse_qs, unquote, urlparse

from src.inspection import InspectionConfig, InspectionService
from src.inspection.config import AccessDenied
from src.sandbox.operator import OperatorBusy, SandboxOperator


STATIC_ROOT = Path(__file__).with_name("static")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class InspectionHandler(BaseHTTPRequestHandler):
    service: InspectionService
    operator_token: str

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/"):
                self._api(parsed.path, parse_qs(parsed.query))
            else:
                self._static(parsed.path)
        except (AccessDenied, ValueError) as error:
            self._json({"error": str(error)}, 400)
        except FileNotFoundError as error:
            self._json({"error": str(error)}, 404)

    def do_POST(self):
        if self.headers.get("X-Sandbox-Token") != self.operator_token:
            return self._json({"error": "invalid sandbox operator token"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            maximum = 2_800_000 if self.path == "/api/maps/import" else 500_000
            if length < 2 or length > maximum:
                raise ValueError("invalid JSON request size")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("JSON request must be an object")
            if self.path == "/api/operator/start":
                return self._json(
                    self.service.operator_start(
                        body.get("action"), body.get("scenario_id"),
                        body.get("parameters"),
                    ),
                    202,
                )
            if self.path == "/api/operator/stop":
                return self._json(self.service.operator_stop(body.get("job_id")))
            if self.path == "/api/maps/draft":
                return self._json(self.service.map_save(body.get("map")))
            if self.path == "/api/maps/draft/delete":
                return self._json(self.service.map_delete(body.get("map_id")))
            if self.path == "/api/maps/revision":
                return self._json(
                    self.service.map_revision_create(body.get("map_id")), 201
                )
            if self.path == "/api/maps/import":
                return self._json(self.service.map_import(body.get("bundle_base64")), 201)
            self._json({"error": "unknown endpoint"}, 404)
        except OperatorBusy as error:
            self._json({"error": str(error)}, 409)
        except (AccessDenied, ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, message, *args):
        print(f"inspection: {message % args}")

    def _api(self, path, query):
        if path == "/api/doctor":
            return self._json(self.service.doctor())
        if path == "/api/profile":
            return self._json(self.service.profile())
        if path == "/api/version":
            return self._json(self.service.version())
        if path == "/api/setup":
            return self._json(self.service.setup())
        if path == "/api/dashboard":
            return self._json(self.service.dashboard())
        if path == "/api/runtime":
            return self._json(self.service.runtime())
        if path == "/api/research":
            return self._json(self.service.research())
        if path == "/api/experiments":
            return self._json(self.service.experiments())
        if path == "/api/workbench":
            return self._json(self.service.workbench())
        if path == "/api/maps":
            return self._json(self.service.maps())
        if path == "/api/lidar":
            return self._json(self.service.lidar())
        if path == "/api/acceptance":
            return self._json(self.service.acceptance())
        if path == "/api/preflight":
            return self._json(self.service.preflight())
        if path == "/api/storage":
            return self._json(self.service.storage())
        if path == "/api/recordings":
            return self._json(self.service.recordings())
        if path == "/api/scenarios":
            return self._json(self.service.scenarios())
        if path == "/api/operator":
            return self._json({
                **self.service.operator_status(),
                "operator_token": self.operator_token,
            })
        parts = [unquote(item) for item in path.split("/") if item]
        if len(parts) == 3 and parts[1] == "maps":
            return self._json(self.service.map_detail(parts[2]))
        if len(parts) == 6 and parts[1:3] == ["maps", "revision"]:
            return self._file(
                self.service.map_revision_file(parts[3], parts[4], parts[5])
            )
        if len(parts) == 5 and parts[1:3] == ["maps", "bundle"]:
            return self._file(self.service.map_bundle_file(parts[3], parts[4]))
        if len(parts) == 4 and parts[1:3] == ["operator", "log"]:
            limit = int(query.get("limit", ["200"])[0])
            return self._json(self.service.operator_log(parts[3], limit))
        if len(parts) == 4 and parts[1:3] == ["workbench", "run"]:
            return self._json(self.service.workbench_run(parts[3]))
        if len(parts) == 4 and parts[1:3] == ["workbench", "inference"]:
            return self._json(self.service.workbench_inference(parts[3]))
        if len(parts) == 5 and parts[1:3] == ["workbench", "inference"]:
            return self._file(
                self.service.workbench_inference_image(parts[3], parts[4])
            )
        if len(parts) == 4 and parts[1] == "logs":
            limit = int(query.get("limit", ["200"])[0])
            return self._json(self.service.logs(parts[2], parts[3], limit))
        if len(parts) == 3 and parts[1] == "frames":
            page = int(query.get("page", ["1"])[0])
            size = int(query.get("page_size", ["24"])[0])
            return self._json(self.service.frames(parts[2], page, size))
        if len(parts) == 3 and parts[1] == "progress":
            return self._json(self.service.progress(parts[2]))
        if len(parts) == 4 and parts[1] == "frame":
            return self._file(self.service.frame_file(parts[2], parts[3]))
        return self._json({"error": "unknown endpoint"}, 404)

    def _static(self, path):
        name = "index.html" if path == "/" else path.lstrip("/")
        if name not in {
            "index.html", "app.js", "style.css", "operator.css", "research.css",
            "experiments.css", "setup.css", "setup.js",
            "profile.css", "ui_state.js", "navigation.css", "navigation.js",
        }:
            return self._json({"error": "not found"}, 404)
        self._file(STATIC_ROOT / name)

    def _file(self, path):
        payload = Path(path).read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)


class SandboxHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, handler, operator):
        self.operator = operator
        super().__init__(address, handler)

    def server_close(self):
        self.operator.shutdown()
        super().server_close()


def create_server(
    config, host="127.0.0.1", port=8765, adapter=None, operator=None
):
    if host not in LOOPBACK_HOSTS:
        raise ValueError("sandbox app must bind to a loopback host")
    operator = operator or SandboxOperator(config, adapter)
    token = secrets.token_urlsafe(24)
    handler = type("ConfiguredInspectionHandler", (InspectionHandler,), {
        "service": InspectionService(config, adapter, operator),
        "operator_token": token,
    })
    return SandboxHTTPServer((host, port), handler, operator)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local research sandbox app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile", choices=("demo", "development", "formal"),
        default="development",
    )
    args = parser.parse_args(argv)
    config = InspectionConfig.for_profile(args.project_root, args.profile)
    server = create_server(config, args.host, args.port)
    print(f"Research sandbox app: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
