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

# Closed allowlists: URL input never selects arbitrary service attributes.
GET_ACTIONS = {
    "/api/doctor": "doctor",
    "/api/profile": "profile",
    "/api/version": "version",
    "/api/setup": "setup",
    "/api/dashboard": "dashboard",
    "/api/runtime": "runtime",
    "/api/research": "research",
    "/api/experiments": "experiments",
    "/api/workbench": "workbench",
    "/api/feedback": "feedback",
    "/api/maps": "maps",
    "/api/map-runs": "map_runs",
    "/api/lidar": "lidar",
    "/api/live-replan": "live_replan",
    "/api/visual-replan": "visual_replan",
    "/api/evidence": "evidence",
    "/api/semantic-flight": "semantic_flight",
    "/api/acceptance": "acceptance",
    "/api/preflight": "preflight",
    "/api/storage": "storage",
    "/api/recordings": "recordings",
    "/api/scenarios": "scenarios",
}
POST_ACTIONS = {
    "/api/feedback/review": ("feedback_review", ("collection_id", "sample_id", "review", "expected_identity"), 200),
    "/api/operator/start": ("operator_start", ("action", "scenario_id", "parameters"), 202),
    "/api/operator/stop": ("operator_stop", ("job_id",), 200),
    "/api/maps/draft": ("map_save", ("map",), 200),
    "/api/maps/draft/delete": ("map_delete", ("map_id",), 200),
    "/api/maps/revision": ("map_revision_create", ("map_id",), 201),
    "/api/maps/import": ("map_import", ("bundle_base64",), 201),
    "/api/maps/register": ("map_recording_register", ("run_id", "dataset_id"), 201),
}


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
            action = POST_ACTIONS.get(self.path)
            if action is not None:
                method, fields, status = action
                result = getattr(self.service, method)(*(body.get(key) for key in fields))
                return self._json(result, status)
            self._json({"error": "unknown endpoint"}, 404)
        except OperatorBusy as error:
            self._json({"error": str(error)}, 409)
        except (AccessDenied, ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, 400)

    def log_message(self, message, *args):
        print(f"inspection: {message % args}")

    def _api(self, path, query):
        method = GET_ACTIONS.get(path)
        if method is not None:
            return self._json(getattr(self.service, method)())
        if path == "/api/operator":
            return self._json({
                **self.service.operator_status(),
                "operator_token": self.operator_token,
            })
        parts = [unquote(item) for item in path.split("/") if item]
        if len(parts) == 3 and parts[1] == "semantic-model":
            return self._json(self.service.semantic_model(parts[2]))
        if len(parts) == 3 and parts[1] == "feedback":
            return self._json(self.service.feedback_collection(parts[2]))
        if len(parts) == 4 and parts[1] == "feedback":
            return self._json(self.service.feedback_sample(parts[2], parts[3]))
        if len(parts) == 5 and parts[1] == "feedback" and parts[4] == "image":
            return self._file(self.service.feedback_image(parts[2], parts[3]))
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
            "index.html", "app.js", "semantic_flight.js", "live_replan.js", "visual_replan.js", "evidence.js", "style.css", "operator.css", "research.css",
            "experiments.css", "setup.css", "setup.js", "feedback.js", "feedback.css",
            "profile.css", "ui_state.js", "navigation.css", "navigation.js",
            "map_studio.css", "map_canvas.js", "map_studio_actions.js",
            "map_studio.js", "map_flight.js", "scenario_selector.js",
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
    # The desktop starts multiple independent panel requests in one burst.
    request_queue_size = 64

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
