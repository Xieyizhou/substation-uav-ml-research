"""Dependency-free local HTTP application for read-only project inspection."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.inspection import InspectionConfig, InspectionService
from src.inspection.config import AccessDenied


STATIC_ROOT = Path(__file__).with_name("static")


class InspectionHandler(BaseHTTPRequestHandler):
    service: InspectionService

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
        self._json({"error": "read-only application"}, 405)

    def log_message(self, message, *args):
        print(f"inspection: {message % args}")

    def _api(self, path, query):
        if path == "/api/doctor":
            return self._json(self.service.doctor())
        if path == "/api/dashboard":
            return self._json(self.service.dashboard())
        if path == "/api/runtime":
            return self._json(self.service.runtime())
        if path == "/api/recordings":
            return self._json(self.service.recordings())
        parts = [unquote(item) for item in path.split("/") if item]
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
        if name not in {"index.html", "app.js", "style.css"}:
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


def create_server(config, host="127.0.0.1", port=8765, adapter=None):
    handler = type("ConfiguredInspectionHandler", (InspectionHandler,), {
        "service": InspectionService(config, adapter)
    })
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the read-only research inspector")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    config = InspectionConfig.defaults(args.project_root)
    server = create_server(config, args.host, args.port)
    print(f"Read-only research inspector: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
