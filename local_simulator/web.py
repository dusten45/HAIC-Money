from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .api import LocalApiHandler, SimulationRegistry
from .map import default_artifact_root


class LocalWebHandler(LocalApiHandler, BaseHTTPRequestHandler):
    static_root: Path

    def do_GET(self) -> None:
        if urlsplit(self.path).path.startswith("/api/"):
            super().do_GET()
            return
        self._serve_static()

    def _serve_static(self) -> None:
        path_value = unquote(urlsplit(self.path).path)
        relative = path_value.lstrip("/") or "index.html"
        candidate = (self.static_root / Path(*relative.split("/"))).resolve()
        try:
            candidate.relative_to(self.static_root)
        except ValueError:
            self.send_error(403, "forbidden")
            return
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            self.send_error(404, "not found")
            return
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args) -> None:
        del format, args


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    project_root: Path | None = None,
    artifact_root: Path | None = None,
) -> ThreadingHTTPServer:
    resolved_project_root = Path(project_root or Path.cwd()).resolve()
    resolved_static_root = resolved_project_root / "web_simulator"
    registry = SimulationRegistry(
        artifact_root=artifact_root,
        project_root=resolved_project_root,
    )
    handler_class = type(
        "ConfiguredLocalWebHandler",
        (LocalWebHandler,),
        {
            "registry": registry,
            "static_root": resolved_static_root,
        },
    )
    server = ThreadingHTTPServer((host, port), handler_class)
    server.registry = registry
    return server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the HAIC Track Lab and its local simulation API."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, default=default_artifact_root())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    server = create_server(
        host=args.host,
        port=args.port,
        project_root=args.project_root,
        artifact_root=args.artifact_root,
    )
    print(f"Track Lab: http://{args.host}:{server.server_address[1]}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.registry.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
