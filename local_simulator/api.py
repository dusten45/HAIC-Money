from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import threading
from typing import Any
from uuid import uuid4

from .logging import run_log_to_dict, save_run_log
from .map import default_artifact_root
from .policies import AgentPolicy, BaselinePolicy, ManualPolicy, Policy
from .preview import build_map_preview
from .schema import (
    CustomMapSpec,
    MapDocument,
    map_from_dict,
    map_to_dict,
)
from .session import SimulationSession
from .track_generator import generate_custom_map


MAX_CONTROL_BODY_BYTES = 256 * 1024
SAFE_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass
class ActiveRun:
    run_id: str
    session: SimulationSession


class SimulationRegistry:
    def __init__(
        self,
        artifact_root: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root or default_artifact_root()).resolve()
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.maps_root = self.artifact_root / "maps"
        self.runs_root = self.artifact_root / "runs"
        self.maps_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active: dict[str, ActiveRun] = {}
        self._completed: dict[str, dict[str, Any]] = {}

    def generate_map(self, payload: dict[str, Any]) -> CustomMapSpec:
        if payload.get("map_kind", "custom") != "custom":
            raise ValueError("the local generator currently creates custom maps")
        design_seed = payload.get("design_seed", 42)
        template = payload.get("template", "oval")
        width = payload.get("width", 8.0)
        map_id = payload.get("map_id") or f"custom-track-{int(design_seed):04d}"
        document = generate_custom_map(
            map_id=map_id,
            design_seed=design_seed,
            template=template,
            width=width,
            max_steps=payload.get("max_steps", 2000),
            frame_skip=payload.get("frame_skip", 4),
        )
        map_payload = map_to_dict(document)
        if "obstacles" in payload:
            map_payload["obstacles"] = payload["obstacles"]
            document = map_from_dict(map_payload)
        return document

    def save_map(self, document: MapDocument) -> Path:
        artifact_id = _safe_artifact_id(document.map_id)
        path = self.maps_root / f"{artifact_id}.json"
        payload = map_to_dict(document)
        payload["preview"] = build_map_preview(document)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def create(
        self,
        document: MapDocument,
        policy_kind: str,
        record_frames: bool = False,
    ) -> str:
        policy = self._policy(policy_kind)
        session = SimulationSession.start(document, policy, record_frames=record_frames)
        run_id = f"run-{uuid4().hex}"
        with self._lock:
            self._active[run_id] = ActiveRun(run_id, session)
        return run_id

    def action(self, run_id: str, action: object) -> dict[str, Any]:
        with self._lock:
            active = self._active.get(run_id)
        if active is None:
            raise KeyError(f"unknown run_id: {run_id}")
        step = active.session.step(action)
        return {
            "run_id": run_id,
            "step": asdict(step),
            "done": active.session.done,
        }

    def run_agent(
        self,
        document: MapDocument,
        record_frames: bool = False,
    ) -> tuple[dict[str, Any], Path]:
        policy = self._policy("agent")
        session = SimulationSession.start(document, policy, record_frames=record_frames)
        try:
            for _ in range(document.max_steps):
                if session.done:
                    break
                session.step()
            result = session.finish()
        except Exception:
            session.close()
            raise
        path = self._save_run(result)
        payload = run_log_to_dict(result)
        payload["output"] = str(path)
        return payload, path

    def finish(self, run_id: str, reason: str | None = None) -> tuple[dict[str, Any], Path]:
        with self._lock:
            active = self._active.pop(run_id, None)
        if active is None:
            completed = self._completed.get(run_id)
            if completed is not None:
                return completed["payload"], Path(completed["output"])
            raise KeyError(f"unknown run_id: {run_id}")
        try:
            result = active.session.finish(reason=reason)
        finally:
            active.session.close()
        path = self._save_run(result, run_id=run_id)
        payload = run_log_to_dict(result)
        payload["output"] = str(path)
        with self._lock:
            self._completed[run_id] = {"payload": payload, "output": str(path)}
        return payload, path

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            active = self._active.get(run_id)
            completed = self._completed.get(run_id)
        if active is not None:
            return {
                "run_id": run_id,
                "status": "running",
                "steps": len(active.session.steps),
                "done": active.session.done,
            }
        if completed is not None:
            return completed["payload"]
        raise KeyError(f"unknown run_id: {run_id}")

    def close(self) -> None:
        with self._lock:
            active = list(self._active.values())
            self._active.clear()
        for item in active:
            item.session.close()

    def _policy(self, policy_kind: str) -> Policy:
        if policy_kind == "manual":
            return ManualPolicy()
        if policy_kind == "baseline":
            return BaselinePolicy()
        if policy_kind == "agent":
            return AgentPolicy(self.project_root / "agent.py", self.project_root)
        raise ValueError("policy must be baseline, agent, or manual")

    def _save_run(self, result, run_id: str | None = None) -> Path:
        resolved_id = run_id or f"run-{result.run['run_id']}"
        path = self.runs_root / f"{_safe_artifact_id(resolved_id)}.json"
        save_run_log(result, path)
        return path


def _safe_artifact_id(value: str) -> str:
    if not isinstance(value, str) or not SAFE_ARTIFACT_ID.fullmatch(value):
        raise ValueError("artifact id contains unsupported characters")
    return value


__all__ = [
    "ActiveRun",
    "LocalApiHandler",
    "MAX_CONTROL_BODY_BYTES",
    "SimulationRegistry",
]


class LocalApiHandler:
    registry: SimulationRegistry

    def _json_response(self, payload: object, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, message: str, status: int) -> None:
        self._json_response({"error": message}, status=status)

    def _read_json(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header or "0")
        except ValueError as error:
            raise ValueError("Content-Length must be an integer") from error
        if length < 0 or length > MAX_CONTROL_BODY_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _document(self, payload: dict[str, Any]) -> MapDocument:
        raw_map = payload.get("map", payload)
        return map_from_dict(raw_map)

    def do_GET(self) -> None:
        from urllib.parse import urlsplit

        path = urlsplit(self.path).path
        if path == "/api/health":
            self._json_response({"status": "ok", "service": "local-simulator"})
            return
        if path.startswith("/api/runs/"):
            run_id = path.rsplit("/", 1)[-1]
            try:
                self._json_response(self.registry.get(run_id))
            except KeyError as error:
                self._error(str(error), 404)
            return
        self._error("not found", 404)

    def do_POST(self) -> None:
        from urllib.parse import urlsplit

        path = urlsplit(self.path).path.rstrip("/")
        try:
            payload = self._read_json()
            if path == "/api/maps/generate":
                document = self.registry.generate_map(payload)
                response = map_to_dict(document)
                response["preview"] = build_map_preview(document)
                self._json_response(response)
                return
            if path == "/api/maps/save":
                document = self._document(payload)
                path_value = self.registry.save_map(document)
                response = map_to_dict(document)
                response["preview"] = build_map_preview(document)
                response["output"] = str(path_value)
                self._json_response(response)
                return
            if path == "/api/runs/agent":
                document = self._document(payload)
                response, _path = self.registry.run_agent(
                    document,
                    record_frames=bool(payload.get("record_frames", False)),
                )
                self._json_response(response)
                return
            if path == "/api/runs/start":
                document = self._document(payload)
                run_id = self.registry.create(
                    document,
                    str(payload.get("policy", "manual")),
                    record_frames=bool(payload.get("record_frames", False)),
                )
                self._json_response({"run_id": run_id, "map": map_to_dict(document)})
                return
            if path.startswith("/api/runs/") and path.endswith("/action"):
                run_id = path.split("/")[-2]
                response = self.registry.action(run_id, payload.get("action"))
                self._json_response(response)
                return
            if path.startswith("/api/runs/") and path.endswith("/finish"):
                run_id = path.split("/")[-2]
                response, _path = self.registry.finish(run_id, payload.get("reason"))
                self._json_response(response)
                return
            self._error("not found", 404)
        except KeyError as error:
            self._error(str(error), 404)
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._error(str(error), 400)
        except RuntimeError as error:
            self._error(str(error), 409)
        except Exception as error:
            self._error(str(error), 500)
