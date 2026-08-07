"""HTTP surface of the console backend.

Routes divide into three groups. ``/api/*`` is the console's own interface,
``/api/stream`` is the single merged Server-Sent Events channel every browser
listens on, and everything else serves the built frontend and the map assets.

Built on ``ThreadingHTTPServer`` from the standard library, so the console runs
with nothing installed beyond the repository itself.
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import queue
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from ui.backend import authoring, history, packages
from ui.backend.session import (
    NO_INTERPRETER_HINT,
    Broadcaster,
    RunSession,
    find_orphan_run,
    simulator_python,
    sumo_available,
)
from ui.backend.sim_client import BridgeUnavailable, SimClient

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = REPO_ROOT / "ui" / "frontend" / "dist"
ASSETS_DIR = REPO_ROOT / "ui" / "assets"
STATIC_DIR = REPO_ROOT / "static"

VERSION = "1.0.0"


class ConsoleApp:
    """Routing table and shared state for the console backend."""

    def __init__(self):
        self.broadcaster = Broadcaster()
        self.session = RunSession(self.broadcaster)
        self.started_wall = time.time()
        # A backend that restarts while a simulation is still running must not
        # leave it invisible, so the orphan is found at boot and offered.
        self.orphan: Optional[Dict[str, Any]] = find_orphan_run()
        # Set by the quit endpoint and watched by serve(), because a request handler
        # cannot stop the server it is running inside.
        self.shutdown_requested = False
        self.broadcaster.publish(self.session.session_json())

    # ------------------------------------------------------------------
    # API handlers
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        environment = sumo_available()
        return {
            "version": VERSION,
            "uptime_s": round(time.time() - self.started_wall, 1),
            "environment": environment,
            "ready": (
                environment["sumo_home_exists"]
                and bool(environment["sumo_binary"])
                and bool(environment["simulator_python"])
            ),
            "frontend_built": FRONTEND_DIST.is_dir(),
            "preview_assets": sorted(p.stem for p in (ASSETS_DIR / "previews").glob("*.json"))
            if (ASSETS_DIR / "previews").is_dir() else [],
            "orphan_run": self.orphan if self.session.phase == "idle" else None,
        }

    def adopt_orphan(self) -> Tuple[int, Dict[str, Any]]:
        if not self.orphan:
            return 404, {"error": "no_orphan", "detail": "no earlier run is still answering"}
        try:
            session = self.session.adopt(self.orphan)
        except RuntimeError as exc:
            return 409, {"error": "run_in_progress", "detail": str(exc)}
        self.orphan = None
        return 200, {"session": session}

    def discard_orphan(self) -> Tuple[int, Dict[str, Any]]:
        """End an earlier run without adopting it, so it exports and exits."""
        if not self.orphan:
            return 404, {"error": "no_orphan", "detail": "no earlier run is still answering"}
        client = SimClient("127.0.0.1", int(self.orphan["port"]))
        try:
            client.control("end")
        except BridgeUnavailable as exc:
            return 503, {"error": "bridge_unavailable", "detail": str(exc)}
        self.orphan = None
        return 200, {"ok": True}

    def packages_payload(self) -> Dict[str, Any]:
        built = {p.stem for p in (ASSETS_DIR / "previews").glob("*.json")} \
            if (ASSETS_DIR / "previews").is_dir() else set()
        rows = []
        for package in packages.list_packages():
            row = package.to_json()
            row["preview_ready"] = package.id in built
            rows.append(row)
        return {
            "packages": rows,
            "scenarios": [{"id": s, "description": packages.SCENARIO_DESCRIPTIONS[s]}
                          for s in packages.SCENARIO_CHOICES],
            "engines": [{"id": e, "description": packages.ENGINE_DESCRIPTIONS[e]}
                        for e in packages.ENGINE_CHOICES],
            "defaults": packages.DEFAULT_CONFIG,
        }

    def package_schedule(self, package_id: str) -> Optional[Dict[str, Any]]:
        package = packages.load_package(package_id)
        if package.problems and not package.households:
            return None
        return {"package": package.to_json(), "schedule": packages.incident_schedule(package)}

    def validate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return packages.validate_config(body, recordings=history.list_recordings()).to_json()

    def validate_package(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Check an authored package without writing anything."""
        validation, _files = authoring.validate_draft(body)
        return validation.to_json()

    def create_package(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Create a package from a map selection.

        Refuses any name already under ``configs/``, so authoring can add packages and
        can never alter one a campaign has run against.
        """
        validation, target = authoring.write_package(body)
        payload = validation.to_json()
        if target is None:
            taken = any(p.get("field") == "id" and "already exists" in p.get("message", "")
                        for p in validation.problems)
            return (409 if taken else 400), payload
        payload["package"] = target.name
        payload["path"] = str(target.relative_to(authoring.REPO_ROOT))
        payload["preview_ready"] = (ASSETS_DIR / "previews" / f"{target.name}.json").exists()
        return 201, payload

    def quit(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Stop the console and release its port.

        A run still in flight is ended first, which lets the simulator export its metrics
        and timeline through its own finally block, exactly as a natural end would.
        Refused while a run is active unless the caller says to end it, so a shutdown can
        never silently discard a campaign.
        """
        active = bool(self.session.session_json().get("active"))
        if active and not bool((body or {}).get("end_run")):
            return 409, {
                "error": "run_in_progress",
                "detail": "a simulation is still running",
                "session": self.session.session_json(),
            }
        if active:
            try:
                self.session.control("end")
            except Exception:
                pass
        self.shutdown_requested = True
        return 200, {"ok": True, "ended_run": active}

    def package_buildings(self, package_id: str) -> Optional[Dict[str, Any]]:
        """The building layer a selection is drawn against, without the road geometry."""
        index = authoring.load_buildings_index(package_id)
        if not index:
            return None
        return {
            "package": package_id,
            "count": len(index),
            "buildings": list(index.values()),
        }

    def launch(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        validation = packages.validate_config(body, recordings=history.list_recordings())
        if simulator_python() is None:
            # Caught here rather than inside the spawned process, so the operator
            # reads a named problem in the Setup form.
            validation.ok = False
            validation.problems.append({
                "field": "engine",
                "message": "no interpreter on this machine can import traci and sumolib",
                "hint": NO_INTERPRETER_HINT,
            })
        if not validation.ok:
            return 400, validation.to_json()
        try:
            session = self.session.launch(validation.normalized)
        except RuntimeError as exc:
            return 409, {"error": "run_in_progress", "detail": str(exc),
                         "session": self.session.session_json()}
        except Exception as exc:
            return 500, {"error": "launch_failed", "detail": str(exc)}
        return 200, {"session": session, "warnings": validation.warnings}

    def control(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        action = str(body.get("action", ""))
        if action not in ("pause", "resume", "toggle_pause", "set_speed", "end"):
            return 400, {"error": "unknown_action", "detail": action}
        try:
            result = self.session.control(action, body.get("value"))
        except RuntimeError as exc:
            return 409, {"error": "not_controllable", "detail": str(exc)}
        except BridgeUnavailable as exc:
            return 503, {"error": "bridge_unavailable", "detail": str(exc)}
        return 200, {"ok": True, "result": result, "session": self.session.session_json()}

    def history_payload(self, limit: int) -> Dict[str, Any]:
        return {"runs": history.list_runs(limit=limit)}

    def run_metrics(self, run_id: str) -> Optional[Dict[str, Any]]:
        metrics = history.load_metrics(run_id)
        if metrics is None:
            return None
        record = history.find_run(run_id) or {}
        return {
            "run_id": run_id,
            "record": record,
            "metrics": metrics,
            "curve": history.evacuation_curve(run_id),
            "artifacts": history.run_artifacts(run_id),
        }

    def export_zip(self, run_id: str) -> Optional[bytes]:
        artifacts = history.run_artifacts(run_id)
        if not artifacts:
            return None
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, relative in artifacts.items():
                if not relative:
                    continue
                path = REPO_ROOT / relative
                if path.is_file():
                    archive.write(path, arcname=f"{run_id}/{name}_{path.name}")
        return buffer.getvalue()


def _serve_file(handler: BaseHTTPRequestHandler, path: Path, *, download_name: Optional[str] = None) -> bool:
    if not path.is_file():
        return False
    content_type, _ = mimetypes.guess_type(str(path))
    try:
        data = path.read_bytes()
    except OSError:
        return False
    handler.send_response(200)
    handler.send_header("Content-Type", content_type or "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    if download_name:
        handler.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
    else:
        handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(data)
    return True


def make_handler(app: ConsoleApp):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"AgentEvacConsole/{VERSION}"

        def log_message(self, fmt, *args):
            return

        # -- helpers --------------------------------------------------------

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return {}
            if length <= 0:
                return {}
            try:
                parsed = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}
            return parsed if isinstance(parsed, dict) else {}

        # -- routes ---------------------------------------------------------

        def do_GET(self):  # noqa: N802 — stdlib naming
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path == "/api/health":
                self._json(app.health())
                return
            if path == "/api/packages":
                self._json(app.packages_payload())
                return
            if path.startswith("/api/packages/") and path.endswith("/schedule"):
                package_id = unquote(path[len("/api/packages/"):-len("/schedule")])
                payload = app.package_schedule(package_id)
                self._json(payload or {"error": "unknown_package"}, 200 if payload else 404)
                return
            if path.startswith("/api/packages/") and path.endswith("/buildings"):
                package_id = unquote(path[len("/api/packages/"):-len("/buildings")])
                payload = app.package_buildings(package_id)
                if payload is None:
                    self._json({"error": "no_building_layer", "package": package_id}, 404)
                else:
                    self._json(payload)
                return
            if path.startswith("/api/packages/") and path.endswith("/preview"):
                package_id = unquote(path[len("/api/packages/"):-len("/preview")])
                asset = ASSETS_DIR / "previews" / f"{package_id}.json"
                if _serve_file(self, asset):
                    return
                self._json({
                    "error": "preview_not_built",
                    "detail": f"no map bundle for {package_id}",
                    "hint": "Run python -m ui.tools.build_map_assets to generate it.",
                }, 404)
                return
            if path == "/api/recordings":
                self._json({"recordings": history.list_recordings()})
                return
            if path == "/api/runs/current":
                self._json(app.session.session_json())
                return
            if path.startswith("/api/runs/current/agents/"):
                agent_id = unquote(path[len("/api/runs/current/agents/"):])
                detail = app.session.agent_detail(agent_id)
                self._json(detail or {"error": "agent_unavailable", "agent_id": agent_id},
                           200 if detail else 404)
                return
            if path == "/api/history":
                try:
                    limit = int(params.get("limit", ["200"])[0])
                except (TypeError, ValueError):
                    limit = 200
                self._json(app.history_payload(limit))
                return
            if path.startswith("/api/history/") and path.endswith("/metrics"):
                run_id = unquote(path[len("/api/history/"):-len("/metrics")])
                payload = app.run_metrics(run_id)
                self._json(payload or {"error": "unknown_run", "run_id": run_id},
                           200 if payload else 404)
                return
            if path.startswith("/api/history/") and path.endswith("/export"):
                run_id = unquote(path[len("/api/history/"):-len("/export")])
                blob = app.export_zip(run_id)
                if blob is None:
                    self._json({"error": "unknown_run", "run_id": run_id}, 404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(blob)))
                self.send_header("Content-Disposition", f'attachment; filename="{run_id}.zip"')
                self.end_headers()
                self.wfile.write(blob)
                return
            if path.startswith("/api/files/"):
                self._serve_run_artifact(unquote(path[len("/api/files/"):]))
                return
            if path == "/api/stream":
                self._stream()
                return

            if self._serve_static(path):
                return
            self._json({"error": "not_found", "path": path}, 404)

        def do_POST(self):  # noqa: N802 — stdlib naming
            path = urlparse(self.path).path
            body = self._body()
            if path == "/api/packages/validate":
                self._json(app.validate_package(body))
                return
            if path == "/api/packages":
                status, payload = app.create_package(body)
                self._json(payload, status)
                return
            if path == "/api/runs/validate":
                self._json(app.validate(body))
                return
            if path == "/api/runs":
                status, payload = app.launch(body)
                self._json(payload, status)
                return
            if path == "/api/runs/current/control":
                status, payload = app.control(body)
                self._json(payload, status)
                return
            if path == "/api/runs/adopt":
                status, payload = app.adopt_orphan()
                self._json(payload, status)
                return
            if path == "/api/quit":
                status, payload = app.quit(body)
                self._json(payload, status)
                return
            if path == "/api/runs/discard-orphan":
                status, payload = app.discard_orphan()
                self._json(payload, status)
                return
            self._json({"error": "not_found", "path": path}, 404)

        def _serve_run_artifact(self, relative: str) -> None:
            """Hand back one file a run produced.

            Only paths inside the outputs directory resolve, so a download link
            cannot be turned into a reader for the rest of the repository.
            """
            outputs = (REPO_ROOT / "outputs").resolve()
            candidate = _safe_join(outputs, relative[len("outputs/"):] if relative.startswith("outputs/") else relative)
            if candidate is None or not candidate.is_file():
                self._json({"error": "file_not_found", "path": relative}, 404)
                return
            _serve_file(self, candidate, download_name=candidate.name)

        # -- merged event stream --------------------------------------------

        def _stream(self) -> None:
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            sub = app.broadcaster.subscribe()
            try:
                while True:
                    try:
                        message = sub.get(timeout=10.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    payload = json.dumps(message, ensure_ascii=False, default=str)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ValueError, OSError):
                pass
            finally:
                app.broadcaster.unsubscribe(sub)

        # -- static files ----------------------------------------------------

        def _serve_static(self, path: str) -> bool:
            if path.startswith("/assets/"):
                candidate = _safe_join(ASSETS_DIR, path[len("/assets/"):])
                if candidate is not None and _serve_file(self, candidate):
                    return True
            if path.startswith("/brand/"):
                candidate = _safe_join(STATIC_DIR, path[len("/brand/"):])
                return candidate is not None and _serve_file(self, candidate)

            if not FRONTEND_DIST.is_dir():
                if path in ("/", "/index.html"):
                    self._send_build_notice()
                    return True
                return False

            relative = path.lstrip("/") or "index.html"
            candidate = _safe_join(FRONTEND_DIST, relative)
            if candidate is not None and candidate.is_file():
                return _serve_file(self, candidate)
            # The console is a single page, so unknown paths fall back to it.
            return _serve_file(self, FRONTEND_DIST / "index.html")

        def _send_build_notice(self) -> None:
            body = (
                "<!doctype html><meta charset='utf-8'>"
                "<title>AgentEvac operator console</title>"
                "<style>body{font:16px/1.6 system-ui;margin:60px auto;max-width:44rem;"
                "background:#12161C;color:#E6EAF0}code{background:#1A2028;padding:2px 6px;"
                "border-radius:4px;color:#E69F00}</style>"
                "<h1>The console has not been built</h1>"
                "<p>The backend is running, so the API is available. Build the interface once with:</p>"
                "<pre><code>cd ui/frontend &amp;&amp; npm install &amp;&amp; npm run build</code></pre>"
                "<p>Then reload this page.</p>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _safe_join(root: Path, relative: str) -> Optional[Path]:
    """Resolve ``relative`` under ``root``, refusing anything that escapes it."""
    candidate = (root / unquote(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    app = ConsoleApp()
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    url = f"http://{host}:{port}"
    print(f"AgentEvac operator console on {url}")
    # Resolved at boot rather than on the first launch, so a machine that cannot
    # run a simulation says so before anyone configures one.
    runner = simulator_python()
    if runner:
        print(f"  runs simulations with {runner}")
    else:
        print(f"  no interpreter here can start a simulation. {NO_INTERPRETER_HINT}")
    if not FRONTEND_DIST.is_dir():
        print("  the interface is not built yet: cd ui/frontend && npm install && npm run build")
    previews = sorted(p.stem for p in (ASSETS_DIR / "previews").glob("*.json")) \
        if (ASSETS_DIR / "previews").is_dir() else []
    print(f"  map bundles ready: {', '.join(previews) if previews else 'none, run python -m ui.tools.build_map_assets'}")
    # The quit endpoint cannot stop the server from inside a handler thread, so it
    # raises a flag and this loop acts on it.
    def watch_for_quit() -> None:
        while not app.shutdown_requested:
            time.sleep(0.25)
        print("\nstopping, asked to quit from the console")
        server.shutdown()

    threading.Thread(target=watch_for_quit, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        app.session.shutdown()
        server.server_close()
