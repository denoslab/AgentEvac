"""HTTP surface of the in-process bridge.

Built on the standard library's ``ThreadingHTTPServer``, the same mechanism the
simulator already uses for its own dashboard, so running the console adds no
dependency to the simulation process.

Handler threads only read from :class:`~ui.bridge.control.BridgeState`. They
never call TraCI and never touch simulator globals, which is what keeps the
single-threaded TraCI contract intact.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from ui.bridge.control import BridgeState


class BridgeServer:
    """Serves bridge state to the console backend on a loopback port."""

    def __init__(self, state: BridgeState, host: str = "127.0.0.1", port: int = 0):
        self.state = state
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> int:
        """Bind and serve. Returns the port actually bound."""
        self._server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name="agentevac-ui-bridge")
        self._thread.start()
        return self.port

    def close(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None

    def _make_handler(self):
        state = self.state
        stop = self._stop

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # keep the simulation stdout readable
                return

            # -- response helpers -------------------------------------------------

            def _json(self, payload: Any, status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _read_json_body(self) -> Dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    return {}
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return {}
                return parsed if isinstance(parsed, dict) else {}

            # -- routes -----------------------------------------------------------

            def do_GET(self):  # noqa: N802 — stdlib naming
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"

                if path in ("/", "/status"):
                    payload = state.status()
                    payload["round"] = state.round_progress()
                    self._json(payload)
                    return

                if path == "/snapshot":
                    snapshot = state.snapshot()
                    if snapshot is None:
                        self._json({"pending": True, "status": state.status()}, status=202)
                    else:
                        self._json(snapshot)
                    return

                if path == "/preview":
                    preview = state.preview()
                    if preview is None:
                        self._json({"pending": True}, status=202)
                    else:
                        self._json(preview)
                    return

                if path == "/events":
                    self._stream_events(parse_qs(parsed.query))
                    return

                if path.startswith("/agent/"):
                    agent_id = unquote(path[len("/agent/"):]).strip()
                    detail = state.request_agent(agent_id)
                    if detail is None:
                        self._json({"error": "agent_unavailable", "agent_id": agent_id}, status=404)
                    else:
                        self._json(detail)
                    return

                self._json({"error": "not_found", "path": path}, status=404)

            def do_POST(self):  # noqa: N802 — stdlib naming
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path != "/control":
                    self._json({"error": "not_found", "path": path}, status=404)
                    return
                body = self._read_json_body()
                action = str(body.get("action", ""))
                try:
                    intent = state.apply_control(action, body.get("value"))
                except ValueError as exc:
                    self._json({"error": "bad_control", "detail": str(exc)}, status=400)
                    return
                self._json({"ok": True, "action": action, "intent": intent})

            # -- server-sent events ------------------------------------------------

            def _stream_events(self, params: Dict[str, Any]) -> None:
                try:
                    backlog = int(params.get("backlog", ["200"])[0])
                except (TypeError, ValueError):
                    backlog = 200
                # An event stream has no length to declare, so the connection is
                # closed at the end rather than reused. Without this a keep-alive
                # client waits for a body it will never be told the size of.
                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                sub = state.subscribe(backlog=backlog)
                try:
                    while not stop.is_set():
                        try:
                            record = sub.get(timeout=5.0)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        payload = json.dumps(record, ensure_ascii=False, default=str)
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ValueError, OSError):
                    pass
                finally:
                    state.unsubscribe(sub)

        return Handler
