"""HTTP client for the bridge running inside a simulation process.

Uses ``urllib`` from the standard library so the console adds no package to the
environment the simulator runs in.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BridgeUnavailable(Exception):
    """The bridge did not answer, which is normal while SUMO is still loading."""


class SimClient:
    """Talks to one simulation process's bridge."""

    def __init__(self, host: str, port: int, timeout_s: float = 5.0):
        self.base = f"http://{host}:{port}"
        self.timeout_s = timeout_s

    def _get(self, path: str, timeout_s: Optional[float] = None) -> Any:
        request = Request(self.base + path, method="GET")
        try:
            with urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 202:  # the bridge is up but the payload is not ready
                return None
            if exc.code == 404:
                return None
            raise BridgeUnavailable(f"GET {path} failed with {exc.code}") from exc
        except (URLError, OSError, ValueError) as exc:
            raise BridgeUnavailable(f"GET {path} failed: {exc}") from exc

    def status(self) -> Dict[str, Any]:
        payload = self._get("/status", timeout_s=3.0)
        if not isinstance(payload, dict):
            raise BridgeUnavailable("status returned no object")
        return payload

    def snapshot(self) -> Optional[Dict[str, Any]]:
        payload = self._get("/snapshot")
        return payload if isinstance(payload, dict) and "sim_t_s" in payload else None

    def preview(self) -> Optional[Dict[str, Any]]:
        payload = self._get("/preview", timeout_s=30.0)
        return payload if isinstance(payload, dict) and "households" in payload else None

    def agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        from urllib.parse import quote

        payload = self._get(f"/agent/{quote(agent_id, safe='')}", timeout_s=8.0)
        return payload if isinstance(payload, dict) else None

    def control(self, action: str, value: Any = None) -> Dict[str, Any]:
        body = json.dumps({"action": action, "value": value}).encode("utf-8")
        request = Request(self.base + "/control", data=body, method="POST",
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, ValueError) as exc:
            raise BridgeUnavailable(f"control {action} failed: {exc}") from exc

    def events(self, backlog: int = 200) -> Iterator[Dict[str, Any]]:
        """Yield events from the bridge stream until the connection ends."""
        request = Request(f"{self.base}/events?backlog={int(backlog)}", method="GET")
        try:
            response = urlopen(request, timeout=None)
        except (HTTPError, URLError, OSError) as exc:
            raise BridgeUnavailable(f"event stream failed: {exc}") from exc
        try:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if not line.startswith("data: "):
                    continue
                try:
                    yield json.loads(line[len("data: "):])
                except ValueError:
                    continue
        finally:
            try:
                response.close()
            except Exception:
                pass
