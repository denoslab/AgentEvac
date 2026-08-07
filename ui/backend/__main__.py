"""Start the operator console backend.

    python -m ui.backend --port 8000
"""

from __future__ import annotations

import argparse
import sys

from ui.backend.server import serve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ui.backend",
        description="Serve the AgentEvac operator console.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. Use 0.0.0.0 to reach the console from a projector machine.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
