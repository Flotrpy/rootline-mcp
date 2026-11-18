"""Run the connector: `rootline-mcp` (HTTP) or `rootline-mcp --stdio` (local clients)."""

from __future__ import annotations

import argparse
import os

from rootline_mcp.config import Settings


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="rootline-mcp")
    ap.add_argument(
        "--stdio", action="store_true", help="talk over stdin/stdout (Claude Code, Desktop)"
    )
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument(
        "--mock", action="store_true", help="use made-up data and a pretend run backend"
    )
    args = ap.parse_args(argv)
    if args.mock:
        os.environ["ROOTLINE_MCP_MOCK"] = "1"
    settings = Settings.from_env()
    if args.stdio:
        from rootline_mcp.server import build_server

        build_server(settings, runs_allowed=True).run("stdio")
        return
    import uvicorn

    from rootline_mcp.server import http_app

    if not settings.mcp_token and settings.backend in ("github", "local"):
        print("warning: no ROOTLINE_MCP_TOKEN set; read tools work but starting runs is off")
    uvicorn.run(http_app(settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
