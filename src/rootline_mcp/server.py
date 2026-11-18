"""The MCP server: tool definitions, endpoint protection and the HTTP app."""

from __future__ import annotations

import hmac
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from rootline_mcp import __version__
from rootline_mcp.config import Settings
from rootline_mcp.runner import build_backend
from rootline_mcp.store import build_store
from rootline_mcp.tools import ToolError, Tools

INSTRUCTIONS = """\
Rootline is a benchmark of free AI models diagnosing microservice failures: 42 frozen
incident snapshots x 4 evidence conditions (metrics, traces, logs, combined), each answer
scored top-1/top-3 (right service and category), layer, evidence validity and calibration.
Start with rootline_overview. Before start_run with confirm=true, show the person the plan
from a confirm=false call. Runs use free-tier API quota only. Report numbers with their
sample size; 42 snapshots is small, so prefer compare_models' paired test to eyeballing.
"""

READ = ToolAnnotations(read_only_hint=True, open_world_hint=True)
Model = Annotated[
    str, Field(description="Model id, e.g. gpt-oss-120b@groq (see rootline_list_models)")
]
Condition = Annotated[
    str, Field(description="Evidence condition: metrics, traces, logs or combined")
]


def _call(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except ToolError as e:
        raise ValueError(str(e)) from None  # surfaces as a clean tool error


def build_server(settings: Settings, runs_allowed: bool, tools: Tools | None = None) -> MCPServer:
    t = tools or Tools(settings, build_store(settings), build_backend(settings), runs_allowed)
    mcp = MCPServer(
        name="rootline", title="Rootline benchmark", version=__version__, instructions=INSTRUCTIONS
    )

    @mcp.tool(name="rootline_overview", annotations=READ)
    def overview() -> dict[str, Any]:
        """What this connector can see: result files, runs per model, queue progress and whether
        it can start runs. Call this first."""
        return _call(t.overview)

    @mcp.tool(name="rootline_list_models", annotations=READ)
    def list_models() -> dict[str, Any]:
        """Every model the harness knows, where it runs (free cloud API or Ollama on your
        computer), its free-tier jobs per day, runs already scored and days needed to finish."""
        return _call(t.list_models)

    @mcp.tool(
        name="rootline_start_run",
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
        ),
    )
    def start_run(
        model: Model,
        confirm: Annotated[
            bool, Field(description="false = show the plan only; true = start it")
        ] = False,
        perturbation: Annotated[
            str, Field(description="p0 (original names) or p1 (anonymised)")
        ] = "p0",
        max_wait_s: Annotated[
            int,
            Field(
                description="Stop for the day rather than wait longer than this for a rate limit",
                ge=0,
                le=3600,
            ),
        ] = 900,
        conditions: Annotated[
            list[str] | None, Field(description="Local runs only: subset of conditions")
        ] = None,
        incidents: Annotated[
            list[str] | None, Field(description="Local runs only: subset of snapshot ids")
        ] = None,
    ) -> dict[str, Any]:
        """Plan or start a scoring run for one model. Without confirm it only returns the plan:
        jobs left, free jobs per day and estimated days. With confirm=true it starts the run
        (GitHub Actions or this computer). The queue resumes, so starting twice never
        double-scores a snapshot."""
        return _call(t.start_run, model, confirm, perturbation, max_wait_s, conditions, incidents)

    @mcp.tool(name="rootline_run_status", annotations=READ)
    def run_status(
        model: Annotated[str | None, Field(description="Only this model")] = None,
        limit: Annotated[int, Field(ge=1, le=20)] = 5,
    ) -> dict[str, Any]:
        """Progress of each job queue (done, pending, failed, percent, days left at the free
        cap, last activity in Eastern time) and the latest runs with their links."""
        return _call(t.run_status, model, limit)

    @mcp.tool(name="rootline_list_failures", annotations=READ)
    def list_failures(
        model: Annotated[str | None, Field(description="Only this model")] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 20,
    ) -> dict[str, Any]:
        """Failed jobs grouped by error (rate limits, timeouts, provider errors) with examples,
        plus how many answers failed to parse."""
        return _call(t.list_failures, model, limit)

    @mcp.tool(
        name="rootline_merge_results",
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
        ),
    )
    def merge_results(
        write: Annotated[
            bool, Field(description="Also write the merged runs.parquet to the working dataset")
        ] = False,
    ) -> dict[str, Any]:
        """Check and merge per-provider exports: duplicate or conflicting run ids, queue done
        count vs export rows, missing snapshot x condition cells per model and unparsed
        answers. With write=true it saves runs.parquet to the private working dataset (never
        the public release); it refuses if any run id conflicts."""
        return _call(t.merge_results, write)

    @mcp.tool(name="rootline_score_summary", annotations=READ)
    def score_summary(
        models: Annotated[list[str] | None, Field(description="Model ids; default all")] = None,
        conditions: Annotated[
            list[str] | None, Field(description="Conditions; default all")
        ] = None,
        group_by: Annotated[
            list[str] | None,
            Field(
                description="Any of model, condition, fault, group, perturbation, incident; default model + condition"
            ),
        ] = None,
        perturbation: Annotated[str, Field(description="p0 or p1")] = "p0",
        source: Annotated[str, Field(description="all, published, work or local")] = "all",
    ) -> dict[str, Any]:
        """Scores table: top-1 with a 95% bootstrap interval, top-3, layer accuracy, evidence
        validity, parse rate, mean confidence and healthy false alarms, grouped as asked.
        group_by=["model","group"] splits application vs network vs hardware faults."""
        return _call(t.score_summary, models, conditions, group_by, perturbation, source)

    @mcp.tool(name="rootline_compare_models", annotations=READ)
    def compare_models(
        model_a: Model,
        model_b: Model,
        condition: Condition = "combined",
        metric: Annotated[str, Field(description="top1, top3, layer_ok or parsed_ok")] = "top1",
        perturbation: Annotated[str, Field(description="p0 or p1")] = "p0",
    ) -> dict[str, Any]:
        """Paired comparison of two models on the same snapshots: both right, only one right,
        both wrong, and McNemar's exact test, with example snapshots where they differ."""
        return _call(t.compare_models, model_a, model_b, condition, metric, perturbation)

    @mcp.tool(name="rootline_condition_effect", annotations=READ)
    def condition_effect(
        model: Model,
        metric: Annotated[str, Field(description="top1, top3, layer_ok or parsed_ok")] = "top1",
        perturbation: Annotated[str, Field(description="p0 or p1")] = "p0",
        source: Annotated[
            str, Field(description="all, published or work (private working results)")
        ] = "all",
    ) -> dict[str, Any]:
        """Does giving the model all evidence help? Compares the combined condition with each
        single signal (metrics, traces, logs) on the same snapshots, names the best single
        signal and tests combined vs it with McNemar's exact test (the paper's RQ1)."""
        return _call(t.condition_effect, model, metric, perturbation, source)

    @mcp.tool(name="rootline_show_cases", annotations=READ)
    def show_cases(
        model: Model,
        condition: Condition = "combined",
        kind: Annotated[
            str,
            Field(
                description="wrong, right, unparsed, overconfident (wrong with confidence >= 0.8) or false_alarm (fault claimed on a healthy snapshot)"
            ),
        ] = "wrong",
        fault: Annotated[
            str | None, Field(description="Only this fault type, e.g. toxiLatency")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=20)] = 5,
        include_raw: Annotated[
            bool, Field(description="Include the model's raw answer (long)")
        ] = False,
        perturbation: Annotated[str, Field(description="p0 or p1")] = "p0",
    ) -> dict[str, Any]:
        """Concrete answers to look at: what the model blamed (service, category, confidence,
        evidence cited, action) next to the true fault, filtered by kind and fault."""
        return _call(t.show_cases, model, condition, kind, fault, limit, include_raw, perturbation)

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "version": __version__})

    @mcp.custom_route("/", methods=["GET"])
    async def root(_: Request) -> PlainTextResponse:
        return PlainTextResponse(
            "Rootline MCP connector. The MCP endpoint is /mcp (or /<token>/mcp)."
        )

    return mcp


class TokenGate:
    """Protects the MCP endpoint with a shared secret.

    Accepted: /<token>/mcp (for clients that can only take a URL, like Claude's custom
    connectors) or an Authorization: Bearer <token> header. /healthz and / stay open.
    """

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.token:
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        if path in ("/", "/healthz"):
            await self.app(scope, receive, send)
            return
        prefix = f"/{self.token}"
        if path == f"{prefix}/mcp" or path.startswith(f"{prefix}/mcp/"):
            scope = dict(scope, path=path[len(prefix) :], raw_path=path[len(prefix) :].encode())
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode()
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], self.token):
            await self.app(scope, receive, send)
            return
        resp = JSONResponse({"error": "not found"}, status_code=404)
        await resp(scope, receive, send)


def http_app(settings: Settings) -> ASGIApp:
    runs_allowed = settings.mock or bool(settings.mcp_token) or settings.allow_unauthenticated_runs
    server = build_server(settings, runs_allowed)
    hosts = ["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*", *settings.allowed_hosts]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=["https://claude.ai", "http://localhost:*", "http://127.0.0.1:*"],
    )
    app = server.streamable_http_app(
        stateless_http=True, json_response=True, transport_security=security, host="0.0.0.0"
    )
    return TokenGate(app, settings.mcp_token)
