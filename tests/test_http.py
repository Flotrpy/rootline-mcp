"""End to end over Streamable HTTP, the way Claude talks to the deployed connector."""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.testclient import TestClient

from rootline_mcp.catalog import MODELS
from rootline_mcp.config import Settings
from rootline_mcp.runner import GitHubBackend, MockBackend, RunError, RunRequest

HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _rpc(client: TestClient, path: str, method: str, params: dict | None = None, i: int = 1):
    body = {"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}
    return client.post(path, headers=HEADERS, content=json.dumps(body))


INIT = {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "test", "version": "0"},
}


@pytest.fixture()
def app(monkeypatch, mock_root):
    import rootline_mcp.mock as mock

    monkeypatch.setattr(mock, "mock_dir", lambda: mock_root)
    from rootline_mcp.server import http_app

    return http_app(Settings(mock=True, mcp_token="s3cret", allowed_hosts=("testserver",)))


def test_token_gate(app):
    with TestClient(app) as c:
        assert c.get("/healthz").json()["ok"] is True
        assert _rpc(c, "/mcp", "initialize", INIT).status_code == 404
        assert _rpc(c, "/wrong/mcp", "initialize", INIT).status_code == 404
        assert _rpc(c, "/s3cret/mcp", "initialize", INIT).status_code == 200
        ok = c.post(
            "/mcp",
            headers={**HEADERS, "Authorization": "Bearer s3cret"},
            content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": INIT}),
        )
        assert ok.status_code == 200


def test_list_and_call_tools(app):
    with TestClient(app) as c:
        _rpc(c, "/s3cret/mcp", "initialize", INIT)
        tools = _rpc(c, "/s3cret/mcp", "tools/list", i=2).json()["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {
            "rootline_overview",
            "rootline_list_models",
            "rootline_start_run",
            "rootline_run_status",
            "rootline_list_failures",
            "rootline_merge_results",
            "rootline_score_summary",
            "rootline_compare_models",
            "rootline_condition_effect",
            "rootline_layer_gap",
            "rootline_hardest_snapshots",
            "rootline_show_cases",
        }
        assert all(t.get("description") for t in tools)
        res = _rpc(
            c,
            "/s3cret/mcp",
            "tools/call",
            {"name": "rootline_score_summary", "arguments": {"conditions": ["combined"]}},
            3,
        ).json()
        rows = res["result"]["structuredContent"]["rows"]
        assert len(rows) == 3
        bad = _rpc(
            c,
            "/s3cret/mcp",
            "tools/call",
            {
                "name": "rootline_compare_models",
                "arguments": {"model_a": "nobody@x", "model_b": "gpt-oss-120b@groq"},
            },
            4,
        ).json()
        assert bad["result"]["isError"] is True


def test_github_backend_dispatch_and_status():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        assert request.headers["Authorization"] == "Bearer tok"
        if request.method == "POST":
            return httpx.Response(204)
        return httpx.Response(
            200,
            json={
                "workflow_runs": [
                    {
                        "id": 9,
                        "status": "queued",
                        "conclusion": None,
                        "display_title": "score",
                        "created_at": "2999-01-01T00:00:00Z",
                        "updated_at": None,
                        "html_url": "https://github.com/Flotrpy/RootLine/actions/runs/9",
                    }
                ]
            },
        )

    gh = GitHubBackend(
        "Flotrpy/RootLine",
        "rootline-score.yml",
        "main",
        "tok",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda s: None,
    )
    out = gh.start(RunRequest(MODELS["gpt-oss-120b@groq"], max_wait_s=600))
    assert out["run"]["id"] == 9
    method, path, body = seen[0]
    assert path.endswith("/actions/workflows/rootline-score.yml/dispatches")
    assert json.loads(body)["inputs"] == {"model": "gpt-oss-120b@groq", "max_wait": "600"}
    with pytest.raises(RunError, match="Ollama"):
        gh.start(RunRequest(MODELS["gemma3-4b-q4@ollama"]))
    with pytest.raises(RunError, match="every snapshot"):
        gh.start(RunRequest(MODELS["gpt-oss-120b@groq"], conditions=("logs",)))


def test_mock_backend_validates():
    with pytest.raises(RunError):
        MockBackend().start(RunRequest(MODELS["gpt-oss-120b@groq"], perturbation="p9"))
