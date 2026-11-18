"""Ways to start a scoring run.

github: dispatches the rootline-score workflow in the Rootline repository.
        GitHub's runners are free for public repositories, the provider key
        and Hugging Face token live in the repository's Actions secrets, and
        the workflow runs scripts/score_daily.py, which resumes the queue from
        the private working dataset. Nothing runs on the connector's host.
local:  runs the harness on this computer (needed for Ollama models).
mock:   pretends, for tests and demos.
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx

from rootline_mcp.catalog import CONDITIONS, Model, valid_incident


class RunError(Exception):
    pass


@dataclass(frozen=True)
class RunRequest:
    model: Model
    perturbation: str = "p0"
    max_wait_s: int = 900
    conditions: tuple[str, ...] = ()
    incidents: tuple[str, ...] = ()

    def validate(self) -> None:
        bad = [c for c in self.conditions if c not in CONDITIONS]
        if bad:
            raise RunError(f"unknown conditions {bad}; use {list(CONDITIONS)}")
        bad = [i for i in self.incidents if not valid_incident(i)]
        if bad:
            raise RunError(f"incident ids look wrong: {bad} (expected like paymentFailure-r1)")
        if self.perturbation not in ("p0", "p1"):
            raise RunError("perturbation must be p0 or p1")
        if not 0 <= self.max_wait_s <= 3600:
            raise RunError("max_wait_s must be between 0 and 3600")


class Backend(Protocol):
    name: str

    def start(self, req: RunRequest) -> dict[str, Any]: ...

    def recent(self, limit: int = 5) -> list[dict[str, Any]]: ...


class NoBackend:
    name = "none"

    def start(self, req: RunRequest) -> dict[str, Any]:
        raise RunError(
            "This connector has no way to start runs yet. Set ROOTLINE_GITHUB_TOKEN"
            " (a fine-grained token with Actions read and write on the Rootline repo)"
            " or ROOTLINE_REPO_DIR (a local checkout) and restart it."
        )

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        return []


@dataclass
class GitHubBackend:
    repo: str
    workflow: str
    ref: str
    token: str
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=20))
    name: str = "github"
    sleep: object = time.sleep

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _url(self, tail: str) -> str:
        return f"https://api.github.com/repos/{self.repo}/actions/workflows/{self.workflow}{tail}"

    def start(self, req: RunRequest) -> dict[str, Any]:
        req.validate()
        if req.model.runs_on == "local":
            raise RunError(
                f"{req.model.id} runs in Ollama on your own machine; GitHub's runners can't serve"
                " it. Start it with the local backend (ROOTLINE_REPO_DIR) instead."
            )
        if req.conditions or req.incidents:
            raise RunError(
                "The GitHub workflow scores every snapshot and condition for a model (that is"
                " what scripts/score_daily.py does). Leave conditions and incidents empty, or"
                " use the local backend for a partial run."
            )
        # GitHub's clock and ours can differ a little; look a bit before the dispatch.
        since = (datetime.now(UTC) - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        r = self.client.post(
            self._url("/dispatches"),
            headers=self._headers(),
            json={
                "ref": self.ref,
                "inputs": {"model": req.model.id, "max_wait": str(req.max_wait_s)},
            },
        )
        if r.status_code not in (200, 204):
            raise RunError(f"GitHub refused the dispatch ({r.status_code}): {r.text[:300]}")
        run = None
        for _ in range(4):  # the run takes a moment to appear
            self.sleep(1.5)  # type: ignore[operator]
            runs = [x for x in self.recent(5) if x.get("created_at", "") >= since]
            if runs:
                run = runs[0]
                break
        return {"started": True, "backend": "github", "model": req.model.id, "run": run}

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        r = self.client.get(
            self._url("/runs"),
            headers=self._headers(),
            params={"per_page": limit, "event": "workflow_dispatch"},
        )
        if r.status_code == 404:
            raise RunError(
                f"No workflow {self.workflow} in {self.repo} (or the token can't see it)."
                " Add deploy/rootline-score.yml to the repo's .github/workflows/ first."
            )
        r.raise_for_status()
        return [
            {
                "id": w["id"],
                "status": w["status"],
                "conclusion": w.get("conclusion"),
                "title": w.get("display_title"),
                "created_at": w["created_at"],
                "updated_at": w.get("updated_at"),
                "url": w["html_url"],
            }
            for w in r.json().get("workflow_runs", [])
        ]


@dataclass
class LocalBackend:
    """Runs the harness CLI as a background job on this machine."""

    repo_dir: Path
    name: str = "local"
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _commands(self, req: RunRequest) -> list[list[str]]:
        prov = req.model.provider
        base = ["uv", "run", "--project", "harness", "python", "-m", "rootline_harness.run"]
        db = ["--db", f"data/queue-{prov}.sqlite"]
        enq = [*base, *db, "enqueue", "--models", req.model.id, "--perturb", req.perturbation]
        if req.conditions:
            enq += ["--conditions", *req.conditions]
        if req.incidents:
            enq += ["--incidents", *req.incidents]
        return [
            enq,
            [*base, *db, "work", "--max-wait", str(req.max_wait_s)],
            [*base, *db, "export", "--out", f"data/runs/runs-{prov}.parquet"],
        ]

    def start(self, req: RunRequest) -> dict[str, Any]:
        req.validate()
        if not (self.repo_dir / "harness" / "rootline_harness").is_dir():
            raise RunError(f"{self.repo_dir} is not a Rootline checkout")
        busy = [j for j in self.jobs.values() if j["status"] == "running"]
        if busy:
            raise RunError(f"a local run is already going ({busy[0]['id']}); wait for it")
        job = {
            "id": uuid.uuid4().hex[:8],
            "model": req.model.id,
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "log_tail": "",
        }
        self.jobs[job["id"]] = job
        threading.Thread(target=self._run, args=(job, self._commands(req)), daemon=True).start()
        return {"started": True, "backend": "local", "model": req.model.id, "run": job}

    def _run(self, job: dict[str, Any], commands: list[list[str]]) -> None:
        log = ""
        for cmd in commands:
            p = subprocess.run(cmd, check=False, cwd=self.repo_dir, capture_output=True, text=True)
            log = (log + p.stdout + p.stderr)[-3000:]
            job["log_tail"] = log
            if p.returncode != 0:
                job["status"] = "failed"
                return
        job["status"] = "completed"

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        return sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True)[:limit]


@dataclass
class MockBackend:
    name: str = "mock"
    runs: list[dict[str, Any]] = field(default_factory=list)

    def start(self, req: RunRequest) -> dict[str, Any]:
        req.validate()
        run = {
            "id": len(self.runs) + 1,
            "status": "queued",
            "conclusion": None,
            "title": f"score {req.model.id}",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "url": f"https://github.com/example/rootline/actions/runs/{len(self.runs) + 1}",
        }
        self.runs.append(run)
        return {"started": True, "backend": "mock", "model": req.model.id, "run": run}

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        return list(reversed(self.runs))[:limit]


def build_backend(settings) -> Backend:
    s = settings
    if s.backend == "mock":
        return MockBackend()
    if s.backend == "local":
        return LocalBackend(s.repo_dir)
    if s.backend == "github":
        return GitHubBackend(s.github_repo, s.github_workflow, s.github_ref, s.github_token)
    return NoBackend()
