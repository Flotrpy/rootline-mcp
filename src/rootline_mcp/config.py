"""Settings, all read from environment variables. No secret is ever logged or returned."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _path(env: Mapping[str, str], name: str) -> Path | None:
    v = env.get(name, "").strip()
    return Path(v).expanduser() if v else None


@dataclass(frozen=True)
class Settings:
    mock: bool = False
    data_dir: Path | None = None  # local folder with runs/*.parquet and queue-*.sqlite
    repo_dir: Path | None = None  # local Rootline checkout, for the local run backend
    hf_dataset: str = "Flotrpy/rootline"  # public release dataset
    hf_work: str = "Flotrpy/rootline-work"  # private working dataset (queue + exports)
    hf_token: str | None = None
    github_repo: str = "Flotrpy/RootLine"
    github_workflow: str = "rootline-score.yml"
    github_ref: str = "main"
    github_token: str | None = None
    mcp_token: str | None = None  # shared secret guarding the HTTP endpoint
    allow_unauthenticated_runs: bool = False
    cache_ttl: float = 120.0
    allowed_hosts: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        hosts = [h.strip() for h in env.get("ROOTLINE_ALLOWED_HOSTS", "").split(",") if h.strip()]
        if env.get("RENDER_EXTERNAL_HOSTNAME"):
            hosts.append(env["RENDER_EXTERNAL_HOSTNAME"])
        return cls(
            mock=_flag(env, "ROOTLINE_MCP_MOCK"),
            data_dir=_path(env, "ROOTLINE_DATA_DIR"),
            repo_dir=_path(env, "ROOTLINE_REPO_DIR"),
            hf_dataset=env.get("ROOTLINE_HF_DATASET", cls.hf_dataset),
            hf_work=env.get("ROOTLINE_HF_WORK", cls.hf_work),
            hf_token=env.get("HF_TOKEN") or None,
            github_repo=env.get("ROOTLINE_GITHUB_REPO", cls.github_repo),
            github_workflow=env.get("ROOTLINE_GITHUB_WORKFLOW", cls.github_workflow),
            github_ref=env.get("ROOTLINE_GITHUB_REF", cls.github_ref),
            github_token=env.get("ROOTLINE_GITHUB_TOKEN") or env.get("GITHUB_TOKEN") or None,
            mcp_token=env.get("ROOTLINE_MCP_TOKEN") or None,
            allow_unauthenticated_runs=_flag(env, "ROOTLINE_ALLOW_UNAUTHENTICATED_RUNS"),
            cache_ttl=float(env.get("ROOTLINE_CACHE_TTL", cls.cache_ttl)),
            allowed_hosts=tuple(hosts),
        )

    @property
    def backend(self) -> str:
        """Which run backend this deployment uses: mock, local, github or none."""
        if self.mock:
            return "mock"
        if self.repo_dir is not None:
            return "local"
        if self.github_token:
            return "github"
        return "none"

    def describe(self) -> dict[str, object]:
        """A safe summary: which pieces are configured, never their values."""
        return {
            "mode": "mock" if self.mock else ("local" if self.data_dir else "huggingface"),
            "run_backend": self.backend,
            "published_dataset": self.hf_dataset,
            "work_dataset": self.hf_work,
            "work_dataset_readable": bool(self.hf_token) or self.mock or self.data_dir is not None,
            "github_repo": self.github_repo,
            "github_workflow": self.github_workflow,
            "endpoint_protected": bool(self.mcp_token),
        }
