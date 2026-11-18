"""Where runs and queues live, and how to read them.

Rootline keeps its state in files: per-provider exports (runs-<provider>.parquet),
the merged release file (runs.parquet) and per-provider SQLite job queues
(queue-<provider>.sqlite). A deployed connector reads them from Hugging Face
(the public release dataset and the private working dataset that
scripts/score_daily.py writes); on your own computer it reads a local folder.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import polars as pl

RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "incident_id",
    "condition",
    "perturbation",
    "model_id",
    "tier",
    "provider",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "raw_output",
    "parsed_ok",
    "top1",
    "top3",
    "layer_ok",
    "evidence_validity",
    "confidence",
    "action_type",
    "action_target",
    "started_at",
)


class FileSource(Protocol):
    name: str

    def list(self) -> list[str]: ...

    def fetch(self, name: str) -> Path: ...

    def upload(self, local: Path, name: str, message: str) -> str: ...


@dataclass
class LocalSource:
    """A folder on disk: files at its top level or in its runs/ subfolder."""

    name: str
    root: Path

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        out = []
        for p in sorted(self.root.rglob("*")):
            rel = p.relative_to(self.root)
            top = len(rel.parts) == 1 or (len(rel.parts) == 2 and rel.parts[0] == "runs")
            if p.is_file() and top and p.suffix in {".parquet", ".sqlite"}:
                out.append(rel.as_posix())
        return out

    def fetch(self, name: str) -> Path:
        p = (self.root / name).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"{name!r} is outside {self.root}")
        return p

    def upload(self, local: Path, name: str, message: str) -> str:
        dest = self.fetch(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local.resolve() != dest:
            dest.write_bytes(local.read_bytes())
        return str(dest)


@dataclass
class HFSource:
    """A Hugging Face dataset repository, read with a short cache."""

    name: str
    repo: str
    token: str | None
    ttl: float = 120.0
    clock: Callable[[], float] = time.monotonic
    _listing: tuple[float, list[str]] | None = field(default=None, repr=False)
    _files: dict[str, tuple[float, Path]] = field(default_factory=dict, repr=False)

    def _api(self):  # imported lazily: tests and mock mode never touch the network
        from huggingface_hub import HfApi

        return HfApi(token=self.token)

    def list(self) -> list[str]:
        now = self.clock()
        if self._listing and now - self._listing[0] < self.ttl:
            return self._listing[1]
        files = [
            f
            for f in self._api().list_repo_files(self.repo, repo_type="dataset")
            if f.endswith((".parquet", ".sqlite")) and (f.count("/") == 0 or f.startswith("runs/"))
        ]
        self._listing = (now, sorted(files))
        return self._listing[1]

    def fetch(self, name: str) -> Path:
        now = self.clock()
        hit = self._files.get(name)
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(self.repo, name, repo_type="dataset", token=self.token))
        self._files[name] = (now, path)
        return path

    def upload(self, local: Path, name: str, message: str) -> str:
        api = self._api()
        if not api.repo_info(self.repo, repo_type="dataset").private:
            raise PermissionError(f"{self.repo} is public; refusing to upload working files there")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=name,
            repo_id=self.repo,
            repo_type="dataset",
            commit_message=message,
        )
        self._listing = None
        self._files.pop(name, None)
        return f"{self.repo}/{name}"


@dataclass(frozen=True)
class RunFile:
    source: str
    name: str

    @property
    def merged(self) -> bool:
        return Path(self.name).name == "runs.parquet"


@dataclass(frozen=True)
class QueueSnapshot:
    source: str
    name: str
    counts: dict[str, dict[str, int]]  # model_id -> status -> n
    failures: list[dict[str, object]]
    last_update: float | None
    done_records: int


def _empty_runs() -> pl.DataFrame:
    return pl.DataFrame(schema={c: pl.String for c in RUN_COLUMNS})


def read_queue(path: Path, source: str, name: str, failure_limit: int = 500) -> QueueSnapshot:
    # immutable=1: a downloaded copy is never written to, so no WAL files appear.
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        counts: dict[str, dict[str, int]] = {}
        for model, status, n in con.execute(
            "SELECT model_id, status, COUNT(*) FROM jobs GROUP BY model_id, status"
        ):
            counts.setdefault(model, {})[status] = n
        failures = [
            {"key": k, "model_id": m, "attempts": a, "error": e or ""}
            for k, m, a, e in con.execute(
                "SELECT key, model_id, attempts, last_error FROM jobs WHERE status='failed'"
                " ORDER BY key LIMIT ?",
                (failure_limit,),
            )
        ]
        last = con.execute("SELECT MAX(updated_at) FROM jobs").fetchone()[0]
        done = con.execute(
            "SELECT COUNT(*) FROM jobs WHERE status='done' AND record IS NOT NULL"
        ).fetchone()[0]
    finally:
        con.close()
    return QueueSnapshot(source, name, counts, failures, last, done)


class Store:
    """All sources together, with the results as one table."""

    def __init__(self, sources: list[FileSource]) -> None:
        self.sources = {s.name: s for s in sources}

    def source(self, name: str) -> FileSource:
        try:
            return self.sources[name]
        except KeyError:
            raise ValueError(f"unknown source {name!r}; have {sorted(self.sources)}") from None

    def run_files(self, source: str | None = None) -> list[RunFile]:
        names = [source] if source else list(self.sources)
        out: list[RunFile] = []
        for n in names:
            src = self.source(n)
            try:
                files = src.list()
            except Exception as e:  # one unreadable source must not hide the others
                raise RuntimeError(f"could not list {n}: {type(e).__name__}: {e}") from e
            out += [RunFile(n, f) for f in files if f.endswith(".parquet")]
        return out

    def read_runs(self, f: RunFile) -> pl.DataFrame:
        df = pl.read_parquet(self.source(f.source).fetch(f.name))
        missing = [c for c in RUN_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{f.source}:{f.name} lacks columns {missing}")
        return df.select(RUN_COLUMNS).with_columns(
            pl.lit(f.source).alias("_source"), pl.lit(f.name).alias("_file")
        )

    def runs(self, source: str = "all") -> pl.DataFrame:
        """Every run, one row per run_id. Per-provider exports win over the merged
        release file, because they are newer."""
        files = self.run_files(None if source == "all" else source)
        frames = []
        for f in files:
            frames.append(
                self.read_runs(f).with_columns(pl.lit(1 if f.merged else 0).alias("_rank"))
            )
        if not frames:
            return _empty_runs()
        df = pl.concat(frames, how="vertical_relaxed")
        return (
            df.sort(["run_id", "_rank"])
            .unique(subset="run_id", keep="first", maintain_order=True)
            .drop("_rank")
        )

    def queues(self) -> list[QueueSnapshot]:
        out = []
        for name, src in self.sources.items():
            try:
                files = src.list()
            except Exception:
                continue
            for f in files:
                if f.endswith(".sqlite"):
                    out.append(read_queue(src.fetch(f), name, f))
        return out


def build_store(settings) -> Store:
    from rootline_mcp.config import Settings

    s: Settings = settings
    if s.mock:
        from rootline_mcp.mock import mock_dir

        root = mock_dir()
        return Store(
            [LocalSource("published", root / "published"), LocalSource("work", root / "work")]
        )
    if s.data_dir is not None:
        return Store([LocalSource("local", s.data_dir)])
    sources: list[FileSource] = [HFSource("published", s.hf_dataset, s.hf_token, s.cache_ttl)]
    if s.hf_token:
        sources.append(HFSource("work", s.hf_work, s.hf_token, s.cache_ttl))
    return Store(sources)
