"""Synthetic Rootline data for mock mode and tests.

Same file names and columns as the real thing, made up numbers. Nothing here
is a real result.
"""

from __future__ import annotations

import json
import random
import sqlite3
import tempfile
import time
from functools import cache
from pathlib import Path

import polars as pl

from rootline_mcp.catalog import APP_FAULTS, CONDITIONS, FAULTS, HW_FAULTS, MODELS, NET_FAULTS
from rootline_mcp.store import RUN_COLUMNS

SERVICE = {
    "paymentFailure": ("payment", "error"),
    "paymentUnreachable": ("payment", "dependency_unavailable"),
    "productCatalogFailure": ("product-catalog", "error"),
    "productCatalogLockContention": ("product-catalog", "lock_contention"),
    "emailMemoryLeak": ("email", "memory_leak"),
    "recommendationCacheFailure": ("recommendation", "error"),
    "kafkaQueueProblems": ("kafka", "queue_lag"),
    "intlShippingSlowdown": ("shipping", "latency"),
    "imageSlowLoad": ("image-provider", "latency"),
    "adHighCpu": ("ad", "cpu_saturation"),
    "toxiLatency": ("payment", "network_latency"),
    "netemLoss": ("payment", "network_loss"),
    "toxiReset": ("payment", "network_reset"),
    "toxiBandwidth": ("product-catalog", "network_throughput"),
    "hwWeakSignal": ("payment", "network_loss"),
    "hwContention": ("payment", "network_latency"),
    "hwWifiOutage": ("payment", "network_reset"),
}
SKILL = {"gpt-oss-120b@groq": 0.22, "gemma3-4b-q4@ollama": 0.1, "qwen3-8b-q4@ollama": 0.06}


def incidents() -> list[str]:
    pilot = [f"{f}-r{r}" for f in FAULTS for r in (1, 2)]
    hw = [f"{f}-r{r}" for f in HW_FAULTS for r in (1, 2, 3, 4)]
    return pilot + hw


def _answer(rng: random.Random, fault: str, right: bool) -> tuple[str, bool, str]:
    if fault == "healthy":
        svc, cat, layer = "checkout", "latency", "application"  # the usual false alarm
    elif right:
        svc, cat = SERVICE[fault]
        layer = "application" if fault in APP_FAULTS else "network"
    else:
        svc = rng.choice(["checkout", "frontend", "cart", "currency"])
        cat = rng.choice(["latency", "error", "network_latency"])
        layer = rng.choice(["application", "network"])
    conf = round(rng.uniform(0.6, 0.95), 2)
    body = {
        "abstain": False,
        "layer": layer,
        "hypotheses": [
            {
                "rank": 1,
                "service": svc,
                "category": cat,
                "confidence": conf,
                "evidence": ["M-0001"],
            },
            {
                "rank": 2,
                "service": "frontend",
                "category": "error",
                "confidence": 0.2,
                "evidence": [],
            },
        ],
        "action": {"type": "restart", "target": svc},
    }
    return (
        json.dumps(body),
        layer == ("network" if fault in (*NET_FAULTS, *HW_FAULTS) else "application"),
        svc,
    )


def make_runs(models: list[str], seed: int = 7) -> pl.DataFrame:
    rng = random.Random(seed)
    rows = []
    for model_id in models:
        m = MODELS[model_id]
        for inc in incidents():
            fault = inc.rsplit("-r", 1)[0]
            for cond in CONDITIONS:
                right = fault != "healthy" and rng.random() < SKILL.get(model_id, 0.1)
                parsed = not (m.tier != "C" and cond == "metrics" and rng.random() < 0.1)
                raw, layer_ok, svc = _answer(rng, fault, right)
                if not parsed:
                    raw = "Sure! Here is my analysis: the checkout service"
                    right, layer_ok = False, False
                rows.append(
                    {
                        "run_id": f"{inc}|{cond}|p0|{model_id}",
                        "incident_id": inc,
                        "condition": cond,
                        "perturbation": "p0",
                        "model_id": model_id,
                        "tier": m.tier,
                        "provider": m.provider,
                        "prompt_tokens": rng.randint(4000, 6000),
                        "completion_tokens": rng.randint(200, 1200),
                        "latency_ms": rng.randint(800, 30000),
                        "raw_output": raw,
                        "parsed_ok": parsed,
                        "top1": right,
                        "top3": right or (parsed and rng.random() < 0.08),
                        "layer_ok": parsed and layer_ok,
                        "evidence_validity": round(rng.uniform(0, 0.4), 3) if parsed else 0.0,
                        "confidence": round(rng.uniform(0.6, 0.95), 2) if parsed else None,
                        "action_type": "restart" if parsed else None,
                        "action_target": svc if parsed else None,
                        "started_at": "2026-11-15T12:00:00Z",
                    }
                )
    return pl.DataFrame(rows).select(RUN_COLUMNS)


def make_queue(path: Path, runs: pl.DataFrame, pending: int, failed: int) -> None:
    """A harness-shaped queue: the given runs are done, then some pending and failed jobs."""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE jobs (key TEXT PRIMARY KEY, incident_id TEXT, condition TEXT,
          perturbation TEXT, model_id TEXT, provider TEXT, status TEXT, attempts INTEGER,
          next_eligible_at REAL, last_error TEXT, record TEXT, updated_at REAL);
        CREATE TABLE usage (provider TEXT, ts REAL, tokens INTEGER);
        """
    )
    now = time.time()
    rows = runs.to_dicts()
    for i, r in enumerate(rows):
        status, err, rec = "done", None, json.dumps(r)
        if i >= len(rows) - pending - failed:
            status, rec = ("pending", None) if i < len(rows) - failed else ("failed", None)
            err = None if status == "pending" else "RateLimitError: tokens per day exceeded"
        con.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["run_id"],
                r["incident_id"],
                r["condition"],
                r["perturbation"],
                r["model_id"],
                r["provider"],
                status,
                1,
                0,
                err,
                rec,
                now - (len(rows) - i) * 60,
            ),
        )
    con.commit()
    con.close()


def build(root: Path) -> Path:
    """published/runs/runs.parquet holds the two Ollama models; work/ holds the
    Groq export and queue, with 168 jobs of which 150 are done."""
    (root / "published" / "runs").mkdir(parents=True, exist_ok=True)
    (root / "work").mkdir(parents=True, exist_ok=True)
    make_runs(["gemma3-4b-q4@ollama", "qwen3-8b-q4@ollama"]).write_parquet(
        root / "published" / "runs" / "runs.parquet"
    )
    groq = make_runs(["gpt-oss-120b@groq"], seed=11)
    make_queue(root / "work" / "queue-groq.sqlite", groq, pending=12, failed=6)
    groq.head(150).write_parquet(root / "work" / "runs-groq.parquet")
    return root


@cache
def mock_dir() -> Path:
    return build(Path(tempfile.mkdtemp(prefix="rootline-mock-")))
