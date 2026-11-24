"""What each MCP tool does, kept apart from the MCP plumbing so it is easy to test."""

from __future__ import annotations

import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from rootline_mcp import outputs, stats
from rootline_mcp.catalog import (
    CONDITIONS,
    EDGES,
    MODELS,
    estimate_days,
    fault_of,
    get_model,
    group_of,
    jobs_per_day,
)
from rootline_mcp.config import Settings
from rootline_mcp.runner import Backend, RunError, RunRequest
from rootline_mcp.store import RUN_COLUMNS, Store

SCORED = ("parsed_ok", "top1", "top3", "layer_ok", "raw_output")
GROUP_KEYS = {
    "model": "model_id",
    "condition": "condition",
    "fault": "fault",
    "group": "fault_group",
    "perturbation": "perturbation",
    "incident": "incident_id",
}
CASE_KINDS = ("wrong", "right", "unparsed", "overconfident", "false_alarm")


class ToolError(Exception):
    """A problem the person can fix; shown as the tool's error message."""


def _tz() -> ZoneInfo:
    return ZoneInfo(os.environ.get("ROOTLINE_TZ", "America/New_York"))


def _local_time(ts: float | str | None) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return ts
    else:
        dt = datetime.fromtimestamp(ts, UTC)
    return dt.astimezone(_tz()).strftime("%a %b %d, %I:%M %p %Z")


def _with_facets(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("incident_id").map_elements(fault_of, return_dtype=pl.String).alias("fault"),
    ).with_columns(
        pl.col("fault").map_elements(group_of, return_dtype=pl.String).alias("fault_group"),
    )


def _bools(series: pl.Series) -> list[bool]:
    return [bool(x) for x in series.fill_null(False).to_list()]


def _rescore_either_end(df: pl.DataFrame) -> pl.DataFrame:
    """top1/top3 under the either-end rule for link faults; other faults unchanged.

    Only answers the harness parsed (parsed_ok) can gain, so an answer that breaks the
    response contract stays an abstention, as under the official rule.
    """

    def hit(k: int) -> pl.Expr:
        return pl.struct("raw_output", "fault", "parsed_ok").map_elements(
            lambda r: bool(
                r["parsed_ok"]
                and r["fault"] in EDGES
                and outputs.names_edge_end(r["raw_output"], *EDGES[r["fault"]], k)
            ),
            return_dtype=pl.Boolean,
        )

    return df.with_columns(
        (pl.col("top1").fill_null(False) | hit(1)).alias("top1"),
        (pl.col("top3").fill_null(False) | hit(3)).alias("top3"),
    )


@dataclass
class Tools:
    settings: Settings
    store: Store
    backend: Backend
    runs_allowed: bool = True

    # --- helpers -------------------------------------------------------------

    def _runs(self, source: str = "all") -> pl.DataFrame:
        try:
            df = self.store.runs(source)
        except (RuntimeError, ValueError) as e:
            raise ToolError(str(e)) from e
        return _with_facets(df) if df.height else df

    def _filter(
        self,
        df: pl.DataFrame,
        models: list[str] | None = None,
        conditions: list[str] | None = None,
        perturbation: str | None = "p0",
    ) -> pl.DataFrame:
        if models:
            unknown = sorted(set(models) - set(df["model_id"].unique().to_list()))
            if unknown:
                have = sorted(df["model_id"].unique().to_list())
                raise ToolError(f"no results for {unknown}; results exist for {have}")
            df = df.filter(pl.col("model_id").is_in(models))
        if conditions:
            bad = [c for c in conditions if c not in CONDITIONS]
            if bad:
                raise ToolError(f"unknown conditions {bad}; use {list(CONDITIONS)}")
            df = df.filter(pl.col("condition").is_in(conditions))
        if perturbation:
            df = df.filter(pl.col("perturbation") == perturbation)
        return df

    def _known_incidents(self, df: pl.DataFrame | None = None) -> list[str]:
        df = self._runs() if df is None else df
        return sorted(df["incident_id"].unique().to_list()) if df.height else []

    # --- tools ---------------------------------------------------------------

    def overview(self) -> dict[str, Any]:
        files = []
        for f in self.store.run_files():
            try:
                n = self.store.read_runs(f).height
            except Exception as e:
                n = f"unreadable: {e}"
            files.append({"source": f.source, "file": f.name, "rows": n})
        df = self._runs()
        by_model = (
            df.group_by("model_id")
            .agg(pl.len().alias("runs"), pl.col("condition").n_unique().alias("conditions"))
            .sort("model_id")
            .to_dicts()
            if df.height
            else []
        )
        return {
            "setup": self.settings.describe(),
            "can_start_runs": self.runs_allowed and self.backend.name != "none",
            "result_files": files,
            "runs_by_model": by_model,
            "snapshots_seen": len(self._known_incidents(df)),
            "queues": self.run_status()["queues"],
            "note": "Mock data: made-up numbers." if self.settings.mock else None,
        }

    def list_models(self) -> dict[str, Any]:
        df = self._runs()
        incidents = self._known_incidents(df) or [f"x-r{i}" for i in range(42)]
        sweep = len(incidents) * len(CONDITIONS)
        done = Counter(df["model_id"].to_list()) if df.height else Counter()
        out = []
        for m in MODELS.values():
            out.append(
                {
                    "model": m.id,
                    "provider": m.provider,
                    "tier": m.tier,
                    "runs_on": "your computer (Ollama)"
                    if m.runs_on == "local"
                    else "free cloud API",
                    "free_jobs_per_day": jobs_per_day(m),
                    "runs_in_results": done.get(m.id, 0),
                    "full_sweep_jobs": sweep,
                    "days_for_remaining": estimate_days(m, max(0, sweep - done.get(m.id, 0))),
                }
            )
        return {"models": out, "conditions": list(CONDITIONS)}

    def start_run(
        self,
        model: str,
        confirm: bool = False,
        perturbation: str = "p0",
        max_wait_s: int = 900,
        conditions: list[str] | None = None,
        incidents: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            m = get_model(model)
        except ValueError as e:
            raise ToolError(str(e)) from e
        req = RunRequest(
            m, perturbation, max_wait_s, tuple(conditions or ()), tuple(incidents or ())
        )
        try:
            req.validate()
        except RunError as e:
            raise ToolError(str(e)) from e
        df = self._runs()
        incs = list(incidents or self._known_incidents(df))
        conds = list(conditions or CONDITIONS)
        have = set()
        if df.height:
            mine = df.filter(
                (pl.col("model_id") == m.id) & (pl.col("perturbation") == perturbation)
            )
            have = set(zip(mine["incident_id"].to_list(), mine["condition"].to_list(), strict=True))
        todo = [(i, c) for i in incs for c in conds if (i, c) not in have]
        plan = {
            "model": m.id,
            "backend": self.backend.name,
            "jobs_already_done": len(incs) * len(conds) - len(todo),
            "jobs_to_run": len(todo),
            "free_jobs_per_day": jobs_per_day(m),
            "estimated_days": estimate_days(m, len(todo)),
            "what_happens": (
                "GitHub Actions runs scripts/score_daily.py for this model: it resumes the queue from"
                " the private Hugging Face working dataset, scores until the free daily cap would"
                " make it wait longer than max_wait_s, then uploads the queue and runs export."
                if self.backend.name == "github"
                else "Pretend run (mock mode): nothing is scored."
                if self.backend.name == "mock"
                else "The harness runs enqueue, work and export on this computer."
            ),
            "cost": "Free-tier API quota only; no paid calls.",
        }
        if not confirm:
            return {
                "plan": plan,
                "started": False,
                "next": "Call again with confirm=true to start.",
            }
        if not self.runs_allowed:
            raise ToolError(
                "Starting runs is switched off on this endpoint because it has no ROOTLINE_MCP_TOKEN."
                " Set one (and use the URL with the token) to allow it."
            )
        if not todo:
            return {"plan": plan, "started": False, "next": "Nothing left to run for this model."}
        try:
            result = self.backend.start(req)
        except RunError as e:
            raise ToolError(str(e)) from e
        return {"plan": plan, **result}

    def run_status(self, model: str | None = None, limit: int = 5) -> dict[str, Any]:
        queues = []
        for q in self.store.queues():
            for mid, c in sorted(q.counts.items()):
                if model and mid != model:
                    continue
                total = sum(c.values())
                left = c.get("pending", 0) + c.get("running", 0)
                eta = estimate_days(MODELS[mid], left) if mid in MODELS else None
                queues.append(
                    {
                        "queue": f"{q.source}:{q.name}",
                        "model": mid,
                        "done": c.get("done", 0),
                        "pending": c.get("pending", 0),
                        "running": c.get("running", 0),
                        "failed": c.get("failed", 0),
                        "total": total,
                        "percent_done": round(100 * c.get("done", 0) / total, 1) if total else 0,
                        "days_left_at_free_cap": eta,
                        "last_activity": _local_time(q.last_update),
                    }
                )
        try:
            recent = self.backend.recent(limit)
        except RunError as e:
            recent = [{"error": str(e)}]
        except Exception as e:
            recent = [{"error": f"could not read runs: {type(e).__name__}"}]
        for r in recent:
            for k in ("created_at", "updated_at"):
                if r.get(k):
                    r[k] = _local_time(r[k])
        return {"queues": queues, "recent_runs": recent, "backend": self.backend.name}

    def list_failures(self, model: str | None = None, limit: int = 20) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for q in self.store.queues():
            for f in q.failures:
                if model and f["model_id"] != model:
                    continue
                kind = str(f["error"]).split(":", 1)[0].strip() or "unknown"
                groups[kind].append(f)
        df = self._runs()
        unparsed = 0
        if df.height:
            sub = df.filter(~pl.col("parsed_ok").fill_null(False))
            if model:
                sub = sub.filter(pl.col("model_id") == model)
            unparsed = sub.height
        return {
            "failed_jobs": sum(len(v) for v in groups.values()),
            "by_error": [
                {
                    "error": k,
                    "count": len(v),
                    "examples": [x["key"] for x in v[:3]],
                    "first_message": str(v[0]["error"])[:240],
                }
                for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))
            ][:limit],
            "answers_that_failed_to_parse": unparsed,
            "hint": "Failed jobs stay in the queue; a retryable error is tried again on the next run.",
        }

    def merge_results(self, write: bool = False) -> dict[str, Any]:
        files = [f for f in self.store.run_files() if not f.merged]
        frames = {f: self.store.read_runs(f) for f in files}
        checks: list[str] = []
        per_file = []
        for f, d in frames.items():
            dup = d.height - d["run_id"].n_unique()
            per_file.append(
                {
                    "file": f"{f.source}:{f.name}",
                    "rows": d.height,
                    "models": sorted(d["model_id"].unique().to_list()),
                    "duplicate_ids": dup,
                }
            )
            if dup:
                checks.append(f"{f.name} repeats {dup} run_ids")
        conflicts: list[str] = []
        if len(frames) > 1:
            allrows = pl.concat(list(frames.values()), how="vertical_relaxed")
            dupes = allrows.filter(pl.col("run_id").is_duplicated())
            for rid, grp in dupes.group_by("run_id"):
                if grp.select(SCORED).unique().height > 1:
                    conflicts.append(str(rid[0]))
        if conflicts:
            checks.append(f"{len(conflicts)} run_ids differ between exports")
        # The queue must hold exactly what its export holds.
        queue_vs_export = []
        exports = {Path(f.name).stem.removeprefix("runs-"): d for f, d in frames.items()}
        for q in self.store.queues():
            prov = Path(q.name).stem.removeprefix("queue-")
            exp = exports.get(prov)
            rows = exp.height if exp is not None else 0
            same = rows == q.done_records
            queue_vs_export.append(
                {
                    "provider": prov,
                    "done_in_queue": q.done_records,
                    "rows_in_export": rows,
                    "match": same,
                }
            )
            if not same:
                checks.append(
                    f"{prov}: queue has {q.done_records} done jobs but the export has {rows} rows"
                    " (run export again)"
                )
        merged = self._runs()
        incidents = self._known_incidents(merged)
        completeness = []
        for mid in sorted(merged["model_id"].unique().to_list()) if merged.height else []:
            mine = merged.filter((pl.col("model_id") == mid) & (pl.col("perturbation") == "p0"))
            have = set(zip(mine["incident_id"].to_list(), mine["condition"].to_list(), strict=True))
            missing = [f"{i}|{c}" for i in incidents for c in CONDITIONS if (i, c) not in have]
            completeness.append(
                {
                    "model": mid,
                    "runs": mine.height,
                    "expected": len(incidents) * len(CONDITIONS),
                    "missing": len(missing),
                    "missing_examples": missing[:5],
                    "unparsed": int((~mine["parsed_ok"].fill_null(False)).sum()),
                }
            )
        result: dict[str, Any] = {
            "exports": per_file,
            "queue_vs_export": queue_vs_export,
            "conflicting_run_ids": conflicts[:10],
            "completeness": completeness,
            "merged_rows": merged.height,
            "problems": checks or ["none"],
            "written_to": None,
        }
        if write:
            if conflicts:
                raise ToolError(
                    "Not writing: some run_ids differ between exports. Fix those first."
                )
            target = next((f.source for f in files), None) or next(iter(self.store.sources))
            if target == "published":
                raise ToolError("Refusing to write into the public release dataset.")
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "runs.parquet"
                merged.select(RUN_COLUMNS).sort("run_id").write_parquet(out, compression="zstd")
                name = "runs/runs.parquet" if target == "local" else "runs.parquet"
                try:
                    result["written_to"] = self.store.source(target).upload(
                        out, name, f"merge: {merged.height} runs"
                    )
                except Exception as e:
                    raise ToolError(
                        f"could not write the merged file: {type(e).__name__}: {e}"
                    ) from e
        return result

    def score_summary(
        self,
        models: list[str] | None = None,
        conditions: list[str] | None = None,
        group_by: list[str] | None = None,
        perturbation: str = "p0",
        source: str = "all",
        link_rule: str = "official",
    ) -> dict[str, Any]:
        keys = group_by or ["model", "condition"]
        bad = [k for k in keys if k not in GROUP_KEYS]
        if bad:
            raise ToolError(f"can't group by {bad}; choose from {list(GROUP_KEYS)}")
        if link_rule not in ("official", "either_end"):
            raise ToolError("link_rule must be official or either_end")
        if link_rule == "either_end" and perturbation != "p0":
            raise ToolError("either_end needs real service names, so use perturbation p0")
        df = self._filter(self._runs(source), models, conditions, perturbation)
        if not df.height:
            return {"rows": [], "note": "No runs match."}
        if link_rule == "either_end":
            df = _rescore_either_end(df)
        cols = [GROUP_KEYS[k] for k in keys]
        rows = []
        for key, g in df.sort(cols).group_by(cols, maintain_order=True):
            top1 = _bools(g["top1"])
            healthy = g.filter(pl.col("fault") == "healthy")
            row = dict(zip(keys, key, strict=True))
            row.update(
                runs=g.height,
                top1=stats.rate(top1),
                top1_95ci=stats.bootstrap_ci(top1),
                top3=stats.rate(_bools(g["top3"])),
                layer=stats.rate(_bools(g["layer_ok"])),
                evidence_validity=round(float(g["evidence_validity"].fill_null(0).mean()), 4),
                parsed=stats.rate(_bools(g["parsed_ok"])),
                mean_confidence=(
                    round(float(c), 3)
                    if (c := g["confidence"].drop_nulls().mean()) is not None
                    else None
                ),
                healthy_false_alarms=(
                    f"{int((healthy['parsed_ok'].fill_null(False) & ~healthy['top1'].fill_null(False)).sum())}"
                    f" of {healthy.height}"
                    if healthy.height
                    else None
                ),
            )
            rows.append(row)
        return {
            "rows": rows,
            "perturbation": perturbation,
            "link_rule": link_rule,
            "link_rule_note": (
                "Exploratory, not pre-registered: on a link fault, naming either end of the "
                "link with the right category counts as correct. Hypothesis tests use the "
                "official rule."
                if link_rule == "either_end"
                else None
            ),
            "note": "Mock data: made-up numbers." if self.settings.mock else None,
        }

    def compare_models(
        self,
        model_a: str,
        model_b: str,
        condition: str = "combined",
        metric: str = "top1",
        perturbation: str = "p0",
    ) -> dict[str, Any]:
        if metric not in ("top1", "top3", "layer_ok", "parsed_ok"):
            raise ToolError("metric must be top1, top3, layer_ok or parsed_ok")
        df = self._filter(self._runs(), [model_a, model_b], [condition], perturbation)
        a = df.filter(pl.col("model_id") == model_a).select(
            "incident_id", pl.col(metric).fill_null(False).alias("a")
        )
        b = df.filter(pl.col("model_id") == model_b).select(
            "incident_id", pl.col(metric).fill_null(False).alias("b")
        )
        j = a.join(b, on="incident_id")
        if not j.height:
            raise ToolError("The two models share no scored snapshots under that condition.")
        both = int((j["a"] & j["b"]).sum())
        only_a = int((j["a"] & ~j["b"]).sum())
        only_b = int((~j["a"] & j["b"]).sum())
        neither = j.height - both - only_a - only_b
        diff = round((only_a - only_b) / j.height, 4)
        return {
            "paired_snapshots": j.height,
            "metric": metric,
            model_a: stats.rate(_bools(j["a"])),
            model_b: stats.rate(_bools(j["b"])),
            "difference_a_minus_b": diff,
            "both_right": both,
            "only_a_right": only_a,
            "only_b_right": only_b,
            "both_wrong": neither,
            "mcnemar_exact_p": stats.mcnemar_exact(only_a, only_b),
            "only_a_examples": j.filter(pl.col("a") & ~pl.col("b"))["incident_id"].to_list()[:5],
            "only_b_examples": j.filter(~pl.col("a") & pl.col("b"))["incident_id"].to_list()[:5],
            "reading": (
                "No statistically significant difference at this sample size."
                if stats.mcnemar_exact(only_a, only_b) >= 0.05
                else f"{model_a if diff > 0 else model_b} is reliably better here (p < 0.05)."
            ),
        }

    def condition_effect(
        self,
        model: str,
        metric: str = "top1",
        perturbation: str = "p0",
        source: str = "all",
    ) -> dict[str, Any]:
        """Combined evidence vs each single signal on the same snapshots (the paper's RQ1)."""
        if metric not in ("top1", "top3", "layer_ok", "parsed_ok"):
            raise ToolError("metric must be top1, top3, layer_ok or parsed_ok")
        df = self._filter(self._runs(source), [model], None, perturbation)
        by_cond = {
            c: df.filter(pl.col("condition") == c).select(
                "incident_id", pl.col(metric).fill_null(False).alias(c)
            )
            for c in CONDITIONS
        }
        if not by_cond["combined"].height:
            raise ToolError(f"{model} has no scored runs under the combined condition yet.")
        coverage = {c: by_cond[c].height for c in CONDITIONS}
        # Compare every signal on the same snapshots, or a signal scored on an easier
        # subset (partial runs) could look best.
        common = set(by_cond["combined"]["incident_id"].to_list())
        for c in CONDITIONS:
            common &= set(by_cond[c]["incident_id"].to_list())
        if not common:
            raise ToolError(
                f"No snapshot of {model} is scored under all four conditions yet "
                f"(snapshots per condition: {coverage})."
            )
        keep = pl.col("incident_id").is_in(sorted(common))
        combined = by_cond["combined"].filter(keep)
        singles = []
        for c in (c for c in CONDITIONS if c != "combined"):
            j = combined.join(by_cond[c].filter(keep), on="incident_id")
            only_comb = int((j["combined"] & ~j[c]).sum())
            only_single = int((~j["combined"] & j[c]).sum())
            p = stats.mcnemar_exact(only_comb, only_single)
            singles.append(
                {
                    "condition": c,
                    "paired_snapshots": j.height,
                    "single_rate": stats.rate(_bools(j[c])),
                    "combined_rate": stats.rate(_bools(j["combined"])),
                    "difference_combined_minus_single": round(
                        (only_comb - only_single) / j.height, 4
                    ),
                    "only_combined_right": only_comb,
                    "only_single_right": only_single,
                    "mcnemar_exact_p": p,
                }
            )
        best = max(singles, key=lambda r: (r["single_rate"] or 0, -r["paired_snapshots"]))
        diff = best["difference_combined_minus_single"]
        if best["mcnemar_exact_p"] >= 0.05:
            reading = (
                f"No statistically significant difference between combined evidence and {best['condition']}, "
                "the best single signal, at this sample size."
            )
        elif diff > 0:
            reading = f"Combined evidence reliably beats {best['condition']} (p < 0.05)."
        else:
            reading = f"{best['condition']} alone reliably beats combined evidence (p < 0.05)."
        return {
            "model": model,
            "metric": metric,
            "perturbation": perturbation,
            "common_snapshots": len(common),
            "snapshots_per_condition": coverage,
            "best_single": best["condition"],
            "combined_vs_best": best,
            "combined_vs_each": singles,
            "reading": reading,
            "caveat": (
                "The best single signal is picked after looking at these results, so its "
                "p-value is optimistic; treat it as exploratory unless the paper fixed the "
                "comparison in advance."
            ),
            "note": "Mock data: made-up numbers." if self.settings.mock else None,
        }

    def layer_gap(
        self,
        model: str,
        condition: str = "combined",
        perturbation: str = "p0",
        source: str = "all",
    ) -> dict[str, Any]:
        """Layer accuracy on network vs application faults, with Fisher's exact test (RQ2)."""
        conditions = None if condition == "all" else [condition]
        df = self._filter(self._runs(source), [model], conditions, perturbation)
        if not df.height:
            raise ToolError(f"{model} has no scored runs under {condition} yet.")
        groups = {}
        parts = {
            "application": ["application"],
            "network": ["network"],
            "hardware": ["hardware"],
            "network_and_hardware": ["network", "hardware"],
        }
        for name, members in parts.items():
            sub = df.filter(pl.col("fault_group").is_in(members))
            ok = _bools(sub["layer_ok"])
            groups[name] = {
                "runs": sub.height,
                "layer_right": sum(ok),
                "layer_accuracy": stats.rate(ok),
                "layer_95ci": stats.bootstrap_ci(ok),
                "top1": stats.rate(_bools(sub["top1"])),
            }
        # The paper's RQ2 counts the hardware-arm cases on the network side.
        app, net = groups["application"], groups["network_and_hardware"]
        if not app["runs"] or not net["runs"]:
            raise ToolError("Need runs on both application and network faults to compare.")
        p = stats.fisher_exact(
            net["layer_right"],
            net["runs"] - net["layer_right"],
            app["layer_right"],
            app["runs"] - app["layer_right"],
        )
        gap = round(net["layer_accuracy"] - app["layer_accuracy"], 4)
        pooled = condition == "all"
        if pooled:
            reading = (
                "Pooled conditions repeat each snapshot, so no significance reading is given; "
                "run it per condition for a test."
            )
        elif p >= 0.05:
            reading = (
                "No statistically significant evidence of a layer-accuracy gap between "
                "network and application faults; with small groups that is not proof they match."
            )
        else:
            way = "less" if gap < 0 else "more"
            reading = (
                f"The model places network faults in the right layer {way} often "
                "(unadjusted p < 0.05)."
            )
        caveat = (
            "Network includes the hardware-arm cases, as in the paper's RQ2. Snapshots differ "
            "between the groups, so this is an unpaired test, and p is not Holm-adjusted "
            "across RQ1 and RQ2 as the paper does."
        )
        if pooled:
            caveat += (
                " With condition='all' the same snapshot counts once per condition, so the "
                "p-value and the 95% intervals are too small."
            )
        return {
            "model": model,
            "condition": condition,
            "perturbation": perturbation,
            "groups": groups,
            "gap_network_minus_application": gap,
            "fisher_exact_p": p,
            "reading": reading,
            "caveat": caveat,
            "note": "Mock data: made-up numbers." if self.settings.mock else None,
        }

    def hardest_snapshots(
        self,
        metric: str = "top1",
        perturbation: str = "p0",
        source: str = "all",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Snapshots ranked by how many model x condition runs got them right."""
        if metric not in ("top1", "top3", "layer_ok"):
            raise ToolError("metric must be top1, top3 or layer_ok")
        df = self._filter(self._runs(source), perturbation=perturbation)
        if not df.height:
            raise ToolError(f"No scored runs under {perturbation} yet.")
        per = (
            df.group_by("incident_id", "fault", "fault_group")
            .agg(
                pl.len().alias("runs"),
                pl.col(metric).fill_null(False).cast(pl.Int64).sum().alias("right"),
            )
            .with_columns((pl.col("right") / pl.col("runs")).round(4).alias("rate"))
            .sort(["rate", "runs", "incident_id"], descending=[False, True, False])
        )
        rows = per.rename({"incident_id": "snapshot"}).to_dicts()
        runs = per["runs"]
        return {
            "metric": metric,
            "perturbation": perturbation,
            "snapshots": per.height,
            "never_right": int((per["right"] == 0).sum()),
            "always_right": int((per["right"] == per["runs"]).sum()),
            "hardest": rows[:limit],
            "easiest": rows[::-1][:limit],
            "caveat": (
                None
                if runs.min() == runs.max()
                else "Snapshots have different run counts (some results are partial), so "
                "compare rates, not counts."
            ),
            "note": "Mock data: made-up numbers." if self.settings.mock else None,
        }

    def snapshot_brief(
        self, incident: str, perturbation: str = "p0", source: str = "all"
    ) -> dict[str, Any]:
        """One snapshot: its fault and group, and which model x condition runs got it right."""
        df = self._filter(self._runs(source), perturbation=perturbation)
        known = self._known_incidents(df)
        if incident not in known:
            raise ToolError(f"No scored runs for {incident!r}; snapshots seen: {known[:10]}")
        sub = df.filter(pl.col("incident_id") == incident).sort("model_id", "condition")
        runs = [
            {
                "model": r["model_id"],
                "condition": r["condition"],
                "top1": bool(r["top1"]),
                "top3": bool(r["top3"]),
                "layer_ok": bool(r["layer_ok"]),
                "parsed": bool(r["parsed_ok"]),
                "blamed": outputs.summarise(r["raw_output"], max_hypotheses=1),
            }
            for r in sub.to_dicts()
        ]
        right = [f"{r['model']} / {r['condition']}" for r in runs if r["top1"]]
        return {
            "snapshot": incident,
            "fault": sub["fault"][0],
            "fault_group": sub["fault_group"][0],
            "perturbation": perturbation,
            "runs": len(runs),
            "top1_right": len(right),
            "right_by": right,
            "results": runs,
            "note": "Mock data: made-up numbers." if self.settings.mock else None,
        }

    def calibration(
        self,
        model: str,
        condition: str = "combined",
        perturbation: str = "p0",
        source: str = "all",
        bins: int = 5,
    ) -> dict[str, Any]:
        """Stated confidence vs top-1 accuracy in equal-width bins, with expected calibration error."""
        if not 2 <= bins <= 10:
            raise ToolError("bins must be between 2 and 10")
        conditions = None if condition == "all" else [condition]
        df = self._filter(self._runs(source), [model], conditions, perturbation)
        df = df.filter(
            (pl.col("fault_group") != "healthy")
            & pl.col("parsed_ok").fill_null(False)
            & pl.col("confidence").is_not_null()
        )
        if not df.height:
            raise ToolError(f"{model} has no parsed answers with a confidence under {condition}.")
        conf = [min(max(float(c), 0.0), 1.0) for c in df["confidence"].to_list()]
        right = _bools(df["top1"])
        n = len(conf)
        rows, ece = [], 0.0
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
            if not idx:
                rows.append({"bin": f"{lo:.4g}-{hi:.4g}", "answers": 0})
                continue
            acc = sum(right[i] for i in idx) / len(idx)
            mc = sum(conf[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(acc - mc)
            rows.append(
                {
                    "bin": f"{lo:.4g}-{hi:.4g}",
                    "answers": len(idx),
                    "mean_confidence": round(mc, 4),
                    "top1_accuracy": round(acc, 4),
                    "gap": round(mc - acc, 4),
                }
            )
        mean_conf = sum(conf) / n
        acc = sum(right) / n
        over = round(mean_conf - acc, 4)
        return {
            "model": model,
            "condition": condition,
            "perturbation": perturbation,
            "answers": n,
            "mean_confidence": round(mean_conf, 4),
            "top1_accuracy": round(acc, 4),
            "overconfidence": over,
            "expected_calibration_error": round(ece, 4),
            "bins": rows,
            "reading": (
                "Overconfident: it states more confidence than its accuracy."
                if over > 0.05
                else "Underconfident: it is right more often than it says."
                if over < -0.05
                else "Confidence roughly matches accuracy overall."
            ),
            "caveat": (
                "Uses parsed answers on fault snapshots only (healthy ones left out), with the "
                "top hypothesis's confidence; bins with few answers are noisy."
            ),
            "note": "Mock data: made-up numbers." if self.settings.mock else None,
        }

    def blame_map(
        self,
        model: str,
        condition: str = "combined",
        perturbation: str = "p0",
        source: str = "all",
        top: int = 3,
    ) -> dict[str, Any]:
        """For each fault, what the model's top hypothesis blamed most often."""
        conditions = None if condition == "all" else [condition]
        df = self._filter(self._runs(source), [model], conditions, perturbation)
        if not df.height:
            raise ToolError(f"{model} has no scored runs under {condition} yet.")
        blames: dict[str, Counter] = {}
        groups: dict[str, str] = {}
        for r in df.select("fault", "fault_group", "raw_output").to_dicts():
            a = outputs.summarise(r["raw_output"], max_hypotheses=1)
            if not a["readable"]:
                label = "unreadable answer"
            elif a.get("abstain") or not a["hypotheses"]:
                label = "abstained"
            else:
                h = a["hypotheses"][0]
                svc = outputs.norm_service(str(h.get("service") or "?"))
                cat = outputs.norm_category(str(h.get("category") or "?"))
                label = f"{svc} / {cat}"
            blames.setdefault(r["fault"], Counter())[label] += 1
            groups[r["fault"]] = r["fault_group"]
        faults = []
        for fault in sorted(blames, key=lambda f: (groups[f], f)):
            c = blames[fault]
            n = sum(c.values())
            faults.append(
                {
                    "fault": fault,
                    "fault_group": groups[fault],
                    "runs": n,
                    "most_blamed": [
                        {"blamed": k, "times": v, "share": round(v / n, 4)}
                        for k, v in c.most_common(top)
                    ],
                }
            )
        return {
            "model": model,
            "condition": condition,
            "perturbation": perturbation,
            "faults": faults,
            "caveat": "Counts the top hypothesis only, as service / category after the harness's normalisation.",
            "note": "Mock data: made-up numbers." if self.settings.mock else None,
        }

    def show_cases(
        self,
        model: str,
        condition: str = "combined",
        kind: str = "wrong",
        fault: str | None = None,
        limit: int = 5,
        include_raw: bool = False,
        perturbation: str = "p0",
    ) -> dict[str, Any]:
        if kind not in CASE_KINDS:
            raise ToolError(f"kind must be one of {list(CASE_KINDS)}")
        df = self._filter(self._runs(), [model], [condition], perturbation)
        parsed = pl.col("parsed_ok").fill_null(False)
        top1 = pl.col("top1").fill_null(False)
        healthy = pl.col("fault") == "healthy"
        pick = {
            "wrong": parsed & ~top1 & ~healthy,
            "right": top1 & ~healthy,
            "unparsed": ~parsed,
            "overconfident": parsed & ~top1 & ~healthy & (pl.col("confidence").fill_null(0) >= 0.8),
            "false_alarm": parsed & ~top1 & healthy,
        }[kind]
        sub = df.filter(pick)
        if fault:
            sub = sub.filter(pl.col("fault") == fault)
        sub = sub.sort("incident_id")
        cases = []
        for r in sub.head(max(1, min(limit, 20))).to_dicts():
            case = {
                "incident": r["incident_id"],
                "fault": r["fault"],
                "fault_group": r["fault_group"],
                "top1": r["top1"],
                "top3": r["top3"],
                "layer_ok": r["layer_ok"],
                "evidence_validity": r["evidence_validity"],
                "answer": outputs.summarise(r["raw_output"]),
            }
            if include_raw:
                case["raw_output"] = (r["raw_output"] or "")[:4000]
            cases.append(case)
        return {
            "matching": sub.height,
            "shown": len(cases),
            "kind": kind,
            "cases": cases,
            "note": "The fault name is the incident's ground truth; the model never saw it.",
        }
