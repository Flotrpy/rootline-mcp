from __future__ import annotations

import polars as pl
import pytest

from rootline_mcp.config import Settings
from rootline_mcp.runner import NoBackend
from rootline_mcp.tools import ToolError, Tools


def test_overview_sees_both_sources(tools):
    o = tools.overview()
    assert {f["source"] for f in o["result_files"]} == {"published", "work"}
    assert o["snapshots_seen"] == 42
    assert o["queues"][0]["total"] == 168


def test_runs_are_unique_and_exports_win(store):
    df = store.runs()
    assert df["run_id"].n_unique() == df.height == 168 * 2 + 150


def test_list_models_plans_the_remaining_groq_jobs(tools):
    rows = {m["model"]: m for m in tools.list_models()["models"]}
    groq = rows["gpt-oss-120b@groq"]
    assert groq["runs_in_results"] == 150 and groq["full_sweep_jobs"] == 168
    assert groq["days_for_remaining"] == 0.7  # 18 jobs at 25 a day
    assert rows["qwen3-8b-q4@ollama"]["runs_on"].startswith("your computer")


def test_start_run_plans_before_starting(tools):
    plan = tools.start_run("gpt-oss-120b@groq")
    assert plan["started"] is False and plan["plan"]["jobs_to_run"] == 18
    started = tools.start_run("gpt-oss-120b@groq", confirm=True)
    assert started["started"] is True and started["run"]["status"] == "queued"
    assert tools.run_status()["recent_runs"][0]["title"] == "score gpt-oss-120b@groq"


def test_start_run_refuses_without_protection(store):
    t = Tools(Settings(), store, NoBackend(), runs_allowed=False)
    with pytest.raises(ToolError, match="ROOTLINE_MCP_TOKEN"):
        t.start_run("gpt-oss-120b@groq", confirm=True)
    with pytest.raises(ToolError, match="unknown model"):
        t.start_run("gpt-5@nowhere")
    with pytest.raises(ToolError, match="incident ids"):
        t.start_run("gpt-oss-120b@groq", incidents=["../etc"])


def test_status_and_failures(tools):
    q = tools.run_status("gpt-oss-120b@groq")["queues"][0]
    assert (q["done"], q["pending"], q["failed"]) == (150, 12, 6)
    f = tools.list_failures()
    assert f["failed_jobs"] == 6 and f["by_error"][0]["error"] == "RateLimitError"


def test_merge_checks_and_writes_locally(tools, mock_root):
    r = tools.merge_results()
    assert r["queue_vs_export"] == [
        {"provider": "groq", "done_in_queue": 150, "rows_in_export": 150, "match": True}
    ]
    groq = next(c for c in r["completeness"] if c["model"] == "gpt-oss-120b@groq")
    assert groq["missing"] == 18
    assert r["problems"] == ["none"]
    w = tools.merge_results(write=True)
    out = pl.read_parquet(mock_root / "work" / "runs.parquet")
    assert w["written_to"].endswith("runs.parquet") and out.height == 486


def test_merge_flags_a_queue_export_mismatch(tools, mock_root):
    df = pl.read_parquet(mock_root / "work" / "runs-groq.parquet")
    df.head(140).write_parquet(mock_root / "work" / "runs-groq.parquet")
    r = tools.merge_results()
    assert any("run export again" in p for p in r["problems"])


def test_merge_refuses_conflicts(tools, mock_root):
    df = pl.read_parquet(mock_root / "work" / "runs-groq.parquet")
    df.head(3).with_columns(~pl.col("top1")).write_parquet(
        mock_root / "work" / "runs-cerebras.parquet"
    )
    r = tools.merge_results()
    assert len(r["conflicting_run_ids"]) == 3
    with pytest.raises(ToolError, match="differ"):
        tools.merge_results(write=True)


def test_score_summary_groups_and_intervals(tools):
    rows = tools.score_summary(group_by=["model", "group"], conditions=["combined"])["rows"]
    groups = {r["group"] for r in rows}
    assert groups == {"application", "network", "hardware", "healthy"}
    r = next(r for r in rows if r["model"] == "gemma3-4b-q4@ollama" and r["group"] == "healthy")
    assert r["runs"] == 2 and r["healthy_false_alarms"] == "2 of 2"
    for r in rows:
        lo, hi = r["top1_95ci"]
        assert lo <= r["top1"] <= hi
    with pytest.raises(ToolError, match="can't group"):
        tools.score_summary(group_by=["colour"])
    with pytest.raises(ToolError, match="no results"):
        tools.score_summary(models=["nobody@nowhere"])


def test_compare_models_is_paired(tools):
    c = tools.compare_models("gemma3-4b-q4@ollama", "qwen3-8b-q4@ollama")
    assert c["paired_snapshots"] == 42
    assert c["both_right"] + c["only_a_right"] + c["only_b_right"] + c["both_wrong"] == 42
    assert 0 <= c["mcnemar_exact_p"] <= 1


def test_show_cases(tools):
    wrong = tools.show_cases("gpt-oss-120b@groq", kind="wrong", limit=3)
    assert wrong["shown"] == 3
    case = wrong["cases"][0]
    assert case["answer"]["readable"] and case["fault"] != "healthy" and not case["top1"]
    fa = tools.show_cases("gemma3-4b-q4@ollama", kind="false_alarm")
    assert fa["matching"] == 2 and all(c["fault"] == "healthy" for c in fa["cases"])
    un = tools.show_cases(
        "qwen3-8b-q4@ollama", condition="metrics", kind="unparsed", include_raw=True
    )
    assert all(not c["answer"]["readable"] and "raw_output" in c for c in un["cases"])


def test_condition_effect_pairs_combined_with_each_single(tools):
    r = tools.condition_effect("qwen3-8b-q4@ollama")
    singles = {s["condition"]: s for s in r["combined_vs_each"]}
    assert set(singles) == {"metrics", "traces", "logs"}
    assert all(s["paired_snapshots"] == 42 for s in singles.values())
    assert r["common_snapshots"] == 42
    best = r["combined_vs_best"]
    assert r["best_single"] == best["condition"]
    assert best["single_rate"] == max(s["single_rate"] for s in singles.values())
    for s in singles.values():
        diff = (s["only_combined_right"] - s["only_single_right"]) / 42
        assert s["difference_combined_minus_single"] == round(diff, 4)
        assert 0 <= s["mcnemar_exact_p"] <= 1
    assert r["reading"] and "exploratory" in r["caveat"]


def test_condition_effect_errors(tools):
    with pytest.raises(ToolError, match="metric"):
        tools.condition_effect("qwen3-8b-q4@ollama", metric="speed")
    with pytest.raises(ToolError, match="no results"):
        tools.condition_effect("gpt-5@nowhere")


def test_condition_effect_uses_snapshots_common_to_all_conditions(tools):
    # The mock Groq results are partial (150 of 168 jobs), so conditions cover
    # different snapshots; every comparison must use the same shared set.
    r = tools.condition_effect("gpt-oss-120b@groq")
    n = r["common_snapshots"]
    assert 0 < n < 42 and n <= min(r["snapshots_per_condition"].values())
    assert all(s["paired_snapshots"] == n for s in r["combined_vs_each"])
    assert len({s["combined_rate"] for s in r["combined_vs_each"]}) == 1
