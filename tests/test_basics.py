from __future__ import annotations

import json

from rootline_mcp import outputs, stats
from rootline_mcp.catalog import MODELS, estimate_days, fault_of, group_of, jobs_per_day
from rootline_mcp.config import Settings


def test_groq_free_tier_math():
    m = MODELS["gpt-oss-120b@groq"]
    assert m.tokens_per_job == 7536
    assert jobs_per_day(m) == 25  # 195,000 tokens a day / 7,536
    assert estimate_days(m, 168) == 6.7
    assert estimate_days(MODELS["gemma3-4b-q4@ollama"], 168) is None


def test_fault_groups():
    assert fault_of("toxiLatency-r2") == "toxiLatency"
    assert group_of("toxiLatency") == "network"
    assert group_of("paymentFailure") == "application"
    assert group_of(fault_of("hwWifiOutage-r4")) == "hardware"
    assert group_of("healthy") == "healthy"


def test_stats():
    assert stats.rate([True, False, False, False]) == 0.25
    lo, hi = stats.bootstrap_ci([True] * 4 + [False] * 38)
    assert 0 <= lo < 4 / 42 < hi <= 0.25
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(0, 6) == 0.0312  # 2 * 0.5**6
    assert stats.mcnemar_exact(3, 3) == 1.0


def test_output_summary_tolerates_fences_and_junk():
    body = {
        "abstain": False,
        "layer": "network",
        "hypotheses": [
            {"rank": 2, "service": "a", "category": "x", "confidence": 0.1, "evidence": []},
            {
                "rank": 1,
                "service": "payment",
                "category": "network_loss",
                "confidence": 0.9,
                "evidence": ["M-1", "T-2"],
            },
        ],
        "action": {"type": "clear_network_fault", "target": "payment"},
    }
    s = outputs.summarise("```json\n" + json.dumps(body) + "\n```")
    assert s["readable"] and s["hypotheses"][0]["service"] == "payment"
    assert s["hypotheses"][0]["evidence_cited"] == 2
    assert outputs.summarise("not json at all")["readable"] is False


def test_settings_never_expose_secrets():
    s = Settings.from_env(
        {
            "HF_TOKEN": "hf_secret",
            "ROOTLINE_GITHUB_TOKEN": "ghp_secret",
            "ROOTLINE_MCP_TOKEN": "abc",
            "RENDER_EXTERNAL_HOSTNAME": "x.onrender.com",
        }
    )
    assert s.backend == "github"
    assert "x.onrender.com" in s.allowed_hosts
    text = json.dumps(s.describe())
    assert "secret" not in text and "abc" not in text
