"""Read a model's raw answer leniently, for display.

Scores come from the harness (parsed_ok, top1, ...); this only pulls out what
a person wants to see: the layer, the first hypotheses and the action.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def parse(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    m = _FENCE.match(raw)
    text = m.group(1) if m else raw.strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def summarise(raw: str | None, max_hypotheses: int = 3) -> dict[str, Any]:
    data = parse(raw)
    if data is None:
        snippet = (raw or "").strip().replace("\n", " ")
        return {"readable": False, "start_of_answer": snippet[:200]}
    hyps = data.get("hypotheses") if isinstance(data.get("hypotheses"), list) else []
    ranked = sorted(
        (h for h in hyps if isinstance(h, dict)),
        key=lambda h: h.get("rank", 99) if isinstance(h.get("rank"), int) else 99,
    )
    action = data.get("action") if isinstance(data.get("action"), dict) else {}
    return {
        "readable": True,
        "abstain": data.get("abstain"),
        "layer": data.get("layer"),
        "hypotheses": [
            {
                "rank": h.get("rank"),
                "service": h.get("service"),
                "category": h.get("category"),
                "confidence": h.get("confidence"),
                "evidence_cited": len(h.get("evidence") or []),
            }
            for h in ranked[:max_hypotheses]
        ],
        "action": {"type": action.get("type"), "target": action.get("target")},
    }


def norm_service(s: str) -> str:
    """Same normalisation as the harness: case, spaces and underscores."""
    return "-".join(s.strip().lower().replace("_", "-").split())


def norm_category(s: str) -> str:
    return "_".join(s.strip().lower().replace("-", " ").split())


def names_edge_end(raw: str | None, ends: tuple[str, str], category: str, k: int) -> bool:
    """True if one of the top-k hypotheses names either end of the link with the
    right category (the README's exploratory either-end rescoring)."""
    data = parse(raw)
    if data is None or data.get("abstain"):
        return False
    hyps = data.get("hypotheses") if isinstance(data.get("hypotheses"), list) else []
    ranked = sorted(
        (h for h in hyps if isinstance(h, dict)),
        key=lambda h: h.get("rank", 99) if isinstance(h.get("rank"), int) else 99,
    )
    want = {norm_service(e) for e in ends}
    return any(
        isinstance(h.get("service"), str)
        and isinstance(h.get("category"), str)
        and norm_service(h["service"]) in want
        and norm_category(h["category"]) == category
        for h in ranked[:k]
    )
