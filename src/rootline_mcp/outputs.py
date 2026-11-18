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
