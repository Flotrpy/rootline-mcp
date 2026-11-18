"""If a Rootline checkout is around, check this connector's copy of the catalog against it."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from rootline_mcp.catalog import APP_FAULTS, FAULTS, HW_FAULTS, LIMITS, MODELS

REPO = Path(os.environ.get("ROOTLINE_REPO_DIR", "/home/claude/flotrpy/rootline"))
HARNESS = REPO / "harness" / "rootline_harness"
pytestmark = pytest.mark.skipif(not HARNESS.is_dir(), reason="no Rootline checkout")


def test_faults_match_harness():
    text = (HARNESS / "catalog.py").read_text()
    block = text.split("FAULTS: Final[tuple[str, ...]] = (", 1)[1].split(")", 1)[0]
    assert tuple(re.findall(r'"(\w+)"', block)) == FAULTS
    hw = text.split("HW_FAULTS: Final[tuple[str, ...]] = (", 1)[1].split(")", 1)[0]
    assert tuple(re.findall(r'"(\w+)"', hw)) == HW_FAULTS
    assert "FLAGS: Final[tuple[str, ...]] = FAULTS[:10]" in text and len(APP_FAULTS) == 10


def test_models_and_limits_match_harness():
    import yaml

    providers = (HARNESS / "scheduler" / "providers.py").read_text()
    for m in MODELS.values():
        assert f'"{m.id}"' in providers and f'"{m.litellm}"' in providers
    caps = yaml.safe_load((HARNESS / "scheduler" / "ratelimits.yaml").read_text())["providers"]
    for name, lim in LIMITS.items():
        assert (caps[name]["rpm"], caps[name]["rpd"], caps[name]["tpm"], caps[name]["tpd"]) == (
            lim.rpm,
            lim.rpd,
            lim.tpm,
            lim.tpd,
        )
