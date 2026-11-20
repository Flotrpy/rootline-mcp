"""What the benchmark contains: faults, conditions, models and free-tier caps.

Mirrors harness/rootline_harness/{catalog.py, scheduler/providers.py,
scheduler/ratelimits.yaml} in the Rootline repository (v0.1.0). Keep them in
step when the harness changes; tests/test_catalog.py lists what to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

CONDITIONS: Final[tuple[str, ...]] = ("metrics", "traces", "logs", "combined")
PERTURBATIONS: Final[tuple[str, ...]] = ("p0", "p1")

APP_FAULTS: Final[tuple[str, ...]] = (
    "paymentFailure",
    "paymentUnreachable",
    "productCatalogFailure",
    "productCatalogLockContention",
    "emailMemoryLeak",
    "recommendationCacheFailure",
    "kafkaQueueProblems",
    "intlShippingSlowdown",
    "imageSlowLoad",
    "adHighCpu",
)
NET_FAULTS: Final[tuple[str, ...]] = ("toxiLatency", "netemLoss", "toxiReset", "toxiBandwidth")
HW_FAULTS: Final[tuple[str, ...]] = ("hwWeakSignal", "hwContention", "hwWifiOutage")
FAULTS: Final[tuple[str, ...]] = (*APP_FAULTS, *NET_FAULTS, "healthy")

INCIDENT_RE: Final = re.compile(r"^(?P<fault>[A-Za-z]+)-r(?P<replica>\d+)$")


def fault_of(incident_id: str) -> str:
    m = INCIDENT_RE.match(incident_id)
    return m.group("fault") if m else incident_id


# Faults on a link between two services (mirrors the controller catalog). The official
# label names the downstream end; the exploratory "either_end" rule also accepts the
# caller, with the same category.
EDGES: Final[dict[str, tuple[tuple[str, str], str]]] = {
    "paymentUnreachable": (("checkout", "payment"), "dependency_unavailable"),
    "toxiLatency": (("checkout", "payment"), "network_latency"),
    "toxiReset": (("checkout", "payment"), "network_reset"),
    "netemLoss": (("frontend", "product-catalog"), "network_loss"),
    "toxiBandwidth": (("frontend", "product-catalog"), "network_throughput"),
    "hwWeakSignal": (("checkout", "payment"), "network_loss"),
    "hwContention": (("checkout", "payment"), "network_latency"),
    "hwWifiOutage": (("checkout", "payment"), "network_reset"),
}


def group_of(fault: str) -> str:
    """application, network, hardware, healthy or unknown."""
    if fault in APP_FAULTS:
        return "application"
    if fault in NET_FAULTS:
        return "network"
    if fault in HW_FAULTS:
        return "hardware"
    if fault == "healthy":
        return "healthy"
    return "unknown"


def valid_incident(incident_id: str) -> bool:
    return bool(INCIDENT_RE.match(incident_id))


@dataclass(frozen=True)
class Limits:
    rpm: int | None
    rpd: int | None
    tpm: int | None
    tpd: int | None


# Free-tier caps as the harness sets them (a little below the published figures).
LIMITS: Final[dict[str, Limits]] = {
    "cerebras": Limits(rpm=4, rpd=None, tpm=25_000, tpd=900_000),
    "groq": Limits(rpm=30, rpd=950, tpm=8_000, tpd=195_000),
    "gemini": Limits(rpm=10, rpd=250, tpm=250_000, tpd=None),
    "openrouter": Limits(rpm=20, rpd=50, tpm=None, tpd=None),
    "ollama": Limits(rpm=None, rpd=None, tpm=None, tpd=None),
    "vllm": Limits(rpm=None, rpd=None, tpm=None, tpd=None),
}

# Providers that need a machine with the model on it; GitHub's runners can't serve them.
LOCAL_ONLY_PROVIDERS: Final[frozenset[str]] = frozenset({"ollama", "vllm"})

PROMPT_TOKENS: Final = 6_000  # every evidence pack is cut to this budget


@dataclass(frozen=True)
class Model:
    id: str
    litellm: str
    tier: str
    provider: str
    max_tokens: int = 2048

    @property
    def tokens_per_job(self) -> int:
        """Upper bound the scheduler reserves for one request."""
        return PROMPT_TOKENS + self.max_tokens

    @property
    def runs_on(self) -> str:
        return "local" if self.provider in LOCAL_ONLY_PROVIDERS else "cloud"


MODELS: Final[dict[str, Model]] = {
    m.id: m
    for m in (
        Model("gpt-oss-120b@cerebras", "cerebras/gpt-oss-120b", "C", "cerebras"),
        Model("gpt-oss-120b@groq", "groq/openai/gpt-oss-120b", "C", "groq", max_tokens=1536),
        Model("qwen3-8b-q4@ollama", "ollama_chat/qwen3:8b", "B", "ollama"),
        Model("gemma3-4b-q4@ollama", "ollama_chat/gemma3:4b", "A", "ollama"),
    )
}


def get_model(model_id: str) -> Model:
    try:
        return MODELS[model_id]
    except KeyError:
        raise ValueError(
            f"unknown model {model_id!r}; known: {', '.join(sorted(MODELS))}"
        ) from None


def jobs_per_day(model: Model) -> int | None:
    """How many jobs the free tier allows in 24 hours, or None for no cap."""
    lim = LIMITS.get(model.provider)
    if lim is None:
        return None
    bounds = []
    if lim.rpd is not None:
        bounds.append(lim.rpd)
    if lim.tpd is not None:
        bounds.append(lim.tpd // model.tokens_per_job)
    return min(bounds) if bounds else None


def jobs_per_hour(model: Model) -> int | None:
    lim = LIMITS.get(model.provider)
    if lim is None:
        return None
    bounds = []
    if lim.rpm is not None:
        bounds.append(lim.rpm * 60)
    if lim.tpm is not None:
        bounds.append((lim.tpm // model.tokens_per_job) * 60)
    return min(bounds) if bounds else None


def estimate_days(model: Model, jobs: int) -> float | None:
    """Days of free quota needed for `jobs` jobs (None when nothing caps it)."""
    if jobs <= 0:
        return 0.0
    per_day = jobs_per_day(model)
    per_hour = jobs_per_hour(model)
    if per_hour is not None:
        per_day = min(per_day, per_hour * 24) if per_day is not None else per_hour * 24
    if not per_day:
        return None
    return round(jobs / per_day, 1)
