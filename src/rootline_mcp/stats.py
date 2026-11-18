"""Small, dependency-free statistics: rates, bootstrap intervals, McNemar's exact test."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def rate(xs: Sequence[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def bootstrap_ci(
    xs: Sequence[bool], n_boot: int = 2000, alpha: float = 0.05, seed: int = 20261107
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for a proportion (seeded, so answers repeat)."""
    n = len(xs)
    if n == 0:
        return None
    rng = random.Random(seed)
    k = sum(xs)
    vals = [1] * k + [0] * (n - k)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return round(lo, 4), round(hi, 4)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)
