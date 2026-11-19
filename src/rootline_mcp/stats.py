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


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p-value for the 2x2 table [[a, b], [c, d]]."""
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    if n == 0 or row1 == 0 or row2 == 0 or col1 in (0, n):
        return 1.0

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    total = sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= observed * (1 + 1e-7))
    return round(min(1.0, total), 4)
