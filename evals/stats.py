"""stats - intervals and paired tests, with no bare percentages anywhere.

Wilson for proportions, percentile bootstrap for means, paired bootstrap for
arm deltas, exact McNemar for paired binary outcomes, Cohen's kappa for judge
calibration. Pure standard library so every number is reproducible.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    def as_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "n": self.n,
        }


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Handles 0/n and n/n."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    low = max(0.0, centre - spread)
    high = min(1.0, centre + spread)
    # exact bounds, so 0/n and n/n report 0.0 and 1.0 rather than 0.9999...
    if successes <= 0:
        low = 0.0
    if successes >= n:
        high = 1.0
    return (low, high)


def proportion_interval(successes: int, n: int) -> Interval:
    low, high = wilson_interval(successes, n)
    return Interval(mean=successes / n if n else 0.0, low=low, high=high, n=n)


def bootstrap_mean_ci(
    values: list[float], *, resamples: int = 2000, seed: int = 13
) -> Interval:
    """Percentile bootstrap CI for a mean. Deterministic for a given seed."""
    if not values:
        return Interval(mean=0.0, low=0.0, high=0.0, n=0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low = means[int(0.025 * (resamples - 1))]
    high = means[int(0.975 * (resamples - 1))]
    return Interval(mean=sum(values) / n, low=low, high=high, n=n)


def paired_bootstrap_delta(
    pairs: list[tuple[float, float]], *, resamples: int = 2000, seed: int = 13
) -> dict:
    """Mean paired difference (B - A) with a bootstrap CI and P(delta > 0)."""
    if not pairs:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "p_positive": 0.0}
    rng = random.Random(seed)
    deltas = [b - a for a, b in pairs]
    n = len(deltas)
    samples = []
    for _ in range(resamples):
        drawn = [deltas[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(drawn) / n)
    samples.sort()
    return {
        "delta": sum(deltas) / n,
        "ci_low": samples[int(0.025 * (resamples - 1))],
        "ci_high": samples[int(0.975 * (resamples - 1))],
        "n": n,
        "p_positive": sum(1 for s in samples if s > 0) / resamples,
    }


def mcnemar_exact(b: int, c: int) -> dict:
    """Two-sided exact McNemar for discordant pairs ``b`` and ``c``."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return {"b": b, "c": c, "p_value": min(1.0, 2 * tail)}


def cohens_kappa(labels_a: list, labels_b: list) -> dict:
    """Cohen's kappa for two raters over the same items."""
    if len(labels_a) != len(labels_b):
        raise ValueError("kappa needs two label lists of equal length")
    n = len(labels_a)
    if n == 0:
        return {"kappa": 0.0, "n": 0, "observed_agreement": 0.0, "expected_agreement": 0.0}
    categories = sorted({*labels_a, *labels_b}, key=str)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = 0.0
    for category in categories:
        p_a = sum(1 for a in labels_a if a == category) / n
        p_b = sum(1 for b in labels_b if b == category) / n
        expected += p_a * p_b
    kappa = 0.0 if expected >= 1.0 else (observed - expected) / (1 - expected)
    return {
        "kappa": kappa,
        "n": n,
        "observed_agreement": observed,
        "expected_agreement": expected,
    }
