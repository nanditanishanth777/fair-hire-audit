"""Fairness metrics for counterfactual (paired) audits."""
from __future__ import annotations

import numpy as np


def rank_in_pool(scores, pool_scores) -> np.ndarray:
    """Rank (1 = best) each score would take among a fixed reference pool.

    Rank = 1 + number of pool members scoring strictly higher, so equal scores
    always get equal ranks.
    """
    pool = np.sort(np.asarray(pool_scores, dtype=float))
    s = np.asarray(scores, dtype=float)
    return 1 + (len(pool) - np.searchsorted(pool, s, side="right"))


def rank_change_mask(a, b, pool_scores) -> np.ndarray:
    """True where the two variants of a pair land on different ranks."""
    return rank_in_pool(a, pool_scores) != rank_in_pool(b, pool_scores)


def rank_shift_pct(a, b, pool_scores) -> np.ndarray:
    """Absolute rank difference as a percentage of the pool size."""
    diff = np.abs(rank_in_pool(a, pool_scores) - rank_in_pool(b, pool_scores))
    return 100.0 * diff / (len(pool_scores) + 1)


def shortlist_flip_mask(a, b, pool_scores, frac: float) -> np.ndarray:
    """True where exactly one variant clears the top-``frac`` pool threshold."""
    thr = np.quantile(np.asarray(pool_scores, dtype=float), 1 - frac)
    return (np.asarray(a) >= thr) != (np.asarray(b) >= thr)


def selection_rates(scores_by_group: dict, frac: float, idx=None) -> dict:
    """Counterfactual selection rate per group.

    Every base resume is scored under every group, the top ``frac`` of all those
    renderings is selected, and a group's rate is the share of its renderings
    selected. Merit is identical across groups by construction, so any gap is
    attributable to the identity cue alone.
    """
    arrays = {
        g: (np.asarray(s, dtype=float) if idx is None else np.asarray(s, dtype=float)[idx])
        for g, s in scores_by_group.items()
    }
    thr = np.quantile(np.concatenate(list(arrays.values())), 1 - frac)
    return {g: float(np.mean(v >= thr)) for g, v in arrays.items()}


def adverse_impact_ratio(rates: dict) -> float:
    """Lowest group selection rate divided by the highest (nan if nobody is selected)."""
    hi, lo = max(rates.values()), min(rates.values())
    return float("nan") if hi == 0 else lo / hi


def passes_four_fifths(ratio: float, threshold: float = 0.8) -> bool:
    return bool(ratio >= threshold)  # nan compares False


def bootstrap_ci(stat_fn, n: int, n_boot: int, rng: np.random.Generator, alpha: float = 0.05):
    """Percentile bootstrap over base resumes. ``stat_fn`` takes an index array."""
    if n_boot <= 0:
        return float("nan"), float("nan")
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = stat_fn(rng.integers(0, n, size=n))
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
