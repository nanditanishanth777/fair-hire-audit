"""Run the counterfactual audit across redaction levels."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import metrics as M
from .data import (
    BiasConfig, historical_labels, identity_tokens, make_variant, random_candidate, render,
)
from .matcher import ResumeMatcher
from .redact import LEVELS, redact

# The pair compared for the rank-change metric, and every group used for selection rates.
PAIR_GROUPS = {"gender": ("female", "male"), "name": ("A", "B"), "college": (1, 3)}
ALL_GROUPS = {"gender": ("female", "male"), "name": ("A", "B"), "college": (1, 2, 3)}


@dataclass
class AuditConfig:
    n_train: int = 6000      # historical decisions the matcher learns from
    n_base: int = 1000       # base resumes; each yields one pair per dimension
    n_pool: int = 500        # reference pool used to turn scores into ranks
    n_test: int = 3000       # held-out resumes for the utility (AUC) check
    select_frac: float = 0.25
    n_boot: int = 200
    seed: int = 7
    bias: BiasConfig = field(default_factory=BiasConfig)


@dataclass
class AuditResult:
    config: AuditConfig
    per_dimension: pd.DataFrame
    overall: pd.DataFrame
    identity_weights: pd.DataFrame


def run_audit(cfg: AuditConfig | None = None, matcher=None) -> AuditResult:
    """Audit ``matcher`` (anything with ``score(texts)``); trains the reference one if None."""
    cfg = cfg or AuditConfig()
    rng = np.random.default_rng(cfg.seed)
    boot_rng = np.random.default_rng(cfg.seed + 1)

    train = [random_candidate(rng) for _ in range(cfg.n_train)]
    y_hist = historical_labels(train, cfg.bias, rng)
    if matcher is None:
        matcher = ResumeMatcher().fit([render(c) for c in train], y_hist)

    pool_texts = [render(random_candidate(rng)) for _ in range(cfg.n_pool)]
    test = [random_candidate(rng) for _ in range(cfg.n_test)]
    test_texts = [render(c) for c in test]
    merit = np.array([c.merit for c in test])
    qualified = (merit >= np.median(merit)).astype(int)  # ground truth independent of identity

    base = [random_candidate(rng) for _ in range(cfg.n_base)]
    variant_texts = {
        (dim, g): [render(make_variant(c, dim, g)) for c in base]
        for dim, groups in ALL_GROUPS.items()
        for g in groups
    }

    n = cfg.n_base
    dim_rows, overall_rows = [], []
    for level in LEVELS:
        def score(texts, level=level):
            return np.asarray(matcher.score([redact(t, level) for t in texts]), dtype=float)

        pool = score(pool_texts)
        scores = {key: score(texts) for key, texts in variant_texts.items()}
        by_group = {dim: {g: scores[(dim, g)] for g in groups} for dim, groups in ALL_GROUPS.items()}

        changed = np.zeros((n, len(PAIR_GROUPS)), dtype=bool)
        flipped = np.zeros_like(changed)
        shifts = np.zeros((n, len(PAIR_GROUPS)))
        airs = {}

        for k, (dim, (g0, g1)) in enumerate(PAIR_GROUPS.items()):
            a, b = scores[(dim, g0)], scores[(dim, g1)]
            changed[:, k] = M.rank_change_mask(a, b, pool)
            flipped[:, k] = M.shortlist_flip_mask(a, b, pool, cfg.select_frac)
            shifts[:, k] = M.rank_shift_pct(a, b, pool)

            groups = by_group[dim]
            rates = M.selection_rates(groups, cfg.select_frac)
            air = M.adverse_impact_ratio(rates)
            airs[dim] = air
            rc_lo, rc_hi = M.bootstrap_ci(lambda i, k=k: 100 * changed[i, k].mean(), n, cfg.n_boot, boot_rng)
            air_lo, air_hi = M.bootstrap_ci(
                lambda i, groups=groups: M.adverse_impact_ratio(M.selection_rates(groups, cfg.select_frac, i)),
                n, cfg.n_boot, boot_rng,
            )
            dim_rows.append({
                "level": level,
                "dimension": dim,
                "n_pairs": n,
                "rank_change_pct": 100 * changed[:, k].mean(),
                "rank_change_ci_lo": rc_lo,
                "rank_change_ci_hi": rc_hi,
                "mean_abs_rank_shift_pct": shifts[:, k].mean(),
                "shortlist_flip_pct": 100 * flipped[:, k].mean(),
                "mean_abs_score_gap": float(np.abs(a - b).mean()),
                "selection_rates": json.dumps({str(g): round(r, 4) for g, r in rates.items()}),
                "selection_rate_ratio": air,
                "ratio_ci_lo": air_lo,
                "ratio_ci_hi": air_hi,
                "passes_four_fifths": M.passes_four_fifths(air),
            })

        def worst_ratio(i):
            return np.nanmin([
                M.adverse_impact_ratio(M.selection_rates(by_group[d], cfg.select_frac, i))
                for d in ALL_GROUPS
            ])

        worst = float(np.nanmin(list(airs.values())))
        worst_dim = min(airs, key=lambda d: np.inf if np.isnan(airs[d]) else airs[d])
        rc_lo, rc_hi = M.bootstrap_ci(lambda i: 100 * changed[i].mean(), n, cfg.n_boot, boot_rng)
        w_lo, w_hi = M.bootstrap_ci(worst_ratio, n, cfg.n_boot, boot_rng)
        overall_rows.append({
            "level": level,
            "n_pairs": int(changed.size),
            "rank_change_pct": 100 * changed.mean(),
            "rank_change_ci_lo": rc_lo,
            "rank_change_ci_hi": rc_hi,
            "mean_abs_rank_shift_pct": shifts.mean(),
            "shortlist_flip_pct": 100 * flipped.mean(),
            "worst_selection_rate_ratio": worst,
            "worst_ratio_ci_lo": w_lo,
            "worst_ratio_ci_hi": w_hi,
            "worst_dimension": worst_dim,
            "passes_four_fifths": M.passes_four_fifths(worst),
            "auc_vs_merit": float(roc_auc_score(qualified, score(test_texts))),
        })

    weights = (
        matcher.token_weights(identity_tokens())
        if hasattr(matcher, "token_weights")
        else pd.DataFrame(columns=["token", "weight"])
    )
    return AuditResult(cfg, pd.DataFrame(dim_rows), pd.DataFrame(overall_rows), weights)
