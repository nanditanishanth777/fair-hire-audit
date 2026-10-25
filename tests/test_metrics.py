import math

import numpy as np

from fairhire import metrics as M


def test_rank_in_pool_counts_strictly_higher():
    pool = [0.1, 0.5, 0.9]
    assert list(M.rank_in_pool([0.95, 0.7, 0.5, 0.05], pool)) == [1, 2, 2, 4]


def test_rank_change_mask_and_shift():
    pool = [0.1, 0.5, 0.9]
    a, b = np.array([0.7, 0.7]), np.array([0.7, 0.95])
    assert list(M.rank_change_mask(a, b, pool)) == [False, True]
    assert list(M.rank_shift_pct(a, b, pool)) == [0.0, 25.0]  # one rank of (3 + 1)


def test_equal_scores_never_change_rank():
    pool = np.random.default_rng(0).random(200)
    s = np.random.default_rng(1).random(50)
    assert not M.rank_change_mask(s, s.copy(), pool).any()


def test_shortlist_flip():
    pool = np.arange(100) / 100            # top 25% threshold = 0.75
    a, b = np.array([0.80, 0.10]), np.array([0.70, 0.20])
    assert list(M.shortlist_flip_mask(a, b, pool, 0.25)) == [True, False]


def test_selection_rates_and_ratio():
    even = M.selection_rates({"g1": [1, 2, 5, 6], "g2": [3, 4, 7, 8]}, 0.5)
    assert even == {"g1": 0.5, "g2": 0.5}
    assert M.adverse_impact_ratio(even) == 1.0

    skew = M.selection_rates({"g1": [1, 2, 3, 8], "g2": [4, 5, 6, 7]}, 0.5)
    assert skew == {"g1": 0.25, "g2": 0.75}
    assert math.isclose(M.adverse_impact_ratio(skew), 1 / 3)


def test_selection_rates_respect_bootstrap_index():
    groups = {"g1": [1, 2, 3, 8], "g2": [4, 5, 6, 7]}
    rates = M.selection_rates(groups, 0.5, idx=np.array([0, 0, 3, 3]))
    assert set(rates) == {"g1", "g2"}


def test_four_fifths_rule():
    assert M.passes_four_fifths(0.8) and M.passes_four_fifths(1.0)
    assert not M.passes_four_fifths(0.79)
    assert not M.passes_four_fifths(float("nan"))


def test_ratio_is_nan_when_nobody_selected():
    assert math.isnan(M.adverse_impact_ratio({"a": 0.0, "b": 0.0}))


def test_bootstrap_ci_brackets_the_mean():
    x = np.random.default_rng(2).normal(10, 1, 400)
    lo, hi = M.bootstrap_ci(lambda i: x[i].mean(), len(x), 200, np.random.default_rng(3))
    assert lo < x.mean() < hi
    assert math.isnan(M.bootstrap_ci(lambda i: 0.0, 10, 0, np.random.default_rng(0))[0])
