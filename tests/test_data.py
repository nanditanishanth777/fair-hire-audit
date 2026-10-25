import numpy as np

from fairhire.data import (
    COLLEGES, BiasConfig, historical_labels, random_candidate, render,
    with_gender, with_name_group, with_tier,
)


def _cands(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return [random_candidate(rng) for _ in range(n)]


def test_generation_and_render_are_deterministic():
    assert [render(c) for c in _cands(seed=3)] == [render(c) for c in _cands(seed=3)]


def test_variants_never_change_merit():
    for c in _cands():
        for v in (with_gender(c, "female"), with_gender(c, "male"),
                  with_name_group(c, "A"), with_name_group(c, "B"),
                  with_tier(c, 1), with_tier(c, 3)):
            assert v.merit == c.merit
            assert (v.years, v.skills, v.certs, v.projects) == (c.years, c.skills, c.certs, c.projects)


def test_name_group_swap_keeps_gender_and_college():
    for c in _cands():
        v = with_name_group(c, "B" if c.name_group == "A" else "A")
        assert v.gender == c.gender and v.college == c.college
        assert v.first != c.first


def test_gender_swap_keeps_name_group_and_college():
    for c in _cands():
        v = with_gender(c, "male" if c.gender == "female" else "female")
        assert v.name_group == c.name_group and v.college == c.college


def test_tier_swap_changes_only_the_education_line():
    for c in _cands():
        v = with_tier(c, 3 if c.tier != 3 else 1)
        assert v.college in COLLEGES[v.tier]
        a, b = render(c).splitlines(), render(v).splitlines()
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        assert len(a) == len(b) and len(diff) == 1 and a[diff[0]].startswith("Education:")


def test_historical_labels_carry_the_injected_bias():
    rng = np.random.default_rng(1)
    cands = [random_candidate(rng) for _ in range(6000)]
    y = historical_labels(cands, BiasConfig(), rng)
    male = np.array([c.gender == "male" for c in cands])
    tier1 = np.array([c.tier == 1 for c in cands])
    tier3 = np.array([c.tier == 3 for c in cands])
    assert y[male].mean() > y[~male].mean()
    assert y[tier1].mean() > y[tier3].mean()
