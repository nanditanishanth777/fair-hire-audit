import pytest

from fairhire.audit import AuditConfig, run_audit
from fairhire.report import make_bullets


@pytest.fixture(scope="module")
def result():
    cfg = AuditConfig(n_train=3000, n_base=200, n_pool=200, n_test=500, n_boot=20, seed=11)
    return run_audit(cfg)


def _dim(result, level, dim):
    df = result.per_dimension
    return df[(df.level == level) & (df.dimension == dim)].iloc[0]


def test_biased_matcher_changes_ranks_and_fails_four_fifths(result):
    base = result.overall.set_index("level").loc["none"]
    assert base.rank_change_pct > 50
    assert base.worst_selection_rate_ratio < 0.8
    assert base.n_pairs == 3 * 200


def test_name_redaction_zeroes_the_name_dimension_only(result):
    assert _dim(result, "names", "name").rank_change_pct == 0
    assert _dim(result, "names", "name").mean_abs_score_gap == 0
    assert _dim(result, "names", "college").rank_change_pct > 0


def test_full_redaction_removes_all_rank_changes(result):
    full = result.overall.set_index("level").loc["names+gender+college"]
    assert full.rank_change_pct == 0
    assert full.worst_selection_rate_ratio == 1.0


def test_redaction_never_increases_rank_changes(result):
    pct = result.overall.set_index("level").rank_change_pct
    assert pct["none"] >= pct["names"] >= pct["names+gender"] >= pct["names+gender+college"]


def test_matcher_stays_useful_after_redaction(result):
    auc = result.overall.set_index("level").auc_vs_merit
    assert auc["names+gender+college"] > 0.8


def test_bullets_are_filled_in(result):
    bullets, values = make_bullets(result, "names")
    assert len(bullets) == 2 and "[" not in "".join(bullets)
    assert values["N"] == 600
