import numpy as np
import pytest

from fairhire.data import (
    FIRST_NAMES, LAST_NAMES, random_candidate, render,
    with_gender, with_name_group, with_tier,
)
from fairhire.redact import LEVELS, redact


def _cands(n=30, seed=5):
    rng = np.random.default_rng(seed)
    return [random_candidate(rng) for _ in range(n)]


def test_none_is_identity():
    t = render(_cands(1)[0])
    assert redact(t, "none") == t


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        redact("x", "everything")


def test_names_removed_from_header_and_body_but_content_kept():
    for c in _cands():
        out = redact(render(c), "names")
        assert c.first not in out and c.last not in out
        assert "Name: [CANDIDATE]" in out
        assert all(s in out for s in c.skills)


def test_gender_cues_removed():
    out = redact("Name: Ms. Emily Walsh\nShe is great. Her work and his notes. Mr Smith, Mrs Lee.", "names+gender")
    for word in ("She", "Her", "his", "Mr", "Mrs", "Ms."):
        assert word not in out
    assert "they is great" in out and "their work" in out


def test_college_removed_only_at_college_level():
    c = _cands(1)[0]
    assert c.college in redact(render(c), "names+gender")
    assert c.college not in redact(render(c), "names+gender+college")
    assert "[INSTITUTION]" in redact(render(c), "names+gender+college")


def test_redaction_is_idempotent():
    for level in LEVELS:
        t = render(_cands(1)[0])
        assert redact(redact(t, level), level) == redact(t, level)


def test_lexicon_covers_every_name_the_generator_can_emit():
    for c in _cands(200, seed=9):
        out = redact(render(c), "names")
        for names in list(FIRST_NAMES.values()) + [LAST_NAMES["A"], LAST_NAMES["B"]]:
            assert not any(n in out for n in names)


# The key invariant: at the matching level, a counterfactual pair is textually identical.
def test_name_pairs_identical_after_name_redaction():
    for c in _cands():
        a, b = render(with_name_group(c, "A")), render(with_name_group(c, "B"))
        assert a != b
        assert redact(a, "names") == redact(b, "names")


def test_gender_pairs_need_gender_redaction_not_just_names():
    for c in _cands():
        a, b = render(with_gender(c, "female")), render(with_gender(c, "male"))
        assert redact(a, "names") != redact(b, "names")           # pronouns still leak
        assert redact(a, "names+gender") == redact(b, "names+gender")


def test_college_pairs_need_college_redaction():
    for c in _cands():
        a, b = render(with_tier(c, 1)), render(with_tier(c, 3))
        assert redact(a, "names+gender") != redact(b, "names+gender")
        assert redact(a, "names+gender+college") == redact(b, "names+gender+college")
