"""Synthetic resumes with swappable identity attributes.

Every resume is rendered from a ``Candidate``. The identity attributes
(gender, name group, college tier) are drawn independently of the merit
attributes (experience, skills, certifications, projects), so a counterfactual
variant changes the identity fields and nothing else.

All names and institutions are illustrative. Edit the lists below to match the
population you want to audit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

FIRST_NAMES = {
    ("female", "A"): ["Emily", "Hannah", "Laura", "Megan", "Sarah", "Rachel"],
    ("male", "A"): ["Greg", "Matthew", "Brad", "Kyle", "Ryan", "Jacob"],
    ("female", "B"): ["Lakshmi", "Anjali", "Fatima", "Meera", "Aisha", "Divya"],
    ("male", "B"): ["Rahul", "Imran", "Arjun", "Karthik", "Omar", "Suresh"],
}
LAST_NAMES = {
    "A": ["Walsh", "Baker", "Keller", "Murphy", "Hughes", "Sullivan"],
    "B": ["Reddy", "Khan", "Nair", "Gupta", "Rao", "Siddiqui"],
}
# Fictional institutions; the tier is an assumption of this benchmark.
COLLEGES = {
    1: ["Arcadia Institute of Technology", "Meridian University", "Halcyon Institute of Science"],
    2: ["Brookfield University", "Silverlake Engineering University", "Redstone Institute of Technology"],
    3: ["Greenfield College of Engineering", "Lakeview Technical College", "Northbridge Institute"],
}
TITLES = {"female": "Ms.", "male": "Mr."}
PRONOUNS = {"female": ("She", "Her"), "male": ("He", "His")}

CORE_SKILLS = [
    "python", "sql", "pandas", "statistics", "excel",
    "tableau", "power bi", "machine learning", "a/b testing", "etl",
]
DOMAINS = [
    "retail analytics", "fintech", "healthcare analytics",
    "logistics", "e-commerce", "telecom",
]
BULLET_TEMPLATES = [
    "Built {s1} pipelines that cut weekly reporting time by {pct}%",
    "Used {s1} and {s2} to find churn drivers, lifting retention by {pct}%",
    "Automated recurring {s1} reports for {n} business teams",
    "Ran {s2} analyses that informed pricing decisions worth {pct}% of revenue",
    "Presented {s1} findings to senior leadership every quarter",
]
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


@dataclass(frozen=True)
class Candidate:
    # identity attributes (what the audit perturbs)
    gender: str
    name_group: str
    first_idx: int
    last_idx: int
    tier: int
    college_idx: int
    # merit attributes (never perturbed)
    years: int
    skills: tuple
    certs: int
    projects: int
    domain_idx: int
    bullet_ids: tuple
    pct: int

    @property
    def first(self) -> str:
        return FIRST_NAMES[(self.gender, self.name_group)][self.first_idx]

    @property
    def last(self) -> str:
        return LAST_NAMES[self.name_group][self.last_idx]

    @property
    def college(self) -> str:
        return COLLEGES[self.tier][self.college_idx]

    @property
    def merit(self) -> float:
        """Job-relevant quality in [0, 1]; independent of identity attributes."""
        return (
            0.40 * min(self.years, 10) / 10
            + 0.30 * len(self.skills) / len(CORE_SKILLS)
            + 0.15 * min(self.certs, 2) / 2
            + 0.15 * min(self.projects, 3) / 3
        )


def random_candidate(rng: np.random.Generator) -> Candidate:
    n_skills = int(rng.integers(3, len(CORE_SKILLS) + 1))
    skills = tuple(str(s) for s in rng.permutation(CORE_SKILLS)[:n_skills])
    return Candidate(
        gender=str(rng.choice(["female", "male"])),
        name_group=str(rng.choice(["A", "B"])),
        first_idx=int(rng.integers(0, 6)),
        last_idx=int(rng.integers(0, 6)),
        tier=int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])),
        college_idx=int(rng.integers(0, 3)),
        years=int(rng.integers(0, 11)),
        skills=skills,
        certs=int(rng.integers(0, 3)),
        projects=int(rng.integers(0, 4)),
        domain_idx=int(rng.integers(0, len(DOMAINS))),
        bullet_ids=tuple(int(i) for i in rng.choice(len(BULLET_TEMPLATES), size=3, replace=False)),
        pct=int(rng.integers(8, 41)),
    )


# ---- counterfactual variants: change identity fields only -------------------

def with_gender(c: Candidate, gender: str) -> Candidate:
    """Swap name, title and pronouns to the other gender; same name group."""
    return replace(c, gender=gender)


def with_name_group(c: Candidate, group: str) -> Candidate:
    """Swap to a name from another group; gender and everything else fixed."""
    return replace(c, name_group=group)


def with_tier(c: Candidate, tier: int) -> Candidate:
    """Swap to a college of another tier; everything else fixed."""
    return replace(c, tier=tier)


def make_variant(c: Candidate, dimension: str, group) -> Candidate:
    if dimension == "gender":
        return with_gender(c, group)
    if dimension == "name":
        return with_name_group(c, group)
    if dimension == "college":
        return with_tier(c, group)
    raise ValueError(f"unknown dimension: {dimension}")


def render(c: Candidate) -> str:
    pro, poss = PRONOUNS[c.gender]
    s1, s2 = c.skills[0], c.skills[1]
    n = 2 + c.pct % 6
    bullets = "\n".join(
        "- " + BULLET_TEMPLATES[i].format(s1=s1, s2=s2, pct=c.pct, n=n) for i in c.bullet_ids
    )
    return (
        f"Name: {TITLES[c.gender]} {c.first} {c.last}\n"
        f"Summary: {pro} is a data analyst with {NUM[c.years]} years of experience in "
        f"{DOMAINS[c.domain_idx]}. {poss} work turns messy data into decisions.\n"
        f"Education: B.Tech, {c.college}\n"
        f"Skills: {', '.join(c.skills)}\n"
        f"Certifications: {NUM[c.certs]}\n"
        f"Projects: {NUM[c.projects]} end-to-end analytics projects\n"
        f"Experience:\n{bullets}\n"
        f"{poss} dashboards are used weekly by the leadership team.\n"
        f"Thanks for considering my application. - {c.first} {c.last}\n"
    )


# ---- biased "historical hiring" labels --------------------------------------

@dataclass
class BiasConfig:
    """Bias injected into the historical decisions the matcher learns from.

    Gaps are in logit units. merit_slope * (merit - merit_center) is the
    legitimate signal; the three gaps are the unfair part.
    """
    merit_slope: float = 9.0
    merit_center: float = 0.55
    gender_gap: float = 0.8      # male favoured
    name_gap: float = 0.6        # name group A favoured
    tier_step: float = 0.7       # +step for tier 1, -step for tier 3


def historical_logit(c: Candidate, bias: BiasConfig) -> float:
    z = bias.merit_slope * (c.merit - bias.merit_center)
    z += bias.gender_gap / 2 * (1 if c.gender == "male" else -1)
    z += bias.name_gap / 2 * (1 if c.name_group == "A" else -1)
    z += bias.tier_step * (2 - c.tier)
    return z


def historical_labels(cands, bias: BiasConfig, rng: np.random.Generator) -> np.ndarray:
    z = np.array([historical_logit(c, bias) for c in cands])
    return (rng.random(len(cands)) < 1 / (1 + np.exp(-z))).astype(int)


def identity_tokens() -> list[str]:
    """Lower-cased tokens that carry identity information (for weight inspection)."""
    toks = {n for names in FIRST_NAMES.values() for n in names}
    toks |= {n for names in LAST_NAMES.values() for n in names}
    toks = {t.lower() for t in toks}
    toks |= {"he", "she", "his", "her", "mr", "ms"}
    toks |= {col.split()[0].lower() for cols in COLLEGES.values() for col in cols}
    return sorted(toks)
