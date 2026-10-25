"""Remove identity cues from resume text before scoring.

Levels are cumulative:
    none                  raw text
    names                 name header line and any name from the name lexicon
    names+gender          + titles (Mr/Ms/...) and gendered pronouns
    names+gender+college  + institution names

Redaction is lexicon-based on purpose: it is transparent and testable, but it
only removes what is on the lexicon. A production system would need NER and a
broader list of proxies (clubs, hobbies, locations, gaps in employment).
"""
from __future__ import annotations

import re

from .data import COLLEGES, FIRST_NAMES, LAST_NAMES

LEVELS = ("none", "names", "names+gender", "names+gender+college")

LEVEL_DESCRIPTIONS = {
    "none": "no redaction",
    "names": "removing names before scoring",
    "names+gender": "removing names and gender cues before scoring",
    "names+gender+college": "removing names, gender cues and college before scoring",
}

_NAME_WORDS = sorted(
    {n for v in FIRST_NAMES.values() for n in v} | {n for v in LAST_NAMES.values() for n in v},
    key=len,
    reverse=True,
)
_NAME_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, _NAME_WORDS)) + r")\b", re.IGNORECASE)
_HEADER_RE = re.compile(r"^Name:.*$", re.MULTILINE | re.IGNORECASE)

_PRONOUNS = {
    "he": "they", "she": "they", "his": "their", "her": "their", "hers": "theirs",
    "him": "them", "himself": "themself", "herself": "themself",
}
_PRONOUN_RE = re.compile(r"\b(?:" + "|".join(_PRONOUNS) + r")\b", re.IGNORECASE)
_TITLE_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\b\.?\s*", re.IGNORECASE)

_COLLEGE_WORDS = sorted({c for v in COLLEGES.values() for c in v}, key=len, reverse=True)
_COLLEGE_RE = re.compile("|".join(map(re.escape, _COLLEGE_WORDS)), re.IGNORECASE)


def redact_names(text: str) -> str:
    text = _HEADER_RE.sub("Name: [CANDIDATE]", text)
    return _NAME_RE.sub("[CANDIDATE]", text)


def redact_gender(text: str) -> str:
    text = _TITLE_RE.sub("", text)
    return _PRONOUN_RE.sub(lambda m: _PRONOUNS[m.group(0).lower()], text)


def redact_college(text: str) -> str:
    return _COLLEGE_RE.sub("[INSTITUTION]", text)


def redact(text: str, level: str = "names") -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown redaction level {level!r}; choose from {LEVELS}")
    if level == "none":
        return text
    text = redact_names(text)
    if level in ("names+gender", "names+gender+college"):
        text = redact_gender(text)
    if level == "names+gender+college":
        text = redact_college(text)
    return text
