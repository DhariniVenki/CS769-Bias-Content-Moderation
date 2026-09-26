"""Shared word-matching and case-preservation helpers.

Kept internal to the `perturb` package (not part of the public API) since both
`gender.py` and `controls.py` need the same word regex and the same
capitalisation rule when substituting a word or phrase into a sentence.
"""
from __future__ import annotations

import re

# A "word" is a maximal run of ASCII letters. This deliberately splits
# contractions ("he's" -> "he" + "s"), which is what lets a swap on "he"
# naturally leave "'s" untouched and produce "she's".
WORD_RE = re.compile(r"[A-Za-z]+")


def match_case(source: str, target: str) -> str:
    """Apply the capitalisation pattern of `source` (a single matched word)
    to `target` (its replacement word or phrase).

    - ALL-CAPS source (len > 1) -> ALL-CAPS target.
    - Capitalized source ("He") -> capitalize target's first letter only.
    - Otherwise -> target unchanged (expected to already be lowercase).
    """
    if len(source) > 1 and source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target
