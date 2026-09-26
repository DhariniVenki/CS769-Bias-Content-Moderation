"""Frozen value types shared by the perturbation modules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Change:
    """A single word-level substitution made by a perturbation."""

    original: str
    replacement: str


@dataclass(frozen=True)
class Perturbation:
    """A perturbed version of a post's text.

    `kind` is "gender" for the targeted swap or "control" for a baseline
    synonym swap.
    """

    kind: str
    text: str
    changes: tuple[Change, ...]

    @property
    def n_changed(self) -> int:
        return len(self.changes)
