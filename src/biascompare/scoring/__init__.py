"""Scoring package: the `Scorer` protocol, probability helpers, and the HF
classifier scorer. See DESIGN.md §4.

Importing this package does not require torch/transformers: `classifier.py`
only imports them lazily, inside `HFClassifierScorer` methods.
"""

from __future__ import annotations

from biascompare.scoring.base import (
    ProbPair,
    Scorer,
    ScorerFactory,
    binary_from_logits,
    resolve_label_index,
)
from biascompare.scoring.classifier import HFClassifierScorer, hf_classifier_factory

__all__ = [
    "ProbPair",
    "Scorer",
    "ScorerFactory",
    "resolve_label_index",
    "binary_from_logits",
    "HFClassifierScorer",
    "hf_classifier_factory",
]
