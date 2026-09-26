"""Scorer protocol and shared probability-conversion helpers. See DESIGN.md §4.

This module has no heavy dependencies (no torch/transformers) so it — and
everything that only imports it — can be imported without either installed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import numpy as np

from biascompare.inputs import ModelSpec

ProbPair = tuple[float, float]


class Scorer(Protocol):
    """A model that turns a batch of texts into `(P(non-toxic), P(toxic))` pairs."""

    name: str
    scorer_type: str
    uses_instruction: bool

    def score(self, texts: Sequence[str]) -> list[ProbPair]: ...

    def close(self) -> None: ...


ScorerFactory = Callable[[ModelSpec, str | None, str], Scorer]
"""(spec, instruction, device) -> Scorer. `hf_classifier_factory` is the default."""


def resolve_label_index(id2label: Mapping[int | str, str], toxic_label: str) -> int:
    """Find the label index matching `toxic_label`, case-insensitively.

    `id2label` keys may be `int` or `str` — Hugging Face configs are
    inconsistent about this after a JSON round-trip, since JSON object keys
    are always strings. Raises `ValueError` listing the available labels if
    no case-insensitive match is found.
    """
    target = toxic_label.strip().lower()
    for key, label in id2label.items():
        if str(label).strip().lower() == target:
            return int(key)

    available = [str(label) for label in id2label.values()]
    raise ValueError(
        f"toxic_label {toxic_label!r} not found among model labels {available}. "
        "Check the ModelSpec.toxic_label value against the model's config.id2label."
    )


def binary_from_logits(logits: Sequence[float], toxic_index: int, multi_label: bool) -> ProbPair:
    """Convert raw classification logits into a `(P(non-toxic), P(toxic))` pair.

    - `multi_label=True`: `p = sigmoid(logits[toxic_index])`; other heads
      (severe_toxic, insult, ...) are ignored.
    - `multi_label=False`: `p = softmax(logits)[toxic_index]`.

    Implemented with numpy for numerical stability (shifted softmax,
    branch-stable sigmoid).
    """
    logits_arr = np.asarray(logits, dtype=np.float64)
    if logits_arr.ndim != 1:
        raise ValueError(
            f"binary_from_logits expects a 1-D sequence of logits, got shape {logits_arr.shape}"
        )
    if not (0 <= toxic_index < logits_arr.shape[0]):
        raise ValueError(
            f"toxic_index {toxic_index} out of range for {logits_arr.shape[0]} logits"
        )

    if multi_label:
        p_toxic = _sigmoid(float(logits_arr[toxic_index]))
    else:
        p_toxic = float(_softmax(logits_arr)[toxic_index])

    return (1.0 - p_toxic, p_toxic)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)
