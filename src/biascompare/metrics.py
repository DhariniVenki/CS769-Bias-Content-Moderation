"""Per-sample divergence and label metrics. See DESIGN.md §6.

`ProbPair` is `(P(non-toxic), P(toxic))`, the shape every `Scorer` returns
per text (see `biascompare.scoring.base`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

ProbPair = tuple[float, float]

_EPS = 1e-12


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence between two discrete distributions, base 2.

    `p` and `q` are eps-clipped and renormalised before computing, so a zero
    probability never causes a log(0). Base-2 JSD is mathematically bounded
    in [0, 1]; the result is clamped to that range to absorb floating-point
    error (e.g. a tiny negative value from cancellation when p == q).
    """
    p_norm = _clip_and_normalise(p)
    q_norm = _clip_and_normalise(q)
    if len(p_norm) != len(q_norm):
        raise ValueError(
            f"js_divergence: p and q must have the same length, got {len(p_norm)} and {len(q_norm)}"
        )

    m = [(pi + qi) / 2.0 for pi, qi in zip(p_norm, q_norm)]
    divergence = 0.5 * _kl_divergence(p_norm, m) + 0.5 * _kl_divergence(q_norm, m)
    return max(0.0, min(1.0, divergence))


def _clip_and_normalise(dist: Sequence[float]) -> list[float]:
    if len(dist) == 0:
        raise ValueError("js_divergence: distributions must be non-empty")
    clipped = [max(_EPS, float(x)) for x in dist]
    total = sum(clipped)
    return [x / total for x in clipped]


def _kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q))


def predicted_label(p: ProbPair, threshold: float = 0.5) -> int:
    """1 if P(toxic) = p[1] >= threshold, else 0."""
    return 1 if p[1] >= threshold else 0


def flipped(p: ProbPair, q: ProbPair, threshold: float = 0.5) -> bool:
    """True if the predicted label of `p` differs from that of `q`."""
    return predicted_label(p, threshold) != predicted_label(q, threshold)
