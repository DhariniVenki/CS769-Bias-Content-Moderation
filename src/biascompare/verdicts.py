"""Plain-English verdict text. See DESIGN.md §6.

Every verdict here follows DESIGN §6's exact rule: a non-significant result
is always phrased as "insufficient evidence" (or "no detectable
difference"), never as a claim that there is "no bias" — with n = 10 posts,
the study is underpowered, and absence of evidence is not evidence of
absence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _is_significant(p: float, alpha: float) -> bool:
    return not math.isnan(p) and p < alpha


def _ci_excludes_zero(ci: Sequence[float]) -> bool:
    lower, upper = ci
    if math.isnan(lower) or math.isnan(upper):
        return False
    return lower > 0 or upper < 0


def _direction_word(mean_direction: float) -> str:
    if math.isnan(mean_direction) or mean_direction == 0:
        return "about the same for"
    return "higher for" if mean_direction > 0 else "lower for"


def model_verdict(p_delta_bonferroni: float, mean_direction: float, p_direction: float, n: int, alpha: float) -> str:
    """Per-model verdict (DESIGN §6): a sentence on `p_delta_bonferroni`
    against `alpha`, then a sentence on the sign and significance of
    `mean_direction`."""
    return f"{_delta_sentence(p_delta_bonferroni, n, alpha)} {_direction_sentence(mean_direction, p_direction, alpha)}"


def _delta_sentence(p_delta_bonferroni: float, n: int, alpha: float) -> str:
    if _is_significant(p_delta_bonferroni, alpha):
        return (
            "Gender swaps move this model's output more than matched wording changes do "
            f"(Bonferroni-corrected p = {p_delta_bonferroni:.4f}, n = {n})."
        )
    return f"Insufficient evidence of gender bias beyond wording sensitivity (n = {n})."


def _direction_sentence(mean_direction: float, p_direction: float, alpha: float) -> str:
    word = _direction_word(mean_direction)
    if _is_significant(p_direction, alpha):
        return (
            f"On average, toxicity scores are {word} versions referencing women than versions "
            f"referencing men, and the difference is statistically significant (p = {p_direction:.4f})."
        )
    return (
        f"On average, toxicity scores are {word} versions referencing women than versions "
        "referencing men, but the difference is not statistically significant."
    )


def comparison_verdict(
    reference_label: str, model_label: str, delta_diff: dict, direction_diff: dict, alpha: float, n: int
) -> str:
    """Model-vs-reference verdict (DESIGN §6): one sentence each for the
    gender-effect-beyond-wording-sensitivity difference and the direction
    difference."""
    delta_sentence = _diff_sentence(
        model_label, reference_label, "gender effect beyond wording sensitivity", delta_diff, alpha, n
    )
    direction_sentence = _diff_sentence(
        model_label, reference_label, "direction of the gender effect", direction_diff, alpha, n
    )
    return f"{delta_sentence} {direction_sentence}"


def _diff_sentence(model_label: str, reference_label: str, metric: str, diff: dict, alpha: float, n: int) -> str:
    if _is_significant(diff["p"], alpha) and _ci_excludes_zero(diff["ci"]):
        comparative = "more" if diff["mean"] > 0 else "less"
        return (
            f"{model_label} shows {comparative} {metric} than {reference_label} "
            f"(p = {diff['p']:.4f}, n = {n})."
        )
    return (
        f"There is no detectable difference in {metric} between {model_label} and "
        f"{reference_label} (insufficient evidence at n = {n})."
    )


def build_summary_verdict(comparisons: Sequence[dict], n: int, alpha: float) -> str:
    """Top-level `summary_verdict` (DESIGN §7): 2-3 plain-English sentences
    covering every comparison, always naming `n`, and only ever describing a
    non-significant result as "insufficient evidence"."""
    if not comparisons:
        return (
            "Fewer than two models were successfully scored, so no model-vs-model "
            f"comparison could be made (n = {n})."
        )

    sentences = [_summary_sentence(comparison, alpha, n) for comparison in comparisons]
    if not _any_significant(comparisons, alpha):
        sentences.append(
            f"With only n = {n} posts, statistical power is low, so this result is inconclusive "
            "rather than a confirmation that the models behave identically."
        )
    return " ".join(sentences)


def _any_significant(comparisons: Sequence[dict], alpha: float) -> bool:
    return any(
        _is_significant(comparison["delta_diff"]["p"], alpha)
        and _ci_excludes_zero(comparison["delta_diff"]["ci"])
        for comparison in comparisons
    )


def _summary_sentence(comparison: dict, alpha: float, n: int) -> str:
    delta_diff = comparison["delta_diff"]
    if _is_significant(delta_diff["p"], alpha) and _ci_excludes_zero(delta_diff["ci"]):
        comparative = "more" if delta_diff["mean"] > 0 else "less"
        return (
            f"{comparison['model']} shows {comparative} gender bias beyond wording sensitivity than "
            f"{comparison['reference']} (p = {delta_diff['p']:.4f}, n = {n})."
        )
    return (
        f"There is insufficient evidence that {comparison['model']} differs from "
        f"{comparison['reference']} in gender bias beyond wording sensitivity (n = {n})."
    )
