"""Model-vs-reference comparisons. See DESIGN.md §6.

Every successfully scored model is compared against the first successfully
scored model (the reference, "model 0" in DESIGN's terms — not necessarily
`models[0]` if that one failed to load). Comparisons are paired per post
over the same eligible-post arrays `model_metrics.compute_model_summary`
already produced, so no metric is recomputed here.
"""

from __future__ import annotations

from biascompare.rng import DELTA_DIFF_CI, DIRECTION_DIFF_CI, derive_rng
from biascompare.stats import bootstrap_ci, mean_or_nan, wilcoxon_two_sided
from biascompare.verdicts import comparison_verdict


def build_comparison(
    reference_label: str,
    reference_deltas: list[float],
    reference_directions: list[float],
    model_label: str,
    model_deltas: list[float],
    model_directions: list[float],
    seed: int,
    comparison_index: int,
    alpha: float,
    n: int,
) -> dict:
    """Build one `comparisons[]` entry (DESIGN §7) for `model_label` against
    `reference_label`. The two delta/direction arrays must already be paired
    by post (same eligible posts, same order) — `compare_models` guarantees
    this since every model is scored over the identical, shared plan."""
    delta_diff = [k - r for k, r in zip(model_deltas, reference_deltas)]
    direction_diff = [k - r for k, r in zip(model_directions, reference_directions)]

    delta_summary = _diff_summary(delta_diff, derive_rng(seed, DELTA_DIFF_CI, comparison_index))
    direction_summary = _diff_summary(direction_diff, derive_rng(seed, DIRECTION_DIFF_CI, comparison_index))

    return {
        "reference": reference_label,
        "model": model_label,
        "delta_diff": delta_summary,
        "direction_diff": direction_summary,
        "verdict": comparison_verdict(reference_label, model_label, delta_summary, direction_summary, alpha, n),
    }


def _diff_summary(diff: list[float], rng) -> dict:
    return {
        "mean": mean_or_nan(diff),
        "ci": list(bootstrap_ci(diff, rng)),
        "p": wilcoxon_two_sided(diff),
    }
