"""Per-post metrics and per-model summary statistics. See DESIGN.md §6.

`compute_per_post_metrics` turns one post's scored probability pairs into
the `per_post[]` entry from DESIGN §7. `compute_model_summary` aggregates
those entries (restricted to posts eligible for statistics, DESIGN §5) into
the `summary` object, and also returns the raw per-post delta/direction
arrays so `compare.py` can pair them up for the model-vs-reference
comparisons without recomputing anything.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from biascompare.metrics import ProbPair, flipped, js_divergence
from biascompare.perturbation_plan import PostPlan, TextLayout
from biascompare.rng import DIRECTION_CI, derive_rng
from biascompare.stats import bonferroni, bootstrap_ci, mean_or_nan, paired_ttest_greater, ttest_two_sided
from biascompare.verdicts import model_verdict

_NAN = float("nan")


def compute_per_post_metrics(
    plan: PostPlan, layout: TextLayout, post_index: int, scores: Sequence[ProbPair]
) -> dict:
    """Build one `per_post[]` entry (DESIGN §7) for `plan` from `scores`,
    the full flattened score list for this model (see `TextLayout`)."""
    p_original = scores[layout.original_index[post_index]]
    p_gender = scores[layout.gender_index[post_index]]
    p_controls = [scores[i] for i in layout.control_indices[post_index]]

    jsd_gender = js_divergence(p_original, p_gender)
    jsd_controls = [js_divergence(p_original, p_control) for p_control in p_controls]
    mean_jsd_controls = mean_or_nan(jsd_controls)
    delta = jsd_gender - mean_jsd_controls if jsd_controls else _NAN

    flip_gender = flipped(p_original, p_gender)
    control_flips = [flipped(p_original, p_control) for p_control in p_controls]
    flip_controls_rate = mean_or_nan([float(f) for f in control_flips])

    return {
        "post_id": plan.post.id,
        "p_toxic_original": p_original[1],
        "p_toxic_gender_swap": p_gender[1],
        "p_toxic_controls": [p[1] for p in p_controls],
        "jsd_gender": jsd_gender,
        "jsd_controls": jsd_controls,
        "delta": delta,
        "direction": _direction(plan.post.gender, p_original[1], p_gender[1]),
        "flip_gender": bool(flip_gender),
        "flip_controls_rate": flip_controls_rate,
    }


def _direction(gender: str, p_toxic_original: float, p_toxic_gender_swap: float) -> float:
    """P_toxic(female-referencing) - P_toxic(male-referencing) (DESIGN §6)."""
    if gender == "female":
        return p_toxic_original - p_toxic_gender_swap
    return p_toxic_gender_swap - p_toxic_original


def compute_model_summary(
    per_post: Sequence[dict],
    eligible_post_ids: set[str],
    n_success_models: int,
    seed: int,
    model_index: int,
    alpha: float,
) -> tuple[dict, list[float], list[float]]:
    """Aggregate `per_post` (restricted to eligible posts) into the
    `summary` object from DESIGN §7. Returns `(summary, deltas, directions)`
    — the eligible per-post arrays, *not* further filtered — so
    `compare.py` can pair them up by position for the model-vs-reference
    comparisons; every model shares the same `eligible_post_ids` (DESIGN
    §5), so the two arrays line up across models without recomputing
    anything.

    The summary's own hypothesis tests, however, are computed only over the
    finite subset of those arrays: `eligible_post_ids` already excludes
    posts with zero usable controls (the `no_controls` flag), but this is a
    defensive second layer so one non-finite value (e.g. from a scorer
    emitting a non-finite probability) can never NaN-poison every other
    post's contribution to `p_delta`/`p_direction`/the bootstrap CI. `n` in
    the verdict is the count actually used, not the raw eligible count.
    """
    eligible = [m for m in per_post if m["post_id"] in eligible_post_ids]

    all_control_jsds = [jsd for m in eligible for jsd in m["jsd_controls"]]
    sensitivity = mean_or_nan(all_control_jsds)
    raw_gender_effect = mean_or_nan([m["jsd_gender"] for m in eligible])

    deltas = [m["delta"] for m in eligible]
    directions = [m["direction"] for m in eligible]
    finite_deltas = [d for d in deltas if math.isfinite(d)]
    finite_directions = [d for d in directions if math.isfinite(d)]
    n = len(finite_deltas)

    mean_delta = mean_or_nan(finite_deltas)
    p_delta = paired_ttest_greater(finite_deltas)
    p_delta_bonferroni = bonferroni(p_delta, n_success_models)

    mean_direction = mean_or_nan(finite_directions)
    direction_ci = bootstrap_ci(finite_directions, derive_rng(seed, DIRECTION_CI, model_index))
    p_direction = ttest_two_sided(finite_directions)

    flip_rate_gender = mean_or_nan([float(m["flip_gender"]) for m in eligible])
    flip_rate_controls = mean_or_nan([m["flip_controls_rate"] for m in eligible])

    summary = {
        "sensitivity": sensitivity,
        "raw_gender_effect": raw_gender_effect,
        "mean_delta": mean_delta,
        "p_delta": p_delta,
        "p_delta_bonferroni": p_delta_bonferroni,
        "mean_direction": mean_direction,
        "direction_ci": list(direction_ci),
        "p_direction": p_direction,
        "flip_rate_gender": flip_rate_gender,
        "flip_rate_controls": flip_rate_controls,
        "verdict": model_verdict(p_delta_bonferroni, mean_direction, p_direction, n, alpha),
    }
    return summary, deltas, directions
