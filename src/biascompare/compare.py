"""Orchestration: run the comparative bias test and assemble results. See DESIGN.md.

`compare_models` is the only entry point the notebook calls for the actual
run (DESIGN §8: "the notebook only calls src"). It loads the dataset, builds
the perturbation plan once (shared across every model, DESIGN §5), scores
each model in turn against that identical input set, and assembles the
`results.json` shape from DESIGN §7 — including verdict text and, for any
model that fails to load or score, a recorded error instead of a crash.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from biascompare.comparisons import build_comparison
from biascompare.inputs import ModelSpec, Post, load_posts
from biascompare.model_metrics import compute_model_summary, compute_per_post_metrics
from biascompare.perturbation_plan import (
    PostPlan,
    TextLayout,
    build_post_plans,
    build_text_layout,
    is_eligible,
    post_plan_to_dict,
)
from biascompare.results import save_results
from biascompare.scoring.base import ProbPair, Scorer, ScorerFactory
from biascompare.scoring.classifier import hf_classifier_factory
from biascompare.verdicts import build_summary_verdict

__all__ = ["compare_models", "preview_perturbations", "save_results"]

_PAPER_CITATION = "Yang et al. 2026, arXiv:2605.01048"


@dataclass(frozen=True)
class _ScoreOutcome:
    """The result of trying to load and run one model (DESIGN §7)."""

    error: str | None
    scores: list[ProbPair] | None
    scorer_type: str | None
    uses_instruction: bool | None


def compare_models(
    models: Sequence[ModelSpec],
    dataset: str | Path,
    instruction: str | None = None,
    n_controls: int = 5,
    seed: int = 0,
    alpha: float = 0.05,
    device: str = "auto",
    scorer_factory: ScorerFactory | None = None,
) -> dict:
    """Run the comparative gender-bias test and return the DESIGN §7 dict.

    Every model is scored over the *same* seeded perturbations (built once,
    before any model is loaded). A model that fails to load or score gets an
    `error` string and is excluded from the summary/comparisons; the run
    continues for the rest. Every successfully scored model is compared
    against the first successfully scored model (the reference).
    """
    _validate_inputs(models, n_controls, alpha)
    factory = scorer_factory or hf_classifier_factory

    posts = load_posts(dataset)
    plans = build_post_plans(posts, n_controls=n_controls, seed=seed)
    layout = build_text_layout(plans)
    eligible_ids = {plan.post.id for plan in plans if is_eligible(plan)}
    n_eligible = len(eligible_ids)

    outcomes = [_score_model(spec, instruction, device, factory, layout.texts) for spec in models]
    n_success = sum(1 for outcome in outcomes if outcome.error is None)

    model_entries, run_model_entries, successful = _process_models(
        models, outcomes, plans, layout, eligible_ids, n_success, seed, alpha
    )
    comparisons = _build_comparisons(successful, seed, alpha, n_eligible)

    return {
        "schema_version": "0.1",
        "run": _run_block(dataset, posts, n_controls, seed, alpha, instruction, run_model_entries),
        "posts": [post_plan_to_dict(plan) for plan in plans],
        "models": model_entries,
        "comparisons": comparisons,
        "summary_verdict": build_summary_verdict(comparisons, n_eligible, alpha),
    }


def _process_models(
    models: Sequence[ModelSpec],
    outcomes: list[_ScoreOutcome],
    plans: Sequence[PostPlan],
    layout: TextLayout,
    eligible_ids: set[str],
    n_success: int,
    seed: int,
    alpha: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Turn each model's `_ScoreOutcome` into its `models[]`/`run.models[]`
    entries, and collect the successfully scored models' deltas/directions
    for the comparisons step. One pass, in the caller's model order."""
    model_entries: list[dict] = []
    run_model_entries: list[dict] = []
    successful: list[dict] = []
    model_index = 0

    for spec, outcome in zip(models, outcomes):
        run_model_entries.append(_run_model_entry(spec, outcome))
        if outcome.error is not None:
            model_entries.append(
                {"name": spec.name, "error": outcome.error, "per_post": [], "summary": None}
            )
            continue

        per_post = [
            compute_per_post_metrics(plan, layout, index, outcome.scores)
            for index, plan in enumerate(plans)
        ]
        summary, deltas, directions = compute_model_summary(
            per_post, eligible_ids, n_success, seed, model_index, alpha
        )
        model_entries.append({"name": spec.name, "error": None, "per_post": per_post, "summary": summary})
        successful.append({"label": spec.name, "deltas": deltas, "directions": directions})
        model_index += 1

    return model_entries, run_model_entries, successful


def preview_perturbations(dataset: str | Path, n_controls: int = 5, seed: int = 0) -> list[dict]:
    """Build and render the perturbation plan without scoring any model.

    Lets the notebook show the gender swaps and controls it is about to
    score. Uses the exact same plan-building code as `compare_models`, so
    the `posts` it previews are identical to the ones a real run would use.
    """
    _validate_n_controls(n_controls)
    posts = load_posts(dataset)
    plans = build_post_plans(posts, n_controls=n_controls, seed=seed)
    return [post_plan_to_dict(plan) for plan in plans]


def _validate_inputs(models: Sequence[ModelSpec], n_controls: int, alpha: float) -> None:
    if len(models) < 2:
        raise ValueError(f"compare_models requires at least 2 models, got {len(models)}")
    _validate_n_controls(n_controls)
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha}")


def _validate_n_controls(n_controls: int) -> None:
    if n_controls < 1:
        raise ValueError(f"n_controls must be >= 1, got {n_controls}")


def _score_model(
    spec: ModelSpec,
    instruction: str | None,
    device: str,
    factory: ScorerFactory,
    texts: Sequence[str],
) -> _ScoreOutcome:
    """Load `spec`, score every text in one call, and always `close()` the
    scorer. Any failure in loading or scoring is captured as `error` instead
    of propagating, so one bad model never aborts the run (DESIGN §7)."""
    scorer: Scorer | None = None
    try:
        scorer = factory(spec, instruction, device)
        scores = scorer.score(texts)
        return _ScoreOutcome(None, scores, scorer.scorer_type, scorer.uses_instruction)
    except Exception as exc:  # noqa: BLE001 - a failing model must not abort the run
        return _ScoreOutcome(str(exc), None, None, None)
    finally:
        if scorer is not None:
            try:
                scorer.close()
            except Exception:  # noqa: BLE001 - never let cleanup mask the real outcome
                pass


def _run_model_entry(spec: ModelSpec, outcome: _ScoreOutcome) -> dict:
    return {
        "name": spec.name,
        "toxic_label": spec.toxic_label,
        "scorer": outcome.scorer_type,
        "instruction_used": outcome.uses_instruction,
    }


def _run_block(
    dataset: str | Path,
    posts: Sequence[Post],
    n_controls: int,
    seed: int,
    alpha: float,
    instruction: str | None,
    run_model_entries: list[dict],
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "n_posts": len(posts),
        "axis": "gender",
        "n_controls": n_controls,
        "seed": seed,
        "alpha": alpha,
        "instruction": instruction,
        "paper": _PAPER_CITATION,
        "models": run_model_entries,
    }


def _build_comparisons(successful: list[dict], seed: int, alpha: float, n_eligible: int) -> list[dict]:
    if len(successful) < 2:
        return []
    reference = successful[0]
    comparisons = []
    for index, model in enumerate(successful[1:]):
        comparisons.append(
            build_comparison(
                reference["label"],
                reference["deltas"],
                reference["directions"],
                model["label"],
                model["deltas"],
                model["directions"],
                seed,
                index,
                alpha,
                n_eligible,
            )
        )
    return comparisons
