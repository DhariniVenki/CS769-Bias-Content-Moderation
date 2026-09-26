"""Comparative counterfactual bias testing for HF models. See DESIGN.md.

Public API (DESIGN §8). `plot_comparison`/`summary_rows` are imported from
`biascompare.viz`, which only imports matplotlib lazily inside
`plot_comparison` — importing this package never requires matplotlib.
"""

from __future__ import annotations

from biascompare.compare import compare_models, preview_perturbations
from biascompare.inputs import ModelSpec, Post, load_posts
from biascompare.results import save_results
from biascompare.viz import plot_comparison, summary_rows

__all__ = [
    "ModelSpec",
    "Post",
    "load_posts",
    "compare_models",
    "save_results",
    "preview_perturbations",
    "plot_comparison",
    "summary_rows",
]
