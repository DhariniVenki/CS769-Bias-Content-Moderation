"""Tests for biascompare.viz: plot_comparison, summary_rows.

`matplotlib` is not installed in this project's dev environment, so every
test that needs a real figure is skipped via `pytest.importorskip` rather
than failing the suite (see pyproject.toml / DESIGN.md §8). `summary_rows`
needs no plotting library at all and is always tested.
"""

from __future__ import annotations

import math

import pytest

from biascompare import ModelSpec, compare_models
from biascompare.viz import summary_rows
from test_compare import BIASED_VS_UNBIASED, DATASET, _fake_factory, _failing_factory

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from biascompare.viz import plot_comparison  # noqa: E402


@pytest.fixture()
def results() -> dict:
    return compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=3, seed=0, scorer_factory=_fake_factory)


@pytest.fixture()
def results_with_failure() -> dict:
    models = [ModelSpec(name="broken"), ModelSpec(name="ok-a"), ModelSpec(name="ok-b")]
    return compare_models(models, DATASET, n_controls=2, seed=0, scorer_factory=_failing_factory)


# ---------------------------------------------------------------------------
# plot_comparison
# ---------------------------------------------------------------------------


def test_plot_comparison_returns_a_figure_with_three_panels(results):
    fig = plot_comparison(results)
    assert len(fig.axes) == 3


def test_plot_comparison_handles_failed_models_by_skipping_them(results_with_failure):
    fig = plot_comparison(results_with_failure)
    assert len(fig.axes) == 3
    # The bar panel's legend should list only the two successful models.
    bar_ax = fig.axes[0]
    legend_labels = {text.get_text() for text in bar_ax.get_legend().get_texts()}
    assert legend_labels == {"ok-a", "ok-b"}
    assert "broken" in fig.get_suptitle()


def test_plot_comparison_raises_when_every_model_failed():
    def always_fails(spec, instruction, device):
        raise RuntimeError("nope")

    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=2, seed=0, scorer_factory=always_fails)
    with pytest.raises(ValueError, match="no successfully scored models"):
        plot_comparison(results)


def test_plot_comparison_uses_consistent_colours_across_panels(results):
    fig = plot_comparison(results)
    bar_ax, delta_ax, direction_ax = fig.axes

    # Each of the 2 models contributes bars of one colour to the bar panel...
    bar_colors = {tuple(patch.get_facecolor()) for patch in bar_ax.patches}
    assert len(bar_colors) == 2

    # ...and the same two colours must reappear on the scatter points in the
    # other two panels (one scatter collection per model).
    delta_colors = {
        tuple(color) for collection in delta_ax.collections for color in collection.get_facecolor()
    }
    direction_colors = {
        tuple(color) for collection in direction_ax.collections for color in collection.get_facecolor()
    }
    assert delta_colors == bar_colors
    assert direction_colors == bar_colors


def test_plot_comparison_direction_panel_has_zero_line(results):
    fig = plot_comparison(results)
    direction_ax = fig.axes[2]
    zero_lines = [line for line in direction_ax.get_lines() if set(line.get_ydata()) == {0.0}]
    assert len(zero_lines) >= 1


# ---------------------------------------------------------------------------
# summary_rows (no matplotlib needed)
# ---------------------------------------------------------------------------


def test_summary_rows_one_row_per_model(results):
    rows = summary_rows(results)
    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"biased", "unbiased"}


def test_summary_rows_flattens_direction_ci(results):
    rows = summary_rows(results)
    for row in rows:
        assert "direction_ci" not in row
        assert "direction_ci_low" in row
        assert "direction_ci_high" in row


def test_summary_rows_handles_failed_models(results_with_failure):
    rows = summary_rows(results_with_failure)
    assert len(rows) == 3
    broken_row = next(row for row in rows if row["name"] == "broken")
    assert broken_row["error"] is not None
    assert "sensitivity" not in broken_row


def test_summary_rows_values_are_scalars_or_strings(results):
    rows = summary_rows(results)
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue
            assert isinstance(value, (str, int, float, bool)) or (
                isinstance(value, float) and math.isnan(value)
            )
