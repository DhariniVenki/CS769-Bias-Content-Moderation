"""Tests for biascompare.metrics: js_divergence, predicted_label, flipped."""

from __future__ import annotations

import pytest

from biascompare.metrics import flipped, js_divergence, predicted_label

# ---------------------------------------------------------------------------
# js_divergence
# ---------------------------------------------------------------------------


def test_js_divergence_zero_on_identical_distributions():
    assert js_divergence((0.3, 0.7), (0.3, 0.7)) == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_one_on_disjoint_distributions():
    assert js_divergence((1.0, 0.0), (0.0, 1.0)) == pytest.approx(1.0, abs=1e-6)


def test_js_divergence_is_symmetric():
    p = (0.1, 0.9)
    q = (0.6, 0.4)
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p), abs=1e-12)


@pytest.mark.parametrize(
    "p,q",
    [
        ((0.5, 0.5), (0.5, 0.5)),
        ((0.9, 0.1), (0.1, 0.9)),
        ((0.99, 0.01), (0.01, 0.99)),
        ((0.2, 0.8), (0.8, 0.2)),
    ],
)
def test_js_divergence_bounded_in_unit_interval(p, q):
    value = js_divergence(p, q)
    assert 0.0 <= value <= 1.0


def test_js_divergence_handles_exact_zero_probabilities_without_crashing():
    # Would divide by zero / log(0) without eps-clipping.
    value = js_divergence((1.0, 0.0), (1.0, 0.0))
    assert value == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_renormalises_unnormalised_input():
    # Not a proper distribution (sums to 2), but should still renormalise
    # and produce a value in [0, 1] rather than raising.
    value = js_divergence((1.0, 1.0), (1.0, 1.0))
    assert value == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        js_divergence((0.5, 0.5), (0.3, 0.3, 0.4))


def test_js_divergence_raises_on_empty_input():
    with pytest.raises(ValueError):
        js_divergence((), ())


# ---------------------------------------------------------------------------
# predicted_label
# ---------------------------------------------------------------------------


def test_predicted_label_above_threshold_is_toxic():
    assert predicted_label((0.3, 0.7)) == 1


def test_predicted_label_below_threshold_is_non_toxic():
    assert predicted_label((0.7, 0.3)) == 0


def test_predicted_label_at_threshold_boundary_is_toxic():
    assert predicted_label((0.5, 0.5), threshold=0.5) == 1


def test_predicted_label_custom_threshold():
    assert predicted_label((0.8, 0.2), threshold=0.1) == 1
    assert predicted_label((0.8, 0.2), threshold=0.3) == 0


# ---------------------------------------------------------------------------
# flipped
# ---------------------------------------------------------------------------


def test_flipped_true_when_labels_differ():
    assert flipped((0.9, 0.1), (0.1, 0.9)) is True


def test_flipped_false_when_labels_agree():
    assert flipped((0.9, 0.1), (0.8, 0.2)) is False


def test_flipped_respects_threshold():
    assert flipped((0.6, 0.4), (0.3, 0.7), threshold=0.5) is True
    assert flipped((0.6, 0.4), (0.3, 0.7), threshold=0.9) is False
