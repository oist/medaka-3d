"""Independent, synthetic record checks; no renderer, fitting or actual data."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "work/three_body"))
import mask_transport_gap_tracker as m  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic validation must not fit")

    monkeypatch.setattr(m.fitter, "fit_with_gap", forbidden)
    monkeypatch.setattr(m.fitter, "frozen_fit", forbidden)


def active_fit():
    """Two candidates: first minimum tied at 4/5, second first minimized at 14."""
    step = np.arange(15, dtype=np.float64)
    owner = np.column_stack((0.125 + step / 64, 0.25 + step / 32))
    gap = np.column_stack((0.5 + step / 16, 0.25 + step / 32))
    total = np.column_stack(
        ([30, 29, 28, 27, 20, 20, 25, 26, 27, 28, 29, 30, 31, 32, 33], 40 - step)
    ).astype(np.float64)
    base = total - 3 * (owner + gap)
    best = np.array([4, 14], np.int64)
    at_best = (best, np.arange(2))
    final_gap = np.zeros((2, 3, 3), np.float64)
    final_gap[0, 0, 0] = final_gap[0, 1, 1] = 0.375
    final_gap[1, 2, 2] = 0.6875
    mass = np.zeros_like(final_gap)
    mass[0, 0, 0], mass[0, 1, 1], mass[0, 1, 2] = 0.25, 0.5, 0.125
    mass[1, 2, 2] = 0.75
    count = np.zeros((2, 3, 3), np.int64)
    count[0, 0, 0], count[0, 1, 1], count[1, 2, 2] = 1, 2, 4
    parts = np.arange(10, dtype=np.float64).reshape(2, 5) / 16
    diagnostic = dict(
        active=True,
        reason="fixed_transport_owner_plus_gap",
        coefficient=3.0,
        base_score=base[at_best].copy(),
        owner_loss=owner[at_best].copy(),
        gap_loss=gap[at_best].copy(),
        total_score=total[at_best].copy(),
        original_loss_parts=parts.copy(),
        iteration_owner_loss=owner,
        iteration_gap_loss=gap,
        iteration_base_score=base,
        iteration_total_score=total,
        iteration_owner_gradient_norm=np.arange(28, dtype=np.float64).reshape(14, 2) / 64,
        iteration_gap_gradient_norm=np.arange(28, dtype=np.float64).reshape(14, 2) / 32,
        best_iteration=best,
        target_diagnostics_computed=True,
        target_mass=np.array([0.875, 0.75], np.float64),
        final_gap_by_view_body=final_gap,
        target_mass_by_view_body=mass,
        positive_gap_pixel_count=count,
    )
    return dict(
        parent=np.array([0, 0], np.int64),
        cost=total[at_best].copy(),
        parts=parts,
        fit_pixels=np.arange(24, dtype=np.float64).reshape(3, 4, 2),
        owner=diagnostic,
    )


def inactive_fit(coefficient, reason):
    scores = np.array([-0.0, 3.5], np.float64)
    parts = np.arange(10, dtype=np.float64).reshape(2, 5)
    diagnostic = dict(
        active=False,
        reason=reason,
        coefficient=coefficient,
        base_score=scores.copy(),
        owner_loss=np.zeros(2, np.float64),
        gap_loss=np.zeros(2, np.float64),
        total_score=scores.copy(),
        original_loss_parts=parts.copy(),
        best_iteration=np.full(2, -1, np.int64),
        target_diagnostics_computed=False,
        target_mass=np.empty(0, np.float64),
        final_gap_by_view_body=np.empty((0, 3, 3), np.float64),
        target_mass_by_view_body=np.empty((0, 3, 3), np.float64),
        positive_gap_pixel_count=np.empty((0, 3, 3), np.int64),
    )
    for name in (
        "owner_loss",
        "gap_loss",
        "base_score",
        "total_score",
        "owner_gradient_norm",
        "gap_gradient_norm",
    ):
        diagnostic["iteration_" + name] = np.empty((0, 2), np.float64)
    return dict(parent=np.array([0, 0], np.int64), cost=scores, parts=parts, owner=diagnostic)


def validate(fit, coefficient=3.0):
    return m.validate_gap_diagnostics(fit, coefficient=coefficient, opt_steps=14)


def fingerprint(value):
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.strides, value.tobytes())
    if isinstance(value, dict):
        return {key: fingerprint(item) for key, item in value.items()}
    return value


def test_two_candidate_best_tie_final_step_and_distinct_terms_are_read_only():
    fit = active_fit()
    before = fingerprint(fit)
    for value in (*fit.values(), *fit["owner"].values()):
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    assert validate(fit) is None
    assert fingerprint(fit) == before
    np.testing.assert_array_equal(fit["owner"]["best_iteration"], [4, 14])
    np.testing.assert_array_equal(fit["owner"]["owner_loss"], [0.1875, 0.6875])
    np.testing.assert_array_equal(fit["owner"]["gap_loss"], [0.75, 0.6875])
    assert fit["owner"]["positive_gap_pixel_count"][1, 2, 2] == 4
    # Positive target mass below the gap boundary is valid, with no positive gap pixel.
    assert fit["owner"]["target_mass_by_view_body"][0, 1, 2] == 0.125
    assert fit["owner"]["positive_gap_pixel_count"][0, 1, 2] == 0


@pytest.mark.parametrize(
    "coefficient,reason",
    [(0.0, "coefficient_zero"), (3.0, "zero_effective_target"), (3.0, "zero_valid_ray_target")],
)
def test_inactive_direct_dispatch_preserves_signed_zero_and_uncomputed_shapes(coefficient, reason):
    fit = inactive_fit(coefficient, reason)
    before = fingerprint(fit)
    assert validate(fit, coefficient) is None
    assert fingerprint(fit) == before
    assert np.signbit(fit["cost"][0])
    assert fit["owner"]["target_mass"].shape == (0,)
    assert fit["owner"]["final_gap_by_view_body"].shape == (0, 3, 3)


@pytest.mark.parametrize("term", ["base_score", "owner_loss", "gap_loss", "total_score"])
def test_each_iterate_decomposition_is_checked_even_away_from_best(term):
    fit = active_fit()
    fit["owner"]["iteration_" + term][11, 0] += 0.25
    with pytest.raises(ValueError, match="iterate objective decomposition"):
        validate(fit)


def test_equal_total_cannot_hide_owner_gap_mislabeling_in_final_diagnostic():
    fit = active_fit()
    fit["owner"]["owner_loss"][0] += 0.125
    fit["owner"]["gap_loss"][0] -= 0.125
    with pytest.raises(ValueError, match="best owner_loss"):
        validate(fit)


def test_strict_best_tie_requires_first_iteration_even_when_final_terms_are_rewritten():
    fit = active_fit()
    d = fit["owner"]
    d["best_iteration"][0] = 5
    for name in ("base_score", "owner_loss", "gap_loss", "total_score"):
        d[name][0] = d["iteration_" + name][5, 0]
    with pytest.raises(ValueError, match="first best iteration"):
        validate(fit)


@pytest.mark.parametrize("field", ["iteration_owner_gradient_norm", "iteration_gap_gradient_norm"])
@pytest.mark.parametrize("bad", ["negative", "nan", "fifteen_gradients"])
def test_each_gradient_record_has_exact_fourteen_finite_nonnegative_rows(field, bad):
    fit = active_fit()
    if bad == "fifteen_gradients":
        fit["owner"][field] = np.zeros((15, 2), np.float64)
    else:
        fit["owner"][field][13, 1] = -0.1 if bad == "negative" else np.nan
    with pytest.raises(ValueError, match="gradient_norm"):
        validate(fit)


@pytest.mark.parametrize("field", ["final_gap_by_view_body", "target_mass_by_view_body"])
def test_per_view_body_totals_cannot_drift_from_final_candidate_totals(field):
    fit = active_fit()
    fit["owner"][field][1, 2, 2] += 0.25
    with pytest.raises(ValueError, match="view/body"):
        validate(fit)


@pytest.mark.parametrize(
    "bad", ["above_draws", "negative", "gap_without_count", "count_without_mass"]
)
def test_positive_count_respects_draw_denominator_and_zero_mass_cells(bad):
    fit = active_fit()
    count = fit["owner"]["positive_gap_pixel_count"]
    if bad == "above_draws":
        count[1, 2, 2] = 5
    elif bad == "negative":
        count[0, 0, 0] = -1
    elif bad == "gap_without_count":
        count[0, 0, 0] = 0
    else:
        count[0, 2, 0] = 1
    with pytest.raises(ValueError, match="pixel"):
        validate(fit)


@pytest.mark.parametrize("field", ["base_score", "best_iteration", "positive_gap_pixel_count"])
def test_numeric_value_equality_does_not_permit_wrong_array_dtype(field):
    fit = active_fit()
    fit["owner"][field] = fit["owner"][field].astype(
        np.float32 if field == "base_score" else np.float64
    )
    with pytest.raises(ValueError, match="array differs"):
        validate(fit)


@pytest.mark.parametrize("field", ["active", "target_diagnostics_computed", "coefficient"])
def test_policy_flags_and_coefficient_require_real_python_scalar_types(field):
    fit = active_fit()
    fit["owner"][field] = np.bool_(True) if field != "coefficient" else np.float32(3)
    with pytest.raises(ValueError, match="policy"):
        validate(fit)


@pytest.mark.parametrize(
    "bad", ["reason", "computed", "shaped_zeros", "best", "loss", "signed_zero"]
)
def test_inactive_is_uncomputed_and_exact_original_return_not_fabricated_measured_zero(bad):
    fit = inactive_fit(0.0, "coefficient_zero")
    d = fit["owner"]
    if bad == "reason":
        d["reason"] = "zero_effective_target"
    elif bad == "computed":
        d["target_diagnostics_computed"] = True
    elif bad == "shaped_zeros":
        d["target_mass"] = np.zeros(2, np.float64)
    elif bad == "best":
        d["best_iteration"][0] = 0
    elif bad == "loss":
        d["gap_loss"][0] = 1.0
    else:
        d["base_score"][0] = 0.0
    with pytest.raises(ValueError):
        validate(fit, 0.0)


def test_lambda_zero_cannot_label_an_active_fit_or_compute_target_diagnostics():
    fit = active_fit()
    fit["owner"]["coefficient"] = 0.0
    fit["owner"]["base_score"] = fit["cost"].copy()
    with pytest.raises(ValueError, match="active owner-plus-gap policy"):
        validate(fit, 0.0)


def test_cost_and_original_parts_are_exact_saved_array_bindings():
    for field in ("cost", "parts"):
        fit = active_fit()
        fit[field].flat[0] = np.nextafter(fit[field].flat[0], np.inf)
        with pytest.raises(ValueError, match=r"gap (total cost|original five parts)"):
            validate(fit)


def test_small_score_tolerance_does_not_rewrite_stored_terms():
    fit = active_fit()
    fit["owner"]["base_score"][0] += 5e-11
    before = copy.deepcopy(fit)
    assert validate(fit) is None
    assert fingerprint(fit) == fingerprint(before)
    fit["owner"]["base_score"][0] += 2e-10
    with pytest.raises(ValueError, match="final objective decomposition"):
        validate(fit)


@pytest.mark.parametrize("bad", ["extra", "missing", "no_target"])
def test_exact_diagnostic_fields_and_positive_active_mass(bad):
    fit = active_fit()
    if bad == "extra":
        fit["owner"]["combined_owner_loss"] = np.ones(2)
    elif bad == "missing":
        del fit["owner"]["gap_loss"]
    else:
        fit["owner"]["target_mass"][:] = 0
    with pytest.raises(ValueError, match=r"exact fields|no effective target"):
        validate(fit)
