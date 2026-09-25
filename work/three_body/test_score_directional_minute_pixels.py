"""Check regenerated fitting evidence and disjoint assessment sampling."""

from types import SimpleNamespace

import numpy as np
import pytest
import score_directional_minute_pixels as score


def frame(empty=False):
    mask = np.zeros((1, 80, 80), bool)
    mask[:, 25:40, 28:46] = True
    return SimpleNamespace(
        frame_idx=66483,
        masks={view: None if empty else mask.copy() for view in score.VIEWS},
        images_bgr={view: SimpleNamespace(shape=(80, 80, 3)) for view in score.VIEWS},
    )


def fit(value):
    return score.fit_arrays(value, dict(positive_samples=384, near_samples=384, global_samples=256))


def diagnostic(value):
    return {
        **{key: value[key].copy() for key in ("fit_target", "fit_weights", "fit_foreground")},
        **{
            view + "_fit_pixels": value["fit_pixels"][i].copy()
            for i, view in enumerate(score.VIEWS)
        },
    }


def test_uniform_minute_has_one_center_per_second_and_fixed_diagnostics():
    np.testing.assert_array_equal(score.SCORED, 66400 + 83 + 166 * np.arange(60))
    assert len(score.DIAGNOSTIC) == 166
    assert len(score.PREPARED) == 225
    assert np.isin(score.DIAGNOSTIC, score.PREPARED).all()


def test_actual_sampler_repeatability_and_unused_membership():
    value = frame()
    first, second = fit(value), fit(value)
    for key in first:
        assert score.checks.equal_bytes(first[key], second[key])
    pixels, target, weights, overlap, audit = score.unused_samples(value, first["fit_pixels"])
    assert not overlap.any()
    assert sum(row["overlap_count"] for row in audit) == 0
    for vi, view in enumerate(score.VIEWS):
        selected = {tuple(row) for row in pixels[vi]}
        fitted = {tuple(row) for row in first["fit_pixels"][vi]}
        assert selected.isdisjoint(fitted)
        held = weights[vi] > 0
        x, y = pixels[vi, held].astype(int).T
        np.testing.assert_array_equal(target[vi, held].astype(bool), value.masks[view].any(0)[y, x])


def test_unavailable_detector_is_neutral_in_both_samplers():
    value = frame(empty=True)
    first = fit(value)
    _, _, weights, overlap, audit = score.unused_samples(value, first["fit_pixels"])
    assert not first["fit_weights"].any() and not weights.any() and not overlap.any()
    assert all(not row["source_mask_available"] for row in audit)


@pytest.mark.parametrize(
    "field", ["fit_target", "fit_weights", "fit_foreground", "top_cam_fit_pixels"]
)
def test_saved_fit_evidence_mismatch_is_rejected(field):
    regenerated = fit(frame())
    saved = diagnostic(regenerated)
    score.validate_diagnostic(regenerated, saved)
    saved[field].flat[0] = 1 - saved[field].flat[0]
    with pytest.raises(ValueError, match="regenerated"):
        score.validate_diagnostic(regenerated, saved)


def test_unlisted_required_file_cannot_be_used(tmp_path):
    path = tmp_path / "candidate.npz"
    path.write_bytes(b"not a sealed output")
    with pytest.raises(ValueError, match="not sealed"):
        score.require_member(tmp_path, path.name, {})


@pytest.mark.parametrize("fault", ["source", "seed", "counts", "config"])
def test_preparation_cannot_substitute_different_sampler_implementation(fault):
    config = dict(mode="detectron2", width_ratio=0.06, positive_samples=384)
    prepared = dict(
        scored_frames=score.SCORED.tolist(),
        prepared_frames=score.PREPARED.tolist(),
        diagnostic_frames=score.DIAGNOSTIC.tolist(),
        counts=list(score.COUNTS),
        seed_offset=score.SEED_OFFSET,
        sampler_config=config.copy(),
    )
    seal = dict(inputs={"sampler.py": dict(sha256="a" * 64)})
    code = {"sampler.py": "a" * 64}
    score.validate_preparation(prepared, seal, code, config)
    if fault == "source":
        seal["inputs"]["sampler.py"]["sha256"] = "b" * 64
    elif fault == "seed":
        prepared["seed_offset"] += 1
    elif fault == "counts":
        prepared["counts"][0] += 1
    else:
        prepared["sampler_config"]["positive_samples"] += 1
    with pytest.raises(ValueError, match="prepared"):
        score.validate_preparation(prepared, seal, code, config)
