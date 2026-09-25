"""Bounded CPU tests of the new auditor; no experiment data or renderer used."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_reflection_temporal as audit  # noqa: E402


def error_fixture():
    target = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    weight = np.tile([1.0, 0.75, 0.25, 0.0], (3, 1))
    union = np.full((2, 3, 4), 0.5)
    union[1] = target
    valid = np.ones((3, 4), bool)
    valid[1, 1] = False
    parts = np.array(
        [[[0.25, 0.1875, 0.0625], [0.25, 0, 0.0625], [0.25, 0.1875, 0.0625]], np.zeros((3, 3))]
    )
    by_view = parts.sum(-1)
    saved = dict(
        target=target,
        weights=weight,
        calibrated_valid=valid,
        strata=np.tile([0, 1, 2, -1], (3, 1)),
        direct_union=union,
        direct_loss_by_stratum=parts,
        direct_loss_by_view=by_view,
        direct_mean_view_loss=by_view.mean(1),
        direct_foreground_loss_by_view=parts[..., 0],
        direct_background_loss_by_view=parts[..., 1:].sum(-1),
        direct_valid_counts=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.int64),
    )
    return saved, parts


def test_fixed_denominator_and_invalid_ray_are_independently_recomputed():
    saved, expected = error_fixture()
    parts, maximum = audit.recompute_error(saved, "direct")
    assert maximum == 0
    np.testing.assert_array_equal(parts, expected)
    assert parts[0, 1].sum() == 0.3125  # Invalid ring does not rescale other strata.
    saved["direct_loss_by_stratum"][0, 1, 0] += 0.01
    with pytest.raises(ValueError, match="numerical value differs"):
        audit.recompute_error(saved, "direct")


def test_evaluation_flag_mismatch_rejected_and_empty_view_neutral():
    saved, _ = error_fixture()
    saved["direct_valid_counts"][1, 1] = 1
    with pytest.raises(ValueError, match="valid evaluation counts"):
        audit.recompute_error(saved, "direct")
    empty = dict(
        target=np.empty((3, 0)),
        weights=np.empty((3, 0)),
        calibrated_valid=np.empty((3, 0), bool),
        strata=np.empty((3, 0), np.int8),
        direct_union=np.empty((2, 3, 0)),
        direct_loss_by_stratum=np.zeros((2, 3, 3)),
        direct_loss_by_view=np.zeros((2, 3)),
        direct_mean_view_loss=np.zeros(2),
        direct_foreground_loss_by_view=np.zeros((2, 3)),
        direct_background_loss_by_view=np.zeros((2, 3)),
        direct_valid_counts=np.zeros((3, 3), np.int64),
    )
    parts, maximum = audit.recompute_error(empty, "direct")
    assert maximum == 0 and not parts.any()


def test_native_gap_counts_review_and_each_slot_gate():
    frames = np.arange(100, 105)
    observed = np.ones((5, 3, 3), bool)
    observed[:2, 0, 1] = False
    observed[-1, 2, 1] = False
    review = np.zeros((5, 3), bool)
    review[-2:, 1] = True
    accepted = np.where((observed[:, :, 1] & ~review)[..., None], np.ones((5, 3, 3)), np.nan)
    counts = audit.statistics(
        dict(observed=observed, identity_review=review, accepted_xyz=accepted), frames
    )
    assert counts["chest_gap_ranges"] == [[[100, 102]], [], [[104, 105]]]
    assert counts["review_ranges"] == [[], [[103, 105]], []]
    assert counts["direct_chest_total"] == 12 and counts["accepted_chest_total"] == 10
    both = {arm: counts.copy() for arm in audit.ARMS}
    assert audit.gate(np.array([[0.5, 0.4], [0.5, 0.3]]), both)["passed"]
    assert not audit.gate(np.array([[0.5, 0.4], [0.5, 0.4]]), both)["passed"]
    both["fresnel"]["accepted_chest_per_slot"] = [2, 5, 4]
    both["fresnel"]["accepted_chest_total"] = 11
    assert not audit.gate(np.array([[0.5, 0.4], [0.5, 0.3]]), both)["passed"]


class Camera:
    def rays(self, pixels):
        origin = np.concatenate((pixels, np.zeros((*pixels.shape[:-1], 1))), -1)
        direction = np.broadcast_to([0.0, 0.0, 1.0], origin.shape).copy()
        return origin, direction

    def project(self, pose):
        return pose[..., :2]


def support_fixture():
    pose = np.arange(27, dtype=float).reshape(1, 3, 3, 3)
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], (1, 3, 3, 3)).copy()
    reader = SimpleNamespace(nodes=SimpleNamespace(at=lambda view, frame: pose[0, ..., :2]))
    rig = SimpleNamespace(camera=lambda view: Camera())
    data = dict(
        frame_idx=np.array([100]), pose_xyz=pose, sources=sources, observed=np.ones((1, 3, 3), bool)
    )
    return data, reader, rig


def test_strict_support_single_use_and_observed_not_invented():
    data, reader, rig = support_fixture()
    summary, arrays = audit.strict_support(data, reader, rig)
    assert summary["source_count"] == 27 and arrays["valid_source"].all()
    data["observed"][0, 1, 0] = False
    with pytest.raises(ValueError, match="observed flags"):
        audit.strict_support(data, reader, rig)
    data["observed"][:] = True
    data["sources"][0, 1, 0, 0] = 0
    with pytest.raises(ValueError, match="two bodies claim"):
        audit.strict_support(data, reader, rig)


def test_incomplete_seal_rejected_before_candidate_or_output(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "experiment.provenance.json").write_text(json.dumps(dict(status="running")))
    output = tmp_path / "audit"
    monkeypatch.setattr(audit, "REPO", tmp_path)
    monkeypatch.setattr(
        audit.producer,
        "code_hashes",
        lambda: {str(audit.LAB / "run_reflection_temporal.py"): audit.PRODUCER_SHA},
    )
    monkeypatch.setattr(audit.np, "load", lambda *args, **kwargs: pytest.fail("candidate opened"))
    with pytest.raises(ValueError, match="must be complete"):
        audit.main(["--run", str(run), "--output", str(output)])
    assert not output.exists()


def test_signed_zero_is_not_an_exact_sample_copy():
    with pytest.raises(ValueError, match="bytes differ"):
        audit.exact(np.array([0.0]), np.array([-0.0]), "pixel")
