"""Source ownership, missing-evidence safeguards and bounded node proposals."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.types import VIEWS

LAB = Path(__file__).resolve().parents[1] / "labs" / "20260916_multiview_node_review"


@pytest.fixture(scope="module")
def candidates():
    if not (LAB / "candidates.py").exists():
        pytest.skip("multiview node review experiment has been retired")
    previous = sys.modules.get("geometry")
    for name in ("geometry", "candidates"):
        module_name = "geometry" if name == "geometry" else "_multiview_candidates"
        spec = importlib.util.spec_from_file_location(module_name, LAB / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    yield module
    if previous is None:
        sys.modules.pop("geometry", None)
    else:
        sys.modules["geometry"] = previous


def observations(rig):
    centers = np.array([[50, 40, 60], [80, 120, 90], [140, 70, 130]])
    pose = centers[:, None] + np.array([[-5, 0, 0], [0, 0, 0], [10, 0, 0]])
    rows = [rig.camera(view).project(pose) for view in VIEWS]
    source = np.tile(np.arange(3)[:, None], (1, 3))
    return pose, rows, source


def prior():
    return {"median": np.array([5.0, 10.0]), "scale": np.array([0.75, 0.75])}


def no_motion():
    return np.full((3, 3), np.nan)


def score_inputs(candidates, rig, n=2):
    _, rows, source = observations(rig)
    base = np.repeat(source[..., None], 3, axis=-1)
    owners = np.repeat(base[None], n, axis=0)
    uv = candidates.assemble(rows, owners)
    metrics = candidates.geometry_metrics(rig, uv)
    return metrics, owners, base, uv[0]


def test_whole_and_node_candidates_never_reuse_a_row_node(candidates, rig):
    _, rows, source = observations(rig)
    owners, families = candidates.owner_candidates(source, rows)
    assert owners.shape == (216, 3, 3, 3)
    assert len(families) == 216
    assert candidates.ownership_bijective(owners).all()
    base = np.repeat(source[..., None], 3, axis=-1)
    np.testing.assert_array_equal(owners[0], base)
    node_owners, _ = candidates.node_candidates(base, owners[17])
    assert len(node_owners) == 72
    assert candidates.ownership_bijective(node_owners).all()
    for candidate in node_owners:
        for view in range(3):
            for node in range(3):
                assert sorted(candidate[:, view, node]) == [0, 1, 2]


def test_unused_whole_detection_does_not_mix_its_nodes_or_reuse_a_row(candidates, rig):
    _, rows, source = observations(rig)
    rows[1] = np.concatenate([rows[1], rows[1][:2] + 5])
    owners, families = candidates.owner_candidates(source, rows)
    assert len(owners) == 216 + 2 * 3
    substitutions = owners[np.asarray(families) == "unused_whole_detection"]
    assert candidates.ownership_bijective(substitutions).all()
    for alternative in substitutions:
        changed_slots = np.flatnonzero(alternative[:, 1, 0] >= 3)
        assert len(changed_slots) == 1
        values = alternative[changed_slots[0], 1]
        assert (values == values[0]).all()


def test_single_view_tail_exchange_recovers_geometry_as_unaccepted_proposal(candidates, rig):
    pose, rows, source = observations(rig)
    rows[1][[0, 1], 2] = rows[1][[1, 0], 2]
    original_rows = [r.copy() for r in rows]
    owners, _, measured, detail = candidates.propose(
        rig, source, rows, prior(), no_motion(), no_motion()
    )
    expected = np.repeat(source[..., None], 3, axis=-1)
    expected[[0, 1], 1, 2] = [1, 0]
    np.testing.assert_array_equal(owners, expected)
    np.testing.assert_allclose(measured["pose"], pose, atol=1e-6)
    assert detail["review_worthy"]
    assert not detail["accepted_identity_correction"]
    assert detail["candidate_components"]["changed_nodes"] == 2
    assert detail["runnerup_margin"] > 0
    for original, unchanged in zip(original_rows, rows, strict=True):
        np.testing.assert_array_equal(original, unchanged)


def test_whole_view_wrong_rows_are_distinct_from_biological_identity(candidates, rig):
    pose, rows, source = observations(rig)
    source[:, 1] = [1, 0, 2]
    owners, _, measured, detail = candidates.propose(
        rig, source, rows, prior(), no_motion(), no_motion()
    )
    np.testing.assert_array_equal(owners[:, 1, 0], [0, 1, 2])
    np.testing.assert_allclose(measured["pose"], pose, atol=1e-6)
    assert detail["family"] == "whole_view_permutation"
    assert not detail["accepted_identity_correction"]


@pytest.mark.parametrize("failure", ["lost_support", "one_invalid_projection", "duplicate_owner"])
def test_bad_evidence_cannot_win_by_lowering_residual(candidates, rig, failure):
    metrics, owners, base, base_uv = score_inputs(candidates, rig)
    metrics["ray_rms_mm"][0] = 5  # A tempting geometric improvement is insufficient.
    if failure == "lost_support":
        metrics["view_counts"][1, 0, 0] -= 1
        metrics["reprojection_px"][1, 0, 0, 0] = np.nan
    elif failure == "one_invalid_projection":
        # The other two projections remain finite: accepting "any finite"
        # projection would incorrectly let this nonphysical hypothesis win.
        metrics["reprojection_px"][1, 0, 0, 0] = np.nan
    else:
        owners[1, 1, 0, 2] = owners[1, 0, 0, 2]
    costs, _ = candidates.score(metrics, owners, base, prior(), no_motion(), no_motion(), base_uv)
    assert np.isfinite(costs[0])
    assert np.isinf(costs[1])


def test_body_cost_preserves_real_bends_and_reduces_over_fish_and_segments(candidates, rig):
    pose, rows, source = observations(rig)
    bent = pose.copy()
    bent[:, 2] = bent[:, 1] + np.array([0, 10, 0])
    uv = np.stack(
        [np.stack(rows, axis=1), np.stack([rig.camera(v).project(bent) for v in VIEWS], axis=1)]
    )
    metrics = candidates.geometry_metrics(rig, uv)
    base = np.repeat(source[..., None], 3, axis=-1)
    owners = np.repeat(base[None], 2, axis=0)
    costs, components = candidates.score(
        metrics, owners, base, prior(), no_motion(), no_motion(), uv[0]
    )
    np.testing.assert_allclose(metrics["bend_deg"][1], 90, atol=1e-6)
    np.testing.assert_allclose(costs, 0, atol=1e-9)
    assert components["body"].shape == (2,)
    # Exactly one of six segment lengths has normalized deviation 2.
    metrics["mp_mm"][1, 2] += 1.5
    _, components = candidates.score(
        metrics, owners, base, prior(), no_motion(), no_motion(), uv[0]
    )
    assert components["body"][1] == pytest.approx(2 / 6)


def test_missing_duplicates_do_not_create_a_fake_runnerup_margin(candidates, rig):
    rows = [np.empty((0, 3, 2)) for _ in VIEWS]
    source = np.full((3, 3), -1)
    owners, _, measured, detail = candidates.propose(
        rig, source, rows, prior(), no_motion(), no_motion()
    )
    assert (owners == -1).all()
    assert np.isnan(measured["pose"]).all()
    assert detail["candidate_count"] == 1
    assert detail["runnerup_margin"] is None
    assert not detail["review_worthy"]


def test_all_invalid_ownership_candidates_abstain_without_inf_minus_inf(candidates, rig):
    _, rows, _ = observations(rig)
    rows = [r[:1] for r in rows]
    source = np.zeros((3, 3), dtype=int)
    with np.errstate(invalid="raise"):
        owners, _, _, detail = candidates.propose(
            rig, source, rows, prior(), no_motion(), no_motion()
        )
    assert (owners == 0).all()  # Original invalid baseline retained, not certified.
    assert detail["finite_candidate_count"] == 0
    assert detail["runnerup_margin"] is None
    assert not detail["review_worthy"]


def test_flank_missing_rows_preserve_real_elapsed_time_not_compressed_steps(candidates):
    frames = np.arange(1000, 1017)
    xyz = np.zeros((17, 3, 3))
    xyz[:, :, 0] = np.arange(17)[:, None] * 2.5
    xyz[8, 0] = np.nan
    models = candidates.fit_flank(xyz, frames)
    assert all(m["valid"] for m in models)
    np.testing.assert_allclose(candidates.predictions(models, 1020)[:, 0], 50)
    # The true adjacent 4.1 mm jump must still invalidate its fish.
    xyz[10:, 1, 0] += 1.6
    models = candidates.fit_flank(xyz, frames)
    assert models[0]["valid"]
    assert models[1]["reason"] == "unstable_reference_step"


@pytest.mark.parametrize("failure", ["partial", "frame_gap", "too_many_missing"])
def test_incomplete_or_gapped_flanks_are_explicitly_unavailable(candidates, failure):
    frames = np.arange(17)
    xyz = np.zeros((17, 3, 3))
    xyz[:, :, 0] = frames[:, None] * 0.1
    if failure == "partial":
        xyz, frames = xyz[:14], frames[:14]
    elif failure == "frame_gap":
        frames[8:] += 1
    else:
        xyz[4:8] = np.nan
    models = candidates.fit_flank(xyz, frames)
    assert not any(m["valid"] for m in models)
    assert np.isnan(candidates.predictions(models, 20)).all()


def test_invalid_owner_indices_are_not_silently_truncated_or_wraparound(candidates, rig):
    _, rows, source = observations(rig)
    base = np.repeat(source[..., None], 3, axis=-1).astype(float)
    base[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="integer"):
        candidates.assemble(rows, base)
    base[0, 0, 0] = -2
    with pytest.raises(ValueError, match="only -1"):
        candidates.assemble(rows, base)
    base[0, 0, 0] = 10
    with pytest.raises(ValueError, match="nonexistent"):
        candidates.assemble(rows, base)
