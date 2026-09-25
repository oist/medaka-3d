"""New saved-array/solver/export boundaries only; no real data or rendering."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK.parents[1] / "labs/20260909_three_body"))
import body_point_identity as helper  # noqa: E402
import evaluate_body_point_identity200 as m  # noqa: E402

from test_render_mask_transport200 import candidate  # noqa: E402


def geometry():
    result = {k: v[:4].copy() for k, v in candidate().items()}
    result["observed"][1:3, 2, 1] = False
    result["xyz"] = np.where(result["observed"][:, :, 1, None], result["pose_xyz"][:, :, 1], np.nan)
    result["accepted_xyz"] = np.where(result["identity_review"][:, :, None], np.nan, result["xyz"])
    return result


def assert_result_same(a, b):
    for field in dataclasses.fields(a):
        left, right = getattr(a, field.name), getattr(b, field.name)
        if isinstance(left, np.ndarray):
            m.previous.same(left, right, "old motion-only direct call " + field.name)
        else:
            assert left == right


def test_zero_unary_wrapper_is_old_solver_exact_and_keeps_raw21_and_review():
    raw = geometry()
    before = m.signatures(raw)
    zero = np.zeros((4, 6), np.float64)
    old = m.memory.solve_identity_memory_dp(
        raw["frame_idx"], zero, raw["xyz"], raw["observed"][:, :, 1], **m.DP_CONFIG
    )
    current, proposal, links, costs = m.solve(raw, zero)
    assert_result_same(current, old)
    assert m.signatures(raw) == before
    assert not any(k.startswith("raw_") for k in proposal)
    assert proposal["prior_review_for_proposed_identity"][:, 2].all()
    assert np.isnan(proposal["conservative_accepted_xyz"][:, 2]).all()
    assert links["motion_cost_applied"].sum() == 7
    assert costs["unary_cost"] == 0 and costs["independently_recomputed"]


def test_CT_changes_only_unary_graph_control_rejects_even_one_motion_edge_change():
    raw = geometry()
    baseline = m.solve(raw, np.zeros((4, 6)))[0]
    unary = np.tile(np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), (4, 1))
    unary[0] = 0
    current = m.solve(raw, unary)[0]
    proof = m.same_motion_graph(baseline, current)
    assert proof["byte_exact"] and len(proof["fields"]) == 11
    assert proof["states"] == len(baseline.state_permutation)
    current.edge_cost[0] += 1
    with pytest.raises(ValueError, match="motion graph edge_cost"):
        m.same_motion_graph(baseline, current)


def point_frame():
    native = m.FIRST
    weight = np.zeros((2, 3, 2, 3), np.float64)
    weight[:, 0, 0, 0] = 2.5
    valid = np.ones((3, 2), bool)
    valid[2, 1] = False
    packet = SimpleNamespace(
        native=native,
        initial_native=native,
        query_sha256=tuple(c * 64 for c in "abc"),
        anchor_ids=np.arange(2, dtype=np.int64),
        release_horizons=np.full(3, m.previous.horizon(native), np.int64),
        point_pixels=np.zeros((3, 2, 2), np.float64),
        point_weight=weight,
        ray_valid=valid,
        query_counts=np.full(3, 2, np.int64),
        old_mass=weight.sum(2),
        single_count=np.ones((2, 3, 3), np.int64),
        valid_single_count=np.ones((2, 3, 3), np.int64),
    )
    # The unused evidence/pose columns differ substantially, detecting a wrong slice.
    own = np.arange(2 * 7 * 3 * 2 * 3, dtype=np.float64).reshape(2, 7, 3, 2, 3)
    values = dict(
        point_rendered_ownership=own,
        point_rendered_q=own + 1,
        point_ray_valid=np.stack([valid, valid]),
        point_frame_idx=np.asarray(native, np.int64),
        point_anchor_ids=packet.anchor_ids.copy(),
        point_query_counts=packet.query_counts.copy(),
        point_mass_by_view_body=np.stack([packet.old_mass, packet.old_mass]),
        point_combined_loss=np.arange(28, dtype=np.float64).reshape(2, 7, 2),
    )
    return values, packet, dict(query_sha256=list(packet.query_sha256))


@pytest.mark.parametrize("fault", [None, "release", "raw_axes", "mass", "native"])
def test_CT_Original_allraw_slice_native_mass_and_release_bindings(fault):
    values, packet, plan = point_frame()
    if fault == "release":
        packet.release_horizons[1] += 1
    elif fault == "raw_axes":
        values["point_rendered_ownership"] = values["point_rendered_ownership"][:, 0]
    elif fault == "mass":
        values["point_mass_by_view_body"][1, 0, 0, 0] += 1
    elif fault == "native":
        values["point_frame_idx"] += 1
    if fault:
        with pytest.raises(ValueError):
            m.selected_frame(values, packet, m.FIRST, plan)
    else:
        result = m.selected_frame(values, packet, m.FIRST, plan)
        m.previous.same(
            result["ownership"], values["point_rendered_ownership"][1, 0], "CT Original"
        )
        m.previous.same(result["point_weight"], packet.point_weight, "no renormalization")
        m.previous.same(
            result["saved_point_combined_loss"], values["point_combined_loss"][1, 0], "diagonal"
        )
        assert result["ownership"].shape == (3, 2, 3)
        assert not np.shares_memory(result["ownership"], values["point_rendered_ownership"])


def test_diagonal_uses_measured_t0_with_roundoff_tolerance_but_rejects_wrong_slice():
    measured = np.zeros((4, 2, 6), np.float64)
    measured[:, :, 0] = np.arange(1, 9).reshape(4, 2) * 3
    costs = dict(measured_unary=measured, unary=measured.copy())
    costs["unary"][0] = 0
    saved = measured[:, :, 0] / 3
    saved[1, 0] += 1e-13
    checked = m.diagonal_control(costs, saved)
    assert checked["measured_P012_divided_by_coefficient"][0].tolist() == [1.0, 2.0]
    saved[0, 0] = 0
    with pytest.raises(ValueError, match="P012"):
        m.diagonal_control(costs, saved)


def test_full_DP_codec_roundtrip_and_separate_proposal_output(tmp_path):
    raw = geometry()
    bundle = m.solve(raw, np.zeros((4, 6)))
    output = m.base.Output(tmp_path / "new_solver_output")
    summary = m.save_solution(output, "motion_only", bundle, raw, bundle[0], helper)
    members = {r["path"]: r for r in output.records.values()}
    loaded = m.evaluated.old_evaluation.load_tree(output.root, members, "motion_only/dp")
    for field in dataclasses.fields(bundle[0]):
        old = getattr(bundle[0], field.name)
        if isinstance(old, np.ndarray):
            m.previous.same(loaded[field.name], old, "full DP saved " + field.name)
        else:
            assert loaded[field.name] == old
    assert not any("candidate.npz" in p for p in output.records)
    assert summary["comparisons"]["motion_only"]["changed_identity_times"] == 0
    with np.load(output.root / "motion_only/identity_proposal.npz", allow_pickle=False) as arrays:
        assert "proposed_sources" in arrays and "prior_review_for_proposed_identity" in arrays
        assert "raw_pose_xyz" not in arrays


def test_failed_input_preserves_receipt_and_does_not_solve(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "SOURCE_PINS", {})
    monkeypatch.setattr(m.base, "Evidence", lambda: SimpleNamespace(add=lambda *a: None, paths={}))

    def reject(_):
        raise ValueError("synthetic completed-input mismatch")

    def trap(*args, **kwargs):
        raise AssertionError("solver cannot run before successful input binding")

    monkeypatch.setattr(m, "load_inputs", reject)
    monkeypatch.setattr(m, "solve", trap)
    output = tmp_path / "failure"
    with pytest.raises(ValueError, match="completed-input mismatch"):
        m.run(output)
    seal = m.base.read_json(output / "evaluation.provenance.json")
    assert seal["status"] == "failed" and seal["completed_runs"] == []
    assert seal["outputs"] == {}
