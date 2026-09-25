"""Bounded first-loss selection and saved-pool decomposition, synthetic CPU only."""

import copy
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import diagnose_body_point_first_losses as m

from test_body_point_transport import fixture


def test_first_three_native_not_first_three_cells_or_worst_times():
    old = np.ones((200, 3, 3), bool)
    new = old.copy()
    new[18, 0, 0] = new[18, 2, 2] = False
    new[26, 1, 1] = new[31, 2, 0] = new[32, 0, 2] = False
    old[2, 0, 0] = new[2, 0, 0] = False
    candidates = {
        arm: dict(frame_idx=np.arange(m.FIRST, m.STOP, dtype=np.int64), observed=obs)
        for arm, obs in zip(m.ARMS, (old, old, new), strict=True)
    }
    offsets, lost = m.first_loss_times(candidates)
    assert offsets.tolist() == [18, 26, 31] and lost.sum() == 5
    candidates[m.producer.ARM]["frame_idx"] = np.arange(200, dtype=np.int64)
    with pytest.raises(ValueError):
        m.first_loss_times(candidates)


def pool():
    fit = dict(
        pose=np.zeros((4, 3, 3, 3)),
        parent=np.array([0, 0, 1, 2]),
        permutation=np.tile(np.arange(3), (4, 1)),
        path_cost=np.arange(4, dtype=float),
    )
    fit["pose"][1] += 0.1
    fit["selected"], fit["order"] = m.selection.selected_indices(
        fit["pose"], fit["path_cost"], fit["parent"], fit["permutation"], 2
    )
    return fit


def test_dedup_and_after_capacity_are_distinct_without_inventing_examined_candidates():
    fit = pool()
    reasons, partner = m.retention(fit, 2)
    assert reasons.tolist() == [
        "retained",
        "same_parent_permutation_pose_dedup",
        "retained",
        "after_beam_capacity_not_examined",
    ]
    assert partner.tolist() == [-1, 0, -1, -1]
    fit["selected"] = np.array([0, 1])
    with pytest.raises(ValueError):
        m.retention(fit, 2)


def test_final_trace_all_matches_preserved_and_unicode_storage_width_is_not_identity():
    records = [{k: np.asarray(0, np.int64) for k in m.previous.RECORD_FIELDS} for _ in range(3)]
    for record in records:
        record["proposal_kind"] = np.asarray("same", dtype="U8")
    records[0]["costs"] = np.asarray(1, np.int64)
    candidate = {k: np.stack([v]) for k, v in records[1].items() if k != "parent"}
    candidate["proposal_kind"] = candidate["proposal_kind"].astype("U100")
    assert m.trace_matches(candidate, 0, records).tolist() == [1, 2]
    candidate["costs"][0] = 4
    with pytest.raises(ValueError):
        m.trace_matches(candidate, 0, records)


def test_base_aux_parent_differences_keep_rank0_and_later_chosen_trace_separate():
    fit = pool()
    reasons, partner = m.retention(fit, 2)
    terms = np.array(
        [[5.0, 1.0, 1.0, 10.0], [4.0, 2.0, 2.0, 10.0], [3.0, 1.0, 1.0, 14.0], [2.0, 1.0, 1.0, 20.0]]
    )
    fitted = np.zeros((4, 3, 3), bool)
    fitted[2:, 1, 2] = True
    pre = fitted.copy()
    pre[1, 1, 2] = True
    fit.update(
        base_owner_gap_parent=terms,
        cost=terms[:, :3].sum(1),
        path_cost=terms.sum(1),
        candidate_anchor=np.array([0, 0, 1, 1]),
        kind=np.asarray(["a", "b", "c", "d"]),
        trace_fit_indices=np.array([2]),
        retention_reason=reasons,
        dedup_partner=partner,
        pre_strict_observed=pre,
        post_strict_observed=fitted,
    )
    result = m.alternatives(fit, 1, 2)
    assert result["prefit_supported_indices"] == [1, 2, 3]
    assert result["fitted_supported_indices"] == [2, 3]
    assert result["retained_supported_indices"] == [2]
    row = result["comparisons"][0]
    assert row["reference_is_rank0"] and not row["reference_on_final_trace"]
    assert row["delta_base_owner_gap_parent"] == [-2, 0, 0, 4]
    assert row["delta_path"] == 2 and row["alternative_local_lower"]
    assert not row["alternative_parent_cost_lower"]
    assert len(result["comparisons"]) == 4


def test_point_mass_retains_all_anchors_and_undefined_zero_S_without_renormalizing():
    args = fixture(invalid_x=(1,), second_anchor=True)
    packet = m.points.build_packet(**args)
    before = copy.deepcopy(dataclasses.asdict(packet))
    result = m.point_terms(packet)
    assert result["old_mass"].shape == (2, 3, 3)
    assert result["single_count"][0, 0, 0] == 3
    assert result["valid_single_count"][0, 0, 0] == 2
    assert result["m_over_S"][0, 0, 0] == 5 / 3
    assert result["effective_mass"][0, 0, 0] == pytest.approx(10 / 3)
    assert np.isnan(result["m_over_S"][0, 0, 2])
    assert result["body_mask_row"].shape == result["body_weight"].shape == (2, 3, 3)
    m.points._same(packet, before, "input unmodified")


def test_unsuccessful_evaluation_cannot_reach_candidate_or_fit_arrays(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "PINS", {})
    monkeypatch.setattr(m.base.Evidence, "add", lambda *a, **k: None)
    monkeypatch.setattr(
        m.base, "read_json", lambda *a: {"status": "failed", "stage": m.evaluation.STAGE}
    )
    monkeypatch.setattr(
        m.previous, "read_arrays", lambda *a: pytest.fail("array before successful evaluation")
    )
    destination = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="successful stage"):
        m.run(tmp_path / "eval", "a" * 64, destination)
    import json

    record = json.loads((destination / "diagnosis.provenance.json").read_text())
    assert record["status"] == "failed" and record["outputs"] == {}
