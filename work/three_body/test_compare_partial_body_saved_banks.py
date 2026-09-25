"""Small synthetic caller checks; no original experiment input is reachable."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import compare_partial_body_saved_banks as m
import numpy as np
import partial_body_bundle_score as core
import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    work = tmp_path / "allowed"
    work.mkdir()
    monkeypatch.setattr(m, "WORK", work)
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m.proof, "REPO", tmp_path)
    monkeypatch.setattr(m, "VISUAL_ROOT", work / "visual")
    monkeypatch.setattr(m, "NEGATIVE_ROOT", work / "negative")
    monkeypatch.setattr(m, "TRACKING_ROOTS", {a: work / a for a in m.ARMS})
    for name in ["load_headers", "source_code", "load_banks", "load_observations", "load_rig"]:
        monkeypatch.setattr(m, name, lambda *a, **k: pytest.fail("real input entry forbidden"))
    monkeypatch.setattr(m.prior.cv2, "VideoCapture", lambda *a, **k: pytest.fail("video forbidden"))
    load = np.load

    def guarded(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            assert Path(path).resolve().is_relative_to(tmp_path)
        return load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    return work


def bank(frame=82720):
    poses = np.array(
        [[[[20.0 + 20 * f, 20.0 + 4 * n, 0.0] for n in range(3)] for f in range(3)]] * 3
    )
    poses[1, ..., 0] += 2
    poses[2, ..., 0] += 15
    base = np.array([0.0, 0.5, 1.0])
    six = np.array([3.0, 0.0, 0.0])
    parent = np.array([0.0, 5.0, 0.0])
    return dict(
        schema="instance_rescore_frame_diagnostics_v1",
        frame_idx=frame,
        post_fit_only=True,
        temporal_state=False,
        candidate_index=np.arange(3),
        candidate_pose_xyz=poses,
        candidate_plan=np.zeros((3, 3, 3, 3), np.int32),
        candidate_parent=np.array([0, 1, 0], np.int64),
        candidate_kind=["predict"] * 3,
        candidate_permutation=np.array([[0, 1, 2]] * 3),
        original_five_loss_parts=np.zeros((3, 5)),
        base_local_cost=base,
        sixth_instance_term=six,
        adjusted_local_cost=base + six,
        parent_path_cost=parent,
        adjusted_path_cost=parent + (base + six),
        views={},
    )


def test_shared_normalizer_and_invalid_candidate_never_wins():
    raw = np.array([[np.inf, 0.0, 0.0], [12.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    per_view, total = m.bounded_term(raw, 15.0)
    assert np.isposinf(total[0]) and total[1] == 0.8 and total[2] == 1.0
    ranked = m.rank_costs(np.array([-100.0, 2.0, 3.0]) + total, np.zeros(3))
    assert ranked["path_order"].tolist() == [1, 2, 0]
    assert ranked["path_minimum_indices"].tolist() == [1]
    assert np.isposinf(per_view[0, 0])
    with pytest.raises(ValueError, match="at least one finite"):
        m.rank_costs(np.array([np.inf, np.inf]), np.zeros(2))
    for bad in [np.nan, -np.inf]:
        with pytest.raises(ValueError):
            m.rank_costs(np.array([bad, 1.0]), np.zeros(2))
    with pytest.raises(ValueError):
        m.bounded_term(np.ones((2, 3)), 12.0)


def test_replacement_retains_parent_cost_and_separates_local_minimum():
    b = bank()
    m.validate_bank(b, 82720)
    before = m.codec.split_diagnostic(b)[0]
    baseline = m.rank_costs(b["adjusted_local_cost"], b["parent_path_cost"])
    trial = m.rank_costs(b["base_local_cost"] + np.array([0.1, 0.0, 0.2]), b["parent_path_cost"])
    assert baseline["path_minimum_indices"].tolist() == [2]
    assert baseline["local_minimum_indices"].tolist() == [1]
    assert trial["path_minimum_indices"].tolist() == [0]
    assert trial["best_path_cost"] == 0.1 and trial["best_local_cost"] == 0.1
    assert np.array_equal(trial["path_cost"], [0.1, 5.5, 1.2])
    assert before == m.codec.split_diagnostic(b)[0]


def test_final_pose_match_sets_keep_duplicates_and_source_constraints():
    b = bank()
    b["candidate_pose_xyz"][1] = b["candidate_pose_xyz"][0]
    b["candidate_plan"][1, 0, 0, 0] = 2
    method = dict(
        pose_xyz_mm=b["candidate_pose_xyz"][0].tolist(), sources=b["candidate_plan"][0].tolist()
    )
    result = m.exact_final_matches(b, method)
    assert result["exact_pose_candidate_indices"].tolist() == [0, 1]
    assert result["exact_pose_and_supported_source_candidate_indices"].tolist() == [0]
    method["sources"] = np.full((3, 3, 3), -1).tolist()
    assert m.exact_final_matches(b, method)[
        "exact_pose_and_supported_source_candidate_indices"
    ].tolist() == [0, 1]
    assert not m.exact_final_matches(b, None)["available"]


def test_outside_output_stops_before_inputs(isolated):
    p = isolated.parent / "outside"
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(p)
    assert not p.exists()


def test_synthetic_eleven_case_run_with_real_scorer_codec_and_small_render_payload(
    isolated, monkeypatch
):
    """Only the five real-input entry functions are stubbed with in-memory fixtures."""
    banks = {(arm, f): bank(f) for arm, f in m.case_keys()}
    references = {f"{a}/{f}": dict(synthetic=True) for a, f in banks}
    records = {}
    masks = {}
    points = {}
    for f in m.ALL_FRAMES:
        p = banks[(m.ARMS[0], f)]["candidate_pose_xyz"][0, ..., :2].astype(np.float32)
        points[f] = {v: p.copy() for v in m.VIEWS}
        masks[f] = {v: np.zeros((3, 90, 90), bool) for v in m.VIEWS}
        for view in m.VIEWS:
            for fish in range(3):
                masks[f][view][fish, 17:32, 17 + 20 * fish : 24 + 20 * fish] = True
        records[f] = dict(
            methods={
                a: dict(
                    pose_xyz_mm=banks[(a, f)]["candidate_pose_xyz"][0].tolist(),
                    sources=banks[(a, f)]["candidate_plan"][0].tolist(),
                )
                for a in m.ARMS
                if (a, f) in banks and f in m.FRAMES
            }
        )
    context = dict(records=records)
    rig = SimpleNamespace(camera=lambda v: SimpleNamespace(project=lambda p: p[..., :2]))
    monkeypatch.setattr(m, "load_headers", lambda e: context)
    monkeypatch.setattr(m, "source_code", lambda c, e: {"SYNTHETIC": True})
    monkeypatch.setattr(m, "load_banks", lambda c, e: (banks, references))
    monkeypatch.setattr(m, "load_observations", lambda c: (masks, points, {}))
    monkeypatch.setattr(m, "load_rig", lambda c, e: rig)
    before = {k: m.codec.split_diagnostic(v)[0] for k, v in banks.items()}
    out = isolated / "synthetic_run"
    m.run(out)
    seal = m.proof.read_json(out / "comparison.provenance.json")
    summary = m.proof.read_json(out / "summary.json")
    payload = m.proof.read_json(out / "render_payload.json")
    plan = m.proof.read_json(out / "plan.json")
    assert seal["status"] == "ok" and len(seal["outputs"]) == 25
    assert summary["saved_candidate_banks"] == 11 and len(summary["cases"]) == 11
    assert summary["target_mask_decodes"] == 21 and plan["negative_controls_included"]
    assert plan["normalizers"] == dict(row_only=15.0, row_plus_mask=15.0)
    assert plan["scorer_ceilings"] == dict(row_only=12.0, row_plus_mask=15.0)
    assert len(payload["cases"]) == 11 and not payload["new_trajectory_created"]
    assert before == {k: m.codec.split_diagnostic(v)[0] for k, v in banks.items()}
    for r in payload["cases"]:
        assert r["final_selected_match"]["available"] == (r["frame_idx"] in m.FRAMES)
        for s in r["selections"].values():
            assert s["candidate_index"] in s["path_minimum_indices"]
            assert np.asarray(s["pose_xyz"]).shape == (3, 3, 3)
            assert set(s["projections"]) == set(m.VIEWS)
            assert s["support_not_recomputed"] and s["plan_is_assigned_views_not_direct_support"]
    arm, f = m.case_keys()[0]
    doc = m.proof.read_json(out / f"cases/{arm}_{f}.json")
    with np.load(doc["archive"]["path"], allow_pickle=False) as z:
        result = m.codec.restore_diagnostic(doc["tree"], {k: z[k] for k in z.files})
    for mode in ["row_only", "row_plus_mask"]:
        assert result["modes"][mode]["normalizer"] == 15.0
        assert result["modes"][mode]["positive_infinity_preserved"]
    assert np.array_equal(
        result["modes"]["baseline"]["path_cost"], banks[(arm, f)]["adjusted_path_cost"]
    )


def test_scorer_infinite_projection_stays_disqualified_in_caller():
    b = bank()
    points = b["candidate_pose_xyz"][0, ..., :2].copy()
    config = core.DEFAULT_CONFIG
    row = replace(config, mask_weight=0.0)
    configs = dict(row_only=row, row_plus_mask=config)
    observed = {
        name: {v: core.prepare_partial_bundle_observation(None, points, c) for v in m.VIEWS}
        for name, c in configs.items()
    }

    def project(p):
        out = p[..., :2].copy()
        out[0] = np.nan
        return out

    rig = SimpleNamespace(camera=lambda v: SimpleNamespace(project=project))
    method = dict(
        pose_xyz_mm=b["candidate_pose_xyz"][0].tolist(), sources=b["candidate_plan"][0].tolist()
    )
    before = copy.deepcopy(b["adjusted_path_cost"])
    result = m.compare_bank(b, method, observed, rig, core, configs)
    for name in configs:
        assert np.isposinf(result["modes"][name]["path_cost"][0])
        assert 0 not in result["modes"][name]["path_minimum_indices"]
        assert result["modes"][name]["invalid_projection_candidates"].tolist() == [
            True,
            False,
            False,
        ]
    assert np.array_equal(before, result["modes"]["baseline"]["path_cost"])
