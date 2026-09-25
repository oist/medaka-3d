"""Synthetic wiring/boundary tests only; no real candidate, mask or SL payload."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import diagnose_temporal_refractive_geometry as d
import numpy as np
import pytest


def candidate():
    result = {k: np.zeros((64, *shape), np.float64) for k, shape in d.FIELDS.items()}
    result["frame_idx"] = np.arange(100, 164, dtype=np.int64)
    for k in (
        "observed",
        "identity_review",
        "mask_supported",
        "body_supported",
        "proposal_flow_nodes",
    ):
        result[k] = np.ones(result[k].shape, bool)
    result["identity_review"][:] = False
    for k in (
        "sources",
        "missing",
        "hypotheses",
        "proposal_flow_parent",
        "node_missing",
        "direct_pectoral_age",
    ):
        result[k] = np.zeros(result[k].shape, np.int32)
    result["proposal_kind"] = np.full(64, "predict")
    return result


def test_native_slice_exact_and_nonaliasing():
    a = candidate()
    result = d.select_candidate(a, 100, 164, 100)
    for k in a:
        d.same(a[k], result[k], k)
        assert not np.shares_memory(a[k], result[k])


@pytest.mark.parametrize("fault", ["frame", "source", "accepted", "schema", "dtype", "offset"])
def test_candidate_wiring_rejects_drift(fault):
    a = candidate()
    first = 100
    if fault == "frame":
        a["frame_idx"][31] += 1
    elif fault == "source":
        a["sources"][0, 0, 0, 0] = -2
    elif fault == "accepted":
        a["accepted_xyz"][0, 0, 0] = 1
    elif fault == "schema":
        a.pop("node_missing")
    elif fault == "dtype":
        a["observed"] = a["observed"].astype(np.uint8)
    else:
        first = 101
    with pytest.raises(ValueError):
        d.select_candidate(a, 100, 164, first)


def test_legacy_identical_aliases_and_conflicts(tmp_path):
    path = tmp_path / "candidate.npz"
    value = dict(path=str(path), sha256="a" * 64, size=123)
    seal = dict(outputs={"candidate": value, str(path): dict(value)})
    result = d.output_members(seal, tmp_path)
    assert len(result) == 1 and result[str(path)]["size_bytes"] == 123
    seal["outputs"][str(path)]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="conflicting"):
        d.output_members(seal, tmp_path)


def test_absolute_output_escape_rejected(tmp_path):
    row = dict(path=str(tmp_path.parent / "outside.npz"), sha256="a" * 64, size=1)
    with pytest.raises(ValueError, match="outside"):
        d.output_members(dict(outputs={"bad": row}), tmp_path)


def test_input_claim_alias_cannot_change_path(tmp_path):
    p = tmp_path / "calib.json"
    seal = dict(inputs={str(p): dict(path=str(tmp_path / "other.json"), sha256="a" * 64)})
    with pytest.raises(ValueError, match="role/path"):
        d.input_sha(seal, p)


def geometric_fixture():
    pose = np.asarray([[[x, 2, 1], [x, 3, 1], [x, 4, 1]] for x in (2, 5, 8)], np.float64)
    masks = np.zeros((3, 11, 11), bool)
    for b, x in enumerate((2, 5, 8)):
        masks[b, 1:6, x - 1 : x + 2] = True
    source = np.tile(np.arange(3, dtype=np.int32)[:, None], (1, 3))
    observed = np.zeros((3, 3), bool)
    camera = SimpleNamespace(project=lambda p: np.asarray(p)[..., :2].copy())
    return pose, masks, source, observed, camera


def test_current_single_view_can_complement_missing_mouth_chest():
    pose, masks, sources, observed, camera = geometric_fixture()
    before = observed.copy()
    data, record = d.geometry_frame(pose, observed, sources, masks, True, camera)
    # Initial linkage still requires global observed and is NOT rescued.
    assert not any(record["reference"]["slot_geometry_eligible"])
    current = record["current_single_view"]
    assert current["scope"] == "single_current_view_geometry"
    assert current["eligible"] == [True, True, True]
    assert current["global_two_view_observed_imputed"] is False
    temporal = np.zeros((3, 3), bool)
    temporal[0, 0] = True
    direct = np.zeros((3, 3), bool)
    direct[1] = current["eligible"]
    result = d.arithmetic.availability_cases(temporal, direct, observed)
    assert result[0]["case_B"] and not result[0]["case_A"]
    assert result[0]["complementary_view_pairs"] == [[0, 1]]
    d.same(observed, before, "global support unchanged")
    assert data["coverage"].shape == (3, 3)
    np.testing.assert_array_equal(data["coverage"], np.eye(3))


def test_geometry_keeps_all_masks_and_latent_competition():
    pose, masks, sources, observed, camera = geometric_fixture()
    masks = np.concatenate([masks, masks[:1]], axis=0)
    data, record = d.geometry_frame(pose, observed, sources, masks, True, camera)
    assert data["coverage"].shape == (4, 3)
    assert record["reference"]["slots"][0]["body_coherence"]["best_mask_rows"] == [0, 3]
    assert not record["current_single_view"]["eligible"][0]
    # This copied instance contains body0's node, a competitor for bodies1/2.
    np.testing.assert_array_equal(data["collision"][3], [False, True, True])


def test_node_inventory_uses_raw_source_pixels_not_projected_pose():
    a = candidate()
    a["observed"][0, 0, 0] = False
    a["sources"][0, 1, 2, 2] = -1
    arrays = dict(ray_distance_mm=np.zeros((3, 3, 3)), pixel_error_px=np.zeros((3, 3, 3)))
    for v in d.VIEWS:
        arrays[v + "_points"] = np.arange(18, dtype=np.float32).reshape(3, 3, 2) + 100
        arrays[v + "_finite_node_rows"] = np.full(3, 3)
    context = dict(candidate=a, spec=dict(name="synthetic"))
    before = {k: d.base.signature(x) for k, x in arrays.items()}
    rows, sets = d.node_inventory(context, arrays)
    assert len(rows) == 27 and len(sets) == 9
    row = next(r for r in rows if r["body"] == 0 and r["node"] == 1 and r["view"] == d.VIEWS[0])
    d.same(row["source_xy"], arrays[d.VIEWS[0] + "_points"][0, 1], "source coordinate")
    assert row["eligible"] and row["source_xy_dtype"] == np.dtype(np.float32).str
    mouth = next(r for r in rows if r["body"] == 0 and r["node"] == 0)
    assert not mouth["eligible"] and "node_not_supported_in_two_views" in mouth["reasons"]
    absent = next(r for r in rows if r["body"] == 1 and r["node"] == 2 and r["view"] == d.VIEWS[2])
    assert not absent["eligible"] and np.isnan(absent["source_xy"]).all()
    assert {k: d.base.signature(x) for k, x in arrays.items()} == before


def test_context_rejects_different_refractive_geometry():
    # No real Rig construction; mismatch is rejected before any projection.
    rig = SimpleNamespace(cameras=[], world_units="mm")
    metadata = dict(context=dict(cameras=[], views=list(d.VIEWS), world_units="mm"))
    d.context_geometry(metadata, rig)
    wrong = deepcopy(metadata)
    wrong["context"]["world_units"] = "cm"
    with pytest.raises(ValueError, match="context"):
        d.context_geometry(wrong, rig)


def test_plan_source_pinned_before_output_can_execute(monkeypatch, tmp_path):
    monkeypatch.setattr(d, "WORK", Path(tmp_path))
    monkeypatch.setattr(d, "PLAN_SHA", "TO_BE_FROZEN")
    with pytest.raises(ValueError, match="reviewed plan"):
        d.run(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_linked_summary_does_not_mix_competing_body_scores():
    common = dict(
        window="initial",
        view=d.VIEWS[0],
        relative_frame=1,
        native_frame=66484,
        initial_mask_row=0,
        comparison="ct_first",
        subset="common",
        selected_count=2,
        linked=[True, False, False],
    )
    rows = [
        dict(common, arm="lk", scores=[0.0, 0.5, 0.5]),
        dict(common, arm="ct_first", scores=[0.8, 0.0, 0.0]),
    ]
    result = d.summaries(rows, [])["initial"]["views"][d.VIEWS[0]]["ct_first"]
    assert result["finite_common_body_score_differences"] == 3
    assert result["median_CT_minus_LK"] == -0.5
    assert result["strict_initial_link_common_score_count"] == 1
    assert result["strict_initial_link_median_CT_minus_LK"] == 0.8
