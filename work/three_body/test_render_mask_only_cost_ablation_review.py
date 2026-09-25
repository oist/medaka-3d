"""Pure saved-array and projection guards; no execute(), file fixtures or real inputs."""

from types import SimpleNamespace

import numpy as np
import pytest
import render_mask_only_cost_ablation_review as render


def test_saved_array_signature_and_readonly():
    array = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    descriptor = dict(__array__="a", signature=render.signature(array))
    result = render.get_array({"a": array.copy()}, descriptor)
    assert np.array_equal(result, array) and not result.flags.writeable
    changed = array.copy()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="signature"):
        render.get_array({"a": changed}, descriptor)


def test_winner_must_match_both_saved_local_and_path_rankings():
    costs = {k: np.array([3.0, 2.0, 1.0]) for k in ("old_local_cost", "old_fixed_parent_path_cost")}
    costs.update(
        {k: np.array([1.0, 2.0, 3.0]) for k in ("new_local_cost", "new_fixed_parent_path_cost")}
    )
    rankings = {
        k: dict(stable_candidate_order=np.argsort(v, kind="stable").tolist())
        for k, v in costs.items()
    }
    assert render.select_winners(costs, rankings, 2, 0).tolist() == [2, 0]
    rankings["new_fixed_parent_path_cost"]["stable_candidate_order"] = [1, 0, 2]
    with pytest.raises(ValueError, match="ranking"):
        render.select_winners(costs, rankings, 2, 0)


def test_all_view_projection_and_saved_camera_mismatch():
    poses = np.arange(54.0).reshape(2, 3, 3, 3)
    rig = SimpleNamespace(
        camera=lambda view: SimpleNamespace(
            project=lambda value: value[..., :2] + render.VIEWS.index(view) * 100
        )
    )
    expected = poses[..., :2] + 100
    result, error = render.projected_pairs(rig, poses, "side1_cam", expected)
    assert error == 0 and np.array_equal(result["side2_cam"], poses[..., :2] + 200)
    with pytest.raises(ValueError, match="saved score"):
        render.projected_pairs(rig, poses, "side2_cam", expected)
