"""Local focused audit checks; no minute candidate or GPU is read."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("audit_persistent_minute.py")
spec = importlib.util.spec_from_file_location("independent_persistent_audit_test", SCRIPT)
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)

from node_geometry import RayObservations, supported_nodes  # noqa: E402


class Rig:
    def camera(self, _view):
        def rays(points):
            origin = np.concatenate([points, np.zeros((*points.shape[:-1], 1))], axis=-1)
            direction = np.zeros_like(origin)
            direction[..., 2] = 1
            return origin, direction

        return SimpleNamespace(project=lambda pose: pose[..., :2], rays=rays)


def inputs():
    pose = np.array([
        [[[20 + fish*40 + node*4 + t*.1, 20 + node, 12] for node in range(3)]
         for fish in range(3)] for t in range(3)
    ], np.float64)
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (3, 3, 3, 3)).copy()
    observed = np.ones((3, 3, 3), bool)
    review = np.zeros((3, 3), bool)
    review[1:, 1] = True
    frames = np.arange(76620, 76623, dtype=np.int64)
    arrays = dict(frame_idx=frames, pose_xyz=pose, sources=source, observed=observed,
                  identity_review=review, xyz=pose[:, :, 1].copy(),
                  accepted_xyz=np.where(review[..., None], np.nan, pose[:, :, 1]),
                  proposal_kind=np.full(3, "motion", dtype="U32"))
    nodes = SimpleNamespace(at=lambda view, frame: pose[frame - frames[0], ..., :2].copy())
    return arrays, nodes, Rig()


def test_vectorized_native_support_is_equal_to_per_frame_original_gate():
    arrays, nodes, rig = inputs()
    arrays["sources"][1, 2, :2, 2] = -1
    arrays["observed"][1, 2, 2] = False
    metrics, measured = A.verify_all_support(arrays, nodes, rig)
    assert metrics["all_sources_and_observed_flags_pass"]
    assert metrics["source_rows_checked"] == 79
    for t, frame in enumerate(arrays["frame_idx"]):
        rays = RayObservations.from_points(rig, {v: nodes.at(v, frame) for v in A.base.VIEWS})
        source, observed, distance = supported_nodes(
            arrays["pose_xyz"][t], arrays["sources"][t], rays, rig
        )
        np.testing.assert_array_equal(observed, arrays["observed"][t])
        np.testing.assert_array_equal(source >= 0, measured["valid_source"][t])
        held = source >= 0
        np.testing.assert_array_equal(measured["ray_distance_mm"][t][held], distance[held])


@pytest.mark.parametrize("change", ["ray", "pixel", "observed", "shared", "row"])
def test_geometry_audit_rejects_corrupted_saved_support(change):
    arrays, nodes, rig = inputs()
    if change == "ray":
        original = arrays["pose_xyz"].copy()
        nodes.at = lambda view, frame: original[frame - 76620, ..., :2].copy()
        arrays["pose_xyz"][0, 0, 0, 0] += 3
    elif change == "pixel":
        old_camera = rig.camera

        def camera(view):
            value = old_camera(view)
            value.project = lambda pose: pose[..., :2] + 15
            return value

        rig.camera = camera
    elif change == "observed":
        arrays["observed"][0, 0, 0] = False
    elif change == "shared":
        arrays["sources"][0, 0, 0, 0] = 1
    else:
        arrays["sources"][0, 0, 0, 0] = 3
    with pytest.raises(ValueError):
        A.verify_all_support(arrays, nodes, rig)


def test_independent_counts_and_review_onset():
    arrays, _, _ = inputs()
    result = A.direct_counts(arrays)
    assert result["direct_pectoral_fish_frames"] == 9
    assert result["accepted_pectoral_fish_frames"] == 7
    assert result["first_review_frame_per_slot"] == [None, 76621, None]
    assert result["direct_node_counts_fish_node"] == [[3, 3, 3]]*3
    arrays["accepted_xyz"][1, 1] = arrays["xyz"][1, 1]
    with pytest.raises(ValueError, match="accepted"):
        A.direct_counts(arrays)


def test_first_difference_is_separate_from_displacement_and_not_an_id_verdict():
    arrays, _, _ = inputs()
    reference = {k: v.copy() for k, v in arrays.items()}
    arrays["pose_xyz"][1, 0, 0, 0] += 1e-10
    arrays["pose_xyz"][2, 1, 1, 1] += .1
    arrays["proposal_kind"][2] = "mouth_recovery"
    result = A.compare_trajectory(arrays, reference)
    assert result["first_exact_value_difference_by_array"]["pose_xyz"] == 76621
    assert result["first_any_node_difference_over_0_001_mm"] == 76622
    assert result["first_exact_value_difference_by_array"]["proposal_kind"] == 76622
    assert result["first_exact_value_difference_by_array"]["identity_review"] is None


@pytest.mark.parametrize("states", [(None, None), ("ok", None), ("ok", "failed")])
def test_incomplete_either_seal_is_rejected_without_array_reads(tmp_path, states, monkeypatch):
    run, comparison = tmp_path / "run", tmp_path / "comparison"
    for directory, filename, status in zip(
        (run, comparison), ("experiment.provenance.json", "evaluation.provenance.json"), states,
        strict=True,
    ):
        directory.mkdir()
        if status is not None:
            (directory / filename).write_text(json.dumps({"status": status}), encoding="utf-8")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("incomplete array opened"))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="both run and comparison"):
        A.main(["--run", str(run), "--comparison", str(comparison), "--output", str(output)])
    assert not output.exists()
