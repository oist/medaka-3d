"""CPU caller checks with real constrained geometry and real strict support gates."""

from dataclasses import replace

import diagnose_bounded_grouped_mouth_intersections as diag
import numpy as np
import pytest
from test_diagnose_grouped_mouth_intersections import scene


def inputs(outside=False):
    data = scene()
    if outside:
        truth = data["pose"].copy()
        truth[:, :, 2] = 31
        truth[2, 0] = [130, 20, 29]
        data["pose"][:, :, 2] = 31
        data["pose"][2, 0] = [135, 25, 32]
        points = {v: data["rig"].camera(v).project(truth).astype(np.float32) for v in diag.VIEWS}
        data["rays"] = diag.free.RayObservations.from_points(data["rig"], points)
        data["initial_lengths"][2, 0] = np.sqrt(29)
        data["config"] = replace(data["config"], lower_mm=(0.0, 0.0, 31.0))
    result, archive = diag.free.assess_scene(**data)
    trial = result["trials"][0]
    return (
        archive,
        trial,
        archive["trial_00__mouth_ray_origins"],
        archive["trial_00__mouth_ray_directions"],
        data["rays"],
        data["rig"],
        data["config"],
    )


def test_interior_geometry_stays_supported_and_input_bytes_are_preserved():
    args = inputs()
    before = {k: diag.proof.signature(v) for k, v in args[0].items()}
    row, arrays = diag.assess_subset(*args)
    assert row["passes_existing_spatial_gates"]
    np.testing.assert_allclose(row["point_xyz"], [130, 20, 30], atol=1e-12, rtol=0)
    assert row["mouth_supported_views"] == 3
    assert row["active_bounds"] == [0, 0, 0]
    assert before == {k: diag.proof.signature(v) for k, v in args[0].items()}
    assert all(not np.shares_memory(a, b) for a in arrays.values() for b in args[0].values())


def test_box_solution_can_be_valid_solver_but_rejected_by_existing_rms_gate():
    args = inputs(outside=True)
    row, arrays = diag.assess_subset(*args)
    assert not args[1]["gates"]["box"]
    assert row["point_xyz"][2] == 31
    assert row["active_bounds"] == [0, 0, -1]
    assert row["kkt_violation"] <= row["kkt_tolerance"]
    assert row["gates"]["box"]
    assert row["mouth_supported_views"] == 3
    assert not row["gates"]["ray_rms"]
    assert not row["passes_existing_spatial_gates"]
    assert row["other_direct_support_losses"] == []
    np.testing.assert_array_equal(arrays["pose_xyz"][:2], args[0]["base_pose_xyz"][:2])
    np.testing.assert_array_equal(arrays["pose_xyz"][:, 1:], args[0]["base_pose_xyz"][:, 1:])
    np.testing.assert_array_equal(arrays["direct_sources"][:2], args[0]["base_plan"][:2])


@pytest.mark.parametrize("field", ["origin", "direction", "row"])
def test_saved_ray_or_row_tampering_rejected_before_solve(field):
    args = list(inputs())
    if field == "origin":
        args[2] = args[2].copy()
        args[2][0, 0] += 1
    elif field == "direction":
        args[3] = args[3].copy()
        args[3][0, 0] += 1
    else:
        args[1]["rows"][0] = 1
    with pytest.raises(ValueError, match=r"ray origin differs|ray differs|direct chest row"):
        diag.assess_subset(*args)


def test_solver_failure_is_not_converted_to_a_feasible_or_partial_result(monkeypatch):
    def fail(*args):
        raise RuntimeError("bounded solve failed")

    monkeypatch.setattr(diag.bounded, "solve_bounded_ray_seed", fail)
    with pytest.raises(RuntimeError, match="bounded solve failed"):
        diag.assess_subset(*inputs())
