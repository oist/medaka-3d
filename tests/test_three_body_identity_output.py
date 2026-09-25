"""Identity export permutes observations and prior review, never inventing support."""

import importlib.util
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/identity_output.py"
SPEC = importlib.util.spec_from_file_location("three_body_identity_output_test", PATH)
OUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OUT)
PERMUTATIONS = np.array(list(permutations(range(3))), np.int64)


def geometry(n=6):
    pose = np.arange(n * 27, dtype=np.float64).reshape(n, 3, 3, 3) / 10
    observed = np.ones((n, 3, 3), bool)
    observed[::2, 0, 0] = False  # Mouth missing while pectoral remains observed.
    observed[1::2, 1, 1] = False
    observed[::2, 2, 2] = False
    review = np.zeros((n, 3), bool)
    review[:, 2] = True
    review[1::2, 0] = True
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).copy()
    for t, fish, node in np.argwhere(~observed):
        sources[t, fish, 1:, node] = -1
    xyz = np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan)
    accepted = np.where(~review[..., None], xyz, np.nan)
    return dict(frame_idx=np.arange(76620, 76620+n), pose_xyz=pose, sources=sources,
                observed=observed, identity_review=review, xyz=xyz, accepted_xyz=accepted)


def manual_permute(value, mapping):
    return np.stack([np.stack([value[t, slot] for slot in row])
                     for t, row in enumerate(mapping)])


def test_every_permutation_applies_to_the_same_fish_axis_and_preserves_node_nan_support():
    original = geometry()
    result = OUT.build_identity_output(original, PERMUTATIONS)
    expected = {
        "proposed_pose_xyz": original["pose_xyz"],
        "proposed_sources": original["sources"],
        "proposed_observed": original["observed"],
        "proposed_xyz": original["xyz"],
        "prior_review_for_proposed_identity": original["identity_review"],
        "conservative_accepted_xyz": original["accepted_xyz"],
        "proposed_node_xyz": np.where(original["observed"][..., None],
                                      original["pose_xyz"], np.nan),
    }
    for name, array in expected.items():
        np.testing.assert_array_equal(result[name], manual_permute(array, PERMUTATIONS))
    assert result["proposed_xyz"].shape == (6, 3, 3)
    assert result["proposed_pose_xyz"].shape == result["proposed_node_xyz"].shape == (6, 3, 3, 3)
    # Raw geometry is diagnostic and finite; missing direct nodes stay NaN separately.
    assert np.isfinite(result["proposed_pose_xyz"]).all()
    np.testing.assert_array_equal(np.isfinite(result["proposed_node_xyz"]).all(-1),
                                  result["proposed_observed"])
    assert np.isnan(result["conservative_accepted_xyz"][
        result["prior_review_for_proposed_identity"]
    ]).all()
    assert set(result) == {
        "frame_idx", "identity_to_raw_slot", "raw_pose_xyz", "raw_sources", "raw_observed",
        "raw_identity_review", *expected,
    }


def test_non_self_inverse_cycle_is_identity_to_raw_slot_not_its_inverse():
    original = geometry(1)
    mapping = np.array([[2, 0, 1]])
    result = OUT.build_identity_output(original, mapping)
    np.testing.assert_array_equal(result["proposed_pose_xyz"][0, 0], original["pose_xyz"][0, 2])
    np.testing.assert_array_equal(result["proposed_pose_xyz"][0, 1], original["pose_xyz"][0, 0])
    np.testing.assert_array_equal(result["proposed_pose_xyz"][0, 2], original["pose_xyz"][0, 1])
    np.testing.assert_array_equal(result["prior_review_for_proposed_identity"],
                                  [[True, False, False]])


def test_raw_output_and_proposed_output_are_independent_copies_of_all_input_arrays():
    original = geometry()
    mapping = PERMUTATIONS.copy()
    before = {key: value.copy() for key, value in original.items()}
    result = OUT.build_identity_output(original, mapping)
    for key in ("pose_xyz", "sources", "observed", "identity_review"):
        np.testing.assert_array_equal(result["raw_" + key], original[key])
    for value in result.values():
        assert not np.shares_memory(value, mapping)
        for source in original.values():
            assert not np.shares_memory(value, source)
    for key, value in result.items():
        for other_key, other_value in result.items():
            if key != other_key:
                assert not np.shares_memory(value, other_value)
    result["proposed_pose_xyz"][:] = -99
    result["raw_sources"][:] = -1
    result["prior_review_for_proposed_identity"][:] = False
    result["identity_to_raw_slot"][:] = 2
    for key in original:
        np.testing.assert_array_equal(original[key], before[key])
    np.testing.assert_array_equal(mapping, PERMUTATIONS)
    original["pose_xyz"][:] = -100
    np.testing.assert_array_equal(result["raw_pose_xyz"], before["pose_xyz"])


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_mapping_integer_types_produce_int64_output(dtype):
    result = OUT.build_identity_output(geometry(), PERMUTATIONS.astype(dtype))
    assert result["identity_to_raw_slot"].dtype == np.int64
    np.testing.assert_array_equal(result["identity_to_raw_slot"], PERMUTATIONS)


@pytest.mark.parametrize("mapping", [
    np.zeros((6, 3), int), np.tile([0, 1, 3], (6, 1)), np.tile([-1, 0, 2], (6, 1)),
    PERMUTATIONS.astype(float), PERMUTATIONS.astype(bool), PERMUTATIONS[:, :, None],
    PERMUTATIONS[:-1], np.full((6, 3), np.nan),
])
def test_invalid_or_partial_mapping_is_rejected_without_changing_raw(mapping):
    original = geometry()
    before = {key: value.tobytes() for key, value in original.items()}
    with pytest.raises(ValueError):
        OUT.build_identity_output(original, mapping)
    assert {key: value.tobytes() for key, value in original.items()} == before


@pytest.mark.parametrize("problem", [
    "full_skeleton_xyz", "missing_sources", "float_sources", "bad_source_sentinel",
    "int_observed", "int_review", "latent_nan", "made_up_direct", "cleared_review",
    "frame_gap", "duplicate_frame", "float_frame",
])
def test_inconsistent_frozen_geometry_is_rejected(problem):
    original = geometry()
    if problem == "full_skeleton_xyz":
        original["xyz"] = original["pose_xyz"].copy()
    elif problem == "missing_sources":
        del original["sources"]
    elif problem == "float_sources":
        original["sources"] = original["sources"].astype(float)
    elif problem == "bad_source_sentinel":
        original["sources"][0, 0, 0, 0] = -2
    elif problem == "int_observed":
        original["observed"] = original["observed"].astype(int)
    elif problem == "int_review":
        original["identity_review"] = original["identity_review"].astype(int)
    elif problem == "latent_nan":
        original["pose_xyz"][0, 0, 0] = np.nan
    elif problem == "made_up_direct":
        original["xyz"][1, 1] = original["pose_xyz"][1, 1, 1]
    elif problem == "cleared_review":
        original["accepted_xyz"][0, 2] = original["xyz"][0, 2]
    elif problem == "frame_gap":
        original["frame_idx"][3:] += 1
    elif problem == "duplicate_frame":
        original["frame_idx"][2] = original["frame_idx"][1]
    else:
        original["frame_idx"] = original["frame_idx"].astype(float)
    with pytest.raises(ValueError):
        OUT.build_identity_output(original, PERMUTATIONS)


def test_entirely_missing_observations_and_sticky_review_do_not_gain_coordinates():
    original = geometry()
    original["observed"][:] = False
    original["sources"][:] = -1
    original["identity_review"][:] = True
    original["xyz"][:] = original["accepted_xyz"][:] = np.nan
    result = OUT.build_identity_output(original, PERMUTATIONS)
    assert np.isfinite(result["proposed_pose_xyz"]).all()
    for key in ("proposed_xyz", "proposed_node_xyz", "conservative_accepted_xyz"):
        assert np.isnan(result[key]).all()
    assert result["raw_identity_review"].all()
    assert result["prior_review_for_proposed_identity"].all()


@pytest.mark.parametrize("frames", [
    np.array([-2, -1]), np.array([-1]), np.array([np.iinfo(np.uint64).max, 0], np.uint64),
    np.array([2**63, 2**63+1], np.uint64),
])
def test_negative_and_unsigned_wrap_native_frames_are_rejected(frames):
    original = geometry(len(frames))
    original["frame_idx"] = frames
    with pytest.raises(ValueError, match="nonnegative int64-compatible"):
        OUT.build_identity_output(original, np.tile([0, 1, 2], (len(frames), 1)))


def test_largest_valid_unsigned_native_frames_are_normalized_to_int64_without_mutation():
    original = geometry(2)
    frames = np.array([2**63-2, 2**63-1], np.uint64)
    original["frame_idx"] = frames
    result = OUT.build_identity_output(original, np.tile([0, 1, 2], (2, 1)))
    assert result["frame_idx"].dtype == np.int64
    np.testing.assert_array_equal(result["frame_idx"], frames.astype(np.int64))
    assert original["frame_idx"].dtype == np.uint64
