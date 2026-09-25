"""Synthetic metric/state checks, never biological identity ground truth."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_directional_one_minute as m


def test_unchosen_history_corruption_cannot_hide_behind_same_winner():
    prior = {"history_" + key: np.arange(12).reshape(6, 2) for key in m.RECORD_FIELDS}
    prior["history_offsets"] = np.array([0, 3, 6])
    later = {key: np.concatenate((value, value[-1:])) for key, value in prior.items()}
    later["history_offsets"] = np.array([0, 3, 6, 7])
    m.verify_history_prefix(prior, later)
    later["history_pose_xyz"][1, 0] += 1  # An alternative, not the presumed winning row.
    with pytest.raises(ValueError, match="history_pose_xyz"):
        m.verify_history_prefix(prior, later)


def test_wider_unicode_storage_keeps_same_retained_labels():
    prior = {"history_" + key: np.array([1, 2]) for key in m.RECORD_FIELDS}
    prior.update(history_offsets=np.array([0, 1, 2]), history_proposal_kind=np.array(["a", "b"]))
    later = copy.deepcopy(prior)
    later["history_proposal_kind"] = np.array(["a", "b", "chest_recovery"])
    m.verify_history_prefix(prior, later)
    later["history_proposal_kind"][0] = "changed"
    with pytest.raises(ValueError, match="history_proposal_kind"):
        m.verify_history_prefix(prior, later)


@pytest.mark.parametrize(
    "field", ["initialize_count", "restore_count", "previous_frame", "processed_frames"]
)
def test_continuity_counter_or_horizon_change_rejected(field):
    meta = dict(
        previous_frame=m.STOP - 1,
        processed_frames=m.STOP - m.FIRST,
        initialize_count=1,
        restore_count=4,
    )
    m.validate_lifecycle(meta, m.STOP, 4)
    meta[field] += 1
    with pytest.raises(ValueError, match="checkpoint horizon"):
        m.validate_lifecycle(meta, m.STOP, 4)


def gap_fixture():
    pose = np.zeros((6, 3, 3, 3))
    pose[:, 0, 1, 0] = [0, 1, 2, 15, 16, 17]
    observed = np.ones((6, 3, 3), bool)
    observed[1:3, 0, 1] = False
    observed[4:, 1, 1] = False
    return dict(
        frame_idx=np.arange(100, 106),
        pose_xyz=pose,
        observed=observed,
        identity_review=np.ones((6, 3), bool),
    )


def test_recovery_uses_last_direct_observation_outside_evaluation_window():
    result = m.gap_metrics(gap_fixture(), 101, 106)
    event = result["reacquisitions"][0]
    assert event["frame_idx"] == 103 and event["last_direct_frame"] == 100
    assert event["missing_frames"] == 2
    assert event["previous_latent_to_reacquired_step_mm"] == 13
    assert event["direct_endpoint_distance_mm"] == 15
    assert event["direct_endpoint_elapsed_frames"] == 3
    assert event["direct_endpoint_average_mm_per_s"] == 830
    assert result["directly_supported_adjacent_steps_gt10mm"] == 0
    assert result["maximum_directly_supported_adjacent_step_mm"] == 1
    assert event["identity_review"] and not event["biological_identity_confirmed"]
    assert result["sum_missing_chest_fish_frames"] == 4
    assert sum(r["missing_frames_in_evaluation"] for r in result["missing_runs"]) == 4
    assert result["missing_runs"][1]["closed_by_direct_observation"] is False


def test_gap_at_recording_start_has_no_manufactured_direct_endpoint():
    candidate = gap_fixture()
    candidate["observed"][0, 0, 1] = False
    result = m.gap_metrics(candidate, 100, 106)
    event = result["reacquisitions"][0]
    assert event["last_direct_frame"] is None
    assert event["direct_endpoint_average_mm_per_s"] is None
    assert event["missing_frames"] == 3


def test_non_native_input_is_rejected_before_motion_scoring():
    candidate = gap_fixture()
    candidate["frame_idx"][3:] += 1
    with pytest.raises(ValueError, match="native gaps"):
        m.gap_metrics(candidate, 100, 107)


def import_fixture():
    arrays = {"state_review": np.array([[True, False, True]], bool)}
    meta = dict(restore_count=3, context=dict(implementation="old", cameras=["synthetic_camera"]))
    new = dict(
        implementation="new",
        cameras=["synthetic_camera"],
        experimental_chest_policy={"fixture": True},
    )
    imported = copy.deepcopy(meta)
    imported.update(restore_count=4, context=new)
    signatures = {
        k: dict(
            dtype=v.dtype.str, shape=list(v.shape), sha256=m.hashlib.sha256(v.tobytes()).hexdigest()
        )
        for k, v in arrays.items()
    }
    migration = dict(
        schema="directional_chest_original_state_import_v1",
        status="ok",
        arrays=signatures,
        original_metadata=copy.deepcopy(meta),
        old_context=copy.deepcopy(meta["context"]),
        new_context=new,
        imported_metadata=imported,
    )
    return arrays, meta, migration


@pytest.mark.parametrize("change", ["state", "metadata", "restore", "calibration"])
def test_false_original_state_import_rejected(change):
    arrays, meta, migration = import_fixture()
    m.verify_import(migration, arrays, meta)
    if change == "state":
        arrays["state_review"][0, 0] = False
    elif change == "metadata":
        migration["original_metadata"]["restore_count"] = 4
    elif change == "restore":
        migration["imported_metadata"]["restore_count"] = 5
    else:
        migration["new_context"]["cameras"] = ["wrong_camera"]
    with pytest.raises(ValueError, match="import"):
        m.verify_import(migration, arrays, meta)


def test_unsealed_same_named_file_is_rejected(tmp_path):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unsealed"):
        m.require_bound(tmp_path, ["summary.json"], {})
