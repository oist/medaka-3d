"""Synthetic interim binding and native gap arithmetic; no real checkpoint read."""

import copy
import json

import audit_bounded_first_chunk as m
import numpy as np
import pytest


def import_fixture():
    arrays = {"state_pose": np.zeros((6, 3, 3, 3))}
    old_context = dict(implementation="Original", tracker_config=dict(device="cuda"))
    parent_meta = dict(context=old_context, restore_count=3, previous_frame=66399)
    directional_context = dict(
        **{**old_context, "implementation": "Directional"},
        experimental_chest_policy={"fixture": True},
    )
    directional_meta = {
        **copy.deepcopy(parent_meta),
        "context": directional_context,
        "restore_count": 4,
    }
    directional = dict(
        schema="directional_chest_original_state_import_v1",
        status="ok",
        arrays=m.array_signature(arrays),
        original_metadata=parent_meta,
        old_context=old_context,
        new_context=directional_context,
        imported_metadata=directional_meta,
    )
    new_context = dict(
        **{**directional_context, "implementation": "Bounded"},
        bounded_mouth_policy=dict(enabled=True, clears_identity_review=False),
    )
    imported = {**copy.deepcopy(parent_meta), "context": new_context, "restore_count": 4}
    migration = dict(
        schema="bounded_mouth_original_state_import_v1",
        status="ok",
        directional_validation=directional,
        directional_context=directional_context,
        arrays=m.array_signature(arrays),
        original_metadata=parent_meta,
        old_context=old_context,
        new_context=new_context,
        imported_metadata=imported,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        review_cleared_by_migration=False,
    )
    return migration, arrays, parent_meta, {"context": new_context}


def test_declared_two_stage_migration_preserves_one_logical_restore():
    migration, arrays, meta, new = import_fixture()
    m.verify_bounded_import(migration, arrays, meta, new)
    assert meta["restore_count"] == 3


@pytest.mark.parametrize("case", ["array", "double_restore", "nested", "policy", "review"])
def test_migration_rejects_silent_change(case):
    migration, arrays, meta, new = copy.deepcopy(import_fixture())
    if case == "array":
        arrays["state_pose"].flat[0] = 1
    if case == "double_restore":
        migration["imported_metadata"]["restore_count"] = 5
    if case == "nested":
        migration["directional_validation"]["arrays"] = {}
    if case == "policy":
        migration["new_context"]["bounded_mouth_policy"]["enabled"] = False
    if case == "review":
        migration["review_cleared_by_migration"] = True
    with pytest.raises(ValueError):
        m.verify_bounded_import(migration, arrays, meta, new)


@pytest.mark.parametrize("case", ["empty", "digest", "path", "malformed_sha"])
def test_uncommitted_or_wrong_pointer_rejected_before_payload(tmp_path, case):
    branch = tmp_path
    target = branch / "checkpoints/through_068060"
    row = dict(path=str(target), manifest_sha256="a" * 64)
    if case == "digest":
        row["manifest_sha256"] = "b" * 64
    if case == "path":
        row["path"] = str(branch / "checkpoints/through_069720")
    (branch / "checkpoint_progress.json").write_text(json.dumps([] if case == "empty" else [row]))
    with pytest.raises(ValueError):
        m.committed(branch, "not-a-sha" if case == "malformed_sha" else "a" * 64, {})
    assert not target.exists()


def candidate():
    frames = np.arange(m.START, m.STOP, dtype=np.int64)
    return dict(
        frame_idx=frames,
        observed=np.ones((len(frames), 3, 3), bool),
        identity_review=np.zeros((len(frames), 3), bool),
        sources=np.zeros((len(frames), 3, 3, 3), int),
        pose_xyz=np.zeros((len(frames), 3, 3, 3)),
    )


def test_gain_loss_counts_keep_slot_node_and_review_denominator():
    old, new = candidate(), candidate()
    old["observed"][0, 0, 0] = False
    new["observed"][1, 2, 1] = False
    new["identity_review"][0, 0] = True
    summary, changes = m.compare_support(old, new)
    assert summary["gained_by_node"] == [1, 0, 0]
    assert summary["lost_by_node"] == [0, 1, 0]
    assert len(changes) == 2 and changes[0]["bounded_review"]
    assert changes[0]["change"] == "gain" and changes[1]["slot"] == 2
    new["frame_idx"][0] += 1
    with pytest.raises(ValueError, match="native"):
        m.compare_support(old, new)


def test_all_node_reacquisition_elapsed_and_terminal_unknown():
    data = candidate()
    data["observed"][11:14, 0, 0] = False
    data["pose_xyz"][14, 0, 0, 0] = 4
    data["observed"][-5:, 2, 2] = False
    result = m.node_reacquisitions(data)
    (event,) = result["events"]
    assert event["node"] == "mouth" and event["missing_frames"] == 3
    assert event["direct_endpoint_elapsed_frames"] == 4
    assert event["direct_endpoint_distance_mm"] == 4
    assert event["direct_endpoint_average_mm_per_s"] == 166
    terminal = [r for r in result["missing_runs"] if r["terminal_open"]]
    assert len(terminal) == 1 and terminal[0]["node"] == "tail"
    assert terminal[0]["missing_frames_in_interval"] == 5
