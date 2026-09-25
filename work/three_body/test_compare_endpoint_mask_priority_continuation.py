"""Bounded synthetic headers, motion axes, probe binding and selective codec reads."""

import copy
from pathlib import Path

import compare_endpoint_mask_priority_continuation as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def confined(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "CONTROL", tmp_path / "control")

    def forbidden(*args, **kwargs):
        pytest.fail("actual producer or candidate loading is forbidden")

    for name in ("read_stage", "load_candidate", "load_priority_probes"):
        monkeypatch.setattr(m, name, forbidden)
    real_load = np.load

    def limited(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", limited)


def headers():
    original = dict(implementation="SparseInstanceRescoreContinuousTracker", initial_pose="seed")
    control = dict(
        retained_interval=[m.FIRST, m.STOP],
        comparison_interval=[m.START, m.STOP],
        terminal_horizon=m.STOP,
        parent_tracking_sha256=m.PARENT_SHA,
        control_checkpoint_manifest_sha256=m.CONTROL_CHECKPOINT_SHA,
        tracking_performed=False,
        original_candidate_116200_used=False,
        original_binding={"seed": "fixed"},
    )
    meta = dict(
        previous_frame=m.STOP - 1,
        processed_frames=m.STOP - m.FIRST,
        initialize_count=1,
        restore_count=6,
        context=original,
    )
    initial = dict(meta, previous_frame=m.START - 1, processed_frames=m.START - m.FIRST)
    result = {"sparse_control": dict(summary=control, metadata=meta)}
    for arm in m.ARMS[1:]:
        context = dict(
            original, implementation="BodyRowContinuousTracker", body_row_policy={"start": m.START}
        )
        if arm == "endpoint_priority":
            context.update(
                implementation="EndpointMaskPriorityTracker",
                endpoint_mask_priority_policy=dict(
                    effective_from_native=m.START,
                    original_fit_local_path_costs_and_five_plus_sixth_terms_unchanged=True,
                ),
            )
        source = {str((m.WORK / m.SPECS[arm][1]).resolve()): m.SPECS[arm][2]}
        plan = dict(
            process_interval=[m.START, m.STOP],
            retained_interval=[m.FIRST, m.STOP],
            scientific_interval=[66400, m.STOP],
            parent_tracking_sha256=m.PARENT_SHA,
            parent_checkpoint_sha256=m.INITIAL_SHA,
            initialize_count=1,
            restore_count=7,
            new_begin_calls=0,
            external_pose_resets=0,
            mask_split_priority_applied=arm == "endpoint_priority",
            source_sha256=source,
            original_binding=control["original_binding"],
            tracker_context=context,
        )
        imported = dict(
            status="ok",
            previous_frame=m.START - 1,
            processed_frames=m.START - m.FIRST,
            effective_from_native=m.START,
            added_restore_count=1,
            new_begin_calls=0,
            all_arrays_byte_preserved=True,
            original_metadata_unchanged=True,
            arrays={str(i): {"synthetic_signature": i} for i in range(39)},
            original_metadata=copy.deepcopy(initial),
            new_context=context,
        )
        result[arm] = dict(
            plan=plan,
            summary=dict(
                plan,
                finish_calls=1,
                all39_roundtrip=True,
                prior_history_prefix_unchanged=True,
                finish_matches_checkpoint_backtrace=True,
            ),
            imported=imported,
            seal={"inputs": source},
            index=dict(
                original_binding=control["original_binding"],
                prefix_chunks=[
                    dict(index=n, start=m.FIRST + n * m.CHUNK, stop=m.FIRST + (n + 1) * m.CHUNK)
                    for n in range(25)
                ],
            ),
        )
    return result


def test_three_arm_header_contract_rejects_window_seed_chunk_and_parent_changes():
    good = headers()
    m.validate_headers(good)
    for damage in ("parent", "chunk", "seed", "initial_array", "terminal"):
        bad = copy.deepcopy(good)
        if damage == "parent":
            bad["endpoint_priority"]["plan"]["parent_checkpoint_sha256"] = "0" * 64
        elif damage == "chunk":
            bad["endpoint_priority"]["index"]["prefix_chunks"][24]["start"] += 1
        elif damage == "seed":
            bad["endpoint_priority"]["plan"]["original_binding"] = {"seed": "other"}
        elif damage == "initial_array":
            bad["endpoint_priority"]["imported"]["arrays"]["3"] = {"different": True}
        else:
            bad["sparse_control"]["summary"]["terminal_horizon"] = 116200
        with pytest.raises(ValueError):
            m.validate_headers(bad)


def candidate(monkeypatch):
    for key, value in (("FIRST", 100), ("START", 100), ("STOP", 105), ("FRAMES", (100, 102, 104))):
        monkeypatch.setattr(m, key, value)
    pose = np.zeros((5, 3, 3, 3), np.float64)
    pose[:, :, :, 0] = np.array([0.0, 20.0, 50.0])[None, :, None]
    pose[:, :, :, 1] = [0.0, 2.0, 5.0]
    sources = np.broadcast_to(
        np.arange(3, dtype=np.int32)[None, :, None, None], (5, 3, 3, 3)
    ).copy()
    return dict(
        frame_idx=np.arange(100, 105, dtype=np.int64),
        pose_xyz=pose,
        observed=np.ones((5, 3, 3), bool),
        sources=sources,
        identity_review=np.ones((5, 3), bool),
        costs=np.zeros(5),
        path_costs=np.zeros(5),
        loss_parts=np.zeros((5, 5)),
        proposal_kind=np.full(5, "predict", dtype="<U22"),
    )


def test_support_loss_does_not_turn_latent_motion_into_direct_success(monkeypatch):
    c = candidate(monkeypatch)
    c["pose_xyz"][2:, 0, 2, 2] += 12
    c["observed"][2, 0, 2] = False
    c["sources"][2, 0, :, 2] = -1
    c["pose_xyz"][2:, 1, 2, 2] += 11
    c["pose_xyz"][2:, 2, 2, 2] += 10
    c["sources"][0, 1, 0, 2] = 2
    m.validate_candidate_fields(c)
    result = m.summarize(c)
    assert result["missing_by_slot_node"][0][2] == 1
    assert result["longest_missing_run_by_slot_node"][0][2] == 1
    assert result["directly_supported_adjacent_pairs_by_slot_node"][0][2] == 2
    assert result["all_estimated_max_step_by_slot_node_mm"][0][2] == 12
    assert [
        (e["raw_slot"], e["node"], e["frame_idx"])
        for e in result["direct_adjacent_steps_over_10mm"]
    ] == [(1, 2, 102)]
    assert result["mixed_source_rows_by_slot_camera"][1][0] == 1
    broken = copy.deepcopy(c)
    broken["frame_idx"][2] += 1
    with pytest.raises(ValueError):
        m.validate_candidate_fields(broken)


def test_all_six_chest_permutations_have_correct_non_self_inverse_direction(monkeypatch):
    a = candidate(monkeypatch)
    b = copy.deepcopy(a)
    for field in ("pose_xyz", "observed", "sources", "identity_review"):
        b[field] = b[field][:, [2, 0, 1]].copy()
    candidates = dict(zip(m.ARMS, (a, b, copy.deepcopy(a)), strict=True))
    _, samples, pairs, arrays = m.compare_arrays(candidates)
    assert len(samples) == 27
    key = "sparse_control_vs_nodewise"
    rms = arrays[key + "_chest_rms_mm"]
    choice = np.where((m.PERMUTATIONS == [1, 2, 0]).all(1))[0][0]
    assert rms.shape == (5, 6) and np.all(rms[:, choice] == 0)
    assert np.all(np.argmin(rms, axis=1) == choice)
    assert pairs[key]["raw_slot_pose_delta_mm"]["maximum"] > 0
    assert pairs[key]["unique_nonidentity_chest_permutation_frames"] == 5
    assert pairs[key]["framewise_geometry_is_not_temporal_ID_alignment"]


def fitted_and_endpoint(c):
    n = 3
    row = 2
    fit = dict(
        candidate_index=np.arange(n),
        candidate_pose_xyz=np.repeat(c["pose_xyz"][row : row + 1], n, axis=0),
        candidate_parent=np.zeros(n, dtype=np.int64),
        candidate_plan=np.repeat(c["sources"][row : row + 1], n, axis=0),
        candidate_kind=["predict"] * n,
        adjusted_local_cost=np.zeros(n),
        adjusted_path_cost=np.zeros(n),
        original_five_loss_parts=np.zeros((n, 5)),
    )
    labels = np.full((n, 3, 3), "same_instance", dtype="<U24")
    labels[:, 0, 1] = "split_instances"
    labels[:, 2, 0] = "unavailable"
    endpoint = dict(
        candidate_index=np.arange(n),
        candidate_parent=np.zeros(n, dtype=np.int64),
        candidate_kind=["predict"] * n,
        candidate_local_cost=np.zeros(n),
        candidate_path_cost=np.zeros(n),
        selected_candidate_indices=np.array([1, 2]),
        candidate_split_by_body=(labels == "split_instances").any(-1),
        unknown_by_candidate_body_view=labels == "unavailable",
        views={v: dict(classification_by_body=labels[:, :, i]) for i, v in enumerate(m.VIEWS)},
    )
    return fit, endpoint


def test_final_probe_preserves_all_matches_unknown_and_rejects_conflicting_same_pose(monkeypatch):
    c = candidate(monkeypatch)
    fit, end = fitted_and_endpoint(c)
    row = m.bind_final_probe(c, 102, fit, end)
    assert row["matching_candidate_indices"] == [0, 1, 2]
    assert row["matching_retained_candidate_indices"] == [1, 2]
    assert row["split_by_raw_body"] == [True, False, False]
    assert row["unknown_by_raw_body_view"][2][0]
    assert not row["parent_chain_independently_decoded"]
    end["views"][m.VIEWS[0]]["classification_by_body"][0, 2] = "same_instance"
    end["unknown_by_candidate_body_view"][0, 2, 0] = False
    with pytest.raises(ValueError, match="identical pose"):
        m.bind_final_probe(c, 102, fit, end)


def test_thin_codec_verifies_all_members_but_decodes_only_requested_fields(tmp_path, monkeypatch):
    root = tmp_path / "producer"
    out = m.proof.Output(root)
    value = dict(
        schema="instance_rescore_frame_diagnostics_v1",
        frame_idx=102,
        wanted=np.array([1.0, 2.0]),
        unused=np.ones((50, 50)),
    )
    tree, arrays = m.codec.split_diagnostic(value)
    record = out.write("diagnostics/f102.npz", arrays)
    out.write("diagnostics/f102.json", dict(archive=record, tree=tree))
    loaded = m.thin_companion(
        dict(root=root, seal={"outputs": out.records}),
        "diagnostics",
        102,
        ("wanted",),
        m.proof.Evidence(),
    )
    assert list(loaded) == ["wanted"]
    np.testing.assert_array_equal(loaded["wanted"], [1.0, 2.0])


def test_outside_output_stops_before_any_producer_input(tmp_path):
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(tmp_path / "a", "a" * 64, tmp_path / "b", "b" * 64, tmp_path.parent / "outside")
