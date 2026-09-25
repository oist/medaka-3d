"""Synthetic CPU contracts; no recording, checkpoint payload, human data or CUDA."""

import copy
from pathlib import Path
from types import SimpleNamespace

import compare_sparse_instance_five_minutes as m
import numpy as np
import pytest
from test_compare_fast_tail_two_minutes import candidate as tiny_candidate
from test_compare_fast_tail_two_minutes import refresh_derived


def summary(arm):
    spec = m.SPECS[arm]
    value = dict(
        schema=spec["stage"],
        status="ok",
        scientific_interval=[m.START, m.STOP],
        process_interval=[spec["resume"], m.STOP],
        native_frames=49800,
        process_frames=m.STOP - spec["resume"],
        initialize_count=1,
        restore_count=spec["restores"],
        external_pose_resets=0,
        new_begin_calls=0,
        finish_calls=1,
        physical_ID_truth=False,
        checkpoints=[
            dict(
                arrays=39,
                previous_frame=f - 1,
                processed_frames=f - m.FIRST,
                initialize_count=1,
                restore_count=spec["restores"],
            )
            for f in range(spec["resume"] + m.CHUNK, m.STOP + m.CHUNK, m.CHUNK)
        ],
    )
    if arm == m.ARMS[1]:
        value.update(
            retained_interval=[m.FIRST, m.STOP],
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
            finish_matches_checkpoint_backtrace=True,
            automatic_adoption=False,
        )
    return value


@pytest.mark.parametrize("arm", m.ARMS)
def test_complete_five_minutes_and_ordered_lifecycle(arm):
    document = summary(arm)
    before = copy.deepcopy(document)
    m.completion(document, arm)
    assert document == before and document["native_frames"] == 49800


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(status="failed"),
        lambda v: v.update(native_frames=4980),
        lambda v: v.update(scientific_interval=[66400, 106240]),
        lambda v: v.update(finish_calls=0),
        lambda v: v.update(finish_matches_checkpoint_backtrace=1),
        lambda v: v.update(all39_roundtrip=False),
        lambda v: v.update(external_pose_resets=1),
        lambda v: v.update(restore_count=7),
        lambda v: v["checkpoints"].pop(),
        lambda v: v["checkpoints"][0].update(previous_frame=79678),
    ],
)
def test_partial_false_completion_and_reset_rejected(change):
    value = summary(m.ARMS[1])
    change(value)
    with pytest.raises(ValueError):
        m.completion(value, m.ARMS[1])


def lineage_fixture():
    old = dict(
        implementation="FastTailRecoveryContinuousTracker",
        calibration="synthetic",
        lower=m.support.COMMON.digest_array(np.array([0.0, 0.0, -6.0])),
        upper=m.support.COMMON.digest_array(np.array([192.0, 192.0, 186.0])),
        recovery_config=dict(lower_mm=[0.0, 0.0, -6.0]),
        tracker_config=dict(mask_weight=1),
    )
    policy = dict(effective_from_native=m.MASK_START, semantic=False)
    dense = dict(
        copy.deepcopy(old),
        implementation="InstanceRescoreContinuousTracker",
        instance_rescore_policy=policy,
    )
    backend = dict(
        effective_from_native=m.SPARSE_START,
        saved_workload_qualification_sha256=m.QUALIFICATION_SHA,
    )
    new = dict(
        copy.deepcopy(dense),
        implementation="SparseInstanceRescoreContinuousTracker",
        sparse_distance_backend=backend,
    )
    before = dict(
        context=copy.deepcopy(dense),
        previous_frame=m.SPARSE_START - 1,
        processed_frames=m.SPARSE_START - m.FIRST,
        initialize_count=1,
        restore_count=5,
    )
    imported = dict(
        schema="sparse_instance_dense_state_import_v1",
        status="ok",
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        processed_frames=m.SPARSE_START - m.FIRST,
        previous_frame=m.SPARSE_START - 1,
        effective_from_native=m.SPARSE_START,
        backend_applied_retroactively_to_prior_history=False,
        review_cleared_by_migration=False,
        companion_diagnostics_required_each_future_frame=True,
        arrays={str(i): {} for i in range(39)},
        old_context=copy.deepcopy(dense),
        new_context=copy.deepcopy(new),
        original_metadata=copy.deepcopy(before),
        imported_metadata=dict(copy.deepcopy(before), context=copy.deepcopy(new), restore_count=6),
    )
    final_binding = dict(
        sparse_distance_backend=backend,
        scoring_policy=policy,
        parent_tracking_sha256=m.PARENT_SHA,
        parent_checkpoint_sha256="a" * 64,
    )
    plan = dict(
        retained_interval=[m.FIRST, m.STOP],
        dense_serial_control_interval=[m.SPARSE_START, m.SPARSE_START + 2],
        sparse_rollout_continues_after_control_without_restore=True,
        mask_policy_effective_from_native=m.MASK_START,
        sparse_backend_effective_from_native=m.SPARSE_START,
        semantic_landmark_term_used=False,
        checkpoint_binding=final_binding,
    )
    parent = m.WORK / m.PARENT_STAGE
    qualification = (
        m.WORK / "sparse_instance_saved_ten_second_qualification_v1/qualification.provenance.json"
    )
    control = dict(
        docs=dict(
            metadata=dict(context=old),
            plan=dict(
                checkpoint_binding=dict(
                    parent_checkpoint_sha256=m.common.PARENT_SHA,
                    audited_parent_sha256=m.common.PARENT_AUDIT_SHA,
                )
            ),
        )
    )
    root = m.WORK / m.SPECS[m.ARMS[1]]["stage"]
    variant = dict(
        root=root,
        seal=dict(
            inputs={
                str(parent / "tracking.provenance.json"): m.PARENT_SHA,
                str(qualification): m.QUALIFICATION_SHA,
            }
        ),
        docs=dict(plan=plan, metadata=dict(context=new)),
    )
    serial = dict(
        status="ok",
        native_interval=[m.SPARSE_START, m.SPARSE_START + 2],
        all39_BYTEEXACT=True,
        initialize_count=1,
        restore_count=6,
        metadata_difference="declared backend context and implementation only",
        full_five_minute_equivalence_proven=False,
        backend=copy.deepcopy(backend),
        compared_arrays={str(i): {} for i in range(39)},
    )
    documents = {
        (root, "state_import.json"): imported,
        (root, "serial_dense_sparse_qualification.json"): serial,
        (parent, "checkpoints/through_078020/metadata.json"): before,
        (parent, "summary.json"): dict(
            checkpoint_manifest_sha256="a" * 64,
            parent_checkpoint_sha256=m.common.PARENT_SHA,
            parent_audit_sha256=m.common.PARENT_AUDIT_SHA,
        ),
    }
    seals = {
        parent / "tracking.provenance.json": dict(stage=m.PARENT_STAGE, status="ok"),
        qualification: dict(status="ok"),
    }
    return control, variant, documents, seals


def bind_fixture(monkeypatch, documents, seals):
    monkeypatch.setattr(
        m, "member_doc", lambda root, seal, name, evidence: copy.deepcopy(documents[root, name])
    )
    monkeypatch.setattr(
        m.common, "read", lambda path, evidence, expected: copy.deepcopy(seals[path])
    )


def test_actual_declared_parent_backend_transition_is_not_a_new_initialization(monkeypatch):
    control, variant, documents, seals = lineage_fixture()
    before = copy.deepcopy((control, variant, documents, seals))
    bind_fixture(monkeypatch, documents, seals)
    value = m.validate_sparse_lineage(control, variant, None)
    assert value["serial_two_native"]["all39_BYTEEXACT"] is True
    assert value["sparse_import"]["added_restore_count"] == 1
    assert (control, variant, documents, seals) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("review_cleared_by_migration", True),
        ("all_arrays_byte_preserved", 1),
        ("backend_applied_retroactively_to_prior_history", True),
        ("effective_from_native", 76360),
        ("added_restore_count", 2),
    ],
)
def test_undeclared_import_changes_rejected(monkeypatch, field, value):
    control, variant, documents, seals = lineage_fixture()
    documents[variant["root"], "state_import.json"][field] = value
    bind_fixture(monkeypatch, documents, seals)
    with pytest.raises(ValueError):
        m.validate_sparse_lineage(control, variant, None)


def test_serial_gpu_control_is_mandatory_and_cannot_claim_full_rollout_equivalence(monkeypatch):
    control, variant, documents, seals = lineage_fixture()
    documents[variant["root"], "serial_dense_sparse_qualification.json"]["all39_BYTEEXACT"] = False
    bind_fixture(monkeypatch, documents, seals)
    with pytest.raises(ValueError, match="actual two-frame"):
        m.validate_sparse_lineage(control, variant, None)


def test_new_depth_policy_is_rejected_even_if_both_arms_claim_it(monkeypatch):
    control, variant, documents, seals = lineage_fixture()
    changed = m.support.COMMON.digest_array(np.array([0.0, 0.0, -12.0]))

    # Make every cross-arm/context comparison internally agree: fixed-domain check must still fail.
    def change(tree):
        if isinstance(tree, dict):
            if "lower" in tree:
                tree["lower"] = changed
            for value in tree.values():
                change(value)

    change(control)
    change(variant)
    change(documents)
    bind_fixture(monkeypatch, documents, seals)
    with pytest.raises(ValueError, match="unchanged -6"):
        m.validate_sparse_lineage(control, variant, None)


def test_mixed_consumed_source_chunk_rejected_without_payload_access():
    docs = dict(
        plan=dict(original_binding=dict(seed="same")),
        input_index=dict(
            original_binding=dict(seed="same"), prefix_chunks=[dict(index=0, source="a")]
        ),
    )
    bundles = {a: dict(docs=copy.deepcopy(docs)) for a in m.ARMS}
    assert m.common_observation_binding(bundles) == dict(seed="same")
    bundles[m.ARMS[1]]["docs"]["input_index"]["prefix_chunks"][0]["source"] = "b"
    with pytest.raises(ValueError, match="consumed40"):
        m.common_observation_binding(bundles)


def test_prefix_change_is_same_horizon_not_immutable_history_edit(monkeypatch):
    monkeypatch.setattr(m, "FIRST", 10)
    monkeypatch.setattr(m, "MASK_START", 13)
    monkeypatch.setattr(m, "STOP", 16)
    old = dict(frame_idx=np.arange(10, 16), **{f"field{i}": np.zeros(6) for i in range(20)})
    new = {k: a.copy() for k, a in old.items()}
    new["field0"][0] = -0.0
    new["field1"][-1] = 9
    value, arrays = m.selected_prefix_changes(old, new)
    assert value["terminal_horizon_by_arm"] == dict.fromkeys(m.ARMS, 16)
    assert value["fields"]["field0"]["changed_rows"] == 1
    assert value["fields"]["field1"]["changed_rows"] == 0
    assert "snapshot_horizon" not in value
    assert arrays["frame_idx"].tolist() == [10, 11, 12]


def test_geometry_retains_latent_nodes_and_excludes_interval_boundary_jump():
    pose = np.zeros((6, 3, 3, 3))
    pose[:, :, :, 0] = np.arange(6)[:, None, None]
    observed = np.ones((6, 3, 3), bool)
    observed[:, 2] = False
    old = dict(frame_idx=np.arange(10, 16), pose_xyz=pose, observed=observed)
    new = {k: a.copy() for k, a in old.items()}
    new["pose_xyz"][:3] += 1000
    new["pose_xyz"][3:, 2, 0, 2] += 4
    before = m.common.signature(old), m.common.signature(new)
    value, arrays = m.geometry_deltas(old, new, 13, 16)
    assert arrays["pose_delta_mm"][:, 2, 0].tolist() == [4, 4, 4]
    assert np.all(arrays["all_estimated_speed_mm_s"] == 166)
    assert np.all(arrays["all_estimated_acceleration_mm_s2"] == 0)
    assert not arrays["speed_both_direct"][:, :, 2].any()
    assert set(value["speed_by_arm"]) == set(m.ARMS)
    assert (m.common.signature(old), m.common.signature(new)) == before


def test_guard_runs_before_path_access_or_output_creation(monkeypatch):
    monkeypatch.setattr(m, "EXECUTION_READY", False)
    with pytest.raises(ValueError, match="source review"):
        m.run("a" * 64, "b" * 64, Path("do-not-create"))


def test_actual_reporting_intervals_and_legacy_field_arms_are_explicit():
    assert m.INTERVALS["full"] == (66400, 116200)
    assert [m.INTERVALS[f"minute_{i}"] for i in range(1, 6)] == [
        (66400, 76360),
        (76360, 86320),
        (86320, 96280),
        (96280, 106240),
        (106240, 116200),
    ]
    assert m.LEGACY_DIFFERENCE_FIELD_ARMS["original_sources"] == m.ARMS[0]
    assert m.LEGACY_DIFFERENCE_FIELD_ARMS["tail_sources"] == m.ARMS[1]
    assert len(m.metrics.old.old.ANCHOR_FRAMES) == 11
    assert all(66400 <= f < 76360 for f in m.metrics.old.old.ANCHOR_FRAMES)


def candidate_bundles(tmp_path, monkeypatch, defect=None):
    # Only this NEW caller's extent is reduced; every frozen helper validates actual arrays.
    monkeypatch.setattr(m, "FIRST", 100)
    monkeypatch.setattr(m, "STOP", 108)
    bundles = {}
    for arm in m.ARMS:
        out = m.proof.Output(tmp_path / arm)
        data, _ = tiny_candidate()
        data["identity_review"][2, 1] = True
        refresh_derived(data)
        out.write("candidate.npz", data)
        milestone = {k: v.copy() for k, v in reversed(tuple(data.items()))}
        if defect == "milestone" and arm == m.ARMS[1]:
            milestone["loss_parts"][3, 4] = 1
        out.write("milestones/through_116200/candidate.npz", milestone)
        signatures = m.common.signature(data)
        if defect == "signature" and arm == m.ARMS[1]:
            signatures["loss_parts"]["sha256"] = "0" * 64
        bundles[arm] = dict(
            root=out.root,
            seal=dict(outputs=out.records),
            docs=dict(summary=dict(candidate_arrays=signatures)),
        )
    return bundles


def test_all21_candidate_and_milestone_match_with_different_archive_sha(tmp_path, monkeypatch):
    bundles = candidate_bundles(tmp_path, monkeypatch)
    loaded, refs = m.load_candidates(bundles, m.proof.Evidence())
    for arm in m.ARMS:
        assert len(loaded[arm]) == 21
        assert refs[arm]["sha256"] != refs[arm]["milestone_sha256"]
        assert loaded[arm]["identity_review"][2, 1]
        assert np.isnan(loaded[arm]["accepted_xyz"][2, 1]).all()
        assert all(not a.flags.writeable for a in loaded[arm].values())
        assert loaded[arm]["loss_parts"].shape == (8, 5)


@pytest.mark.parametrize("defect", ["milestone", "signature"])
def test_candidate_rejects_changed_fifth_loss_field_or_sealed_signature(
    tmp_path, monkeypatch, defect
):
    bundles = candidate_bundles(tmp_path, monkeypatch, defect)
    with pytest.raises(ValueError):
        m.load_candidates(bundles, m.proof.Evidence())


def test_failed_final_seal_rejected_before_any_member_or_payload(monkeypatch):
    monkeypatch.setattr(
        m.common, "read", lambda *args: dict(stage=m.SPECS[m.ARMS[1]]["stage"], status="failed")
    )
    monkeypatch.setattr(m, "member_doc", lambda *args: pytest.fail("failed seal opened a member"))
    monkeypatch.setattr(m.np, "load", lambda *args, **kw: pytest.fail("failed seal opened payload"))
    with pytest.raises(ValueError, match="final tracking seal failed"):
        m.load_headers(m.ARMS[1], "a" * 64, None)


def test_no_cuda_was_initialized():
    assert not m.common.sys.modules["torch"].cuda.is_initialized()


def test_source_inventory_ignores_generated_relative_module_filename(monkeypatch):
    monkeypatch.setitem(
        m.sys.modules, "synthetic_generated_module", SimpleNamespace(__file__="_classes.py")
    )
    code = m.source_code()
    assert str(m.REPO / "_classes.py") not in code
    assert code[str(Path(m.__file__).resolve())] == m.proof.sha(m.__file__)


def test_exact_frozen_execution_contract_is_a_bound_source_member():
    path = m.WORK / "FAST_TAIL_TWO_MINUTE_EXECUTION_CONTRACT.md"
    assert m.is_producer_source(path, m.EXECUTION_CONTRACT_SHA)
    assert not m.is_producer_source(path, "0" * 64)
    assert not m.is_producer_source(m.WORK / "another.md", m.EXECUTION_CONTRACT_SHA)
    assert not m.is_producer_source(m.WORK / "arrays.npz", "0" * 64)
