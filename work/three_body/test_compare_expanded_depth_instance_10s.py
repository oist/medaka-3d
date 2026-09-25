"""Synthetic source/domain, depth-distribution and saved-array checks only."""

import copy
from pathlib import Path

import compare_expanded_depth_instance_10s as m
import numpy as np
import pytest
from test_compare_fast_tail_two_minutes import candidate as tiny_candidate
from test_compare_fast_tail_two_minutes import refresh_derived


def completed(stage):
    return dict(
        schema=stage,
        status="ok",
        process_interval=[m.START, m.STOP],
        native_frames=1660,
        retained_interval=[m.FIRST, m.STOP],
        parent_audit_sha256=m.common.PARENT_AUDIT_SHA,
        parent_checkpoint_sha256=m.common.PARENT_SHA,
        initialize_count=1,
        restore_count=5,
        new_begin_calls=0,
        external_pose_resets=0,
        finish_calls=1,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        finish_matches_checkpoint_backtrace=True,
        diagnostic_frames=1660,
        physical_ID_truth=False,
        automatic_adoption=False,
    )


@pytest.mark.parametrize("stage", [m.CONTROL_STAGE, m.VARIANT_STAGE])
def test_complete_same_horizon_has_one_logical_initialization(stage):
    summary = completed(stage)
    before = copy.deepcopy(summary)
    m.completion(summary, stage)
    assert summary == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_frames", 1659),
        ("finish_calls", 0),
        ("all39_roundtrip", 1),
        ("retained_interval", [49800, 86320]),
        ("external_pose_resets", 1),
        ("restore_count", 6),
        ("diagnostic_frames", 1659),
        ("parent_checkpoint_sha256", "0" * 64),
    ],
)
def test_partial_or_changed_parent_rejected(field, value):
    summary = completed(m.VARIANT_STAGE)
    summary[field] = value
    with pytest.raises(ValueError):
        m.completion(summary, m.VARIANT_STAGE)


def domains():
    old = dict(
        implementation="FastTailRecoveryContinuousTracker",
        calibration="synthetic",
        lower=m.support.COMMON.digest_array(np.array([0.0, 0.0, -6.0])),
        upper=m.support.COMMON.digest_array(np.array([192.0, 192.0, 186.0])),
        recovery_config=dict(lower_mm=[0.0, 0.0, -6.0], rms=2.0),
        tracker_config=dict(mask_weight=1.0),
    )
    policy = dict(effective_from_native=m.START, clears_identity_review=False)
    dense = dict(
        copy.deepcopy(old),
        implementation="InstanceRescoreContinuousTracker",
        instance_rescore_policy=policy,
    )
    before = dict(
        context=copy.deepcopy(old),
        previous_frame=m.START - 1,
        processed_frames=m.START - m.FIRST,
        initialize_count=1,
        restore_count=4,
    )
    imported = dict(
        schema="instance_rescore_fast_tail_state_import_v1",
        status="ok",
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        processed_frames=m.START - m.FIRST,
        previous_frame=m.START - 1,
        effective_from_native=m.START,
        review_cleared_by_migration=False,
        companion_diagnostics_required_each_future_frame=True,
        arrays={str(i): {} for i in range(39)},
        old_context=copy.deepcopy(old),
        new_context=copy.deepcopy(dense),
        original_metadata=copy.deepcopy(before),
        imported_metadata=dict(
            copy.deepcopy(before), context=copy.deepcopy(dense), restore_count=5
        ),
    )
    backend = dict(
        effective_from_native=m.START, saved_workload_qualification_sha256=m.QUALIFICATION_SHA
    )
    depth = dict(
        schema="expanded_depth_instance_rescore_policy_v1",
        effective_from_native=m.START,
        original_lower_z_mm=-6.0,
        expanded_lower_z_mm=-12.0,
        scientific_search_domain_changed=True,
        old_history_rewritten=False,
        depth_policy_applied_retroactively_to_prior_history=False,
        measured_floor_or_physical_depth_truth_claimed=False,
        clears_identity_review=False,
    )
    expanded = dict(
        copy.deepcopy(dense),
        implementation="ExpandedDepthInstanceRescoreContinuousTracker",
        sparse_distance_backend=backend,
        expanded_depth_policy=depth,
        lower=m.support.COMMON.digest_array(np.array([0.0, 0.0, -12.0])),
    )
    expanded["recovery_config"]["lower_mm"] = [0.0, 0.0, -12.0]
    imp2 = dict(
        copy.deepcopy(imported),
        schema="expanded_depth_fast_tail_state_import_v1",
        new_context=copy.deepcopy(expanded),
        imported_metadata=dict(
            copy.deepcopy(before), context=copy.deepcopy(expanded), restore_count=5
        ),
        depth_policy_applied_retroactively_to_prior_history=False,
    )
    scope = dict(
        post_fit_candidate_selection_only=False,
        lower_z_changed_mm=[-6.0, -12.0],
        proposal_and_fitting_domains_changed=True,
        mask_loss_definition_and_normalization_unchanged=True,
        new_differentiated_instance_loss=False,
        x_y_upper_bounds_calibration_RMS_anatomy_support_age_review_unchanged=True,
        development_reuse=True,
        physical_depth_or_ID_truth=False,
        automatic_adoption=False,
    )
    policies = dict(
        new_scoring_policy=policy,
        distance_backend=backend,
        expanded_depth_policy=depth,
        scientific_scope=scope,
    )
    index = dict(
        original_binding=dict(seed="same"),
        prefix_chunks=[
            dict(
                index=i, start=m.FIRST + i * 1660, stop=m.FIRST + (i + 1) * 1660, source="synthetic"
            )
            for i in range(17)
        ],
    )
    rationale = {
        role: dict(path="synthetic", sha256="a" * 64, summary={})
        for role in ("free", "bounded", "sparse", "dense_control")
    }
    control = dict(
        docs=dict(
            state_import=imported,
            metadata=dict(context=dense),
            plan=dict(
                new_scoring_policy=policy,
                checkpoint_binding=dict(new_scoring_policy=policy),
                original_binding=dict(seed="same"),
            ),
            input_index=copy.deepcopy(index),
        )
    )
    variant = dict(
        docs=dict(
            state_import=imp2,
            metadata=dict(context=expanded),
            plan=dict(
                copy.deepcopy(policies),
                checkpoint_binding=copy.deepcopy(policies),
                original_binding=dict(seed="same"),
            ),
            summary=copy.deepcopy(policies),
            input_index=dict(copy.deepcopy(index), rationale_references=copy.deepcopy(rationale)),
        )
    )
    for doc in (
        variant["docs"]["plan"],
        variant["docs"]["summary"],
        variant["docs"]["plan"]["checkpoint_binding"],
    ):
        doc["rationale_references"] = copy.deepcopy(rationale)
    return control, variant


def test_only_declared_lower_domain_and_sparse_backend_change():
    args = domains()
    before = copy.deepcopy(args)
    proof = m.validate_domains(*args)
    assert proof["scientific_scope"]["proposal_and_fitting_domains_changed"]
    assert not proof["scientific_scope"]["post_fit_candidate_selection_only"]
    assert proof["retained39_payload_reads"] == 0
    assert args == before
    assert set(args[0]["docs"]["input_index"]) == {"original_binding", "prefix_chunks"}
    assert set(args[1]["docs"]["input_index"]) == {
        "original_binding",
        "prefix_chunks",
        "rationale_references",
    }
    assert len(args[0]["docs"]["input_index"]["prefix_chunks"]) == 17


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["state_import"]["arrays"].pop("38"),
        lambda d: d["state_import"].update(review_cleared_by_migration=True),
        lambda d: d["state_import"]["imported_metadata"].update(restore_count=6),
        lambda d: d["state_import"]["original_metadata"].update(previous_frame=76358),
        lambda d: d["summary"]["scientific_scope"].update(post_fit_candidate_selection_only=True),
        lambda d: d["summary"]["new_scoring_policy"].update(effective_from_native=78020),
        lambda d: d["input_index"]["prefix_chunks"][16].update(source="different"),
        lambda d: d["summary"]["rationale_references"]["free"].update(sha256="b" * 64),
    ],
)
def test_domain_import_and_policy_tampering_rejected(change):
    control, variant = domains()
    change(variant["docs"])
    with pytest.raises(ValueError):
        m.validate_domains(control, variant)


def test_undeclared_RMS_change_rejected_even_if_target_contexts_agree():
    control, variant = domains()
    for context in (
        variant["docs"]["metadata"]["context"],
        variant["docs"]["state_import"]["new_context"],
        variant["docs"]["state_import"]["imported_metadata"]["context"],
    ):
        context["recovery_config"]["rms"] = 3.0
    with pytest.raises(ValueError, match="undeclared calibration"):
        m.validate_domains(control, variant)


def z_candidate():
    pose = np.zeros((6, 3, 3, 3))
    pose[:, :, :, 2] = 20
    pose[:, 2, 0, 2] = [-12.01, -12.0, -11.75, -11.749, -6.0, -5.99]
    observed = np.ones((6, 3, 3), bool)
    observed[0:2, 2, 0] = False
    return dict(frame_idx=np.arange(100, 106), pose_xyz=pose, observed=observed)


def test_z_thresholds_and_inclusive_quarter_mm_band_include_latent_without_truth_claim():
    candidate = z_candidate()
    before = m.common.signature(candidate)
    summary, arrays = m.z_diagnostics(candidate, 100, 106)
    all_nodes = summary["subsets"]["all_estimated_including_latent"]
    direct = summary["subsets"]["directly_observed"]
    assert all_nodes["total"] == 54 and direct["total"] == 52
    assert all_nodes["counts"]["below_old_lower"]["by_slot_node"][2][0] == 4
    assert all_nodes["counts"]["below_expanded_lower"]["total"] == 1
    assert all_nodes["counts"]["within_expanded_lower_band"]["total"] == 2
    assert direct["counts"]["below_old_lower"]["total"] == 2
    assert direct["counts"]["below_expanded_lower"]["total"] == 0
    assert direct["counts"]["within_expanded_lower_band"]["total"] == 1
    assert summary["boundary_band_inclusive_mm"] == [-12.0, -11.75]
    assert summary["boundary_band_is_acceptance_gate"] is False
    assert arrays["z_mm"][0, 2, 0] == -12.01  # Keep out-of-domain diagnostics, no clipping.
    assert m.common.signature(candidate) == before


def test_z_empty_direct_subset_is_reported_without_inventing_observations():
    candidate = z_candidate()
    candidate["observed"][:] = False
    summary, _ = m.z_diagnostics(candidate, 100, 106)
    assert summary["subsets"]["directly_observed"]["total"] == 0
    assert summary["subsets"]["all_estimated_including_latent"]["total"] == 54


def test_z_rejects_discontinuous_same_count_frames():
    candidate = z_candidate()
    candidate["frame_idx"][2] = 101
    with pytest.raises(ValueError, match="complete z"):
        m.z_diagnostics(candidate, 100, 106)


def test_loss_runs_preserve_terminal_and_single_frame_losses():
    loss = np.zeros((6, 3, 3), bool)
    loss[:2, 2, 0] = True
    loss[3:, 2, 0] = True
    loss[2, 0, 1] = True
    rows = m.support_loss_runs(np.arange(100, 106), loss)
    assert [(r["start"], r["stop"], r["frames"], r["slot"]) for r in rows] == [
        (100, 102, 2, 2),
        (102, 103, 1, 0),
        (103, 106, 3, 2),
    ]
    assert sum(r["frames"] for r in rows) == int(loss.sum())


def saved_bundles(tmp_path, monkeypatch, tamper=False):
    monkeypatch.setattr(m, "FIRST", 100)
    monkeypatch.setattr(m, "STOP", 108)
    result = {}
    for arm in m.ARMS:
        out = m.proof.Output(tmp_path / arm)
        data, _ = tiny_candidate()
        data["identity_review"][0, 2] = True
        refresh_derived(data)
        out.write("candidate.npz", data)
        signatures = m.common.signature(data)
        if tamper and arm == m.ARMS[1]:
            signatures["sources"]["sha256"] = "0" * 64
        result[arm] = dict(
            root=out.root,
            seal=dict(outputs=out.records),
            docs=dict(summary=dict(candidate_arrays=signatures)),
            seal_sha256="a" * 64,
        )
    return result


def test_saved_candidate_all21_keeps_review_five_parts_and_raw_slots(tmp_path, monkeypatch):
    bundles = saved_bundles(tmp_path, monkeypatch)
    candidates, refs = m.load_candidates(bundles, m.proof.Evidence())
    for arm in m.ARMS:
        assert len(candidates[arm]) == len(refs[arm]["arrays"]) == 21
        assert candidates[arm]["loss_parts"].shape == (8, 5)
        assert candidates[arm]["identity_review"][0, 2]
        assert np.isnan(candidates[arm]["accepted_xyz"][0, 2]).all()
        assert not candidates[arm]["sources"].flags.writeable


def test_saved_candidate_source_change_rejected(tmp_path, monkeypatch):
    bundles = saved_bundles(tmp_path, monkeypatch, tamper=True)
    with pytest.raises(ValueError, match="all21 signature"):
        m.load_candidates(bundles, m.proof.Evidence())


def test_failed_seal_cannot_open_members_or_candidate(monkeypatch):
    monkeypatch.setattr(
        m.common, "read", lambda *args: dict(stage=m.VARIANT_STAGE, status="failed")
    )
    monkeypatch.setattr(m, "document", lambda *args: pytest.fail("member opened"))
    with pytest.raises(ValueError, match="final tracking seal failed"):
        m.load_headers(m.VARIANT_STAGE, "a" * 64, m.VARIANT_RUNNER_SHA, None)


def test_execution_gate_precedes_any_input_or_output(monkeypatch):
    monkeypatch.setattr(m, "EXECUTION_READY", False)
    with pytest.raises(ValueError, match="source review"):
        m.run("a" * 64, m.VARIANT_RUNNER_SHA, Path("do-not-create"))
    assert not m.sys.modules["torch"].cuda.is_initialized()


def test_frozen_source_inventory_also_contains_the_exact_execution_contract():
    assert m.is_producer_source(m.WORK / "run_instance_rescore_10s.py", "a" * 64)
    path = m.WORK / "FAST_TAIL_TWO_MINUTE_EXECUTION_CONTRACT.md"
    assert m.is_producer_source(path, m.EXECUTION_CONTRACT_SHA)
    assert not m.is_producer_source(path, "a" * 64)
    assert not m.is_producer_source(m.WORK / "another_contract.md", m.EXECUTION_CONTRACT_SHA)
    assert not m.is_producer_source(m.WORK / "arrays.npz", "a" * 64)


@pytest.mark.parametrize("defect", [None, "input_edge", "failed_seal"])
def test_rationale_summary_has_successful_consumed_seal_binding(tmp_path, defect):
    root = tmp_path / "synthetic_feasibility"
    out = m.proof.Output(root)
    summary = out.write("summary.json", dict(schema=root.name, status="ok", synthetic=True))
    seal = dict(
        stage=root.name,
        status="failed" if defect == "failed_seal" else "ok",
        outputs=dict(out.records),
    )
    path = root / "feasibility.provenance.json"
    path.write_bytes(m.proof.json_bytes(seal))
    digest = m.proof.sha(path)
    variant = dict(
        seal=dict(inputs={str(path): digest, summary["path"]: summary["sha256"]}),
        docs=dict(
            input_index=dict(
                rationale_references=dict(free=dict(path=str(path), sha256=digest, summary=summary))
            )
        ),
    )
    if defect == "input_edge":
        variant["seal"]["inputs"][summary["path"]] = "0" * 64
    evidence = m.proof.Evidence()
    if defect is None:
        m.bind_rationale_edges(variant, evidence)
        assert set(evidence.paths) == {str(path), summary["path"]}
    else:
        with pytest.raises(ValueError):
            m.bind_rationale_edges(variant, evidence)
