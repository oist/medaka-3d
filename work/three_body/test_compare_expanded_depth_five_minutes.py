"""Focused synthetic CPU contracts; no final scientific payloads or GPU."""

import copy
import itertools
from pathlib import Path

import compare_expanded_depth_five_minutes as m
import expanded_depth_instance_rescore_tracker as expanded
import instance_rescore_tracker as dense
import numpy as np
import pytest
import sparse_instance_rescore_tracker as sparse
import test_instance_rescore_tracker as toy
from test_compare_fast_tail_two_minutes import candidate as tiny_candidate
from test_compare_fast_tail_two_minutes import refresh_derived


def completion():
    return dict(
        schema=m.VARIANT_STAGE,
        status="ok",
        scientific_interval=[m.START, m.STOP],
        process_interval=[m.RESUME, m.STOP],
        retained_interval=[m.FIRST, m.STOP],
        native_frames=m.STOP - m.START,
        process_frames=m.STOP - m.RESUME,
        initialize_count=1,
        restore_count=6,
        external_pose_resets=0,
        new_begin_calls=0,
        finish_calls=1,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        finish_matches_checkpoint_backtrace=True,
        diagnostic_frames=m.STOP - m.RESUME,
        policy_transition=False,
        physical_ID_truth=False,
        physical_depth_truth=False,
        automatic_adoption=False,
        parent_tracking_sha256=m.PARENT_SHAS[m.ARMS[1]],
        checkpoints=[
            dict(
                arrays=39,
                previous_frame=f - 1,
                processed_frames=f - m.FIRST,
                initialize_count=1,
                restore_count=6,
            )
            for f in range(m.RESUME + m.CHUNK, m.STOP + m.CHUNK, m.CHUNK)
        ],
    )


def test_source_disabled_and_intervals_female_scope_are_fixed(monkeypatch):
    assert not m.EXECUTION_READY
    with pytest.raises(ValueError, match="source review"):
        m.run("a" * 64, "b" * 64, "c" * 64, Path("not-created"))
    assert m.INTERVALS["full"] == (66400, 116200)
    assert m.INTERVALS["before_depth_change"] == (66400, 76360)
    assert m.INTERVALS["development_ten_seconds"] == (76360, 78020)
    assert m.INTERVALS["continuation_after_development"] == (78020, 116200)
    minutes = [m.INTERVALS[f"minute_{i}"] for i in range(1, 6)]
    assert sum(b - a for a, b in minutes) == 49800
    assert all(b == c for (_, b), (c, _) in itertools.pairwise(minutes))
    assert len(m.metrics.old.old.ANCHOR_FRAMES) == 11
    assert all(66400 <= f < 76360 for f in m.metrics.old.old.ANCHOR_FRAMES)
    code = m.source_code()
    assert code[str(Path(m.__file__).resolve())] == m.proof.sha(m.__file__)
    assert not m.sys.modules["torch"].cuda.is_initialized()


def test_successful_completion_keeps23_contiguous_saved_checkpoints():
    doc = completion()
    m.variant_completion(doc)
    assert len(doc["checkpoints"]) == 23


@pytest.mark.parametrize(
    "damage", ["stage", "finish", "restore", "process", "missing", "checkpoint"]
)
def test_completion_rejects_partial_or_reset_run(damage):
    doc = completion()
    if damage == "stage":
        doc["schema"] = m.saved.STAGE
    elif damage == "finish":
        doc["finish_calls"] = 0
    elif damage == "restore":
        doc["restore_count"] = 7
    elif damage == "process":
        doc["process_interval"][0] = 76360
    elif damage == "missing":
        doc["checkpoints"].pop(3)
    else:
        doc["checkpoints"][3]["previous_frame"] += 1
    with pytest.raises(ValueError):
        m.variant_completion(doc)


def test_failed_depth_seal_and_bad_cli_source_rejected_before_candidate(monkeypatch):
    monkeypatch.setattr(
        m.common, "read", lambda *args: dict(stage=m.VARIANT_STAGE, status="failed")
    )
    monkeypatch.setattr(
        m.saved, "member_doc", lambda *args: pytest.fail("failed seal opened member")
    )
    monkeypatch.setattr(
        m.np, "load", lambda *args, **kwargs: pytest.fail("failed seal opened candidate")
    )
    with pytest.raises(ValueError, match="failed"):
        m.load_variant_headers("a" * 64, "b" * 64, None)
    with pytest.raises(ValueError, match="explicit"):
        m.load_variant_headers("a" * 64, "missing reviewed SHA", None)


@pytest.mark.parametrize("damage", [None, "CLI", "edge", "tracker", "current_file"])
def test_CLI_source_actual_file_and_consumed_seal_agree(damage):
    # Source files only; no successful/future tracking data is read.
    digest = m.proof.sha(m.VARIANT_RUNNER)
    source = {
        str(m.VARIANT_RUNNER): digest,
        str(m.WORK / "expanded_depth_instance_rescore_tracker.py"): m.PINS[
            "expanded_depth_instance_rescore_tracker.py"
        ],
    }
    plan, seal = dict(source_sha256=copy.deepcopy(source)), dict(inputs=copy.deepcopy(source))
    if damage == "CLI":
        digest = "0" * 64
    elif damage == "edge":
        seal["inputs"][str(m.VARIANT_RUNNER)] = "0" * 64
    elif damage == "tracker":
        plan["source_sha256"][str(m.WORK / "expanded_depth_instance_rescore_tracker.py")] = "0" * 64
    elif damage == "current_file":
        digest = plan["source_sha256"][str(m.VARIANT_RUNNER)] = seal["inputs"][
            str(m.VARIANT_RUNNER)
        ] = "0" * 64
    evidence = m.proof.Evidence()
    if damage:
        with pytest.raises(ValueError):
            m.bind_variant_sources(plan, seal, digest, evidence)
    else:
        m.bind_variant_sources(plan, seal, digest, evidence)
        assert evidence.paths == source


def restoration_fixture():
    inputs = toy.inputs.__wrapped__()
    dense_ctx = dense.InstanceRescoreContinuousTracker(*inputs)._context()
    sparse_ctx = sparse.SparseInstanceRescoreContinuousTracker(*inputs)._context()
    depth_ctx = expanded.ExpandedDepthInstanceRescoreContinuousTracker(*inputs)._context()
    source_arrays = {f"array{i}": np.array([i, 0], np.float64) for i in range(39)}
    signatures = m.common.signature(source_arrays)

    def metadata(ctx):
        return dict(
            context=ctx,
            initialize_count=1,
            restore_count=5,
            previous_frame=78019,
            processed_frames=28220,
            synthetic_other_metadata="kept",
        )

    old_parent, new_parent = metadata(dense_ctx), metadata(depth_ctx)
    imported = dict(
        schema="sparse_instance_dense_state_import_v1",
        status="ok",
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        previous_frame=78019,
        processed_frames=28220,
        effective_from_native=78020,
        review_cleared_by_migration=False,
        backend_applied_retroactively_to_prior_history=False,
        arrays=copy.deepcopy(signatures),
        original_metadata=copy.deepcopy(old_parent),
        old_context=copy.deepcopy(dense_ctx),
        new_context=copy.deepcopy(sparse_ctx),
        imported_metadata={
            **copy.deepcopy(old_parent),
            "context": copy.deepcopy(sparse_ctx),
            "restore_count": 6,
        },
    )
    restored = dict(
        schema=m.VARIANT_STAGE + "_same_policy_restore",
        status="ok",
        all39_byte_preserved=True,
        source_inputs_unchanged=True,
        context_unchanged=True,
        initialize_count=1,
        source_restore_count=5,
        restore_count=6,
        new_begin_calls=0,
        policy_transition=False,
        review_cleared=False,
        resume_native=78020,
        arrays=copy.deepcopy(signatures),
        original_metadata=copy.deepcopy(new_parent),
        restored_metadata={**copy.deepcopy(new_parent), "restore_count": 6},
    )
    old_policy = dict(
        scoring_policy=copy.deepcopy(sparse_ctx["instance_rescore_policy"]),
        sparse_distance_backend=copy.deepcopy(sparse_ctx["sparse_distance_backend"]),
    )
    new_policy = dict(
        scoring_policy=copy.deepcopy(depth_ctx["instance_rescore_policy"]),
        sparse_distance_backend=copy.deepcopy(depth_ctx["sparse_distance_backend"]),
        expanded_depth_policy=copy.deepcopy(depth_ctx["expanded_depth_policy"]),
    )
    control = dict(
        docs=dict(
            metadata=dict(context=copy.deepcopy(sparse_ctx)),
            state_import=imported,
            plan=dict(checkpoint_binding=old_policy),
        )
    )
    variant = dict(
        docs=dict(
            metadata=dict(context=copy.deepcopy(depth_ctx)),
            state_restore=restored,
            plan={**copy.deepcopy(new_policy), "checkpoint_binding": copy.deepcopy(new_policy)},
            summary=copy.deepcopy(new_policy),
            milestone=copy.deepcopy(new_policy),
        )
    )
    parents = {
        m.ARMS[0]: dict(metadata=old_parent, arrays=copy.deepcopy(signatures)),
        m.ARMS[1]: dict(
            metadata=new_parent,
            arrays=copy.deepcopy(signatures),
            summary=dict(expanded_depth_policy=copy.deepcopy(depth_ctx["expanded_depth_policy"])),
        ),
    }
    return control, variant, parents


def test_real_class_contexts_only_depth_domain_and_prior_sparse_start_differ():
    args = restoration_fixture()
    before = copy.deepcopy(args)
    result = m.validate_restorations(*args)
    assert args == before
    assert result["full39_payload_reads"] == 0
    assert result["same_audited76360_parent"]


@pytest.mark.parametrize(
    "damage",
    [
        "parent_byte",
        "source_meta",
        "review",
        "restore",
        "context",
        "sparse_start",
        "policy_binding",
    ],
)
def test_restoration_metadata_and_any_parent_byte_tampering_rejected(damage):
    control, variant, parents = restoration_fixture()
    docs = variant["docs"]
    if damage == "parent_byte":
        parents[m.ARMS[1]]["arrays"]["array38"]["sha256"] = "0" * 64
    elif damage == "source_meta":
        docs["state_restore"]["original_metadata"]["synthetic_other_metadata"] = "lost"
    elif damage == "review":
        docs["state_restore"]["review_cleared"] = True
    elif damage == "restore":
        docs["state_restore"]["restored_metadata"]["restore_count"] = 7
    elif damage == "context":
        docs["metadata"]["context"]["tracker_config"]["mask_weight"] = 0.5
    elif damage == "sparse_start":
        docs["metadata"]["context"]["sparse_distance_backend"]["effective_from_native"] = 78020
    else:
        docs["milestone"]["expanded_depth_policy"]["expanded_lower_z_mm"] = -20
    with pytest.raises(ValueError):
        m.validate_restorations(control, variant, parents)


def test_manifest_byte_signature_conversion_is_exact_for_nan_signedzero_unicode():
    values = {f"v{k}": np.array([float(k), -0.0, np.nan], np.float64) for k in range(38)}
    values["text"] = np.array(["メダカ", "尾"], dtype="U12")
    import hashlib

    manifest = dict(
        arrays={
            k: dict(
                dtype=dict(str=v.dtype.str),
                shape=list(v.shape),
                bytes_sha256=hashlib.sha256(v.tobytes()).hexdigest(),
            )
            for k, v in values.items()
        }
    )
    assert m.manifest_signatures(manifest) == m.common.signature(values)
    manifest["arrays"].pop("text")
    with pytest.raises(ValueError, match="all39"):
        m.manifest_signatures(manifest)


@pytest.mark.parametrize("damage", [None, "milestone", "signature"])
def test_final_and_milestone_all21_real_archive_bytes(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(m, "FIRST", 100)
    monkeypatch.setattr(m, "STOP", 108)
    bundles = {}
    for arm in m.ARMS:
        out = m.proof.Output(tmp_path / arm)
        candidate, _ = tiny_candidate()
        candidate["identity_review"][2, 1] = True
        refresh_derived(candidate)
        out.write("candidate.npz", candidate)
        milestone = {k: v.copy() for k, v in reversed(tuple(candidate.items()))}
        if arm == m.ARMS[1] and damage == "milestone":
            milestone["loss_parts"][2, 4] = -0.0
        out.write("milestones/through_116200/candidate.npz", milestone)
        signatures = m.common.signature(candidate)
        if arm == m.ARMS[1] and damage == "signature":
            signatures["sources"]["sha256"] = "0" * 64
        bundles[arm] = dict(
            root=out.root,
            seal=dict(outputs=out.records),
            seal_sha256="a" * 64,
            docs=dict(
                summary=dict(candidate_arrays=signatures),
                milestone=dict(candidate_arrays=signatures),
            ),
        )
    if damage:
        with pytest.raises(ValueError):
            m.load_candidates(bundles, m.proof.Evidence())
    else:
        candidates, references = m.load_candidates(bundles, m.proof.Evidence())
        assert all(len(v) == 21 for v in candidates.values())
        assert all(not a.flags.writeable for c in candidates.values() for a in c.values())
        assert all(r["sha256"] != r["milestone_sha256"] for r in references.values())
        assert candidates[m.ARMS[1]]["identity_review"][2, 1]
        assert np.isnan(candidates[m.ARMS[1]]["accepted_xyz"][2, 1]).all()


def test_native_source_equality_does_not_accept_renamed_fake_arm():
    docs = dict(
        plan=dict(original_binding={"same": True}),
        input_index=dict(original_binding={"same": True}, prefix_chunks=[{"source": "a"}]),
    )
    bundles = {a: dict(docs=copy.deepcopy(docs)) for a in m.ARMS}
    assert m.observation_binding(bundles) == {"same": True}
    bundles[m.ARMS[1]]["docs"]["input_index"]["prefix_chunks"][0]["source"] = "b"
    with pytest.raises(ValueError, match="actual40"):
        m.observation_binding(bundles)


def test_interval_report_keeps_latent_z_and_clips_missing_runs_at_requested_interval():
    candidate, seed = tiny_candidate()
    candidate["observed"][1:7, 2, 0] = False
    candidate["sources"][1:7, 2, :, 0] = -1
    candidate["pose_xyz"][2:6, 2, 0, 2] = [-12.01, -12.0, -11.75, -6.0]
    refresh_derived(candidate)
    before = m.common.signature(candidate)
    summary, motion, geometry, z = m.interval_report(candidate, seed, 102, 106)
    full = summary["z_distribution"]["subsets"]["all_estimated_including_latent"]
    direct = summary["z_distribution"]["subsets"]["directly_observed"]
    assert full["counts"]["below_expanded_lower"]["total"] == 1
    assert full["counts"]["within_expanded_lower_band"]["total"] == 2
    assert full["counts"]["at_or_below_old_lower"]["by_slot_node"][2][0] == 4
    assert direct["counts"]["below_expanded_lower"]["total"] == 0
    mouth = next(
        row for row in motion["by_slot_node"] if row["slot"] == 2 and row["node"] == "mouth"
    )
    assert mouth["longest_missing_frames_in_interval"] == 4
    assert z["frame_idx"].tolist() == geometry["frame_idx"].tolist() == [102, 103, 104, 105]
    assert not summary["z_distribution"]["physical_depth_truth"]
    assert m.common.signature(candidate) == before


def test_geometry_delta_uses_actual_arm_names_and_excludes_outside_interval_jump():
    pose = np.zeros((6, 3, 3, 3))
    pose[..., 0] = np.arange(6)[:, None, None]
    a = dict(frame_idx=np.arange(100, 106), pose_xyz=pose, observed=np.ones((6, 3, 3), bool))
    b = {k: v.copy() for k, v in a.items()}
    b["pose_xyz"][:3] += 1000
    b["pose_xyz"][3:, 2, 0, 2] += 4
    b["observed"][:, 2, 0] = False
    summary, arrays = m.geometry_delta(a, b, 103, 106)
    assert set(summary["speed_by_arm"]) == set(m.ARMS)
    assert summary["array_axis0_arm_order"] == list(m.ARMS)
    assert np.all(arrays["all_estimated_speed_mm_s"] == 166)
    assert np.all(arrays["all_estimated_acceleration_mm_s2"] == 0)
    assert arrays["pose_delta_mm"][:, 2, 0].tolist() == [4, 4, 4]
    assert not arrays["speed_both_direct"][1, :, 2, 0].any()
