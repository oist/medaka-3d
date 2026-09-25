"""Synthetic schema and numerical tests; never load actual source candidates or masks."""

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import score_mask_only_saved_transition as caller


def policy():
    return dict(
        schema="instance_rescore_policy_v1",
        effective_from_native=76360,
        bodies_per_view=3,
        fixed_camera_denominator=3,
        cap_per_view=9.0,
        weight=3.0,
        camera_order=list(caller.VIEWS),
        unavailable_view_contribution=0.0,
        nonfinite_projection_contribution=9.0,
        post_fit_only=True,
        gradient_objective_unchanged=True,
        original_five_loss_parts_unchanged=True,
        permanent_IDs_assigned=False,
        clears_identity_review=False,
        observation_config=json.loads(json.dumps(dataclasses.asdict(caller.core.DEFAULT_CONFIG))),
    )


def bank(count=4):
    contributions = np.tile([0.25, 0.0, 1.0], (count, 1)).astype(np.float64)
    base = np.arange(count, dtype=np.float64) / 10
    parent = np.arange(count, dtype=np.float64)[::-1] * 2
    sixth = 3.0 * contributions.sum(axis=1) / 3
    return dict(
        schema="instance_rescore_frame_diagnostics_v1",
        frame_idx=90258,
        policy=policy(),
        distance_backend={"synthetic": True},
        candidate_index=np.arange(count, dtype=np.int64),
        candidate_pose_xyz=np.zeros((count, 3, 3, 3), dtype=np.float64),
        candidate_parent=np.arange(count, dtype=np.int64),
        candidate_plan=np.zeros((count, 3, 3, 3), dtype=np.int32),
        candidate_kind=["synthetic"] * count,
        candidate_permutation=np.tile(np.arange(3, dtype=np.int64), (count, 1)),
        base_local_cost=base,
        original_five_loss_parts=np.zeros((count, 5)),
        sixth_instance_term=sixth,
        adjusted_local_cost=base + sixth,
        parent_path_cost=parent,
        adjusted_path_cost=parent + (base + sixth),
        capped_contributions_by_view=contributions,
        views={
            v: dict(capped_view_contribution=contributions[:, k].copy())
            for k, v in enumerate(caller.VIEWS)
        },
    )


def score(raw, eligible=True):
    # Typed compact-view input for arithmetic tests, not a scorer-accuracy claim.
    result = {
        k: None
        for k in (
            "best_assignment",
            "assignment_costs",
            "optimal_assignment_mask",
            "body_instance_costs",
        )
    }
    result.update(
        eligible=eligible,
        neutral=not eligible,
        reason="synthetic",
        observation_evidence={},
        best_cost=np.asarray(raw, dtype=np.float64),
    )
    return result


def reset_costs(data, base, parent):
    data["base_local_cost"] = np.asarray(base, dtype=np.float64)
    data["parent_path_cost"] = np.asarray(parent, dtype=np.float64)
    data["sixth_instance_term"] = 3.0 * data["capped_contributions_by_view"].sum(axis=1) / 3
    data["adjusted_local_cost"] = data["base_local_cost"] + data["sixth_instance_term"]
    data["adjusted_path_cost"] = data["parent_path_cost"] + data["adjusted_local_cost"]


def test_normalization_cap_invalid_and_other_view_columns_remain_exact():
    data = bank()
    before = {k: caller.proof.signature(v) for k, v in data.items() if isinstance(v, np.ndarray)}
    result, ranks = caller.replace_view_contribution(data, "side1_cam", score([0, 3, 30, np.inf]))
    assert np.array_equal(result["new_view"]["capped_view_contribution"], [0, 1, 9, 9])
    assert result["new_view"]["cap_applied"].tolist() == [False, False, True, True]
    caller.exact(
        result["new_capped_contributions_by_view"][:, [0, 2]],
        data["capped_contributions_by_view"][:, [0, 2]],
        "other views changed",
    )
    assert np.array_equal(result["new_sixth_instance_term"], [1.25, 2.25, 10.25, 10.25])
    assert ranks["new_local_cost"]["stable_candidate_order"] == [0, 1, 2, 3]
    assert before == {
        k: caller.proof.signature(v) for k, v in data.items() if isinstance(v, np.ndarray)
    }


def test_neutral_nonfinite_is_unassessed_zero_and_preserves_old_costs_when_target_was_zero():
    data = bank()
    result, _ = caller.replace_view_contribution(data, "side1_cam", score([np.inf] * 4, False))
    assert np.all(result["new_view"]["capped_view_contribution"] == 0)
    caller.exact(
        result["new_local_cost"], data["adjusted_local_cost"], "neutral changed local cost"
    )
    caller.exact(
        result["new_fixed_parent_path_cost"], data["adjusted_path_cost"], "neutral changed path"
    )


def test_floating_association_is_parent_plus_base_plus_sixth_inside_parentheses():
    data = bank(1)
    data["capped_contributions_by_view"][:] = 0
    for row in data["views"].values():
        row["capped_view_contribution"][:] = 0
    reset_costs(data, [1.0], [1e16])
    result, _ = caller.replace_view_contribution(data, "side1_cam", score([3.0]))
    expected = data["parent_path_cost"] + (data["base_local_cost"] + 1.0)
    alternative = data["parent_path_cost"] + data["base_local_cost"] + 1.0
    assert expected[0] != alternative[0]
    caller.exact(result["new_fixed_parent_path_cost"], expected, "arithmetic association changed")


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "shape",
        "candidate_order",
        "sixth",
        "local",
        "path",
        "view",
        "permutation",
        "config",
        "camera_order",
    ],
)
def test_saved_schema_and_arithmetic_tamper_rejected(fault):
    data = bank()
    if fault == "dtype":
        data["candidate_pose_xyz"] = data["candidate_pose_xyz"].astype(np.float32)
    elif fault == "shape":
        data["candidate_plan"] = data["candidate_plan"][:, :, :2]
    elif fault == "candidate_order":
        data["candidate_index"] = data["candidate_index"][::-1]
    elif fault in ("sixth", "local", "path"):
        key = {
            "sixth": "sixth_instance_term",
            "local": "adjusted_local_cost",
            "path": "adjusted_path_cost",
        }[fault]
        data[key][0] += 1
    elif fault == "view":
        data["views"]["top_cam"]["capped_view_contribution"][0] = 0
    elif fault == "permutation":
        data["candidate_permutation"][0] = [0, 0, 2]
    elif fault == "camera_order":
        data["policy"]["camera_order"].reverse()
    else:
        data["policy"]["observation_config"]["endpoint_tolerance_px"] = 7
    with pytest.raises(ValueError):
        caller.validate_bank(data)


def test_rank_ties_are_reported_and_fixed_parent_ranking_is_separate_from_local_ranking():
    data = bank(3)
    reset_costs(data, [0, 0, 1], [10, 0, 0])
    _, ranks = caller.replace_view_contribution(data, "side1_cam", score([0, 0, 0]))
    assert ranks["new_local_cost"]["exact_minimum_tied_candidates"] == [0, 1]
    assert ranks["new_local_cost"]["stable_candidate_order"] == [0, 1, 2]
    assert ranks["new_fixed_parent_path_cost"]["stable_candidate_order"] == [1, 2, 0]


def test_reflection_negative_control_is_kept_even_when_false_candidate_ranks_first():
    data = bank(2)
    masks = np.zeros((3, 80, 180), bool)
    pose = np.zeros((3, 3, 3), dtype=np.float64)
    for k, x in enumerate((10, 70, 130)):
        masks[k, 20:40, x : x + 30] = True
        pose[k] = [[x + 3, 25, 0], [x + 13, 28, 0], [x + 25, 30, 0]]
    data["candidate_pose_xyz"][1] = pose
    data["candidate_pose_xyz"][0] = pose
    data["candidate_pose_xyz"][0, 2] = pose[0]
    data["candidate_pose_xyz"][0, 2, :, 2] = 20
    reset_costs(data, [0, 0], [0, 0])
    obs = caller.preparer.prepare_mask_only_instance_observation(masks)
    scored = caller.sparse.score_sparse_instance_bodies(
        data["candidate_pose_xyz"], lambda x: x[..., :2], obs
    )
    result, ranks = caller.replace_view_contribution(data, "side1_cam", scored)
    assert ranks["new_local_cost"]["stable_candidate_order"][0] == 1
    assert result["new_local_cost"][1] < result["new_local_cost"][0]
    assert (90141, "side1_cam") in caller.CASES


def test_lossless_codec_subset_reads_only_requested_arrays_and_is_readonly(tmp_path, monkeypatch):
    data = bank()
    data["unused_payload"] = np.ones((5, 7))
    tree, arrays = caller.codec.split_diagnostic(data)
    path = tmp_path / "synthetic.npz"
    np.savez_compressed(path, **arrays)
    payload = path.read_bytes()
    header = dict(
        saved=dict(
            tree=tree,
            archive=dict(
                path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)
            ),
        ),
        snapshot=dict(candidate_count=4),
    )
    reads = []
    original = np.lib.npyio.NpzFile.__getitem__

    def counted(self, key):
        reads.append(key)
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counted)
    captures = {}
    decoded = caller.load_bank(header, captures)
    assert tree["unused_payload"]["__array__"] not in reads
    assert len(reads) == len(set(reads))
    assert all(not v.flags.writeable for v in decoded.values() if isinstance(v, np.ndarray))
    assert len(captures) == 1
    with pytest.raises(ValueError, match="twice"):
        caller.load_bank(header, captures)


def test_array_descriptor_tamper_and_duplicate_are_rejected(tmp_path):
    data = bank()
    tree, arrays = caller.codec.split_diagnostic(data)
    duplicate = {"a": tree["candidate_index"], "b": tree["candidate_index"]}
    with pytest.raises(ValueError):
        caller.array_descriptors(duplicate)
    tree["candidate_index"]["signature"]["sha256"] = "0" * 64
    path = tmp_path / "bad_signature.npz"
    np.savez_compressed(path, **arrays)
    payload = path.read_bytes()
    header = dict(
        saved=dict(
            tree=tree,
            archive=dict(
                path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)
            ),
        ),
        snapshot=dict(candidate_count=4),
    )
    with pytest.raises(ValueError, match="array differs"):
        caller.load_bank(header, {})


def binding_fixture():
    outputs = {"side2_cam_masks": {"path": "synthetic/masks.npz", "sha256": "a" * 64}}
    detector = {"synthetic_protocol": "x"}
    shared = {"synthetic/calib.json": "b" * 64}
    raw = dict(
        shared_SL_calibration_sha256=shared,
        source_chunk=dict(index=24, start=89640, stop=91300, output_hashes=outputs),
    )
    binding = dict(
        original_binding=dict(
            inputs=dict(seed=dict(SL_and_calibration_sha256=shared), detector_binding=detector)
        )
    )
    manifest = dict(
        binding=detector,
        frames=list(range(89640, 91300)),
        source_interval_half_open=[89640, 91300],
        fps=166,
        image_shape_hw=[1200, 1200],
    )
    stage = dict(
        status="ok",
        stage="prepare_independent_mask_chunk",
        outputs=copy.deepcopy(outputs),
        params=dict(binding=detector),
    )
    stage["outputs"]["side2_cam_masks"]["size_bytes"] = 123
    return binding, raw, manifest, stage


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "wrong_chunk",
        "calibration_mix",
        "detector_mix",
        "output_mix",
        "false_stage",
        "shifted_native",
    ],
)
def test_realistic_rich_output_and_observation_binding_scope(fault):
    binding, raw, manifest, stage = binding_fixture()
    if fault == "wrong_chunk":
        raw["source_chunk"]["index"] = 23
    elif fault == "calibration_mix":
        raw["shared_SL_calibration_sha256"] = {"different.json": "c" * 64}
    elif fault == "detector_mix":
        manifest["binding"] = {"other": True}
    elif fault == "output_mix":
        stage["outputs"]["side2_cam_masks"]["sha256"] = "d" * 64
    elif fault == "false_stage":
        stage["status"] = "failed"
    elif fault == "shifted_native":
        manifest["frames"][0] += 1
    if fault is None:
        caller.validate_observation_binding(binding, raw, manifest, stage)
    else:
        with pytest.raises(ValueError):
            caller.validate_observation_binding(binding, raw, manifest, stage)


def test_detector_scores_are_exact_frame_order_metadata_and_never_used_in_costs():
    packed = dict(
        frame_offsets=np.array([0, 2, 5], np.int64),
        instance_id=np.array([0, 1, 0, 1, 2], np.int64),
        scores=np.array([0.9, 0.8, 0.7, 0.99, 0.51], np.float32),
    )
    diagnostic = caller.aligned_detector_scores(packed, 1, 3)
    caller.exact(diagnostic["scores"], packed["scores"][2:5], "score order changed")
    assert diagnostic["used_for_eligibility_or_ranking"] is False
    packed["scores"][:] = 0.1
    assert diagnostic["scores"][0] == np.float32(0.7)
    packed["instance_id"][2:5] = [2, 1, 0]
    with pytest.raises(ValueError, match="order"):
        caller.aligned_detector_scores(packed, 1, 3)


def test_result_roundtrip_preserves_all_six_costs_nonfinite_distances_and_input_bank(tmp_path):
    out = caller.proof.Output(tmp_path / "output")
    result = dict(
        input_candidate_bank=bank(),
        new_score=dict(
            assignment_costs=np.array([[0, np.inf, 2, 3, 4, 5.0]]),
            distances=np.array([np.nan, np.inf, -np.inf]),
        ),
    )
    row = caller.write_result(out, "case", result)
    saved = json.loads((out.root / "case.json").read_text())
    with np.load(row["path"], allow_pickle=False) as archive:
        restored = caller.codec.restore_diagnostic(
            saved["tree"], {k: archive[k] for k in archive.files}
        )
    caller.exact(
        restored["new_score"]["assignment_costs"],
        result["new_score"]["assignment_costs"],
        "six assignments changed",
    )
    caller.exact(
        restored["new_score"]["distances"],
        result["new_score"]["distances"],
        "nonfinite data changed",
    )


def test_disabled_or_outside_execution_cannot_reach_actual_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(
        caller, "load_headers", lambda *_: pytest.fail("real input loader must never run in tests")
    )
    monkeypatch.setattr(caller, "EXECUTION_READY", False)
    with pytest.raises(ValueError, match="not enabled"):
        caller.execute(SimpleNamespace(execute=True, output=tmp_path / "output"))
    monkeypatch.setattr(caller, "EXECUTION_READY", True)
    monkeypatch.setattr(caller, "WORK", tmp_path / "allowed")
    with pytest.raises(ValueError, match="fresh workspace"):
        caller.execute(SimpleNamespace(execute=True, output=tmp_path / "outside"))


def test_relative_path_escape_rejected_before_archive_read(tmp_path):
    with pytest.raises(ValueError):
        caller.relative(tmp_path, "../escaped.npz")
    with pytest.raises(ValueError):
        caller.relative(tmp_path, str(Path(tmp_path.anchor) / "absolute.npz"))


@pytest.mark.parametrize(
    "fault", [None, "captured_sha", "consumed_sha", "captured_stat", "captured_members"]
)
def test_actual_captured_mask_schema_needs_no_direct_raw_seal_input(tmp_path, fault):
    path = tmp_path / "side1.npz"
    path.write_bytes(b"synthetic header-only path fixture; never decoded")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    captured = dict(
        path=str(path),
        sha256=digest,
        file_stat=caller.parity.file_stat(path),
        arrays={key: None for key in caller.chunk_masks.SCHEMA_DTYPES},
    )
    role = dict(path=str(path), sha256=digest)
    context = dict(
        chunk_path=tmp_path / "manifest.json",
        raw_seal=dict(inputs={}),
        manifest=dict(masks={"side1_cam": dict(path=path.name, sha256=digest)}),
        raw=dict(
            source_chunk=dict(output_hashes={"side1_cam_masks": role}),
            captured_mask_records={"side1_cam": captured},
        ),
    )
    if fault == "captured_sha":
        captured["sha256"] = "0" * 64
    elif fault == "consumed_sha":
        role["sha256"] = "0" * 64
    elif fault == "captured_stat":
        captured["file_stat"][2] += 1
    elif fault == "captured_members":
        del captured["arrays"]["scores"]
    if fault is None:
        result, result_sha, result_record = caller.mask_descriptor(context, "side1_cam")
        assert result == path and result_sha == digest and result_record == captured
    else:
        with pytest.raises(ValueError):
            caller.mask_descriptor(context, "side1_cam")
