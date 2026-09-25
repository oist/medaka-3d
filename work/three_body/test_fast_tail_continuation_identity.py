"""CPU gates, one-call orchestration and a small real streaming-DAG execution."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import run_fast_tail_continuation_identity as m
from test_compare_fast_tail_two_minutes import candidate


@pytest.fixture(autouse=True)
def no_gpu_images_or_original_solver(monkeypatch):
    assert not m.comparison.producer.torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU/images/original whole-DAG solver forbidden")

    monkeypatch.setattr(m.comparison.producer.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(m.comparison.producer.original.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(m.comparison.producer.original.cv2, "imread", forbidden)
    monkeypatch.setattr(m.comparison.producer, "read_checkpoint", forbidden)
    monkeypatch.setattr(m.streaming.frozen, "solve_identity_node_memory_dp", forbidden)
    yield
    assert not m.comparison.producer.torch.cuda.is_initialized()


def qualification_fixture():
    plan = dict(
        schema=m.qualified.STAGE,
        native_interval=[66400, 76360],
        target_block_frames=256,
        node_indices=[1, 0],
        dp_config=m.qualified.proof.DP_CONFIG.copy(),
        resources=dataclasses.asdict(m.streaming.DEFAULT_RESOURCES),
    )
    summary = dict(
        schema=m.qualified.STAGE,
        native_interval=[66400, 76360],
        status="ok",
        native_frames=9960,
        all_29_arrays_exact=True,
        all_19_scalars_exact=True,
        raw21_unchanged=True,
        finite_saved_medaka_minute_qualified=True,
        five_minute_qualified=False,
        all_three_ID_truth=False,
        original_solver_rerun=False,
        intermediate_ID_reset=False,
        resources=dataclasses.asdict(m.streaming.DEFAULT_RESOURCES),
    )
    fields = dict(
        array_count=29,
        scalar_count=19,
        all_exact=True,
        arrays={str(i): dict(exact=True, read_only=True) for i in range(29)},
        scalars={str(i): dict(exact=True) for i in range(19)},
    )
    return plan, summary, fields


def test_qualified_finite_minute_gate():
    m.validate_qualification(*qualification_fixture())


@pytest.mark.parametrize(
    "change",
    (
        "cap",
        "weight",
        "tie",
        "block",
        "node",
        "short",
        "array",
        "scalar",
        "five_minute_claim",
        "truth",
    ),
)
def test_reject_changed_qualification_or_scope(change):
    plan, summary, fields = qualification_fixture()
    if change == "cap":
        plan["resources"]["global_state_limit"] += 1
    elif change == "weight":
        plan["dp_config"]["weight"] = 0.3
    elif change == "tie":
        plan["dp_config"]["tie_rtol"] = 1e-8
    elif change == "block":
        plan["target_block_frames"] = 128
    elif change == "node":
        plan["node_indices"] = [0, 1]
    elif change == "short":
        summary["native_frames"] -= 1
    elif change == "array":
        fields["arrays"]["7"]["exact"] = False
    elif change == "scalar":
        fields["scalars"].pop("7")
    elif change == "five_minute_claim":
        summary["five_minute_qualified"] = True
    else:
        summary["all_three_ID_truth"] = True
    with pytest.raises(ValueError):
        m.validate_qualification(plan, summary, fields)


def comparison_fixture(stop=86320):
    refs = {"original": {"sha256": "a" * 64}, "fast_tail": {"sha256": "b" * 64}}
    plan = dict(
        schema=m.comparison.STAGE,
        horizon_stop=stop,
        milestone_sha256="c" * 64,
        candidate_references=refs,
    )
    summary = dict(
        schema=m.comparison.STAGE,
        horizon_stop=stop,
        status="ok",
        native_interval=[66400, stop],
        native_frames=stop - 66400,
        candidate_all21_and_source_arrays_unchanged=True,
        review_modified=False,
        same_parent_and_horizon=True,
        logical_initialization_count=1,
        restore_count_by_arm={"fast_tail": 5},
        external_pose_resets=0,
        all_three_ID_truth=False,
        candidate_references=copy.deepcopy(refs),
    )
    return plan, summary


@pytest.mark.parametrize("stop", (86320, 116200))
def test_body_comparison_same_milestone_gate(stop):
    m.validate_comparison(*comparison_fixture(stop), stop, "c" * 64)


@pytest.mark.parametrize(
    "change", ("sha", "stage", "short", "mutated", "review", "restore", "truth", "candidate")
)
def test_body_comparison_mismatch_rejected(change):
    plan, summary = comparison_fixture()
    if change == "sha":
        plan["milestone_sha256"] = "d" * 64
    elif change == "stage":
        plan["schema"] = "fast_tail_two_minute_comparison_v1"
    elif change == "short":
        summary["native_frames"] -= 1
    elif change == "mutated":
        summary["candidate_all21_and_source_arrays_unchanged"] = False
    elif change == "review":
        summary["review_modified"] = True
    elif change == "restore":
        summary["restore_count_by_arm"]["fast_tail"] = 4
    elif change == "truth":
        summary["all_three_ID_truth"] = True
    else:
        summary["candidate_references"]["fast_tail"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="matching body comparison"):
        m.validate_comparison(plan, summary, 86320, "c" * 64)


@pytest.mark.parametrize("frames", (19920, 49800))
def test_exactly_one_solver_call_for_whole_native_interval(monkeypatch, tmp_path, frames):
    xyz = np.zeros((frames, 3, 3, 3), np.float64)
    xyz[:, :, 0, :] = 10
    xyz[:, :, 1, :] = 20
    geometry = dict(
        frame_idx=np.arange(66400, 66400 + frames, dtype=np.int64),
        pose_xyz=xyz,
        observed=np.ones((frames, 3, 3), bool),
    )
    calls = []
    result = object()

    def solve(native, unary, positions, direct, **kwargs):
        calls.append(kwargs)
        assert np.array_equal(native, geometry["frame_idx"])
        assert unary.shape == (frames, 6) and unary.dtype == np.float64 and not unary.any()
        assert np.all(positions[:, :, 0] == 20) and np.all(positions[:, :, 1] == 10)
        assert direct.shape == (frames, 3, 2)
        assert all(not a.flags.writeable for a in (native, unary, positions, direct))
        assert kwargs == dict(
            spool_root=tmp_path / "spool",
            resources=m.streaming.DEFAULT_RESOURCES,
            target_block_frames=256,
            node_indices=(1, 0),
            **m.CONFIG,
        )
        return result

    monkeypatch.setattr(m.streaming, "solve_streaming_identity_node_memory_dp", solve)
    returned, signatures = m.solve_once(geometry, tmp_path / "spool")
    assert returned is result and len(calls) == 1
    assert signatures["frame_idx"]["shape"] == [frames]


@pytest.mark.parametrize(
    "key,value", (("weight", 0.3), ("scale_mm", 1.0), ("initial_state", 1), ("tie_rtol", 1e-8))
)
def test_changed_policy_rejected_before_solver(monkeypatch, tmp_path, key, value):
    monkeypatch.setattr(m, "CONFIG", {**m.CONFIG, key: value})
    with pytest.raises(ValueError, match="unchanged"):
        m.solve_once({}, tmp_path / "spool")


def test_missing_initial_mouth_refuses_identity_reinitialization(tmp_path):
    geometry, _ = candidate(66400, 66403)
    geometry["observed"][0, 2, 0] = False
    with pytest.raises(ValueError, match="initial chest and mouth"):
        m.solve_once(geometry, tmp_path / "spool")
    assert not (tmp_path / "spool").exists()


def test_real_streaming_result_all29_19_raw21_and_ambiguity(tmp_path):
    geometry, _ = candidate(66400, 66670)
    geometry["pose_xyz"][:] = 0  # Deliberately ambiguous synthetic identities.
    geometry["observed"][-1, 0, 1] = False
    before = m.comparison.previous.signature(geometry)
    for value in geometry.values():
        value.flags.writeable = False
    spooled, signatures = m.solve_once(geometry, tmp_path / "spool")
    metadata, records, arrays, scalars = m.validate_result(
        spooled, geometry, signatures, tmp_path / "spool"
    )
    assert metadata["block_count"] == 2
    assert len(arrays) == 29 and len(scalars) == 19
    assert m.comparison.previous.signature(geometry) == before
    mappings, report = m.identity_report(spooled.result)
    assert report["multiple_optimal_permutation_frames"] == 269
    assert report["tied_identity_intervals"] == [
        dict(identity=i, start=66401, stop=66670, native_frames=269) for i in range(3)
    ]
    assert report["all_three_ID_truth"] is False
    assert report["biological_swap_zero_verified"] is False
    assert report["review_flags_cleared"] is False
    assert spooled.result.terminal_unsettled_gap.any()
    assert np.array_equal(mappings["identity_to_raw_slot"], spooled.result.permutation_by_id)
    assert records[spooled.manifest_path]["sha256"] == m.proof.sha(spooled.manifest_path)


def test_capacity_failure_is_not_retried_and_artifacts_remain(monkeypatch, tmp_path):
    geometry, _ = candidate(66400, 66403)
    calls = []

    def fail(*args, spool_root, **kwargs):
        calls.append(kwargs)
        spool_root.mkdir()
        (spool_root / "partial.bin").write_bytes(b"completed-prefix")
        raise m.streaming.StreamingCapacityError(2, "layer_states", 100001, 100000)

    monkeypatch.setattr(m.streaming, "solve_streaming_identity_node_memory_dp", fail)
    output = m.proof.Output(tmp_path / "result")
    with pytest.raises(m.streaming.StreamingCapacityError) as caught:
        m.solve_once(geometry, output.root / "spool")
    m.record_spool(output)
    output.final_guard()
    assert len(calls) == 1
    assert (output.root / "spool/partial.bin").read_bytes() == b"completed-prefix"
    assert m.capacity_details(caught.value, 86320) == dict(
        resource="layer_states",
        attempted=100001,
        limit=100000,
        reported_frame_offset=2,
        native_frame=66402,
        fixed_caps_unchanged=True,
        approximate_pruning=False,
        retry_performed=False,
        partial_mapping_published=False,
    )
    assert "spool/partial.bin" in output.records


def test_nonframe_capacity_and_noncapacity_errors():
    error = m.streaming.StreamingCapacityError(-1, "storage_bytes", 5, 4)
    assert m.capacity_details(error, 116200)["native_frame"] is None
    assert m.capacity_details(ValueError("input error"), 116200) is None
