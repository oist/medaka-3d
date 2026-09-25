"""New state-audit edge; tiny CPU-produced state, no real data or CUDA."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import fixed_body_width_state_audit as m  # noqa: E402

import test_fixed_body_width_tracker as state_toy  # noqa: E402
import test_mask_transport_tracker as toy  # noqa: E402

tracking = m.tracking


@pytest.fixture(scope="module")
def qualified():
    # Reuse fixture generation only, not prior test suites. All observations are
    # synthetic; the adapter below must not initialize/advance/fit/restore them.
    return state_toy.qualified.__wrapped__()


def options(qualified):
    data, old, _, _, _, _ = qualified
    original = copy.deepcopy(old[1]["context"])
    original.pop("mask_transport")
    # The old CPU wrapper's original science type is normalized in the same way
    # as the successful producer does against its separate scientific context.
    original["implementation"] = "AcceleratedContinuousPersistentNodeTracker"
    return dict(
        rig=data[0],
        reference_pose=data[1].copy(),
        original_context=original,
        first_native=100,
        group_counts=(0, 0, 0),
        query_sha256=("a" * 64,) * 3,
        input_binding={"fixture": "synthetic CPU only"},
        coefficient=0.0,
        expected_frames=2,
    )


def rebind(bundle, *, device=None, radial_delta=0.0, change=None):
    core, metadata, arrays, header = copy.deepcopy(bundle)
    content = tracking.decode_sidecar(arrays, header)
    width = metadata["context"]["mask_transport"]["fixed_width"]
    if device:
        metadata["context"]["tracker_config"]["device"] = device
        width["reference_device"] = str(m.runtime.resolved_device(device))
    if radial_delta:
        radial = content["fixed_width_reference"]["radial_sigma_mm"]
        radial[0, 0] += radial_delta
        width["radial_sigma_mm"] = tracking.signature(radial)
        width["renderer_signature"][-1] = [
            radial.dtype.str,
            list(radial.shape),
            tracking.signature(radial)["sha256"],
        ]
    if change:
        change(core, metadata, content, width)
    header["fixed_width"] = copy.deepcopy(width)
    content["fixed_width_reference"]["policy"] = copy.deepcopy(width)
    new_arrays = {}
    header["tree"] = tracking._pack(content, new_arrays)
    header["array_signatures"] = toy.signatures(new_arrays)
    header["core_array_signatures"] = toy.signatures(core)
    header["core_metadata_sha256"] = tracking.json_hash(metadata)
    header.pop("content_sha256")
    header["content_sha256"] = tracking.json_hash(header)
    return core, metadata, new_arrays, header


def forbid_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("audit executed tracking/GPU")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    for name in ("begin", "advance", "_fit", "restore_state", "restore_transport_state"):
        monkeypatch.setattr(tracking.FixedBodyWidthTracker, name, forbidden)


def test_real_tiny_cpu_state_is_validated_without_execution_or_input_rewrite(
    qualified, monkeypatch
):
    bundle = qualified[-1]
    params = options(qualified)
    before = toy.signatures(bundle[0]), toy.signatures(bundle[2]), copy.deepcopy(bundle[1:])
    forbid_execution(monkeypatch)
    core, content, report = m.validate_state(*bundle, **params)
    assert core is bundle[0]
    assert report["passed"] and report["core_arrays"] == 39 and report["retained_fields"] == 19
    assert report["history_frames"] == 2 and len(content["fits"]) == 2
    assert report["radial_formula_byte_exact"]
    assert not report["GPU_used"] and not report["cross_device_restore_qualified"]
    assert (
        report["state_initializations"]
        == report["restore_calls"]
        == report["advance_calls"]
        == report["fit_calls"]
        == 0
    )
    assert toy.signatures(bundle[0]) == before[0] and toy.signatures(bundle[2]) == before[1]
    assert tracking.json_hash(bundle[1]) == tracking.json_hash(before[2][0])
    assert tracking.json_hash(bundle[3]) == tracking.json_hash(before[2][2])


@pytest.mark.parametrize("delta", [0.0, 5e-14])
def test_synthetic_saved_cuda_label_keeps_exact_metadata_and_R0_without_cuda(
    qualified, monkeypatch, delta
):
    bundle = rebind(qualified[-1], device="cuda", radial_delta=delta)
    params = options(qualified)
    params["original_context"]["tracker_config"]["device"] = "cuda"
    old_meta = tracking.json_hash(bundle[1])
    old_arrays = toy.signatures(bundle[2])
    forbid_execution(monkeypatch)
    _, content, report = m.validate_state(*bundle, **params)
    assert report["saved_device"] == "cuda:0" and report["computation_device"] == "cpu"
    assert report["radial_formula_max_abs"] <= 1e-12
    assert report["radial_formula_byte_exact"] is (delta == 0.0)
    assert report["original_metadata_sha256"] == old_meta == tracking.json_hash(bundle[1])
    assert toy.signatures(bundle[2]) == old_arrays
    tracking.same(
        content["fixed_width_reference"]["radial_sigma_mm"],
        tracking.decode_sidecar(bundle[2], bundle[3])["fixed_width_reference"]["radial_sigma_mm"],
        "saved R0 stays unchanged",
    )


def test_cpu_formula_tolerance_does_not_accept_changed_width_model(qualified, monkeypatch):
    bundle = rebind(qualified[-1], device="cuda", radial_delta=1e-9)
    params = options(qualified)
    params["original_context"]["tracker_config"]["device"] = "cuda"
    forbid_execution(monkeypatch)
    with pytest.raises(ValueError, match="CPU formula"):
        m.validate_state(*bundle, **params)


@pytest.mark.parametrize(
    "fault",
    [
        "policy_start",
        "reference_pose",
        "renderer_config",
        "R0_signature",
        "nonfinite_R0",
        "other_config",
        "query_binding",
    ],
)
def test_only_declared_device_R0_representation_differences_allowed(qualified, monkeypatch, fault):
    def change(core, metadata, content, width):
        if fault == "policy_start":
            width["apply_from_native"] += 1
        elif fault == "reference_pose":
            content["fixed_width_reference"]["pose"][0, 0, 0] += 1
        elif fault == "renderer_config":
            width["renderer_config"]["width_ratio"] += 0.01
        elif fault == "R0_signature":
            width["radial_sigma_mm"]["sha256"] = "0" * 64
        elif fault == "nonfinite_R0":
            content["fixed_width_reference"]["radial_sigma_mm"][0, 0] = np.nan
        elif fault == "other_config":
            metadata["context"]["tracker_config"]["motion_weight"] += 0.1
        else:
            metadata["context"]["mask_transport"]["query_sha256"][0] = "b" * 64

    bundle = rebind(qualified[-1], change=change)
    forbid_execution(monkeypatch)
    with pytest.raises(ValueError):
        m.validate_state(*bundle, **options(qualified))


def test_successful_plan_binding_is_required_even_when_old_metadata_is_self_consistent(
    qualified, monkeypatch
):
    params = options(qualified)
    params["input_binding"] = {"fixture": "unrelated successful plan"}
    forbid_execution(monkeypatch)
    with pytest.raises(ValueError, match="context"):
        m.validate_state(*qualified[-1], **params)


@pytest.mark.parametrize("fault", ["terminal_velocity", "retained_source", "side_best"])
def test_original_semantic_history_and_new_sidecar_validation_both_run(
    qualified, monkeypatch, fault
):
    def change(core, metadata, content, width):
        if fault == "terminal_velocity":
            core["state_velocity"][0, 0] += 0.1
        elif fault == "retained_source":
            core["history_sources"][-1, 0, :, 0] = -1
        else:
            content["fits"][1]["owner"]["best_iteration"][0] = -1

    bundle = rebind(qualified[-1], change=change)
    forbid_execution(monkeypatch)
    with pytest.raises(ValueError):
        m.validate_state(*bundle, **options(qualified))


@pytest.mark.parametrize(
    "method",
    [
        "begin",
        "advance",
        "track",
        "finish",
        "restore_state",
        "restore_transport_state",
        "export_transport_state",
        "_fit",
        "_proposals",
        "_tensor",
        "_advance_owned",
    ],
)
def test_validation_view_cannot_be_used_for_computation(qualified, method):
    params = options(qualified)
    cpu = m._cpu_tracker(
        params["rig"],
        params["reference_pose"],
        params["original_context"],
        first_native=params["first_native"],
        group_counts=params["group_counts"],
        query_sha256=params["query_sha256"],
        input_binding=params["input_binding"],
        coefficient=0.0,
    )
    bundle = qualified[-1]
    content = tracking.decode_sidecar(bundle[2], bundle[3])
    view = m._validation_view(cpu, bundle[1], content)
    with pytest.raises(RuntimeError, match="cannot execute or restore"):
        getattr(view, method)()
    assert view.initialize_count == view.restore_count == 0 and view._states is None


def test_wrong_count_or_lifecycle_is_not_a_complete_new_arm(qualified):
    params = options(qualified)
    params["expected_frames"] = 200
    with pytest.raises(ValueError, match="scope/lifecycle"):
        m.validate_state(*qualified[-1], **params)
