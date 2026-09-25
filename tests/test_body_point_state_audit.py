"""Audit real synthetic active states without executing the saved tracker."""

import copy
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import body_point_state_audit as m  # noqa: E402

import test_body_point_tracker as fixtures  # noqa: E402


def forbidden(*args, **kwargs):
    raise AssertionError("validation executed a tracker or initialized CUDA")


@pytest.fixture(scope="module")
def example():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.accelerator, "current_accelerator", lambda *a, **k: None)
        patch.setattr(torch.cuda, "_lazy_init", forbidden)
        return fixtures.real_active_packet_bundle.__wrapped__()


def options(example):
    data, bundle, _, _, _ = example
    original = copy.deepcopy(bundle[1]["context"])
    policy = original.pop("mask_transport")
    original["implementation"] = "AcceleratedContinuousPersistentNodeTracker"
    return dict(
        rig=data[0],
        reference_pose=data[1].copy(),
        original_context=original,
        first_native=100,
        group_counts=(3, 3, 3),
        query_sha256=("a" * 64,) * 3,
        input_binding=copy.deepcopy(policy["input_binding"]),
        coefficient=3.0,
        expected_frames=2,
    )


def rebind(bundle, change=None, device=None):
    core, meta, arrays, header = copy.deepcopy(bundle)
    content = m.tracking.decode_sidecar(arrays, header)
    if device:
        meta["context"]["tracker_config"]["device"] = device
    if change:
        change(core, meta, content)
    arrays = {}
    header["tree"] = m.tracking._pack(content, arrays)
    header["array_signatures"] = {k: m.tracking.signature(v) for k, v in arrays.items()}
    header["core_array_signatures"] = {k: m.tracking.signature(v) for k, v in core.items()}
    header["core_metadata_sha256"] = m.tracking.json_hash(meta)
    header.pop("content_sha256")
    header["content_sha256"] = m.tracking.json_hash(header)
    return core, meta, arrays, header


def trap_execution(patch):
    patch.setattr(torch.cuda, "_lazy_init", forbidden)
    for name in ("begin", "advance", "_fit", "restore_state", "restore_transport_state"):
        patch.setattr(m.tracking.BodyPointTracker, name, forbidden)


def signatures(bundle):
    return (
        {k: m.tracking.signature(v) for k, v in bundle[0].items()},
        m.tracking.json_hash(bundle[1]),
        {k: m.tracking.signature(v) for k, v in bundle[2].items()},
        m.tracking.json_hash(bundle[3]),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_active_state_full_semantic_audit_keeps_original_metadata_and_arrays(
    example, monkeypatch, device
):
    bundle = rebind(example[1], device=device)
    params = options(example)
    params["original_context"]["tracker_config"]["device"] = device
    before = signatures(bundle)
    trap_execution(monkeypatch)
    core, content, report = m.validate_state(*bundle, **params)
    assert core is bundle[0] and signatures(bundle) == before
    assert report["passed"] and report["core_arrays"] == 39
    assert report["retained_fields"] == 19 and report["history_frames"] == 2
    assert content["fits"][1]["owner"]["active"]
    assert report["saved_device"] == device
    assert report["computation_device"] == "cpu"
    assert not report["metadata_rewritten"] and not report["GPU_used"]
    assert not report["cross_device_restore_qualified"]
    assert report["state_initializations"] == report["restore_calls"] == report["fit_calls"] == 0
    assert report["initial_geometry_links_verified"]


@pytest.mark.parametrize("fault", ["seed", "science", "binding", "device"])
def test_rehashed_saved_state_cannot_override_independent_science(example, monkeypatch, fault):
    params = options(example)
    bundle = rebind(example[1])
    if fault == "seed":
        params["reference_pose"][0, 0, 0] += 0.1
    elif fault == "science":
        params["original_context"]["tracker_config"]["width_ratio"] *= 2
    elif fault == "binding":
        params["input_binding"]["fixture"] = "different input"
    else:
        params["original_context"]["tracker_config"]["device"] = "cuda"
    before = signatures(bundle)
    trap_execution(monkeypatch)
    with pytest.raises(ValueError):
        m.validate_state(*bundle, **params)
    assert signatures(bundle) == before


@pytest.mark.parametrize("fault", ["unselected_cost", "point_mass", "initial_link"])
def test_rehashed_contents_still_require_parent_point_and_initial_geometry(
    example, monkeypatch, fault
):
    def change(core, metadata, content):
        fit = content["fits"][1]
        if fault == "unselected_cost":
            index = next(i for i in range(len(fit["parent"])) if i not in fit["selected"])
            fit["parent_cost"][index] += 1
        elif fault == "point_mass":
            fit["packet"]["point_weight"][0, 0, 0, 0] += 0.1
        else:
            anchor = content["anchors"][0]
            anchor["links"][0][0, 0] = ~anchor["links"][0][0, 0]

    bundle = rebind(example[1], change=change)
    trap_execution(monkeypatch)
    with pytest.raises(ValueError):
        m.validate_state(*bundle, **options(example))


@pytest.mark.parametrize("name", ["begin", "advance", "_tensor", "restore_transport_state"])
def test_validation_view_has_no_execution_entry(name):
    view = object.__new__(m._ReadOnlyValidationView)
    with pytest.raises(RuntimeError, match="cannot execute"):
        getattr(view, name)()
