"""CPU-only qualification contract tests; no CUDA context or real fits."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "reflection_qualification_tested", Path(__file__).with_name("qualify_reflection_gpu.py")
)
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def saved_context():
    saved = dict(
        frame_idx=np.array(79381, np.int64),
        arms=np.asarray(q.ARMS),
        pose_xyz=np.ones((3, 3, 3, 3), np.float64),
        predicted_pose_xyz=np.ones((3, 3, 3), np.float64),
        fixed_plan=np.full((3, 3, 3), -1, np.int32),
        parent_missing=np.zeros(3, np.int64),
    )
    pixels = dict(
        frame_idx=np.array(79381, np.int64),
        fit_uv=np.zeros((3, 1024, 2), np.float64),
        fit_target=np.zeros((3, 1024), np.float32),
        fit_weights=np.full((3, 1024), 1 / 1024, np.float32),
    )
    return saved, pixels


def output_fields():
    keys = [
        *q.FLOAT_FIELDS,
        *q.WEIGHT_FIELDS,
        *("gradient_" + arm for arm in q.ARMS),
        *("loss_" + arm for arm in q.ARMS),
        "fit_pose_xyz",
        "fit_score",
        "fit_parts",
        "initial_lengths_mm",
    ]
    result = {key: np.zeros(2, np.float64) for key in keys}
    result.update({key: np.zeros(2, bool) for key in (*q.BOOL_FIELDS, "fit_direct_mask_support")})
    return result


def test_fixed_context_retains_float32_labels_without_candidate_reweighting():
    saved, pixels = saved_context()
    q.validate_context(saved, pixels)
    pixels["fit_weights"][0] = 0
    q.validate_context(saved, pixels)  # Missing/neutral evidence remains neutral.


@pytest.mark.parametrize("change", ["frame", "arm_order", "rows", "nan", "weights", "gap"])
def test_invalid_fixed_context_is_rejected(change):
    saved, pixels = saved_context()
    if change == "frame":
        saved["frame_idx"] += 1
    elif change == "arm_order":
        saved["arms"] = saved["arms"][::-1]
    elif change == "rows":
        saved["fixed_plan"][:2, 0, 0] = 1
    elif change == "nan":
        saved["pose_xyz"][0, 0, 0, 0] = np.nan
    elif change == "weights":
        pixels["fit_weights"][0, 0] = -1
    else:
        saved["parent_missing"][0] = -1
    with pytest.raises(ValueError):
        q.validate_context(saved, pixels)


@pytest.mark.parametrize(
    "key,tolerance",
    [
        ("fresnel_union", 1e-9),
        ("gradient_fresnel", 1e-9),
        ("outer_weight", 1e-12),
        ("fit_pose_xyz", 1e-8),
        ("fit_score", 1e-9),
        ("fit_parts", 1e-9),
    ],
)
def test_registered_absolute_tolerance_is_not_relative_or_rounded(key, tolerance):
    cpu = output_fields()
    gpu = {name: value.copy() for name, value in cpu.items()}
    gpu[key][0] = tolerance
    passed, errors = q.compare_outputs(cpu, gpu)
    assert passed and errors[key]["maximum_absolute_difference"] == tolerance
    gpu[key][0] = np.nextafter(tolerance, np.inf)
    passed, errors = q.compare_outputs(cpu, gpu)
    assert not passed and errors[key]["tolerance"] == tolerance


@pytest.mark.parametrize("key", ["ray_valid", "inner_valid", "fit_direct_mask_support"])
def test_one_boolean_difference_fails_even_with_equal_losses(key):
    cpu = output_fields()
    gpu = {name: value.copy() for name, value in cpu.items()}
    gpu[key][0] = True
    passed, errors = q.compare_outputs(cpu, gpu)
    assert not passed and errors[key]["mismatched_elements"] == 1


@pytest.mark.parametrize("change", ["absent_both", "nan", "shape", "dtype"])
def test_incomplete_or_invalid_output_cannot_pass(change):
    cpu = output_fields()
    gpu = {name: value.copy() for name, value in cpu.items()}
    if change == "absent_both":
        cpu.pop("gradient_fresnel")
        gpu.pop("gradient_fresnel")
    elif change == "nan":
        gpu["gradient_fresnel"][0] = np.nan
    elif change == "shape":
        gpu["fit_pose_xyz"] = np.zeros(3, np.float64)
    else:
        cpu["ray_valid"] = cpu["ray_valid"].astype(np.int32)
        gpu["ray_valid"] = gpu["ray_valid"].astype(np.int32)
    with pytest.raises(ValueError):
        q.compare_outputs(cpu, gpu)


def test_plan_sha_is_checked_before_opening_any_execution_context(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"tolerances": {"fit_pose_mm": 1.0}}))
    with pytest.raises(ValueError, match="registered plan SHA"):
        q.verify_registration(tmp_path, "0" * 64, {}, {})


def test_runtime_metadata_does_not_initialize_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU registration must not query/init CUDA")

    monkeypatch.setattr(q.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(q.torch.cuda, "get_device_name", forbidden)
    values = q.runtime("expected GPU only")
    assert values["torch_cpu_threads"] == 2
    assert values["cuda_device_name"] == "expected GPU only"
    assert set(values) == {
        "python",
        "numpy",
        "scipy",
        "opencv",
        "torch",
        "cuda",
        "cuda_device_name",
        "torch_cpu_threads",
    }
