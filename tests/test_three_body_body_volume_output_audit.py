"""Independent bookkeeping audit rejects altered rankings, pixels and support."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
SPEC = importlib.util.spec_from_file_location(
    "body_volume_output_audit_test_module", WORK / "audit_body_volume_probe.py"
)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def fixture():
    original = np.arange(27, dtype=float).reshape(3, 3, 3) + 30
    inputs = np.stack((original, original + 1, original + 2))
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], (3, 3, 3, 3)).copy()
    union = np.linspace(0.05, 0.95, 48).reshape(4, 3, 4)
    target = np.broadcast_to([1.0, 0.0, 0.0, 0.0], (3, 4)).copy()
    weight = np.full((3, 4), 0.25)
    strata = np.broadcast_to([0, 1, 2, 2], (3, 4)).astype(np.int8)
    loss = (union - target) ** 2 * weight
    parts = np.stack([(loss * (strata == k)).sum(-1) for k in range(3)], -1)
    first_silhouette = ((union[0] - target) ** 2 * weight).sum() / 3
    arrays = dict(
        fit_input_pose_xyz=inputs,
        original_pose_xyz=original,
        fit_pose_xyz=inputs.copy(),
        fit_sources=sources,
        fit_initial_sources=sources.copy(),
        fit_observed=np.ones((3, 3, 3), bool),
        fit_initial_observed=np.ones((3, 3, 3), bool),
        fit_scores=np.array([2.0, 1.0, 1.5]),
        fit_initial_scores=np.array([3.0, 4.0, 4.0]),
        fit_initial_loss_parts=np.zeros((3, 5)),
        fit_boundary_coordinates=np.zeros_like(inputs, bool),
        evaluation_union=union,
        evaluation_ray_valid=np.ones((3, 4), bool),
        evaluation_loss_by_stratum=parts,
        evaluation_mean_view_loss=parts.sum(-1).mean(-1),
    )
    arrays["fit_initial_loss_parts"][0, 4] = first_silhouette
    for arm, selected in [("mask", [0, 1]), ("random", [0, 2])]:
        arrays[arm + "_pose_xyz"] = inputs[selected]
        arrays[arm + "_outside_coordinates"] = np.zeros((2, 3, 3, 3), bool)
        arrays[arm + "_admissible"] = np.ones(2, bool)
        arrays[arm + "_fit_index"] = np.array(selected)
    samples = dict(
        fit_target=target,
        fit_weights=weight,
        evaluation_target=target.copy(),
        evaluation_weights=weight.copy(),
        evaluation_strata=strata,
        evaluation_sampled_counts=np.ones((3, 3), np.int64),
        evaluation_pool_counts=np.full((3, 3), 20, np.int64),
        evaluation_excluded_fit_unique_count=np.full(3, 4, np.int64),
        evaluation_view_available=np.ones(3, bool),
        evaluation_foreground_nonempty=np.ones(3, bool),
    )
    b = dict(
        fit_target=target.copy(),
        fit_weights=weight.copy(),
        eval_target=target.copy(),
        eval_weights=weight.copy(),
        eval_strata=strata.copy(),
        eval_counts=np.ones((3, 3), np.int64),
        eval_pool_counts=np.full((3, 3), 20, np.int64),
        eval_excluded_fit_unique_count=np.full(3, 4, np.int64),
        eval_view_available=np.ones(3, bool),
        eval_foreground_nonempty=np.ones(3, bool),
    )
    for view in m.p.base.VIEWS:
        samples["fit_pixels_" + view] = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        samples["evaluation_pixels_" + view] = samples["fit_pixels_" + view] + 10
        b[view + "_eval_uv"] = samples["evaluation_pixels_" + view].copy()
    b["fit_uv"] = np.stack([samples["fit_pixels_" + view] for view in m.p.base.VIEWS])
    curve = dict(
        pose_xyz=original.copy(),
        original_fit_union=union[0].copy(),
        original_eval_union=union[0].copy(),
        fit_ray_valid=np.ones((3, 4), bool),
    )
    summary = dict(
        frame=79372, joint_admitted_fit_count=3, selected=dict(fit_indices=dict(mask=1, random=2))
    )
    return arrays, samples, summary, b, curve


def test_shared_original_image_pixels_and_all_candidate_conservation():
    result = m.audit_frame(*fixture())
    assert result["fitted"] == 3
    assert result["original_silhouette_ab_abs_difference"] == 0
    assert result["shared_pixels_dtype_shape_bytes_equal"]


@pytest.mark.parametrize(
    "bad",
    [
        "ranking",
        "source",
        "support",
        "pixel_order",
        "pixel_dtype",
        "evaluation",
        "render",
        "outside",
        "candidate_map",
    ],
)
def test_independent_audit_rejects_mutated_semantics(bad):
    arrays, samples, summary, b, curve = fixture()
    if bad == "ranking":
        summary["selected"]["fit_indices"]["mask"] = 0
    elif bad == "source":
        arrays["fit_sources"][0, 1, 0, 0] = 0
    elif bad == "support":
        arrays["fit_observed"][0, 0, 0] = False
    elif bad == "pixel_order":
        b["fit_uv"][0] = b["fit_uv"][0, ::-1]
    elif bad == "pixel_dtype":
        b["fit_uv"] = b["fit_uv"].astype(np.float32)
    elif bad == "evaluation":
        arrays["evaluation_mean_view_loss"][1] += 0.01
    elif bad == "render":
        curve["original_fit_union"][0, 0] += 0.1
    elif bad == "outside":
        arrays["mask_outside_coordinates"][1, 1, 0, 0] = True
    else:
        arrays["mask_fit_index"][1] = 2
    with pytest.raises((ValueError, AssertionError)):
        m.audit_frame(arrays, samples, summary, b, curve)


def test_seal_success_sha_and_exact_consumed_output_are_required(tmp_path):
    data = tmp_path / "data.npz"
    np.savez(data, values=np.arange(3))
    entry = dict(path=str(data), sha256=m.p.base.sha256(data))
    seal = dict(status="ok", stage="expected", inputs={}, outputs=dict(data=entry))
    path = tmp_path / "custom.provenance.json"
    path.write_text(json.dumps(seal))
    loaded, _ = m.completed(tmp_path, "expected", "custom")
    assert m.require_output(data, loaded) == data
    missing = tmp_path / "unbound.npz"
    missing.write_bytes(data.read_bytes())
    with pytest.raises(ValueError, match="unique sealed"):
        m.require_output(missing, loaded)
    data.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        m.completed(tmp_path, "expected", "custom")
    seal["status"] = "failed"
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="completed"):
        m.completed(tmp_path, "expected", "custom")


def test_conflicting_dependency_hash_cannot_overwrite_initial_code_guard():
    expected = {"source.py": "a" * 64}
    with pytest.raises(ValueError, match="different"):
        m.merge_hashes(expected, {"source.py": "b" * 64})
    assert expected == {"source.py": "a" * 64}
