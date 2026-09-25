"""Fixed-schema and geometry-only synthetic fixtures; no actual candidate/raw/model."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_dino_minute_geometry as m


def candidate(first=10, count=5, float_dtype=np.float64):
    data = {}
    for key, (shape, kind) in m.CANDIDATE_FIELDS.items():
        dtype = {"i": np.int64, "b": bool, "f": float_dtype, "U": "U32"}[kind]
        data[key] = np.zeros((count, *shape), dtype=dtype)
    data["frame_idx"] = np.arange(first, first + count)
    data["pose_xyz"][:] = np.arange(count * 27).reshape(count, 3, 3, 3)
    data["sources"][:] = -1
    for key in ("xyz", "accepted_xyz", "body_xyz"):
        data[key][:] = np.nan
    data["score_margin"][:] = np.inf
    data["identity_review"][:, 1] = True
    data["proposal_kind"][:] = "synthetic_original"
    return data


@pytest.mark.parametrize("float_dtype", [np.float64, np.float32])
def test_exact21_native_axis_slice_preserves_dtypes_and_does_not_mutate(float_dtype):
    data = candidate(float_dtype=float_dtype)
    before = {key: m.signature(value) for key, value in data.items()}
    sliced = m.validate_candidate(data, first=10, stop=15, frames=(10, 12, 14))
    assert len(sliced) == 21
    for key in data:
        assert m.checks.equal_bytes(sliced[key], data[key][[0, 2, 4]])
        assert not np.shares_memory(sliced[key], data[key])
    assert {key: m.signature(value) for key, value in data.items()} == before


@pytest.mark.parametrize(
    "damage",
    [
        "extra_static",
        "missing",
        "static_shape",
        "dtype",
        "frame_gap",
        "derived",
        "source",
        "nonfinite",
    ],
)
def test_wrong_candidate_schema_never_blindly_slices(damage):
    data = candidate()
    if damage == "extra_static":
        data["views"] = np.asarray(m.VIEWS)
    elif damage == "missing":
        del data["costs"]
    elif damage == "static_shape":
        data["proposal_kind"] = np.asarray("synthetic")
    elif damage == "dtype":
        data["sources"] = data["sources"].astype(float)
    elif damage == "frame_gap":
        data["frame_idx"][1] += 1
    elif damage == "derived":
        data["accepted_xyz"][0, 1] = 0
    elif damage == "source":
        data["sources"][0, 0, 0, 0] = -2
    else:
        data["pose_xyz"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        m.validate_candidate(data, first=10, stop=15, frames=(10, 12, 14))


class Camera:
    def project(self, xyz):
        # Nonlinear x(z) proves interpolation occurs before projection.
        return np.stack((xyz[..., 0] + 0.01 * xyz[..., 2] ** 2, xyz[..., 1]), axis=-1)


def scene():
    return np.asarray(
        [
            [[8.0, 8.0, 0.0], [10.0, 8.0, 1.0], [12.0, 8.0, 2.0]],
            [[26.0, 8.0, 0.0], [28.0, 8.0, 1.0], [30.0, 8.0, 2.0]],
            [[8.0, 26.0, 0.0], [10.0, 26.0, 1.0], [12.0, 26.0, 2.0]],
        ]
    )


def masks_for_scene():
    masks = np.zeros((3, 40, 40), bool)
    masks[0, 3:15, 3:18] = True
    masks[1, 3:15, 21:36] = True
    masks[2, 21:33, 3:18] = True
    return masks


def crops_for(masks, native=10, view="top_cam"):
    return [
        dict(
            frame_idx=native,
            view=view,
            mask_row=i,
            crop_id=f"f{native}_{view}_m{i:03d}",
            mask_sha256=m.signature(mask)["sha256"],
            core_sha256=m.signature(m.partial.eroded_core(mask))["sha256"],
            core_pixels=int(m.partial.eroded_core(mask).sum()),
            path=f"crops/m{i}.npz",
            sha256="0" * 64,
        )
        for i, mask in enumerate(masks)
    ]


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "latent_competitor",
        "duplicate",
        "missing",
        "no_masks",
        "empty",
        "invalid_projection",
        "small_core",
    ],
)
def test_all_frozen_geometry_decisions_exact_same_input_and_review_not_gate(mode):
    pose, masks = scene(), masks_for_scene()
    observed = np.ones((2, 3, 3), bool)
    sources = np.broadcast_to(np.arange(3)[:, None], (3, 3)).copy()
    available = True
    if mode == "latent_competitor":
        pose[1] = pose[0]
        observed[:, 1] = False
    elif mode == "duplicate":
        masks = np.concatenate((masks, masks[:1]))
    elif mode == "missing":
        masks, available = np.empty((0, 40, 40), bool), False
    elif mode == "no_masks":
        masks = np.empty((0, 40, 40), bool)
    elif mode == "empty":
        masks[:] = False
    elif mode == "invalid_projection":
        pose[1, 0] = np.nan
    elif mode == "small_core":
        masks[0] = False
        masks[0, 8, 8:14] = True
    scenes = np.stack([pose, pose])
    plans = np.stack([sources, sources])
    reviews = np.ones((2, 3), bool)
    cores = m.core_rows(masks, crops_for(masks), 10, "top_cam")
    before = [m.signature(x) for x in (masks, scenes, observed, plans, reviews)]
    fields, refs = m.geometry_group(
        masks, available, scenes, observed, plans, reviews, Camera(), cores
    )
    for mi, ref in enumerate(refs):
        metrics = [
            m.coh.coherence_metrics(fields, np.arange(33) + (mi * 3 + slot) * 33)
            for slot in range(3)
        ]
        frozen = m.identity.geometry_reference(
            masks, available, Camera().project(pose), observed[mi], plans[mi], metrics
        )
        for key in frozen:
            if key == "slots":
                for actual, expected in zip(ref[key], frozen[key], strict=True):
                    assert m.coh.encode_nonfinite(
                        {k: actual[k] for k in expected}
                    ) == m.coh.encode_nonfinite(expected)
            else:
                assert m.coh.encode_nonfinite(ref[key]) == m.coh.encode_nonfinite(frozen[key])
        assert all(s["identity_review"] for s in ref["slots"])
        assert ref["review_excluded_from_eligibility"] is False
    if mode == "normal":
        assert all(r["all3_endpoint_eligible"] for r in refs)
        interpolated_uv = (Camera().project(pose[0, 0]) + Camera().project(pose[0, 1])) / 2
        assert not np.array_equal(fields["projected_uv"][0, 0, 8], interpolated_uv)
    if mode == "latent_competitor":
        assert "mask_contains_other_projected_node" in refs[0]["slots"][0]["reasons"]
    if mode == "duplicate":
        assert refs[0]["slots"][0]["body_coherence"]["best_mask_rows"] == [0, 3]
    if mode in ("missing", "no_masks"):
        expected = "missing" if not available else "available_no_instances"
        assert refs[0]["slots"][0]["body_coherence"]["observation_status"] == expected
    assert [m.signature(x) for x in (masks, scenes, observed, plans, reviews)] == before
    assert fields["outside_distance"].shape == (len(masks) + 1, 198)


def test_core_threshold_exact12_and_no_new_color_conditions():
    masks = np.zeros((2, 20, 30), bool)
    masks[0, 4:11, 3:11] = True  # 3 x 4 after radius2 erosion == 12.
    masks[1, 4:11, 16:23] = True  # 3 x 3 == 9.
    rows = m.core_rows(masks, crops_for(masks), 10, "top_cam")
    assert [r["core_pixels"] for r in rows] == [12, 9]
    assert [r["core_valid"] for r in rows] == [True, False]
    assert all(r["crop_payload_opened"] is False for r in rows)
    broken = crops_for(masks)
    broken[1]["core_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="core/mask"):
        m.core_rows(masks, broken, 10, "top_cam")


def test_sparse_native_support_uses_singletons_original_frames_and_empty_support():
    arrays = m.validate_candidate(candidate(), first=10, stop=15, frames=(10, 12, 14))

    class EmptyNodes:
        def at(self, view, frame):
            pytest.fail("empty declared support needs no optical lookup")

    class Rig:
        views = m.VIEWS

    summary, fields = m.verify_selected_support(arrays, EmptyNodes(), Rig())
    assert fields["frame_idx"].tolist() == [10, 12, 14]
    assert summary["temporal_subsampling"] is True and summary["native_frames"] == 3
    assert summary["source_rows_checked"] == summary["observed_nodes_checked"] == 0
    assert fields["valid_source"].shape == (3, 3, 3, 3) and not fields["valid_source"].any()
    assert np.isnan(fields["ray_distance_mm"]).all()
    arrays["observed"][1, 0, 0] = True
    with pytest.raises(ValueError, match="empty source"):
        m.verify_selected_support(arrays, EmptyNodes(), Rig())


def test_selected_support_real_frozen_equations_and_exclusive_rows():
    frames = np.asarray([10, 12, 14], np.int64)
    pose = np.zeros((3, 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(3)[None, :, None] * 10
    pose[..., 1] = np.arange(3)[None, None, :]
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (3, 3, 3, 3)).copy()
    source[:, :, 2] = -1
    arrays = dict(
        frame_idx=frames, pose_xyz=pose, sources=source, observed=np.ones((3, 3, 3), bool)
    )
    seen = []

    class Nodes:
        def at(self, view, frame):
            assert frame in frames
            seen.append(frame)
            return pose[0, ..., :2].copy()

    class OpticalCamera:
        def rays(self, pixels):
            origin = np.concatenate((pixels, np.zeros((*pixels.shape[:-1], 1))), axis=-1)
            direction = np.zeros_like(origin)
            direction[..., 2] = 1
            return origin, direction

        def project(self, points):
            return points[..., :2].copy()

    class Rig:
        views = m.VIEWS

        def camera(self, view):
            return OpticalCamera()

    before = {key: m.signature(value) for key, value in arrays.items()}
    summary, fields = m.verify_selected_support(arrays, Nodes(), Rig())
    assert set(seen) == set(frames)
    assert summary["observed_nodes_checked"] == 27 and summary["source_rows_checked"] == 54
    assert np.array_equal(fields["valid_source"], source >= 0)
    assert {key: m.signature(value) for key, value in arrays.items()} == before
    invalid = {key: value.copy() for key, value in arrays.items()}
    invalid["pose_xyz"][1, 0, 0, 0] += 2.001
    with pytest.raises(ValueError, match="ray/pixel"):
        m.verify_selected_support(invalid, Nodes(), Rig())
    invalid = {key: value.copy() for key, value in arrays.items()}
    invalid["sources"][1, 1, 0, 0] = 0
    with pytest.raises(ValueError, match="multiple fish"):
        m.verify_selected_support(invalid, Nodes(), Rig())


def test_comparison_exact_scope_and_two_candidate_bindings():
    summary = dict(
        schema="tail_one_minute_comparison_v1",
        status="ok",
        native_interval=[66400, 76360],
        horizon_stop=76360,
        native_frames=9960,
        same_parent_and_horizon=True,
        one_minute_completed=True,
        all39_arrays_semantically_decoded=True,
        all21_arrays_exact_backtrace=True,
        candidate_paths={k: str(p) for k, (p, h) in m.crop.CANDIDATES.items()},
        candidate_sha256={k: h for k, (p, h) in m.crop.CANDIDATES.items()},
    )
    m.validate_comparison(summary)
    for key in ("all21_arrays_exact_backtrace", "one_minute_completed"):
        broken = copy.deepcopy(summary)
        broken[key] = False
        with pytest.raises(ValueError, match="completed"):
            m.validate_comparison(broken)
    broken = copy.deepcopy(summary)
    broken["candidate_sha256"]["tail"] = "0" * 64
    with pytest.raises(ValueError, match="origin"):
        m.validate_comparison(broken)


def test_failed_crop_seal_rejected_before_any_npz_or_candidate(tmp_path, monkeypatch):
    root = tmp_path / "crop"
    root.mkdir()
    seal = root / "preparation.provenance.json"
    seal.write_text(json.dumps(dict(stage=m.crop.STAGE, status="failed", finished_at="now")))
    monkeypatch.setattr(m, "CROP_ROOT", root)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no NPZ before crop success"))
    with pytest.raises(ValueError, match="successful pinned"):
        m.load_inputs(m.checks.sha(seal), {}, {})


def test_direct_output_member_path_size_and_sha_binding(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text('{"value":1}')
    seal = dict(
        outputs={
            "summary": dict(
                path=str(path), size_bytes=path.stat().st_size, sha256=m.checks.sha(path)
            )
        }
    )
    hashes = {}
    assert m.json_output(tmp_path, seal, "summary.json", hashes) == {"value": 1}
    assert hashes[str(path)] == seal["outputs"]["summary"]["sha256"]
    path.write_text('{"value":2}')
    with pytest.raises(ValueError, match="changed"):
        m.json_output(tmp_path, seal, "summary.json", {})


@pytest.mark.parametrize("method", m.METHODS)
def test_actual_small_metadata_contract_no_candidate_arrays(method, monkeypatch):
    """Explicitly allowed real JSON fixture only: no NPZ, rig load, image or geometry."""
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("metadata must not open NPZ"))
    hashes = {}
    comparison = m.crop.header(
        m.crop.COMPARISON,
        "comparison.provenance.json",
        m.crop.COMPARISON_SHA,
        "tail_one_minute_comparison_v1",
        hashes,
    )
    root = m.crop.CANDIDATES[method][0].parent
    filename, stage, digest, relative = m.producer_contract(method)
    producer = m.crop.header(root, filename, digest, stage, hashes)
    assert m.crop.member(comparison, "inputs", root / filename, hashes) == digest
    metadata = m.bound_metadata(root, method, producer, comparison, hashes)
    assert metadata["context"]["world_units"] == "mm"
    assert metadata["context"]["views"] == list(m.VIEWS)
    assert len(metadata["record_fields"]) == 19 and len(metadata["state_fields"]) == 7
    assert str(root / relative) in hashes
    assert len(hashes) == 3


def test_metadata_rejects_comparison_conflict_and_old_path_for_tail(tmp_path):
    root = tmp_path / "tail"
    metadata = root / "checkpoints/through_076360/metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"context":{}}')
    item = dict(path=str(metadata), size=metadata.stat().st_size, sha256=m.checks.sha(metadata))
    producer = dict(status="ok", stage="tail_one_minute_continuous", outputs={"metadata": item})
    comparison = dict(inputs={str(metadata): {**item, "sha256": "0" * 64}})
    with pytest.raises(ValueError, match="conflicting"):
        m.bound_metadata(root, "tail", producer, comparison, {})
    producer["stage"] = "three_body_long_continuation"
    with pytest.raises(ValueError, match="stage"):
        m.bound_metadata(root, "tail", producer, comparison, {})


def test_reference_core_addition_does_not_change_original_or_clear_review():
    masks, pose = masks_for_scene(), scene()
    cores = m.core_rows(masks, crops_for(masks), 10, "top_cam")
    _, refs = m.geometry_group(
        masks,
        True,
        np.stack([pose, pose]),
        np.ones((2, 3, 3), bool),
        np.zeros((2, 3, 3), int),
        np.zeros((2, 3), bool),
        Camera(),
        cores,
    )
    original = copy.deepcopy(refs[0])
    revised = m.add_core_conditions(refs[0], cores, np.ones(3, bool))
    assert refs[0] == original
    assert revised["slot_endpoint_eligible"] == original["slot_endpoint_eligible"]
    assert all(s["identity_review"] for s in revised["slots"])


def test_failure_seal_preserves_error_without_success(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "source_code", lambda: {})

    def failed(*args):
        raise ValueError("synthetic incomplete crop")

    monkeypatch.setattr(m, "load_inputs", failed)
    output = tmp_path / "geometry"
    with pytest.raises(ValueError, match="incomplete crop"):
        m.run(output, "0" * 64)
    seal = json.loads((output / "geometry.provenance.json").read_text())
    assert seal["stage"] == m.STAGE and seal["status"] == "failed"
    assert "incomplete crop" in seal["error"] and not (output / "summary.json").exists()
