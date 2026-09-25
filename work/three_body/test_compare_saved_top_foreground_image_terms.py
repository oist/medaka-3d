"""Independent CPU fixtures; every input/PNG/NPZ is synthetic and temporary."""

import hashlib
import io
import json
from pathlib import Path

import compare_saved_top_foreground_image_terms as m
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def isolated_inputs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real input or GPU entry is forbidden")

    for module in (m, m.saved, m.proof):
        for name in ("WORK", "REPO"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, tmp_path)
    for name in ("PRODUCER", "PHOTOS", "MASKS"):
        monkeypatch.setattr(m, name, tmp_path / ("missing_" + name))
    for module, name in (
        (m, "source_code"),
        (m, "load_headers"),
        (m.saved, "load_rig"),
        (m.saved, "load_bank"),
        (m.saved, "capture"),
        (m.observations, "SequenceReader"),
        (m.observations, "CachedNodes"),
        (m.cv2, "VideoCapture"),
        (torch.cuda, "synchronize"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    load = np.load

    def local_load(path, *args, **kwargs):
        assert isinstance(path, io.BytesIO) or Path(path).resolve().is_relative_to(tmp_path)
        assert kwargs.get("allow_pickle") is False
        return load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", local_load)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def config(large=False):
    return dict(
        positive_samples=384 if large else 2,
        near_samples=384 if large else 2,
        global_samples=256 if large else 2,
        width_ratio=0.06,
        mask_weight=3.0,
    )


def frame_fixture(native=90141, size=64, empty=None):
    images = {view: np.zeros((size, size, 3), np.uint8) for view in m.VIEWS}
    background = np.zeros_like(images[m.VIEWS[0]])
    images[m.VIEWS[0]][40:48, 40:49] = 255
    masks = {}
    for view in m.VIEWS:
        mask = np.zeros((1, size, size), bool)
        mask[:, 8:15, 9:17] = True
        masks[view] = mask
    if empty is not None:
        images[m.VIEWS[0]][:] = 0
        masks = {
            view: None if empty == "none" else np.empty((0, size, size), bool) for view in m.VIEWS
        }
    return m.observations.FrameObservation(native, {}, images, masks, {m.VIEWS[0]: background})


class Camera:
    def __init__(self, index):
        self.index = index

    def rays(self, pixels):
        origins = np.column_stack((pixels * 0.05, np.zeros(len(pixels))))
        origins[:, 1] += self.index * 0.1
        directions = np.tile([0.0, 0.0, 1.0], (len(pixels), 1))
        origins[pixels[:, 0] == 0] = np.nan  # finite pixel whose calibrated ray is invalid
        return origins, directions


class Rig:
    views = m.VIEWS
    world_units = "mm"

    def camera(self, view):
        return Camera(self.views.index(view))


def poses(count=2):
    value = np.empty((count, 3, 3, 3), np.float64)
    for i in range(count):
        for fish in range(3):
            value[i, fish] = [
                [0.5 + i * 0.03, 0.5 + fish, 3.0],
                [1.5 + i * 0.03, 0.5 + fish, 3.1],
                [2.5 + i * 0.03, 0.6 + fish, 3.0],
            ]
    return value


def test_actual_sampler_changes_top_only_and_is_deterministic():
    frame = frame_fixture()
    before = {k: a.copy() for k, a in frame.images_bgr.items()}
    first = m.sample_modes(frame, config(True))
    second = m.sample_modes(frame, config(True))
    d2, contrast = [first[key] for key in m.MODES]
    assert d2.target.shape == (3, 1024)
    assert not np.array_equal(d2.pixels[m.VIEWS[0]], contrast.pixels[m.VIEWS[0]])
    for vi, view in enumerate(m.VIEWS):
        for mode in m.MODES:
            assert np.array_equal(first[mode].pixels[view], second[mode].pixels[view])
        assert np.array_equal(frame.images_bgr[view], before[view])
        if vi:
            for name in ("target", "weights", "foreground"):
                assert np.array_equal(getattr(d2, name)[vi], getattr(contrast, name)[vi])
            assert np.array_equal(d2.pixels[view], contrast.pixels[view])
    # First stratum must lie in the independently specified positive pools.
    assert np.all((d2.pixels[m.VIEWS[0]][:384, 0] >= 9) & (d2.pixels[m.VIEWS[0]][:384, 0] < 17))
    assert np.all(
        (contrast.pixels[m.VIEWS[0]][:384, 0] >= 40) & (contrast.pixels[m.VIEWS[0]][:384, 0] < 49)
    )


def test_real_FP64_renderer_weighted_image_formula_and_invalid_ray_oracle():
    pixels = {
        v: np.array(
            [[0.0, 1.0], [10.0, 10.0], [30.0, 10.0], [40.0, 30.0], [10.0, 50.0], [50.0, 50.0]]
        )
        for v in m.VIEWS
    }
    targets = np.array([[1.0, 0.0, 1.0, 0.25, 0.0, 1.0]] * 3, np.float32)
    weights = np.array([[1000.0, 0.4, 0.2, 0.1, 0.05, 0.25]] * 3, np.float32)
    target = m.observations.PixelTargets(pixels, targets, weights, targets > 0.5)
    pose = poses()
    actual = m.image_terms(pose, target, Rig(), config())
    renderer = m.rendering.RefractiveFishRenderer(
        Rig(),
        pixels,
        m.rendering.RendererConfig(width_ratio=0.06),
        device="cpu",
        dtype=torch.float64,
    )
    with torch.no_grad():
        alpha = renderer.render(pose).union_alpha.numpy()
    valid = renderer.ray_valid.numpy()
    assert not valid[:, 0].any() and valid[:, 1:].all()
    residual = (alpha - targets.astype(np.float64)) ** 2 * weights.astype(np.float64) * valid
    expected = np.array([sum(float(x) for x in row.flat) / 3 for row in residual])
    np.testing.assert_allclose(actual["silhouette"], expected, rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        actual["per_view_silhouette"], residual.sum(2) / 3, rtol=0, atol=2e-16
    )
    strata = np.stack(
        [residual[:, :, a:b].sum(2) / 3 for a, b in [(0, 2), (2, 4), (4, 6)]], axis=-1
    )
    np.testing.assert_allclose(actual["per_view_positive_near_global"], strata, rtol=0, atol=2e-16)
    assert actual["silhouette"].dtype == np.float64
    assert np.all(actual["ray_origins"][:, 0] == 0)


@pytest.mark.parametrize("empty", ["none", "empty"])
def test_no_foreground_contributes_zero_weight_not_negative_evidence(empty):
    targets = m.sample_modes(frame_fixture(empty=empty), config())
    for target in targets.values():
        assert not target.weights.any() and not target.foreground.any()
        result = m.image_terms(poses(), target, Rig(), config())
        assert np.array_equal(result["silhouette"], np.zeros(2))


@pytest.mark.parametrize("damage", ["float32", "nan"])
def test_nonFP64_or_nonfinite_pose_rejected(damage):
    pose = poses()
    if damage == "float32":
        pose = pose.astype(np.float32)
    else:
        pose[0, 0, 0, 0] = np.nan
    target = m.sample_modes(frame_fixture(), config())["detectron2"]
    with pytest.raises(ValueError, match="finite saved FP64"):
        m.image_terms(pose, target, Rig(), config())


def bank_fixture(n=3):
    base = 100.0 + np.arange(n, dtype=np.float64)
    sixth = np.full(n, 0.75)
    parent = np.arange(n, 0, -1, dtype=np.float64) * 2
    parts = np.arange(n * 5, dtype=np.float64).reshape(n, 5) / 10
    parts[:, 4] = 1 + np.arange(n) / 10
    return dict(
        candidate_index=np.arange(n),
        candidate_pose_xyz=poses(n),
        candidate_parent=np.zeros(n, dtype=np.int64),
        candidate_plan=np.zeros((n, 3, 3, 3), np.int8),
        candidate_permutation=np.tile(np.arange(3), (n, 1)),
        base_local_cost=base,
        original_five_loss_parts=parts,
        sixth_instance_term=sixth,
        adjusted_local_cost=base + sixth,
        parent_path_cost=parent,
        adjusted_path_cost=parent + (base + sixth),
        capped_contributions_by_view=np.full((n, 3), 0.25),
    )


def fake_image_terms(monkeypatch, *, mismatch=False):
    calls = []

    def evaluate(pose, target, rig, conf):
        mode = len(calls) % 2
        calls.append(mode)
        n = len(pose)
        baseline = 1 + np.arange(n) / 10
        # Known top-only image counterfactual; side costs stay invariant.
        term = baseline + (np.linspace(-0.2, 0.2, n) if mode else 0)
        if mismatch and not mode:
            term = term + 0.1
        per_view = np.column_stack((term - 0.3, np.full(n, 0.1), np.full(n, 0.2)))
        pixels = np.stack([target.pixels[v] for v in m.VIEWS])
        origins = np.concatenate((pixels, np.zeros((*pixels.shape[:-1], 1))), axis=-1)
        return dict(
            silhouette=term,
            per_view_silhouette=per_view,
            per_view_positive_near_global=np.repeat(per_view[..., None] / 3, 3, axis=-1),
            sampled_pixels=pixels,
            sampled_target=target.target.copy(),
            sampled_weights=target.weights.copy(),
            sampled_foreground=target.foreground.copy(),
            ray_origins=origins,
            ray_directions=np.broadcast_to([0.0, 0.0, 1.0], origins.shape).copy(),
            ray_valid=np.ones(origins.shape[:-1], bool),
        )

    monkeypatch.setattr(m, "image_terms", evaluate)
    return calls


@pytest.mark.parametrize("mismatch", [False, True])
def test_fixed_cost_delta_retains_unsaved_orientation_and_sixth(monkeypatch, mismatch):
    bank = bank_fixture()
    before = {k: m.saved.signature(v) for k, v in bank.items()}
    calls = fake_image_terms(monkeypatch, mismatch=mismatch)
    arrays, summary = m.compare_bank(bank, frame_fixture(), Rig(), config())
    assert calls == [0, 1] and before == {k: m.saved.signature(v) for k, v in bank.items()}
    delta = 3 * (arrays["top_contrast__silhouette"] - arrays["detectron2__silhouette"])
    assert np.array_equal(arrays["weighted_image_delta"], delta)
    assert np.array_equal(arrays["counterfactual_local_cost"], bank["adjusted_local_cost"] + delta)
    assert np.array_equal(
        arrays["counterfactual_path_cost"],
        bank["parent_path_cost"] + (bank["adjusted_local_cost"] + delta),
    )
    assert not np.allclose(bank["base_local_cost"], bank["original_five_loss_parts"].sum(1))
    for name in bank:
        assert m.saved.signature(arrays[name]) == before[name]
    assert summary["baseline_CPU_CUDA_within_declared_tolerance"] is (not mismatch)
    if mismatch:
        assert summary["rankings"] == {} and not any(k.endswith("_order") for k in arrays)
    else:
        for key, field in [
            ("saved_local", "adjusted_local_cost"),
            ("counterfactual_path", "counterfactual_path_cost"),
        ]:
            assert np.array_equal(arrays[key + "_order"], np.argsort(arrays[field], kind="stable"))


def setup_run(tmp_path, monkeypatch, *, mismatch=False):
    root = tmp_path / "synthetic_inputs"
    root.mkdir()
    photo_records = []
    masks = {}
    signatures = {}
    load_calls = []
    capture_calls = []

    def write_png(name, arr):
        ok, data = m.cv2.imencode(".png", arr)
        assert ok
        path = root / name
        payload = data.tobytes()
        path.write_bytes(payload)
        return dict(
            path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)
        )

    background = write_png("background.png", np.zeros((1200, 1200, 3), np.uint8))
    for native in m.FRAMES:
        frame = frame_fixture(native, size=1200)
        for view in m.VIEWS:
            image = frame.images_bgr[view]
            ref = write_png(f"f{native}_{view}.png", image)
            photo_records.append(
                dict(frame_idx=native, view=view, image=ref, bgr_signature=m.saved.signature(image))
            )
            key = f"f{native}_{view}"
            mask = frame.masks[view]
            masks[key] = mask
            signature = m.saved.signature(mask)
            signatures[key] = dict(
                frame_idx=native,
                view=view,
                available=True,
                signature=signature,
                original_mask_sha256=signature["sha256"],
            )
    payload = io.BytesIO()
    np.savez_compressed(payload, **masks)
    data = payload.getvalue()
    path = root / "synthetic_masks.npz"
    path.write_bytes(data)
    context = dict(
        bank_references={f: dict(synthetic_frame=f) for f in m.FRAMES},
        rig_refs={"synthetic": "a" * 64},
        config=config(True),
        photo_records=photo_records,
        mask_signatures=signatures,
        mask_archive=dict(path=str(path), sha256=hashlib.sha256(data).hexdigest()),
        top_background=background,
    )
    monkeypatch.setattr(m, "source_code", lambda *a: None)
    monkeypatch.setattr(m, "load_headers", lambda *a: context)
    monkeypatch.setattr(m.saved, "load_rig", lambda *a: Rig())

    def capture(path, expected, records, guards):
        path = Path(path).resolve()
        assert path.is_relative_to(root)
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected
        records[str(path)] = expected
        guards[str(path)] = m.saved.stamp(path)
        capture_calls.append(path.name)
        return payload

    monkeypatch.setattr(m.saved, "capture", capture)

    def load_bank(ref, native, *args):
        assert ref == dict(synthetic_frame=native)
        load_calls.append(native)
        n = m.EXPECTED_COUNTS[native]
        return bank_fixture(n), ["synthetic"] * n

    monkeypatch.setattr(m.saved, "load_bank", load_bank)
    fake_image_terms(monkeypatch, mismatch=mismatch)
    return tmp_path / "synthetic_output", load_calls, capture_calls


@pytest.mark.parametrize("mismatch", [False, True])
def test_synthetic_three_bank_run_saves_measurements_and_failure_gate(
    tmp_path, monkeypatch, mismatch
):
    output, load_calls, captures = setup_run(tmp_path, monkeypatch, mismatch=mismatch)
    if mismatch:
        with pytest.raises(ValueError, match="reproduction tolerance"):
            m.run(output)
    else:
        m.run(output)
    seal = json.loads((output / "comparison.provenance.json").read_bytes())
    summary = json.loads((output / "summary.json").read_bytes())
    assert seal["status"] == ("failed" if mismatch else "ok")
    assert len(seal["outputs"]) == 8 and summary["candidate_count"] == 59
    assert load_calls == list(m.FRAMES)
    assert (
        len(captures) == 11 and len(set(captures)) == 11
    )  # 9 native PNGs+background+one mask archive
    assert summary["refitted"] is summary["physical_accuracy_claimed"] is False
    for native in m.FRAMES:
        case = json.loads((output / f"cases/f{native}.json").read_bytes())
        with np.load(output / f"cases/f{native}.npz", allow_pickle=False) as archive:
            assert len(archive["candidate_index"]) == m.EXPECTED_COUNTS[native]
            assert np.array_equal(
                archive["counterfactual_local_cost"],
                archive["adjusted_local_cost"] + archive["weighted_image_delta"],
            )
            assert (not any(k.endswith("_order") for k in archive.files)) is mismatch
        assert bool(case["rankings"]) is (not mismatch)


def test_output_guard_runs_before_any_input(tmp_path):
    for path in [tmp_path, tmp_path.parent / "outside_forbidden"]:
        with pytest.raises(ValueError, match="fresh workspace"):
            m.run(path)
