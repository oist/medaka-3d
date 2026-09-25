"""Local experiment I/O, arm conservation and measurement semantics on CPU."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "body_volume_probe_tests_driver", REPO / "work/three_body/run_body_volume_probe.py"
)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)
from prepare_sequence import PackedMasks  # noqa: E402


def scene():
    centers = np.array([[40.0, 50.0, 60.0], [85.0, 90.0, 100.0], [135.0, 120.0, 140.0]])
    return centers[:, None] + np.array([[6.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-16.0, 2.0, 1.0]])


def arms():
    poses = np.repeat(scene()[None], 4, axis=0)
    for t in range(1, 4):
        poses[t, t - 1] += [t, 0, 0]
    second = poses.copy()
    second[1:] += [0, 3, 0]
    outside_a = np.zeros_like(poses, bool)
    outside_b = outside_a.copy()
    outside_a[2, 1, 0, 0] = True
    outside_b[1, 0, 0, 0] = True
    return dict(
        pose_xyz=poses,
        outside_coordinates=outside_a,
        admissible=np.array([True, True, False, True]),
        component_slot_sign_roll=np.array(
            [[-1, -1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 2, 1, 0]]
        ),
    ), dict(
        pose_xyz=second,
        outside_coordinates=outside_b,
        admissible=np.array([True, False, True, True]),
        raw_slot=np.array([-1, 0, 1, 2]),
    )


def test_shared_original_and_unequal_rejections_have_exact_arm_index_maps():
    a, b = arms()
    before = [{k: v.tobytes() for k, v in arm.items()} for arm in (a, b)]
    poses, index = m.combine_admissible(a, b)
    assert index["mask"].tolist() == [0, 1, -1, 2]
    assert index["random"].tolist() == [0, -1, 3, 4]
    assert len(poses) == 5
    for name, arm in [("mask", a), ("random", b)]:
        for trial in np.flatnonzero(arm["admissible"]):
            np.testing.assert_array_equal(poses[index[name][trial]], arm["pose_xyz"][trial])
    assert all(
        {k: v.tobytes() for k, v in arm.items()} == saved
        for arm, saved in zip((a, b), before, strict=True)
    )


@pytest.mark.parametrize("bad", ["original", "slots", "outside", "base_rejected"])
def test_incomparable_or_misreported_arms_rejected(bad):
    a, b = arms()
    if bad == "original":
        b["pose_xyz"][0, 0, 0, 0] += 1
    elif bad == "slots":
        b["raw_slot"][2] = 0
    elif bad == "outside":
        a["admissible"][2] = True
    else:
        a["admissible"][0] = False
    with pytest.raises((AssertionError, ValueError)):
        m.combine_admissible(a, b)


def test_no_defined_component_is_retained_as_base_only_control():
    a = m.volume_scene_starts(scene(), [])
    b = m.random_scene_starts(scene(), a["component_slot_sign_roll"][1:, 1], seed=1)
    poses, index = m.combine_admissible(a, b)
    np.testing.assert_array_equal(poses, scene()[None])
    assert index["mask"].tolist() == index["random"].tolist() == [0]
    assert m.unique_pose_count(np.repeat(poses, 8, axis=0)) == 1
    different = np.repeat(poses, 2, axis=0)
    different[1, 0, 0, 0] += 2e-9
    assert m.unique_pose_count(different) == 2


def tiny_frame(empty=False):
    masks = np.zeros((0 if empty else 1, 8, 8), bool)
    if not empty:
        masks[0, 2:5, 2:5] = True
    return m.FrameObservation(
        79372,
        {},
        dict.fromkeys(m.base.VIEWS, np.zeros((8, 8, 3), np.uint8)),
        dict.fromkeys(m.base.VIEWS, masks),
        {},
    )


def fitting(frame):
    return m.sample_pixels(frame, n_positive=2, n_near=2, n_global=1)


def test_saved_spatial_targets_are_disjoint_and_preserve_actual_weights():
    frame = tiny_frame()
    fit = fitting(frame)
    evaluation = m.sample_body_evaluation(frame, fit.pixels)
    arrays = m.target_arrays(fit, evaluation)
    for view in m.base.VIEWS:
        assert arrays["fit_pixels_" + view].tobytes() == fit.pixels[view].tobytes()
        assert arrays["evaluation_pixels_" + view].tobytes() == evaluation.pixels[view].tobytes()
    assert arrays["evaluation_weights"].tobytes() == evaluation.weights.tobytes()
    assert all(not value.dtype.hasobject for value in arrays.values())
    collision = dict(evaluation.pixels)
    collision[m.base.VIEWS[0]] = fit.pixels[m.base.VIEWS[0]][:1]
    with pytest.raises(ValueError, match="fitting pixel"):
        m.target_arrays(fit, replace(evaluation, pixels=collision))


def test_all_empty_targets_are_neutral_without_constructing_renderer(monkeypatch):
    frame = tiny_frame(empty=True)
    targets = m.sample_body_evaluation(frame, fitting(frame).pixels)
    monkeypatch.setattr(m, "RefractiveFishRenderer", lambda *a, **kw: pytest.fail("neutral render"))
    result = m.evaluate_poses(None, scene()[None], targets, {})
    assert result["union"].shape == (1, 3, 0)
    assert result["ray_valid"].shape == (3, 0)
    np.testing.assert_array_equal(result["mean_view_loss"], 0.0)
    assert result["valid_counts"].sum() == 0


def test_full_instance_masses_keep_overlap_and_exclude_invalid_rays(monkeypatch):
    masks = np.zeros((2, 1200, 1200), bool)
    masks[0, 10, [10, 11]] = True
    masks[1, 10, [11, 12]] = True
    frame = m.FrameObservation(79372, {}, {}, dict.fromkeys(m.base.VIEWS, masks), {})
    alpha = np.broadcast_to(
        np.array([[0.5, 0.2, 0.0], [0.25, 0.5, 0.0], [0.9, 0, 0]]), (2, 3, 3, 3)
    ).copy()
    union = 1 - np.prod(1 - alpha, axis=-1)
    visible = alpha.copy()
    visible[..., 1] *= 1 - alpha[..., 0]
    visible[..., 2] *= (1 - alpha[..., 0]) * (1 - alpha[..., 1])
    valid = np.ones((3, 3), bool)
    valid[0, 2] = False
    rendered = SimpleNamespace(
        per_fish_alpha=torch.tensor(alpha),
        visible_alpha=torch.tensor(visible),
        union_alpha=torch.tensor(union),
        ray_valid=torch.tensor(valid),
    )
    monkeypatch.setattr(
        m, "RefractiveFishRenderer", lambda *a, **kw: SimpleNamespace(render=lambda pose: rendered)
    )
    rows = m.instance_mass(
        None, np.repeat(scene()[None], 2, axis=0), frame, dict(width_ratio=0.06, device="cpu")
    )
    assert len(rows) == 6
    first, overlap = rows[:2]
    assert first["valid_pixels"] == 2 and overlap["valid_pixels"] == 1
    np.testing.assert_allclose(first["unexplained_mass"], 0.4 + 0.375)
    np.testing.assert_allclose(first["unique_union_mass"], [[0.525, 0.475, 0]] * 2)
    np.testing.assert_allclose(first["pair_shared_mass"], [[0.225, 0, 0]] * 2)
    np.testing.assert_allclose(overlap["visible_mass"], [[0.25, 0.375, 0]] * 2)
    assert sum(row["valid_pixels"] for row in rows[2:4]) == 4  # Not union's three pixels.


def test_original_support_uses_its_new_plan_not_a_saved_id_flag(monkeypatch):
    starts = np.repeat(scene()[None], 2, axis=0)
    plans = np.arange(54).reshape(2, 3, 3, 3)
    calls = []

    def support(pose, plan, rays, rig):
        calls.append((pose.copy(), plan.copy()))
        return plan + 1, np.full((3, 3), len(calls) == 1), np.zeros((3, 3, 3))

    monkeypatch.setattr(m, "supported_nodes", support)
    result = m.initial_support(None, dict(input_pose_xyz=starts, plans=plans), None)
    np.testing.assert_array_equal(result["initial_sources"], plans + 1)
    assert result["initial_observed"][0].all() and not result["initial_observed"][1].any()
    for index, (pose, plan) in enumerate(calls):
        np.testing.assert_array_equal(pose, starts[index])
        np.testing.assert_array_equal(plan, plans[index])


def decoder_fixture(monkeypatch, tmp_path):
    frames = np.arange(100, 103)
    arrays = {}
    pictures = [np.full((1200, 1200, 3), t, np.uint8) for t in range(3)]
    manifest = dict(raw={}, masks={}, backgrounds={})
    for view in m.base.VIEWS:
        encoder = PackedMasks()
        for t, native in enumerate(frames):
            masks = np.zeros((2 if t == 1 else 0, 1200, 1200), bool)
            if len(masks):
                masks[0, 10:12, 14:16] = True
                masks[1, 20:24, 18:20] = True
            encoder.append(
                native,
                masks,
                np.full(len(masks), 0.75),
                np.zeros((len(masks), 4)),
                np.zeros(len(masks), np.int64),
            )
        arrays[view] = encoder.arrays()
        arrays[view]["view_available"][2] = False
        manifest["masks"][view] = dict(path=view)
        manifest["backgrounds"][view] = dict(path="background_" + view)
        manifest["raw"][view] = dict(
            path="raw_" + view, frame_bgr_sha256=[m.array_sha256(v) for v in pictures]
        )
        for name in (view, "raw_" + view, "background_" + view):
            (tmp_path / name).write_bytes(b"decoder fixture")
    captures = []

    class Capture:
        def __init__(self, *args):
            self.position = 0
            self.released = False
            captures.append(self)

        def isOpened(self):
            return True

        def set(self, key, value):
            assert key == cv2.CAP_PROP_POS_FRAMES
            self.position = int(value)
            return True

        def get(self, key):
            return {
                cv2.CAP_PROP_FRAME_COUNT: 3,
                cv2.CAP_PROP_FPS: 166,
                cv2.CAP_PROP_FRAME_WIDTH: 1200,
                cv2.CAP_PROP_FRAME_HEIGHT: 1200,
                cv2.CAP_PROP_POS_FRAMES: self.position,
            }[key]

        def read(self):
            picture = pictures[self.position]
            self.position += 1
            return True, picture

        def release(self):
            self.released = True

    monkeypatch.setattr(m, "FRAMES", (100, 101, 102))
    monkeypatch.setattr(m, "_npz", lambda path: arrays[Path(path).name])
    monkeypatch.setattr(m.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(m.cv2, "imread", lambda path: pictures[0])
    run = SimpleNamespace(frames=frames, manifest=manifest, sequence=tmp_path)
    nodes = SimpleNamespace(at=lambda v, f: np.full((3, 3, 2), f))
    return run, nodes, arrays, captures


def test_native_seek_row_order_and_missing_vs_empty_are_preserved(monkeypatch, tmp_path):
    run, nodes, arrays, captures = decoder_fixture(monkeypatch, tmp_path)
    results = list(m.decoded_probes(run, nodes))
    assert [frame.frame_idx for _, frame, _ in results] == [100, 101, 102]
    for view in m.base.VIEWS:
        assert results[0][1].masks[view].shape == (0, 1200, 1200)
        assert results[2][1].masks[view] is None
        np.testing.assert_array_equal(
            results[1][1].masks[view], m.decode_frame_masks(arrays[view], 1)
        )
        assert results[1][2][view]["instance_id"] == [0, 1]
        assert results[1][2][view]["bgr_sha256"] == run.manifest["raw"][view]["frame_bgr_sha256"][1]
    assert all(capture.released for capture in captures)


@pytest.mark.parametrize("bad", ["raw_hash", "mask_payload", "unavailable", "frames", "fps"])
def test_corrupt_or_misaligned_inputs_fail_and_release_decoders(monkeypatch, tmp_path, bad):
    run, nodes, arrays, captures = decoder_fixture(monkeypatch, tmp_path)
    view = m.base.VIEWS[0]
    if bad == "raw_hash":
        run.manifest["raw"][view]["frame_bgr_sha256"][1] = "0" * 64
    elif bad == "mask_payload":
        arrays[view]["mask_bits"][0] ^= 1
    elif bad == "unavailable":
        arrays[view]["view_available"][1] = False
    elif bad == "frames":
        arrays[view]["frame_idx"][2] += 1
    else:
        original = m.cv2.VideoCapture.get
        monkeypatch.setattr(
            m.cv2.VideoCapture,
            "get",
            lambda self, key: 165 if key == cv2.CAP_PROP_FPS else original(self, key),
        )
    with pytest.raises((ValueError, AssertionError)):
        list(m.decoded_probes(run, nodes))
    assert all(capture.released for capture in captures)


def test_main_rejects_unsealed_input_before_gpu_and_preserves_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "source_paths", lambda: [Path(m.__file__)])

    def reject(*args, **kwargs):
        assert kwargs == {"appearance": False}
        raise ValueError("producer is not success-sealed")

    monkeypatch.setattr(m, "_verify_run", reject)
    monkeypatch.setattr(m, "LocalBodyFitter", lambda *a, **kw: pytest.fail("GPU before binding"))
    torch_threads, cv_threads = torch.get_num_threads(), cv2.getNumThreads()
    output = tmp_path / "failed_probe"
    with pytest.raises(ValueError, match="success-sealed"):
        m.main(["--run", str(tmp_path / "incomplete"), "--output", str(output)])
    assert torch.get_num_threads() == torch_threads and cv2.getNumThreads() == cv_threads
    seal = json.loads((output / "experiment.provenance.json").read_text())
    assert seal["status"] == "failed"
    assert not (output / "summary.json").exists()
    assert len(seal["outputs"]) == 1
    with pytest.raises(FileExistsError):
        m.main(["--output", str(output)])


def test_preview_all_unknown_projection_retains_full_native_image():
    camera = SimpleNamespace(project=lambda poses: np.full((*poses.shape[:-1], 2), np.nan))
    rig = SimpleNamespace(camera=lambda view: camera)
    frame = m.FrameObservation(
        79372,
        {},
        dict.fromkeys(m.base.VIEWS, np.zeros((1200, 1200, 3), np.uint8)),
        dict.fromkeys(m.base.VIEWS, None),
        {},
    )
    image, native = m.preview(rig, frame, scene()[None], np.zeros((1, 3, 3), bool), ["unknown"])
    assert image.shape == (350, 1830, 3)
    for view in m.base.VIEWS:
        assert native[view]["crop_xyxy"] == [0, 0, 1200, 1200]
