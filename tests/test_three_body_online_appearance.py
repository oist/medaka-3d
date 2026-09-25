"""Prospective templates freeze once; missing and model-reviewed slots stay excluded."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from medaka3d.types import VIEWS


def modules():
    lab = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
    names = ("appearance_diagnostic", "online_appearance")
    previous = {name: sys.modules.get(name) for name in names}
    loaded = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"online_test_{name}", lab / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return loaded


M = modules()
A = M["online_appearance"]
D = M["appearance_diagnostic"]


class Rig:
    views = VIEWS

    def camera(self, view):
        assert view in VIEWS
        return SimpleNamespace(project=lambda xyz: np.asarray(xyz)[..., :2].copy())


def inputs(frame_idx=40):
    pose = np.array([
        [[12+30*i, 15, 8], [17+30*i, 15, 8], [20+30*i, 15, 8]] for i in range(3)
    ], dtype=float)
    masks = np.zeros((3, 30, 100), bool)
    image = np.zeros((30, 100, 3), np.uint8)
    for fish, color in enumerate(([30, 30, 100], [20, 100, 160], [80, 100, 100])):
        masks[fish, 6:25, 5+30*fish:25+30*fish] = True
        image[masks[fish]] = color
    frame = SimpleNamespace(
        frame_idx=frame_idx,
        masks={view: masks.copy() for view in VIEWS},
        images_bgr={view: image.copy() for view in VIEWS},
        # These channels have no role in template registration.
        detections=object(), backgrounds_bgr=object(),
    )
    sources = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    return frame, pose, sources, np.ones((3, 3), bool), np.zeros(3, bool)


def accumulator(n=3, minimum=2):
    return A.OnlineAppearanceAccumulator(Rig(), A.OnlineAppearanceConfig(n, minimum))


def assert_snapshot_equal(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, right[key], err_msg=key)
        else:
            assert value == right[key]


def test_initial_state_zero_counts_no_prefix_template_and_exact_freeze_median():
    acc = accumulator()
    assert acc.training_frames_seen == 0 and not acc.frozen
    assert not acc.counts.any() and not acc.enabled_views.any()
    assert np.isnan(acc.templates).all()
    assert acc.snapshot()["features"].shape == (0, 3, 3, 6)
    samples = []
    for t in range(3):
        args = inputs(40+t)
        for image in args[0].images_bgr.values():
            image[:] = np.minimum(image.astype(int) + t * 7, 255).astype(np.uint8)
        samples.append([
            D.describe_core(args[0].images_bgr[VIEWS[0]], mask)[0]
            for mask in args[0].masks[VIEWS[0]]
        ])
        acc.observe(*args)
        if t < 2:
            assert not acc.frozen and not acc.enabled_views.any()
            assert np.isnan(acc.templates).all()
    assert acc.frozen and acc.enabled_views.all()
    np.testing.assert_array_equal(acc.counts, np.full((3, 3), 3))
    expected = np.median(samples, axis=0)
    np.testing.assert_array_equal(acc.templates, np.broadcast_to(expected, (3, 3, 6)))
    assert acc.snapshot()["first_frame_idx"] == 40
    assert acc.snapshot()["last_training_frame_idx"] == 42


@pytest.mark.parametrize("next_frame", [40, 39, 42])
def test_duplicate_reverse_and_gap_fail_without_committing(next_frame):
    acc = accumulator()
    acc.observe(*inputs(40))
    before = acc.snapshot()
    with pytest.raises(ValueError, match="continuous native"):
        acc.observe(*inputs(next_frame))
    assert_snapshot_equal(before, acc.snapshot())
    acc.observe(*inputs(41))
    assert acc.training_frames_seen == 2


def test_after_freeze_never_reads_new_payload_or_changes_counts_templates_history():
    acc = accumulator(1, 1)
    acc.observe(*inputs(40))
    before = acc.snapshot()
    for frame_idx in (41, 42):
        acc.observe(SimpleNamespace(frame_idx=frame_idx), object(), object(), object(), object())
    assert_snapshot_equal(before, acc.snapshot())
    with pytest.raises(ValueError, match="continuous native"):
        acc.observe(SimpleNamespace(frame_idx=44), None, None, None, None)


def test_empty_masks_and_unavailable_views_keep_full_denominator_and_disable_entire_view():
    acc = accumulator(2, 2)
    for t in range(2):
        args = inputs(40+t)
        args[0].masks[VIEWS[1]] = np.zeros((0, 30, 100), bool)
        args[0].masks[VIEWS[2]] = None
        acc.observe(*args)
    np.testing.assert_array_equal(acc.counts, [[2, 2, 2], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(acc.enabled_views, [True, False, False])
    assert np.isnan(acc.templates[1:]).all()
    reasons = acc.snapshot()["reasons"]
    assert reasons.shape == (2, 3, 3)
    assert np.all(reasons[:, 1] == "no_masks")
    assert np.all(reasons[:, 2] == "view_unavailable")


def test_missing_image_or_dictionary_view_does_not_prevent_other_views_registering():
    acc = accumulator(1, 1)
    args = inputs()
    del args[0].images_bgr[VIEWS[1]]
    del args[0].masks[VIEWS[2]]
    acc.observe(*args)
    np.testing.assert_array_equal(acc.enabled_views, [True, False, False])
    assert np.all(acc.snapshot()["reasons"][0, 1] == "image_unavailable")
    assert np.all(acc.snapshot()["reasons"][0, 2] == "view_unavailable")


def test_one_under_sampled_slot_disables_all_templates_in_its_view():
    acc = accumulator(2, 2)
    for t in range(2):
        args = inputs(40+t)
        if t == 0:
            args[0].masks[VIEWS[0]] = args[0].masks[VIEWS[0]][:2]
        acc.observe(*args)
    np.testing.assert_array_equal(acc.counts[0], [2, 2, 1])
    assert not acc.enabled_views[0] and np.isnan(acc.templates[0]).all()
    assert acc.enabled_views[1:].all()


def test_reviewed_slot_is_excluded_but_remains_as_latent_mask_competitor():
    acc = accumulator(1, 1)
    args = inputs()
    args[4][1] = True
    for view in VIEWS:
        args[0].masks[view][0] |= args[0].masks[view][1]
    acc.observe(*args)
    np.testing.assert_array_equal(acc.counts, [[0, 0, 1]] * 3)
    reasons = acc.snapshot()["reasons"][0]
    assert np.all(reasons[:, 0] == "mask_contains_other_projected_node")
    assert np.all(reasons[:, 1] == "identity_review")
    assert not acc.enabled_views.any()


def test_global_direct_and_local_mouth_pect_sources_are_both_required():
    acc = accumulator(1, 1)
    args = inputs()
    args[3][0, 0] = False  # Global mouth unsupported, even with present local source.
    args[2][1, 0, 0] = -1  # Still globally supported by the other two views.
    acc.observe(*args)
    np.testing.assert_array_equal(acc.counts, [[0, 0, 1], [0, 1, 1], [0, 1, 1]])
    assert not acc.enabled_views.any()


def test_duplicate_instance_tie_and_tiny_eroded_core_do_not_register():
    acc = accumulator(1, 1)
    args = inputs()
    for view in VIEWS:
        masks = args[0].masks[view]
        masks[1] = False
        masks[1, 14:17, 41:48] = True  # Both projected nodes inside, no eroded core.
        args[0].masks[view] = np.concatenate([masks, masks[:1]])
    acc.observe(*args)
    np.testing.assert_array_equal(acc.counts, [[0, 0, 1]] * 3)
    assert np.all(acc.snapshot()["reasons"][0, :, 0] == "multiple_joint_instances")
    assert np.all(acc.snapshot()["reasons"][0, :, 1] == "core_too_small")


def test_invalid_latent_projection_never_hides_competitor():
    acc = accumulator(1, 1)
    acc.rig = SimpleNamespace(
        views=VIEWS,
        camera=lambda view: SimpleNamespace(project=lambda xyz: np.full((3, 2, 2), np.nan)),
    )
    acc.observe(*inputs())
    assert not acc.counts.any()
    assert np.all(acc.snapshot()["reasons"] == "projection_invalid")


def test_copies_defend_against_caller_mutations_of_inputs_and_returned_arrays():
    acc = accumulator(1, 1)
    args = inputs()
    acc.observe(*args)
    before = acc.snapshot()
    args[1][:] = 999
    for image in args[0].images_bgr.values():
        image[:] = 0
    for value in (acc.templates, acc.counts, acc.enabled_views):
        value[:] = 0
    altered = acc.snapshot()
    altered["config"]["training_frames"] = 999
    altered["views"][0] = "wrong"
    altered["features"][:] = 0
    altered["reasons"][:] = "invented"
    for name in ("online_pose_xyz", "online_sources", "online_observed", "online_identity_review"):
        altered[name][:] = 0
    assert_snapshot_equal(before, acc.snapshot())


def test_online_audit_shapes_and_dtypes_are_stable_before_any_observation():
    snapshot = accumulator().snapshot()
    expected = {
        "online_pose_xyz": ((0, 3, 3, 3), np.dtype("float64")),
        "online_sources": ((0, 3, 3, 3), np.dtype("int64")),
        "online_observed": ((0, 3, 3), np.dtype(bool)),
        "online_identity_review": ((0, 3), np.dtype(bool)),
    }
    for key, (shape, dtype) in expected.items():
        assert snapshot[key].shape == shape and snapshot[key].dtype == dtype


def test_online_audit_retains_each_current_state_despite_buffer_reuse_and_later_traceback():
    acc = accumulator(2, 1)
    args = inputs()
    args[3][2, 0] = False
    args[4][1] = True
    keys = ("online_pose_xyz", "online_sources", "online_observed", "online_identity_review")
    first = [value.copy() for value in args[1:]]
    acc.observe(*args)
    permutation = [2, 0, 1]
    for value in args[1:]:
        value[:] = value[permutation]  # Reuse caller buffers for another online state.
    args[0].frame_idx = 41
    second = [value.copy() for value in args[1:]]
    acc.observe(*args)
    snapshot = acc.snapshot()
    for key, old, new in zip(keys, first, second, strict=True):
        np.testing.assert_array_equal(snapshot[key], np.stack([old, new]))
    assert acc.frozen
    # A final traceback can select a different earlier branch. It is not supplied
    # back to registration, and mutations of either input or returned audit arrays
    # must not rewrite the online states preserved above.
    for value in args[1:]:
        value[:] = 0
    for key in keys:
        changed_traceback = snapshot[key][::-1].copy()
        snapshot[key][:] = changed_traceback
    acc.observe(SimpleNamespace(frame_idx=42), object(), object(), object(), object())
    final = acc.snapshot()
    for key, old, new in zip(keys, first, second, strict=True):
        np.testing.assert_array_equal(final[key], np.stack([old, new]))
    np.testing.assert_array_equal(final["frame_idx"], [40, 41])


def test_invalid_later_view_is_atomic_and_same_frame_can_be_retried():
    acc = accumulator(1, 1)
    args = inputs()
    args[0].images_bgr[VIEWS[2]] = np.zeros((30, 100, 3), np.float32)
    with pytest.raises(ValueError, match="uint8"):
        acc.observe(*args)
    assert acc.training_frames_seen == 0 and not acc.counts.any()
    acc.observe(*inputs())
    assert acc.frozen


@pytest.mark.parametrize("field,value", [
    ("training_frames", True), ("training_frames", 0), ("training_frames", 167),
    ("minimum_template_samples", 0), ("minimum_template_samples", float("nan")),
    ("minimum_template_samples", 1.5), ("minimum_template_samples", 167),
])
def test_config_rejects_invalid_prefix_or_sample_counts(field, value):
    with pytest.raises(ValueError):
        replace(A.OnlineAppearanceConfig(), **{field: value})


def test_source_view_order_and_boolean_support_validation():
    with pytest.raises(ValueError, match="view order"):
        A.OnlineAppearanceAccumulator(SimpleNamespace(views=VIEWS[::-1]))
    args = list(inputs())
    args[3] = args[3].astype(int)
    with pytest.raises(ValueError, match="observed"):
        accumulator().observe(*args)
    args = inputs()
    args[2][:, 1:] = -1
    with pytest.raises(ValueError, match="at least two views"):
        accumulator().observe(*args)
