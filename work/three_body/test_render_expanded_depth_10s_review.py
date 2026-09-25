"""Synthetic binding/visual contract tests; no recording, GPU or tracker input."""

import copy

import numpy as np
import pytest
import render_expanded_depth_10s_review as render


def headers():
    plan = dict(
        schema=render.comparison.STAGE,
        native_interval=[76360, 78020],
        native_frames=1660,
        terminal_horizon_by_arm=dict.fromkeys(render.ARMS, 78020),
        candidate_references={a: {"fake": a} for a in render.ARMS},
    )
    summary = copy.deepcopy(plan)
    summary.update(
        status="ok",
        all_three_physical_ID_truth=False,
        physical_depth_truth=False,
        candidate_modified=False,
        review_modified=False,
    )
    return dict(stage=render.comparison.STAGE, status="ok"), plan, summary


def test_new_stage_and_fixed_scope_are_disabled():
    assert not render.EXECUTION_READY
    assert render.STAGE != render.previous.STAGE
    assert len(render.FRAMES) == 6 and tuple(sorted(render.FRAMES)) == render.FRAMES
    assert set(render.FRAMES) == set(render.DEVELOPMENT_FRAMES) | set(render.REGRESSION_SELECTION)
    render.validate_comparison(*headers())
    with pytest.raises(ValueError, match="source review"):
        render.run(render.WORK / "not_created_by_disabled_render")


@pytest.mark.parametrize("change", ["stage", "horizon", "interval", "reference", "truth", "review"])
def test_comparison_tampering_rejected(change):
    seal, plan, summary = headers()
    if change == "stage":
        seal["stage"] = render.previous.COMPARISON_STAGE
    elif change == "horizon":
        summary["terminal_horizon_by_arm"][render.ARMS[1]] = 86320
    elif change == "interval":
        summary["native_interval"][0] += 1
    elif change == "reference":
        summary["candidate_references"][render.ARMS[1]]["fake"] = "other"
    elif change == "truth":
        summary["all_three_physical_ID_truth"] = True
    else:
        summary["review_modified"] = True
    with pytest.raises(ValueError):
        render.validate_comparison(seal, plan, summary)


def indices():
    dense = dict(
        original_binding={"inputs": {"test": 1}},
        prefix_chunks=[
            dict(index=i, start=76360 if i == 16 else i, stop=78020 if i == 16 else i + 1)
            for i in range(17)
        ],
    )
    expanded = copy.deepcopy(dense)
    expanded["rationale_references"] = {"explicit_new_depth_rationale": True}
    return dense, expanded


def test_actual_two_index_schemas_and_same_chunk():
    dense, expanded = indices()
    assert render.validate_indices(dense, expanded) == dense["prefix_chunks"][-1]
    for key in ("original_binding", "prefix_chunks"):
        changed = copy.deepcopy(expanded)
        changed[key] = {}
        with pytest.raises(ValueError, match="observation histories"):
            render.validate_indices(dense, changed)
    changed = copy.deepcopy(expanded)
    del changed["rationale_references"]
    with pytest.raises(ValueError, match="schema"):
        render.validate_indices(dense, changed)


@pytest.fixture
def candidate():
    n = 78020 - 49800
    frames = np.arange(49800, 78020, dtype=np.int64)
    arrays = dict(
        frame_idx=frames,
        pose_xyz=np.zeros((n, 3, 3, 3), np.float64),
        sources=np.full((n, 3, 3, 3), -1, np.int64),
        observed=np.zeros((n, 3, 3), bool),
        identity_review=np.ones((n, 3), bool),
    )
    arrays["pose_xyz"][:, 0, 0, 0] = frames
    return arrays, dict(
        terminal_horizon=78020, arrays={k: render.proof.signature(v) for k, v in arrays.items()}
    )


def test_six_anchor_rows_preserve_geometry_sources_reviews_and_readonly(candidate):
    arrays, reference = candidate
    before = {k: render.proof.signature(v) for k, v in arrays.items()}
    selected = render.anchor_rows(arrays, reference)
    np.testing.assert_array_equal(selected["frame_idx"], render.FRAMES)
    np.testing.assert_array_equal(selected["pose_xyz"][:, 0, 0, 0], render.FRAMES)
    assert selected["identity_review"].all() and (selected["sources"] == -1).all()
    assert not selected["observed"].any()
    assert all(not v.flags.writeable for v in selected.values())
    assert before == {k: render.proof.signature(v) for k, v in arrays.items()}


@pytest.mark.parametrize("key", render.KEYS)
def test_each_required_candidate_field_tampering_rejected(candidate, key):
    arrays, reference = candidate
    arrays[key].flat[0] = not arrays[key].flat[0] if arrays[key].dtype == bool else 100
    with pytest.raises(ValueError, match="signatures"):
        render.anchor_rows(arrays, reference)


def test_signed_but_gapped_candidate_rejected(candidate):
    arrays, reference = candidate
    arrays["frame_idx"][100] += 1
    reference["arrays"]["frame_idx"] = render.proof.signature(arrays["frame_idx"])
    with pytest.raises(ValueError, match="horizon"):
        render.anchor_rows(arrays, reference)


def fixture_image():
    image = np.zeros((80, 90, 3), np.uint8)
    masks = np.zeros((3, 80, 90), bool)
    for i in range(3):
        masks[i, 15 + 20 * i : 20 + 20 * i, 10 + 20 * i : 22 + 20 * i] = True
    points = np.array(
        [
            [[12, 16], [16, 17], [20, 18]],
            [[32, 36], [36, 37], [40, 38]],
            [[52, 56], [56, 57], [60, 58]],
        ],
        np.float32,
    )
    observed = np.ones((3, 3), bool)
    observed[0, 0] = False
    return image, masks, points, observed


@pytest.mark.parametrize("mode", ["raw", "D2"])
def test_panel_has_all_nodes_SL_mouth_and_mask_indices_without_mutation(mode):
    image, masks, points, observed = fixture_image()
    before = [a.tobytes() for a in (image, masks, points, observed)]
    fig, ax = render.prior.plt.subplots()
    try:
        render.draw_panel(ax, image, masks, points, points, observed, mode)
        labels = {t.get_text() for t in ax.texts}
        assert {f"D2#{s}" for s in range(3)} <= labels
        assert {f"SL{s}M" for s in range(3)} <= labels
        assert {f"S{s}{n}" for s in range(3) for n in "MCT"} <= labels
        assert len(ax.images) == (2 if mode == "D2" else 1)
        # One latent node is an open circle, unlike directly supported nodes.
        circles = [line for line in ax.lines if line.get_marker() == "o"]
        assert circles[0].get_markerfacecolor() == "none"
        assert circles[1].get_markerfacecolor() != "none"
    finally:
        render.prior.plt.close(fig)
    assert before == [a.tobytes() for a in (image, masks, points, observed)]


@pytest.mark.parametrize("mask_mode", ["missing", "empty"])
def test_missing_or_empty_masks_and_nonfinite_SL_render(mask_mode):
    image, _, points, observed = fixture_image()
    points[:] = np.nan
    masks = None if mask_mode == "missing" else np.empty((0, 80, 90), bool)
    fig, ax = render.prior.plt.subplots()
    try:
        render.draw_panel(ax, image, masks, points, None, observed, "D2")
        assert not any(t.get_text().startswith("D2#") for t in ax.texts)
        lo, hi = render.display_crop(image, masks, points, [points])
        np.testing.assert_array_equal(lo, [0, 0])
        np.testing.assert_array_equal(hi, [90, 80])
    finally:
        render.prior.plt.close(fig)


def test_old_nine_raw_masks_SL_must_match():
    _, masks, points, _ = fixture_image()
    row = dict(
        bgr_sha256="bgr",
        mask_sha256="masks",
        mask_count=3,
        mask_available=True,
        SLEAP_points=render.prior.finite_json(points),
    )
    old = dict(records=[dict(frame=77709, observations={"top_cam": row})])
    assert render.verify_old_observation(77709, "top_cam", "bgr", "masks", masks, True, points, old)
    assert not render.verify_old_observation(
        77413, "top_cam", "bgr", "masks", masks, True, points, old
    )
    with pytest.raises(ValueError, match="frame evidence"):
        render.verify_old_observation(
            77709, "top_cam", "changed", "masks", masks, True, points, old
        )


def test_composite_png_and_both_modes_same_crops():
    image, masks, points, observed = fixture_image()
    images = dict.fromkeys(render.VIEWS, image)
    mask_map = dict.fromkeys(render.VIEWS, masks)
    point_map = dict.fromkeys(render.VIEWS, points)
    projections = {a: point_map for a in render.ARMS}
    rows = {a: {"observed": observed[None]} for a in render.ARMS}
    png, crop1 = render.render(77413, images, mask_map, point_map, projections, rows, 0, "raw")
    png2, crop2 = render.render(77413, images, mask_map, point_map, projections, rows, 0, "D2")
    assert png[:8] == png2[:8] == b"\x89PNG\r\n\x1a\n"
    assert crop1 == crop2 and png != png2
    assert not render.prior.plt.get_fignums()
