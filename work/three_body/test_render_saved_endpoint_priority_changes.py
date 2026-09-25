"""Tiny synthetic display contracts; scientific input loaders cannot be called."""

import copy

import numpy as np
import pytest
import render_saved_endpoint_priority_changes as m


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved scientific input or raw video forbidden in synthetic tests")

    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "SCORE_ROOT", tmp_path / "score")
    monkeypatch.setattr(m, "PHOTO_ROOT", tmp_path / "photos")
    for name in ("load_context", "load_saved_masks", "photograph", "source_code"):
        monkeypatch.setattr(m, name, forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(m.prior.cv2, "VideoCapture", forbidden)


def fixture(frame):
    points = np.array([[[20, y], [35, y], [50, y]] for y in (25, 60, 95)], float)
    case = dict(arm=m.ARM, frame_idx=frame, selections={})
    for j, (mode, index) in enumerate(zip(m.MODES, m.EXPECTED_WINNERS[frame], strict=True)):
        labels = {v: ["same_instance", "uncovered_endpoint", "ambiguous_endpoint"] for v in m.VIEWS}
        if j == 0:
            labels[m.VIEWS[0]][0] = "split_instances"
        case["selections"][mode] = dict(
            candidate_index=index,
            kind="synthetic",
            candidate_parent=0,
            pose_xyz=np.pad(points, ((0, 0), (0, 0), (0, 1))).tolist(),
            candidate_plan=np.zeros((3, 3, 3), np.int32).tolist(),
            projections={v: (points + j).tolist() for v in m.VIEWS},
            classifications=labels,
            split_by_body=[j == 0, False, False],
            parent_cost=3.0,
            local_cost=2.0 + j,
            path_cost=5.0 + j,
        )
    summary = dict(
        arm=m.ARM,
        frame_idx=frame,
        best_changed=True,
        candidates=20,
        old_best=m.EXPECTED_WINNERS[frame][0],
        priority_best=m.EXPECTED_WINNERS[frame][1],
        selected_priority_path_cost_difference=1.0,
    )
    return case, summary


@pytest.mark.parametrize("frame", m.FRAMES)
def test_exact_saved_selections_and_costs_without_slot_permutation(frame):
    case, summary = fixture(frame)
    original = copy.deepcopy(case)
    selected = m.selected_case(case, summary)
    assert case == original
    assert [selected["selections"][mode]["candidate_index"] for mode in m.MODES] == list(
        m.EXPECTED_WINNERS[frame]
    )
    for mode in m.MODES:
        for view in m.VIEWS:
            np.testing.assert_array_equal(
                selected["projections"][mode][view], case["selections"][mode]["projections"][view]
            )
    case["selections"][m.MODES[1]]["path_cost"] += 0.5
    with pytest.raises(ValueError, match="ledger"):
        m.selected_case(case, summary)


def test_actual_synthetic_render_has_only_open_nodes_no_mask_fill_and_same_crop(monkeypatch):
    case, summary = fixture(m.FRAMES[0])
    selected = m.selected_case(case, summary)
    image = np.full((120, 140, 3), 72, np.uint8)
    masks = np.zeros((3, 120, 140), bool)
    masks[0, 15:35, 15:56] = True
    masks[1, 50:70, 15:56] = True
    masks[2, 85:105, 15:56] = True
    before = image.tobytes(), masks.tobytes()
    close = m.prior.plt.close
    checked = []

    def inspect_then_close(fig):
        axes = fig.axes
        assert len(axes) == 6
        for ax in axes:
            assert len(ax.images) == 1  # photograph only, masks drawn as contours
            nodes = [line for line in ax.lines if line.get_marker() in ("o", "s", "^")]
            assert len(nodes) == 9 and all(line.get_markerfacecolor() == "none" for line in nodes)
        for vi in range(3):
            assert axes[vi].get_xlim() == axes[vi + 3].get_xlim()
            assert axes[vi].get_ylim() == axes[vi + 3].get_ylim()
        assert "Depth -12" in fig._suptitle.get_text()
        assert "uncovered" in axes[3].get_title()
        checked.append(True)
        close(fig)

    monkeypatch.setattr(m.prior.plt, "close", inspect_then_close)
    data, crops = m.render(selected, {v: image for v in m.VIEWS}, {v: masks for v in m.VIEWS})
    decoded = m.prior.cv2.imdecode(np.frombuffer(data, np.uint8), m.prior.cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.ndim == 3 and set(crops) == set(m.VIEWS)
    assert checked == [True]
    assert before == (image.tobytes(), masks.tobytes())
