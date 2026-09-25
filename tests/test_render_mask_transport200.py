"""Synthetic projection/display/codec boundaries; no real tracking payload."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "work/three_body"))
import render_mask_transport200 as m  # noqa: E402


def candidate():
    pose = np.broadcast_to(
        np.array(
            [
                [[200.0, 200.0, 10.0], [230.0, 200.0, 11.0], [260.0, 200.0, 12.0]],
                [[400.0, 400.0, 20.0], [430.0, 400.0, 21.0], [460.0, 400.0, 22.0]],
                [[600.0, 600.0, 30.0], [630.0, 600.0, 31.0], [660.0, 600.0, 32.0]],
            ]
        ),
        (200, 3, 3, 3),
    ).copy()
    observed = np.ones((200, 3, 3), bool)
    review = np.zeros((200, 3), bool)
    review[:, 2] = True
    result = dict(
        frame_idx=np.arange(m.FIRST, m.STOP, dtype=np.int64),
        pose_xyz=pose,
        observed=observed,
        sources=np.zeros((200, 3, 3, 3), np.int32),
        identity_review=review,
        mask_supported=np.zeros((200, 3), bool),
        body_supported=np.ones((200, 3), bool),
        missing=np.zeros((200, 3), np.int64),
        costs=np.zeros(200),
        path_costs=np.zeros(200),
        loss_parts=np.zeros((200, 5)),
        score_margin=np.full(200, np.inf),
        proposal_kind=np.full(200, "predict"),
        hypotheses=np.ones(200, np.int64),
        proposal_flow_nodes=np.zeros((200, 3, 3), bool),
        proposal_flow_parent=np.zeros(200, np.int64),
        node_missing=np.zeros((200, 3, 3), np.int64),
        direct_pectoral_age=np.zeros((200, 3), np.int64),
        xyz=pose[:, :, 1].copy(),
        body_xyz=pose[:, :, 1].copy(),
        accepted_xyz=np.where(review[:, :, None], np.nan, pose[:, :, 1]),
    )
    return result


def test_all21_native_fields_and_derived_nan_review_masks_are_validated():
    value = candidate()
    assert len(value) == 21
    before = {k: m.base.signature(a) for k, a in value.items()}
    m.validate_candidate(value)
    assert {k: m.base.signature(a) for k, a in value.items()} == before


@pytest.mark.parametrize("fault", ["field", "native", "pose", "dtype", "review"])
def test_candidate_mismatch_rejected(fault):
    value = candidate()
    if fault == "field":
        value.pop("proposal_flow_nodes")
    elif fault == "native":
        value["frame_idx"][-1] += 1
    elif fault == "pose":
        value["pose_xyz"][0, 0, 0, 0] = np.nan
    elif fault == "dtype":
        value["pose_xyz"] = value["pose_xyz"].astype(np.float32)
    else:
        value["accepted_xyz"][:, 2] = value["xyz"][:, 2]
    with pytest.raises(ValueError):
        m.validate_candidate(value)


class Camera:
    def __init__(self, view, calls):
        self.view, self.calls = view, calls

    def project(self, xyz):
        self.calls.append((self.view, xyz.copy()))
        result = np.stack(
            (xyz[..., 0] + 0.01 * xyz[..., 2] ** 2, xyz[..., 1] + 0.02 * xyz[..., 0] ** 2), axis=-1
        )
        return result


class Rig:
    def __init__(self):
        self.calls = []

    def camera(self, view):
        return Camera(view, self.calls)


def test_full_pose_camera_projection_preserves_source_axes_and_global_vs_view_flags():
    value, rig = candidate(), Rig()
    value["sources"].fill(-1)
    value["sources"][0, 1, 2, 0] = 5  # Body1, view2, mouth.
    value["sources"][0, 0, 1, 1] = 7
    value["observed"][0, 0, 1] = False
    before = {k: m.base.signature(a) for k, a in value.items()}
    result = m.projection_record(value, 0, rig)
    assert [c[0] for c in rig.calls] == list(m.VIEWS)
    assert all(c[1].shape == (3, 3, 3) for c in rig.calls)
    expected = np.stack(
        (
            value["pose_xyz"][0, ..., 0] + 0.01 * value["pose_xyz"][0, ..., 2] ** 2,
            value["pose_xyz"][0, ..., 1] + 0.02 * value["pose_xyz"][0, ..., 0] ** 2,
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(result["projected_xy"], np.stack([expected] * 3))
    assert result["marker_kind"][2, 1, 0] == 2
    assert result["marker_kind"][1, 0, 1] == 1
    assert np.count_nonzero(result["marker_kind"]) == 2
    assert result["marker_kind"][0, 1, 0] == 0  # Observed globally, no source in this view.
    assert {k: m.base.signature(a) for k, a in value.items()} == before


def test_nonfinite_outside_and_near_edge_coordinates_remain_explicit():
    value = candidate()
    uv = np.full((3, 3, 2), 100.0)
    uv[0, 0] = (np.nan, 10)
    uv[0, 1] = (1200, 0)
    uv[0, 2] = (1199.9, 0)
    uv[1, 0] = (-0.01, 5)
    uv[1, 1] = (1e300, 1e300)
    rig = SimpleNamespace(camera=lambda view: SimpleNamespace(project=lambda pose: uv.copy()))
    result = m.projection_record(value, 0, rig)
    assert not result["finite"][0, 0, 0]
    assert result["finite"][0, 0, 1] and not result["inside"][0, 0, 1]
    assert result["inside"][0, 0, 2] and not result["rounded_inside"][0, 0, 2]
    assert result["rounded_pixel_xy"][0, 0, 2, 0] == 1200
    assert result["projected_xy"][0, 1, 1, 0] == 1e300
    assert np.array_equal(result["rounded_pixel_xy"][0, 1, 1], [-1, -1])
    assert result["marker_kind"][0, 0, 0] == 2  # Status retained even when not drawable.


def test_raw_panel_is_exact_same_resize_and_marker_meanings_are_visible():
    raw = np.full((1200, 1200, 3), (11, 13, 17), np.uint8)
    before = raw.tobytes()
    plain = m.panel(raw, "raw fixture")
    np.testing.assert_array_equal(
        plain[m.IMAGE_Y : m.IMAGE_Y + 600],
        cv2.resize(raw, (600, 600), interpolation=cv2.INTER_AREA),
    )
    projection = dict(
        projected_xy=np.zeros((3, 3, 3, 2)),
        inside=np.zeros((3, 3, 3), bool),
        marker_kind=np.zeros((3, 3, 3), np.int8),
        identity_review=np.array([True, False, True]),
    )
    for b, kind in enumerate((2, 1, 0)):
        projection["projected_xy"][0, b, 0] = (200 + b * 400, 400)
        projection["inside"][0, b, 0] = True
        projection["marker_kind"][0, b, 0] = kind
    marked = m.panel(raw, "anonymous fixture", projection, 0)
    y = m.IMAGE_Y + 200
    np.testing.assert_array_equal(marked[y, 100], m.COLORS[0])
    np.testing.assert_array_equal(marked[y, 300], m.COLORS[1])
    np.testing.assert_array_equal(marked[y, 500], raw[0, 0])
    assert not np.array_equal(marked[y, 504], raw[0, 0])
    composite = m.composite(raw, m.VIEWS[0], m.FIRST, [projection] * 3)
    assert composite.shape == (m.PANEL_H, 2400, 3)
    changed_review = copy.deepcopy(projection)
    changed_review["identity_review"][:] = False
    unflagged = m.panel(raw, "anonymous fixture", changed_review, 0)
    np.testing.assert_array_equal(marked[:720], unflagged[:720])
    assert not np.array_equal(marked[720:], unflagged[720:])
    assert raw.tobytes() == before


def test_png_roundtrip_records_the_exact_annotated_raster(tmp_path):
    output = m.base.Output(tmp_path / "synthetic_png")
    image = np.arange(99 * 81 * 3, dtype=np.uint8).reshape(99, 81, 3)
    m.save_png(output, "view.png", image)
    decoded = cv2.imread(str(output.root / "view.png"))
    np.testing.assert_array_equal(decoded, image)
    assert output.records["view.png"]["sha256"] == m.base.sha(output.root / "view.png")


@pytest.mark.parametrize("fault", [None, "short", "fps", "shape"])
def test_optional_video_full_decode_contract_and_close(monkeypatch, fault):
    class Capture:
        def __init__(self):
            self.count, self.closed = 0, False

        def isOpened(self):
            return True

        def get(self, field):
            return 25 if fault == "fps" else 20

        def read(self):
            self.count += 1
            if self.count > (199 if fault == "short" else 200):
                return False, None
            return True, SimpleNamespace(
                shape=(m.PANEL_H - 1 if fault == "shape" else m.PANEL_H, 2400, 3)
            )

        def release(self):
            self.closed = True

    cap = Capture()
    monkeypatch.setattr(m.cv2, "VideoCapture", lambda path: cap)
    if fault:
        with pytest.raises(ValueError):
            m.verify_video("synthetic_only.mp4")
    else:
        assert m.verify_video("synthetic_only.mp4") == dict(
            frames=200, fps=20, width=2400, height=m.PANEL_H
        )
    assert cap.closed


def completed_metadata():
    plan = dict(
        schema=m.PRODUCER_STAGE,
        native_interval=[m.FIRST, m.STOP],
        arms=list(m.ARMS),
        input_binding=dict(
            preparation_sha256=m.inputs.PREPARATION_SHA,
            implementation_plan_sha256=m.inputs.PLAN_SHA,
        ),
    )
    summary = dict(
        schema=m.PRODUCER_STAGE,
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        original_control_passed=True,
        arms=[dict(arm=a, frames=200, initialize_count=1, restore_count=0) for a in m.ARMS],
    )
    control = dict(
        passed=True, **{k: dict(byte_exact=True) for k in ("candidate", "history", "state")}
    )
    return plan, summary, control


@pytest.mark.parametrize("fault", [None, "v1", "unfinished", "short", "control", "raw"])
def test_only_successful_same_v2_metadata_is_eligible(fault):
    plan, summary, control = completed_metadata()
    before = copy.deepcopy((plan, summary, control))
    if fault == "v1":
        plan["schema"] = "mask_transport200_three_arm_tracking_v1"
    elif fault == "unfinished":
        summary["status"] = "failed"
    elif fault == "short":
        summary["arms"][2]["frames"] = 199
    elif fault == "control":
        control["state"]["byte_exact"] = False
    elif fault == "raw":
        plan["input_binding"]["preparation_sha256"] = "0" * 64
    if fault:
        with pytest.raises(ValueError):
            m.validate_completed(plan, summary, control)
    else:
        m.validate_completed(plan, summary, control)
        assert (plan, summary, control) == before


@pytest.mark.parametrize(
    "fault", [None, "new_count", "old_origin", "incomplete", "source", "sidecar"]
)
def test_v2_reused_original_is_distinct_from_two_new_arms(fault):
    plan, summary, _ = completed_metadata()
    plan.update(
        new_arms_this_process=["lk", "ct_first"], reused_original_source_seal_sha256=m.FAILED_SHA
    )
    summary.update(
        reused_original_source_seal_sha256=m.FAILED_SHA,
        this_process_initialize_calls=2,
        this_process_restore_calls=0,
        new_advance_calls=400,
        reused_original_frames=200,
    )
    for index, arm in enumerate(summary["arms"]):
        arm["computed_in_this_process"] = bool(index)
    summary["arms"][0]["reused_from_seal_sha256"] = m.FAILED_SHA
    proof = dict(
        schema="mask_transport_original_reuse_qualification_v1",
        status="ok",
        passed=True,
        original_source_seal_sha256=m.FAILED_SHA,
        original_source_sha256=m.FAILED_SOURCE_SHA,
        frames=200,
        actual_fit4_byte_exact_frames=200,
        saved_frame_fit_and_records_validated=200,
        original_files_copied_byte_exact=408,
        initial_preparation_SL_masks_byte_exact_views=3,
    )
    for field in (
        "candidate_21_byte_exact",
        "history_19_plus_offsets_byte_exact",
        "core_39_byte_exact",
        "core_metadata_unchanged",
        "semantic_state_decode_passed",
        "sidecar_validation_passed",
        "original_context_and_sidecar_bytes_preserved",
    ):
        proof[field] = True
    if fault == "new_count":
        summary["this_process_initialize_calls"] = 3
    elif fault == "old_origin":
        summary["arms"][0]["computed_in_this_process"] = True
    elif fault == "incomplete":
        proof["saved_frame_fit_and_records_validated"] = 199
    elif fault == "source":
        proof["original_source_sha256"] = "0" * 64
    elif fault == "sidecar":
        proof["sidecar_validation_passed"] = False
    if fault:
        with pytest.raises(ValueError):
            m.validate_reuse(plan, summary, proof)
    else:
        m.validate_reuse(plan, summary, proof)
