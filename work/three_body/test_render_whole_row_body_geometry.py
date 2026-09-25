"""Synthetic projection/drawing tests; real geometry and photograph loaders blocked."""

from __future__ import annotations

import copy
import io

import numpy as np
import pytest
import render_whole_row_body_geometry as m


class Camera:
    def project(self, xyz):
        assert np.isfinite(xyz).all()
        return xyz[..., :2] + 30


class Rig:
    def camera(self, view):
        assert view in m.VIEWS
        return Camera()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(m, "WORK", work)
    monkeypatch.setattr(m, "ROOT", work / "geometry")
    monkeypatch.setattr(m.proof, "REPO", tmp_path)
    monkeypatch.setattr(m, "read_context", lambda *a: pytest.fail("actual metadata forbidden"))
    monkeypatch.setattr(m, "load_rig", lambda *a: pytest.fail("actual rig forbidden"))
    monkeypatch.setattr(
        m.drawing, "photograph_and_points", lambda *a: pytest.fail("actual photo forbidden")
    )
    monkeypatch.setattr(m.prior.cv2, "VideoCapture", lambda *a: pytest.fail("video forbidden"))
    return work


def test_partial_projection_preserves_nan_without_sending_nan_to_camera():
    pose = np.array([[np.nan] * 3, [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    before = pose.copy()
    result = m.project_partial(Rig(), pose, np.array([False, True, True]))
    assert all(
        np.isnan(p[0, 0]).all() and np.array_equal(p[0, 1:], [[32, 33], [34, 35]])
        for p in result.values()
    )
    assert np.array_equal(pose, before, equal_nan=True)
    with pytest.raises(ValueError, match="partial pose"):
        m.project_partial(Rig(), pose, np.ones(3, bool))


def scene_fixture():
    pose = np.arange(27, dtype=float).reshape(3, 3, 3) + 30
    scene = dict(
        rank=0,
        pose_xyz=pose.tolist(),
        source_rows=[[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        pool_indices=[2, 5, 8],
        projections={v: pose[..., :2].tolist() for v in m.VIEWS},
        total_used_rays=27,
        sum_squared_ray_rms_mm=1.2,
    )
    case = dict(arm=m.drawing.ARMS[0], frame_idx=82727, scenes=[scene])
    summary = dict(scenes=dict(scenes_returned=1))
    return case, summary


def test_complete_scenes_keep_saved_projection_and_reject_duplicate_rows():
    case, summary = scene_fixture()
    before = copy.deepcopy(case)
    rows = m.scene_rows(case, summary)
    assert rows[0]["pool_indices"] == [2, 5, 8]
    assert rows[0]["projection_origin"] == "saved geometry scene projections"
    assert case == before
    case["scenes"][0]["source_rows"][1][0] = 0
    with pytest.raises(ValueError, match="reuses"):
        m.scene_rows(case, summary)


def test_anonymous_open_partial_drawing_has_no_imputed_mouth():
    pose = np.array([[np.nan] * 3, [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    row = dict(
        kind="individual_pool_body",
        index=7,
        pool_indices=[7],
        source_rows=np.array([[0, -1, 2]]),
        triangulated_nodes=np.array([[False, True, True]]),
        projections=m.project_partial(Rig(), pose, np.array([False, True, True])),
    )
    fig, ax = m.prior.plt.subplots()
    try:
        m.draw_panel(ax, np.zeros((100, 100, 3), np.uint8), np.empty((0, 3, 2)), row, m.VIEWS[0])
        assert {t.get_text() for t in ax.texts} == {"P7C", "P7T"}
        markers = [line for line in ax.lines if line.get_marker() in m.drawing.MARKERS]
        assert len(markers) == 2 and all(line.get_markerfacecolor() == "none" for line in markers)
    finally:
        m.prior.plt.close(fig)
    images = {v: np.zeros((100, 100, 3), np.uint8) for v in m.VIEWS}
    points = {v: np.empty((0, 3, 2)) for v in m.VIEWS}
    crops = {v: (np.array([0, 0]), np.array([99, 99])) for v in m.VIEWS}
    png = m.render_page(
        dict(arm=m.drawing.ARMS[0], frame_idx=82720), [row], images, points, crops, 1, 1
    )
    assert m.prior.cv2.imdecode(np.frombuffer(png, np.uint8), m.prior.cv2.IMREAD_COLOR) is not None


def test_synthetic_scene_and_pool_run(isolated, monkeypatch):
    scene, summary = scene_fixture()
    partial = dict(arm=m.drawing.ARMS[0], frame_idx=82720, scenes=[])
    out = m.proof.Output(m.ROOT)
    arrays = {}

    def store(a):
        key = f"a{len(arrays):05d}"
        arrays[key] = np.asarray(a)
        return dict(__array__=key, signature=m.proof.signature(arrays[key]))

    pose = np.array([[np.nan] * 3, [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    body = dict(
        pool_index=0,
        same_row_within_each_view=True,
        pose_xyz=store(pose),
        triangulated_nodes=store(np.array([False, True, True])),
        view_rows=store(np.array([0, -1, 2], np.int32)),
        ray_rms_mm=store(np.array([np.nan, 0.1, 0.2])),
    )
    stem = f"cases/{partial['arm']}_{partial['frame_idx']}"
    archive = out.write(stem + ".npz", arrays)
    out.write(
        stem + ".json",
        dict(
            archive=archive,
            tree=dict(
                arm=partial["arm"],
                frame_idx=partial["frame_idx"],
                pool=[body],
                no_temporal_identity_assignment=True,
                direct_support_not_recomputed=True,
            ),
        ),
    )
    summary.update(arm=scene["arm"], frame_idx=scene["frame_idx"])
    partial_summary = dict(
        arm=partial["arm"],
        frame_idx=partial["frame_idx"],
        archive=archive,
        scenes=dict(row_disjoint_complete_triples=0),
        pool=dict(geometry_candidates=1),
    )
    context = dict(
        payload=dict(cases=[scene, partial]),
        source_code={},
        geometry_seal=dict(outputs=out.records),
        geometry_summary=dict(cases=[summary, partial_summary]),
    )
    monkeypatch.setattr(m, "read_context", lambda *a: context)
    monkeypatch.setattr(m, "load_rig", lambda *a: (Rig(), {"fake": "synthetic"}))
    monkeypatch.setattr(
        m.drawing,
        "photograph_and_points",
        lambda context, frame, view, evidence: (
            np.zeros((100, 100, 3), np.uint8),
            np.empty((0, 3, 2)),
            dict(fake=True, frame=frame, view=view),
        ),
    )
    actual_load = np.load

    def safe_load(payload, **kwargs):
        assert isinstance(payload, io.BytesIO)
        return actual_load(payload, **kwargs)

    monkeypatch.setattr(np, "load", safe_load)
    target = isolated / "render"
    m.run(target)
    report = m.proof.read_json(target / "summary.json")
    assert report["contact_sheets"] == 2 and report["pool_archives_decoded"] == 1
    assert not report["missing_nodes_completed"] and not report["physical_ID_truth"]
    assert report["cases"][1]["geometry_rows"][0]["pose_xyz"][0][0] == [None] * 3
    seal = m.proof.read_json(target / "render.provenance.json")
    assert seal["status"] == "ok" and len(seal["outputs"]) == 4
    for record in seal["outputs"].values():
        assert m.proof.sha(record["path"]) == record["sha256"]
