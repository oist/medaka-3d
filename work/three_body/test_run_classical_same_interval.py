"""Isolated window tests; real classical smoke uses only synthesized pixels/rig."""

import importlib.util
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import run_classical_same_interval as c

REPOSITORY = c.REPO


@pytest.fixture(autouse=True)
def isolate_paths(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    work = root / "work/three_body"
    work.mkdir(parents=True)
    monkeypatch.setattr(c, "REPO", root)
    monkeypatch.setattr(c, "WORK", work)
    monkeypatch.setattr(c, "INPUT_ROOT", root / "inputs")
    monkeypatch.setattr(c, "SEED_REFERENCE", root / "seed/registration.json")
    monkeypatch.setattr(c, "EXECUTION_READY", False)


def detections(view="top_cam", frames=(8, 9, 10, 10, 12, 14, 14, 15), n_frames=20):
    frames = np.asarray(frames, np.int32)
    points = np.arange(len(frames) * 6, dtype=np.float32).reshape(-1, 3, 2)
    score = np.arange(len(frames), dtype=np.float32) + 2
    return c.c01.Detections(c.EXPERIMENT, view, frames, points, score, n_frames)


def seed(rows=None):
    return dict(
        source_rows=np.tile(np.arange(3)[:, None], (1, 3)).tolist() if rows is None else rows,
        sha256="a" * 64,
    )


def test_clip_and_rebase_keep_native_rows_stable_and_readonly():
    data = detections()
    before = c.detection_signatures(data)
    window = c.Window(10, 15)
    native, flat = c.clip_native(data, window)
    local = c.local_detections(native, window)
    assert flat == (2, 7)
    assert native.frame_idx.tolist() == [10, 10, 12, 14, 14]
    assert local.frame_idx.tolist() == [0, 0, 2, 4, 4]
    assert native.n_frames == 15 and local.n_frames == 5
    assert np.array_equal(local.points, data.points[2:7])
    assert np.array_equal(local.score, data.score[2:7])
    assert np.array_equal(local.start[[0, 2, 4]], [0, 2, 3])
    for field in ("frame_idx", "points", "score"):
        assert not getattr(native, field).flags.writeable
        assert not getattr(local, field).flags.writeable
    assert c.detection_signatures(data) == before


@pytest.mark.parametrize("kind", ["unsorted", "wrong_dtype", "foreign_view", "foreign_nodes"])
def test_bad_sl_rejected_before_stages(kind):
    data = detections()
    if kind == "unsorted":
        data = c.replace(data, frame_idx=data.frame_idx[::-1])
    elif kind == "wrong_dtype":
        data = c.replace(data, points=data.points.astype(np.float64))
    elif kind == "foreign_view":
        data = c.replace(data, view="side3_cam")
    else:
        data = c.replace(data, nodes=("tail", "pectoral", "mouth"))
    with pytest.raises(ValueError):
        c.clip_native(data, c.Window(10, 15))


def test_native_outside_scope_is_rejected_before_tracker(monkeypatch):
    monkeypatch.setattr(c.c24, "run_tracker", lambda *a, **k: pytest.fail("tracker reached"))
    with pytest.raises(ValueError, match="unclipped"):
        c.stage_a({v: detections(v) for v in c.VIEWS}, c.Window(10, 15))


def test_stage_a_is_one_native_continuous_call_per_camera_and_source_rows_survive(monkeypatch):
    window = c.Window(10, 15)
    native = {v: c.clip_native(detections(v), window)[0] for v in c.VIEWS}
    calls = []

    def tracker(data, nodes, **kw):
        calls.append((data.view, kw["frames"]))
        assert nodes == ("mouth", "pectoral")
        assert kw == dict(window_size=5, scoring="oks", min_match_points=0, frames=range(10, 15))
        return np.array([9, 3, 5, 8, 2], np.int64), {"test": "synthetic"}

    monkeypatch.setattr(c.c24, "run_tracker", tracker)
    local, labels, detail = c.stage_a(native, window)
    assert calls == [(v, range(10, 15)) for v in c.VIEWS]
    assert all(detail[v]["tracker_calls"] == 1 for v in c.VIEWS)
    # Actual compact_layout sorts by labels but preserves original detector offsets.
    _, label_at, row_at, cardinality = c.c25.compact_layout(
        local[c.VIEWS[0]], labels[c.VIEWS[0]], c.NODES
    )
    assert cardinality.tolist() == [2, 0, 1, 0, 2]
    assert label_at[0, :2].tolist() == [3, 9]
    assert row_at[0, :2].tolist() == [1, 0]
    assert label_at[4, :2].tolist() == [2, 8]
    assert row_at[4, :2].tolist() == [1, 0]


def test_degraded_only_uses_inside_scores_and_keeps_original_sequential_dilation():
    window = c.Window(10, 18)
    collection = {}
    for view in c.VIEWS:
        d = detections(view, frames=range(9, 19))
        d.score[:] = 2
        d.score[[0, -1]] = 0.1  # immediately outside both boundaries
        collection[view] = c.local_detections(c.clip_native(d, window)[0], window)
    assert not c.degraded_window(collection, window.size).any()
    d = collection[c.VIEWS[0]]
    scores = d.score.copy()
    scores[0] = 0.1
    collection[c.VIEWS[0]] = c.replace(d, score=scores)
    actual = c.degraded_window(collection, window.size)
    assert actual.tolist() == [True, True, True, True, False, False, False, False]


def cache_bytes(data):
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        **{
            key: getattr(data, key)
            for key in ("experiment", "view", "frame_idx", "points", "score", "n_frames", "nodes")
        },
    )
    return stream.getvalue()


def test_zip_reader_only_deserializes_points_score_inside_window(monkeypatch):
    data, window = detections(), c.Window(10, 15)
    data.points[[0, 1, -1]] = np.inf
    data.score[[0, 1, -1]] = -9999
    actual_reader = c.read_npy_rows
    calls = []

    def guarded(*args):
        calls.append((args[1], args[2], args[3]))
        assert args[2:4] == (2, 7)
        return actual_reader(*args)

    monkeypatch.setattr(c, "read_npy_rows", guarded)
    result, record = c.decode_window(cache_bytes(data), data.view, window)
    assert calls == [("points", 2, 7), ("score", 2, 7)]
    assert np.array_equal(result.points, data.points[2:7])
    assert np.array_equal(result.score, data.score[2:7])
    assert record["outside_points_scores_deserialized"] is False
    assert record["full_native_index_read"] is True


@pytest.mark.parametrize("corrupt", ["score_dtype", "points_shape", "order", "view"])
def test_bad_archive_rejected(corrupt):
    data = detections()
    if corrupt == "score_dtype":
        data = c.replace(data, score=data.score.astype(np.float64))
    elif corrupt == "points_shape":
        data = c.replace(data, points=data.points[:, :2])
    elif corrupt == "order":
        data = c.replace(data, frame_idx=data.frame_idx[::-1])
    else:
        data = c.replace(data, view="side1_cam")
    with pytest.raises(ValueError):
        c.decode_window(cache_bytes(data), "top_cam", c.Window(10, 15))


def test_first_correspondence_only_reports_fixed_naming_and_never_changes_source():
    reference = seed([[0, 2, 2], [2, 0, 0], [1, 1, 1]])
    source = np.asarray(reference["source_rows"], np.int8)[[1, 2, 0]]
    before = source.copy()
    result = c.first_correspondence(source, reference)
    assert result["seed_slot_to_output_slot_permutations"] == [[2, 0, 1]]
    assert result["exact_initial_correspondence_match"]
    assert not result["shared_seed_supplied"]
    assert not result["physical_ID_truth"]
    assert np.array_equal(source, before)
    source[0, 0] = -1
    assert not c.first_correspondence(source, reference)["exact_initial_correspondence_match"]


def test_replay_stage_wiring_preserves_storage_boundaries_and_local_frames(monkeypatch):
    window = c.Window(100, 104)
    source = np.tile(np.arange(3, dtype=np.int8)[None, :, None], (4, 1, 3))
    xyz = np.arange(36, dtype=float).reshape(4, 3, 3) + 0.123456789
    native = {
        v: detections(v, frames=np.repeat(np.arange(100, 104), 3), n_frames=104) for v in c.VIEWS
    }
    local = {v: c.local_detections(d, window) for v, d in native.items()}
    context = dict(
        rays={},
        label_at={},
        row_at={},
        k={},
        frames=np.arange(4),
        base=5,
        degraded=np.zeros(4, bool),
    )
    calls = []

    def solve(*args, diagnostics):
        calls.append("solve")
        assert np.array_equal(args[4], np.arange(4)) and args[5:7] == (0.767, 4.0)
        assert diagnostics == {}
        diagnostics.update(cache_replacements=1, numerically_different_cache_replacements=1)
        return np.zeros(4, np.int32), 2.5, np.zeros(4, int), [], np.ones(4) / 3, {}

    def geometry(*args):
        calls.append("geometry")
        return xyz.copy(), np.full((4, 3), 0.123456789), source.copy(), np.full((4, 3), 3, np.int8)

    def repair(arm, detections, rig):
        calls.append("repair")
        assert arm["xyz"].dtype == arm["residual_mm"].dtype == arm["margin_mm"].dtype == np.float32
        assert all(d.frame_idx.min() == 0 and d.frame_idx.max() == 3 for d in detections.values())
        return {k: arm[k].copy() for k in ("xyz", "residual_mm", "source", "slot_rays")}, {}

    def stitch(xyz, ok, *args, **kwargs):
        calls.append("stitch")
        assert xyz.dtype == np.float64 and np.array_equal(ok, np.arange(4))
        return np.tile(np.array([2, 0, 1], np.int8), (4, 1)), [], []

    def windows(xyz, ok):
        calls.append("windows")
        assert xyz.dtype == np.float64
        return []

    for module, name, value in (
        (c.checked_solver, "solve", solve),
        (c.c25, "chosen_geometry", geometry),
        (c.c33, "repair", repair),
        (c.c30, "stitch", stitch),
        (c.c34, "find_windows", windows),
    ):
        monkeypatch.setattr(module, name, value)
    payload, diagnostics, stages = c.replay_arm(context, local, None, window, 0.767, seed())
    assert calls == ["solve", "geometry", "repair", "stitch", "windows"]
    assert diagnostics["transition_cache_diagnostics"] == dict(
        cache_replacements=1, numerically_different_cache_replacements=1
    )
    assert payload["frame_idx"].tolist() == [100, 101, 102, 103]
    assert np.array_equal(payload["xyz"], xyz.astype(np.float32)[:, [2, 0, 1]])
    assert np.array_equal(payload["source"], source[:, [2, 0, 1]])
    assert stages["c28"]["margin_mm"].tolist() == [1, 1, 1, 1]
    assert diagnostics["initial_C25"]["seed_slot_to_output_slot_permutations"] == [[0, 1, 2]]
    assert diagnostics["initial_C34"]["seed_slot_to_output_slot_permutations"] == [[1, 2, 0]]
    for view in c.VIEWS:
        # Actual frozen propagation reads local starts, then native metadata is restored.
        assert payload[f"track__{view}"].reshape(4, 3).tolist() == [[1, 2, 0]] * 4


def synthetic_rig(tmp_path):
    path = REPOSITORY / "tests/conftest.py"
    spec = importlib.util.spec_from_file_location("classical_test_calib_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rig.__wrapped__(tmp_path), module.project


def test_real_classical_pipeline_short_synthetic_rig_no_payloads(tmp_path):
    rig, project = synthetic_rig(tmp_path)
    window = c.Window(70, 78)
    centres = np.array([[40.0, 60.0, 30.0], [120.0, 90.0, 100.0], [70.0, 150.0, 160.0]])
    world = centres[None] + np.arange(8)[:, None, None] * 0.04
    pose = np.stack([world + np.array([3, 0, 0]), world, world - np.array([5, 0, 0])], axis=2)
    native = {}
    for view in c.VIEWS:
        # Existing refractive camera, synthesized pinhole-index=1 calibration.
        points = project(rig.camera(view), pose).astype(np.float32).reshape(-1, 3, 2)
        assert np.isfinite(points).all()
        native[view] = c.c01.Detections(
            c.EXPERIMENT,
            view,
            np.repeat(window.native_frames, 3).astype(np.int32),
            points,
            np.full(24, 2.0, np.float32),
            window.stop,
        )
    before = {v: c.detection_signatures(d) for v, d in native.items()}
    local, labels, details = c.stage_a(native, window)
    context = c.prepare_correspondence(local, labels, rig, window)
    payload, diagnostics, stages = c.replay_arm(context, local, rig, window, 0.767, seed())
    assert all(details[v]["tracker_calls"] == 1 for v in c.VIEWS)
    assert payload["xyz"].shape == (8, 3, 3)
    assert payload["xyz"].dtype == np.float32 and payload["source"].dtype == np.int8
    assert np.array_equal(payload["frame_idx"], window.native_frames)
    assert np.isfinite(payload["xyz"]).all()
    assert (payload["slot_track"] >= 0).all()
    assert diagnostics["solve_calls"] == 1
    assert diagnostics["transition_cache_diagnostics"] == dict(
        cache_replacements=0, numerically_different_cache_replacements=0
    )
    assert set(stages) == {"c28", "c33", "c30"}
    for frame in range(8):
        rows = payload["source"][frame]
        assert np.array_equal(rows[:, 0], rows[:, 1]) and np.array_equal(rows[:, 0], rows[:, 2])
        assert sorted(rows[:, 0].tolist()) == [0, 1, 2]
        assert np.allclose(payload["xyz"][frame], world[frame, rows[:, 0]], atol=1e-3)
    assert before == {v: c.detection_signatures(d) for v, d in native.items()}
    c.no_cuda()


@pytest.mark.parametrize("enabled,outside", [(False, False), (True, True)])
def test_run_guard_never_reaches_real_sources_or_inputs(monkeypatch, tmp_path, enabled, outside):
    monkeypatch.setattr(c, "EXECUTION_READY", enabled)
    monkeypatch.setattr(c, "source_binding", lambda: pytest.fail("source lookup reached"))
    monkeypatch.setattr(c, "load_inputs", lambda *a: pytest.fail("real loader reached"))
    output = tmp_path / "outside" if outside else c.WORK / "output"
    with pytest.raises(ValueError):
        c.run(SimpleNamespace(output=output, execute=True))
    assert not output.exists()


def test_full_driver_all7_once_and_failed_output_preserved(monkeypatch):
    monkeypatch.setattr(c, "EXECUTION_READY", True)
    monkeypatch.setattr(c, "START", 10)
    monkeypatch.setattr(c, "STOP", 14)
    monkeypatch.setattr(c, "SCIENCE_START", 12)
    monkeypatch.setattr(c, "source_binding", lambda: {"synthetic": True, "sources": {}})
    native = {v: detections(v, frames=[10, 11, 12, 13], n_frames=14) for v in c.VIEWS}
    monkeypatch.setattr(
        c, "load_inputs", lambda w: (native, None, dict(records=[], seed_reference=seed()))
    )
    monkeypatch.setattr(c, "stage_a", lambda n, w: (n, {v: np.arange(4) for v in c.VIEWS}, {}))
    monkeypatch.setattr(c, "prepare_correspondence", lambda *a: {})
    called = []

    def arm(*args):
        called.append(args[4])
        return dict(frame_idx=np.arange(10, 14)), dict(objective=1.0), {}

    monkeypatch.setattr(c, "replay_arm", arm)
    output = c.WORK / "success"
    c.run(SimpleNamespace(output=output, execute=True))
    summary = json.loads((output / "summary.json").read_text())
    assert called == list(c.LAMBDAS)
    assert summary["c25_whole_window_calls"] == 7 and summary["shared_seed"] is False
    assert len(summary["results"]) == 7
    assert json.loads((output / "replay.provenance.json").read_text())["status"] == "ok"

    def fail(*a):
        raise RuntimeError("synthetic stage failure")

    monkeypatch.setattr(c, "replay_arm", fail)
    failed = c.WORK / "failed"
    with pytest.raises(RuntimeError, match="synthetic stage failure"):
        c.run(SimpleNamespace(output=failed, execute=True))
    assert json.loads((failed / "replay.provenance.json").read_text())["status"] == "failed"
    assert (failed / "failure.json").is_file()
    assert not (failed / "summary.json").exists()
