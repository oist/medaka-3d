"""Synthetic tests only; every callable input root is isolated before run tests."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import render_depth_identity_transition_review as m


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    work = tmp_path / "allowed"
    work.mkdir()
    monkeypatch.setattr(m, "WORK", work)
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m.proof, "REPO", tmp_path)
    monkeypatch.setattr(m, "COMPARISON_ROOT", work / "comparison")
    monkeypatch.setattr(m, "EXECUTION_READY", False)
    for name in ("load_headers", "source_code"):
        monkeypatch.setattr(m, name, lambda *a, **k: pytest.fail("real metadata entry forbidden"))
    monkeypatch.setattr(
        m.prior.cv2, "VideoCapture", lambda *a, **k: pytest.fail("real video forbidden")
    )
    load = np.load

    def guarded_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            assert Path(path).resolve().is_relative_to(tmp_path)
        return load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    return work


def full_arrays():
    n = m.STOP - m.FIRST
    return dict(
        frame_idx=np.arange(m.FIRST, m.STOP, dtype=np.int64),
        pose_xyz=np.zeros((n, 3, 3, 3), np.float64),
        observed=np.ones((n, 3, 3), bool),
        sources=np.zeros((n, 3, 3, 3), np.int32),
        identity_review=np.ones((n, 3), bool),
    )


def reference(arrays):
    return dict(arrays={k: m.proof.signature(v) for k, v in arrays.items()})


def test_all_nine_selection_and_readonly_preservation():
    a = full_arrays()
    for i, frame in enumerate(m.FRAMES):
        a["pose_xyz"][frame - m.FIRST] = i + np.arange(27).reshape(3, 3, 3) / 100
        a["sources"][frame - m.FIRST] = i
        a["observed"][frame - m.FIRST, i % 3, i % 3] = False
    before = reference(a)
    rows = m.candidate_rows(a, before)
    assert rows["frame_idx"].tolist() == list(m.FRAMES)
    for k in m.KEYS:
        assert np.array_equal(rows[k], a[k][np.array(m.FRAMES) - m.FIRST])
        assert not rows[k].flags.writeable
    assert before == reference(a)


@pytest.mark.parametrize("field", m.KEYS)
def test_every_selected_field_signature_tamper_rejected(field):
    a = full_arrays()
    ref = reference(a)
    a[field].flat[0] = not a[field].flat[0] if a[field].dtype == bool else 7
    with pytest.raises(ValueError, match="signature"):
        m.candidate_rows(a, ref)


def test_own_signatures_cannot_hide_wrong_native_horizon_or_dtype():
    a = full_arrays()
    a["frame_idx"] += 1
    with pytest.raises(ValueError, match="horizon"):
        m.candidate_rows(a, reference(a))
    a["frame_idx"] -= 1
    a["sources"] = a["sources"].astype(np.int64)
    with pytest.raises(ValueError, match="schema"):
        m.candidate_rows(a, reference(a))


def index_fixture():
    rows = [
        dict(
            index=i,
            start=m.FIRST + i * m.CHUNK,
            stop=m.FIRST + (i + 1) * m.CHUNK,
            manifest_sha256=m.CHUNK_SHAS.get(i, "a" * 64),
        )
        for i in range(40)
    ]
    return dict(original_binding={}, prefix_chunks=rows)


def test_two_chunks_all_nine_and_exact_halfopen_boundary():
    chunks = {i: dict(chunk=c) for i, c in m.select_chunks(index_fixture()).items()}
    assert [m.frame_chunk(chunks, f)["chunk"]["index"] for f in m.FRAMES] == [19] * 8 + [20]
    assert m.frame_chunk(chunks, 82999)["chunk"]["index"] == 19
    assert m.frame_chunk(chunks, 83000)["chunk"]["index"] == 20
    assert [f - m.frame_chunk(chunks, f)["chunk"]["start"] for f in m.FRAMES] == [
        781,
        1350,
        1378,
        1379,
        1380,
        1387,
        1388,
        1410,
        265,
    ]
    with pytest.raises(ValueError, match="exactly one"):
        m.frame_chunk(chunks, 84660)


@pytest.mark.parametrize("change", ("gap", "SHA", "count"))
def test_chunk_order_and_pins_reject_changes(change):
    index = index_fixture()
    if change == "gap":
        index["prefix_chunks"][20]["start"] += 1
    elif change == "SHA":
        index["prefix_chunks"][19]["manifest_sha256"] = "f" * 64
    else:
        index["prefix_chunks"].pop()
    with pytest.raises(ValueError):
        m.select_chunks(index)


def test_native_reader_receives_absolute_frame_not_chunk_local(monkeypatch):
    calls, closed = [], []

    def iterator(videos, manifest, start, stop):
        calls.append((videos, manifest, start, stop))
        try:
            yield start, {"synthetic": start}
        finally:
            closed.append(True)

    monkeypatch.setattr(m.prior.raw, "iter_native", iterator)
    chunk = dict(videos={"fake": "fake"}, manifest=dict(frames=list(range(83000, 84660))))
    assert m.native_images(chunk, 83265) == {"synthetic": 83265}
    assert calls[0][2:] == (83265, 83266)
    assert closed == [True]


@pytest.mark.parametrize("available", (True, False))
def test_mask_availability_and_local_offset_and_SHA(monkeypatch, available):
    masks = np.zeros((int(available), 6, 7), bool)
    digest = m.prior.prepare_sequence.array_sha256(masks)
    data = dict(
        frame_idx=np.array([82999, 83000]),
        frame_mask_sha256=np.array([digest, digest], dtype="S64"),
        view_available=np.array([available, available]),
    )
    chunk = dict(
        chunk=dict(start=82999),
        packed={"v": data},
        manifest=dict(masks={"v": dict(instances_per_frame=[len(masks)] * 2)}),
    )
    calls = []
    monkeypatch.setattr(
        m.prior.prepare_sequence, "decode_frame_masks", lambda d, pos: (calls.append(pos), masks)[1]
    )
    result, record = m.frame_masks(chunk, 83000, "v")
    assert calls == [1] and record["mask_available"] == available
    assert (result is masks) if available else result is None
    data["frame_mask_sha256"][1] = b"0" * 64
    with pytest.raises(ValueError, match="pixel SHA"):
        m.frame_masks(chunk, 83000, "v")


def test_native_png_preserves_channel_order_and_pixels():
    image = np.arange(23 * 31 * 3, dtype=np.uint8).reshape(23, 31, 3)
    encoded = m.png_bytes(image)
    decoded = m.prior.cv2.imdecode(np.frombuffer(encoded, np.uint8), m.prior.cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, image)


def test_panel_contract_uses_final_horizon_same_crops_and_raw_support(monkeypatch):
    titles, panels, axes = [], [], np.empty((3, 3), object)

    class Axis:
        def __init__(self):
            self.bounds = []

        def set_xlim(self, *xy):
            self.bounds.append(xy)

        set_ylim = set_xlim

        def set_title(self, label, **kwargs):
            titles.append(label)

        def tick_params(self, **kwargs):
            pass

    class Figure:
        def suptitle(self, title, **kwargs):
            titles.append(title)

        def tight_layout(self, **kwargs):
            pass

        def savefig(self, stream, **kwargs):
            stream.write(b"SYNTHETIC_FIGURE")

    for index in np.ndindex(axes.shape):
        axes[index] = Axis()
    monkeypatch.setattr(m.prior.plt, "subplots", lambda *a, **k: (Figure(), axes))
    monkeypatch.setattr(m.prior.plt, "close", lambda f: None)
    monkeypatch.setattr(m.visual, "draw_panel", lambda *a: panels.append(a))
    image = np.zeros((60, 70, 3), np.uint8)
    images = dict.fromkeys(m.VIEWS, image)
    masks = dict.fromkeys(m.VIEWS, None)
    points = {v: np.empty((0, 3, 2)) for v in m.VIEWS}
    projections = {a: {v: np.arange(18).reshape(3, 3, 2) for v in m.VIEWS} for a in m.ARMS}
    rows = {a: dict(observed=np.array([[[True, False, True]] * 3])) for a in m.ARMS}
    result, crops = m.render(82728, images, masks, points, projections, rows, 0, "D2")
    assert result == b"SYNTHETIC_FIGURE" and len(panels) == 9
    assert "same final 116200" in titles[0] and "78020" not in titles[0]
    assert "anonymous RAW slots" in titles[0] and set(crops) == set(m.VIEWS)
    for vi in range(3):
        assert axes[0, vi].bounds == axes[1, vi].bounds == axes[2, vi].bounds
        assert panels[vi * 3][4:6] == (None, None)
        assert panels[vi * 3 + 1][5] is rows[m.ARMS[0]]["observed"][0] or np.array_equal(
            panels[vi * 3 + 1][5], rows[m.ARMS[0]]["observed"][0]
        )
        assert np.array_equal(panels[vi * 3 + 2][5], rows[m.ARMS[1]]["observed"][0])


def test_disabled_and_outside_guard_never_loads_real_inputs(isolated, monkeypatch):
    with pytest.raises(ValueError, match="source review"):
        m.run(isolated / "output")
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    outside = isolated.parent / "outside"
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(outside)
    assert not outside.exists() and not (isolated / "output").exists()


def test_synthetic_nine_frame_run_two_chunk_memory_and_output_contract(isolated, monkeypatch):
    """Real five-field loading/output/PNG/selection; external readers explicitly stubbed."""
    m.COMPARISON_ROOT.mkdir()
    arrays = full_arrays()
    for i, f in enumerate(m.FRAMES):
        arrays["pose_xyz"][f - m.FIRST] = np.arange(27).reshape(3, 3, 3) + i
        arrays["sources"][f - m.FIRST] = i
        arrays["observed"][f - m.FIRST, i % 3, 0] = False
    ref = reference(arrays)
    # Object placeholders prove none of the other sixteen fields is decoded.
    extra = {f"unread_{i}": np.array([object()], object) for i in range(16)}
    ref["arrays"].update({k: dict(unread_synthetic=True) for k in extra})
    outputs = {}
    for arm in m.ARMS:
        path = m.COMPARISON_ROOT / f"candidates/{arm}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays, **extra)
        outputs[f"candidates/{arm}.npz"] = dict(
            path=str(path), sha256=m.proof.sha(path), size_bytes=path.stat().st_size
        )
    shared, slpaths = {}, {}
    for view in m.VIEWS:
        path = isolated / f"{view}.npz"
        path.write_bytes(b"SYNTHETIC_CACHE_STUB")
        shared[str(path)] = m.proof.sha(path)
        slpaths[view] = path
    rigpath = isolated / "calibration_manifest.json"
    models = []
    for i in range(3):
        p = isolated / f"cam{i}.json"
        p.write_text("{}")
        models.append(dict(path=p.name))
        shared[str(p)] = m.proof.sha(p)
    rigpath.write_bytes(m.proof.json_bytes(dict(model_files=models)))
    shared[str(rigpath)] = m.proof.sha(rigpath)
    chunks = {i: dict(chunk=c) for i, c in m.select_chunks(index_fixture()).items()}
    image = np.zeros((1200, 1200, 3), np.uint8)
    image[0, 0] = [7, 11, 19]
    bgrsha = m.prior.prepare_sequence.array_sha256(image)
    for i, c in chunks.items():
        c.update(
            videos=dict.fromkeys(m.VIEWS, isolated / f"fake{i}.mp4"),
            manifest=dict(
                experiment="SYNTHETIC",
                frames=list(range(c["chunk"]["start"], c["chunk"]["stop"])),
                raw={v: dict(frame_bgr_sha256=[bgrsha] * m.CHUNK) for v in m.VIEWS},
                masks={
                    v: dict(sha256="f" * 64, instances_per_frame=[0] * m.CHUNK) for v in m.VIEWS
                },
            ),
            mask_paths={v: isolated / f"mask{i}_{v}.npz" for v in m.VIEWS},
        )
    context = dict(
        seal=dict(outputs=outputs),
        references={a: copy.deepcopy(ref) for a in m.ARMS},
        chunks=chunks,
        shared=shared,
        binding=dict(seed=dict(input_root=str(isolated))),
        source_code={},
    )
    monkeypatch.setattr(m, "load_headers", lambda evidence: context)
    node_calls = []

    class Nodes:
        def __init__(self, root, experiment, frames):
            assert root == str(isolated) and experiment == "SYNTHETIC"
            assert np.array_equal(frames, np.arange(82121, 83266))
            node_calls.append(1)
            self.paths = slpaths

        def at(self, view, frame):
            return np.full((2, 3, 2), frame % 100, dtype=np.float32)

    monkeypatch.setattr(m.observations.observations, "CachedNodes", Nodes)
    rig = SimpleNamespace(
        views=m.VIEWS,
        world_units="mm",
        camera=lambda view: SimpleNamespace(project=lambda p: p[..., :2] + 5),
    )
    monkeypatch.setattr(m.observations.calib, "load_rig", lambda p: rig)
    captured = []

    def capture(path, digest, frames):
        assert Path(path).is_relative_to(isolated)
        if int(frames[0]) == 83000:
            assert "packed" not in chunks[19]  # Sequential release is observable.
        captured.append(str(path))
        Path(path).write_bytes(b"SYNTHETIC_PACKED")
        empty = np.empty((0, 1200, 1200), bool)
        d = dict(
            frame_idx=frames,
            view_available=np.ones(len(frames), bool),
            frame_mask_sha256=np.full(
                len(frames), m.prior.prepare_sequence.array_sha256(empty), dtype="S64"
            ),
        )
        return d, dict(
            path=str(path),
            file_stat=m.observations.file_stat(path),
            arrays={k: m.proof.signature(a) for k, a in d.items()},
        )

    monkeypatch.setattr(m.frozen, "capture_masks", capture)
    monkeypatch.setattr(m.previous, "video_stat_guard", lambda videos, c: None)
    mask_calls = []
    monkeypatch.setattr(
        m.prior.prepare_sequence,
        "decode_frame_masks",
        lambda d, pos: (
            mask_calls.append(int(d["frame_idx"][pos])),
            np.empty((0, 1200, 1200), bool),
        )[1],
    )
    raw_calls = []

    def raw(videos, manifest, start, stop):
        assert stop == start + 1 and start in m.FRAMES
        assert manifest["frames"][0] <= start <= manifest["frames"][-1]
        raw_calls.append(start)
        yield start, dict.fromkeys(m.VIEWS, image)

    monkeypatch.setattr(m.prior.raw, "iter_native", raw)
    render_calls = []

    def render(frame, images, masks, points, projections, rows, t, mode):
        assert rows[m.ARMS[0]]["frame_idx"][t] == frame
        for arm in m.ARMS:
            for view in m.VIEWS:
                assert np.array_equal(projections[arm][view], rows[arm]["pose_xyz"][t, ..., :2] + 5)
        render_calls.append((frame, mode))
        return b"SYNTHETIC_RENDER_STUB", dict(crop=[0, 1200])

    monkeypatch.setattr(m, "render", render)
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    output = isolated / "synthetic_result"
    m.run(output)
    seal = m.proof.read_json(output / "render.provenance.json")
    doc = m.proof.read_json(output / "source_and_frame_evidence.json")
    assert seal["status"] == "ok" and len(seal["outputs"]) == 49
    assert (
        raw_calls == list(m.FRAMES)
        and len(mask_calls) == 27
        and len(captured) == 6
        and node_calls == [1]
    )
    assert render_calls == [(f, mode) for f in m.FRAMES for mode in ("raw", "D2")]
    assert doc["terminal_horizon"] == 116200 and not doc["physical_ID_truth"]
    assert not doc["review_modified"] and doc["other_16_candidate_signatures_inherited"]
    assert len(list((output / "native").glob("*.png"))) == 27
    for t, r in enumerate(doc["records"]):
        assert r["frame"] == m.FRAMES[t]
        assert r["observations"][m.VIEWS[0]]["canonical_video_frame"] == m.FRAMES[t]
        assert r["observations"][m.VIEWS[0]]["source_chunk_index"] == (19 if t < 8 else 20)
    for arm in m.ARMS:
        with np.load(output / f"selected/{arm}.npz", allow_pickle=False) as selected:
            assert set(selected.files) == set(m.KEYS)
            assert selected["frame_idx"].tolist() == list(m.FRAMES)
            assert selected["identity_review"].all()
