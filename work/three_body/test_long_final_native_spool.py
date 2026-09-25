"""Tiny synthetic images/masks only; never actual final samples or videos."""

import copy
import json

import long_final_native_spool as m
import numpy as np
import pytest
from prepare_sequence import PackedMasks


def sample(n, view, mode="one"):
    masks = np.zeros((1 if mode == "one" else 0, 4, 5), bool)
    if len(masks):
        masks[0, 1:3, 1:4] = True
    pack = PackedMasks((4, 5))
    count = len(masks)
    pack.append(
        n,
        masks,
        np.full(count, 0.8, np.float32),
        np.zeros((count, 4), np.float32),
        np.zeros(count, np.int64),
    )
    packed = pack.arrays()
    if mode == "missing":
        packed["view_available"][:] = False
        masks = None
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    detections = np.arange(6, dtype=np.float64).reshape(1, 3, 2)
    detections.flat[0] = -0.0
    detections.view(np.uint64).flat[-1] = 0x7FF8000000000012
    row = dict(
        frame_idx=n,
        view=view,
        video_offset=n,
        chunk_index=(n - m.FIRST) // m.CHUNK,
        chunk_frame_offset=(n - m.FIRST) % m.CHUNK,
        raw_bgr_sha256=m.native.array_sha(image),
        mask_available=masks is not None,
        mask_count=None if masks is None else len(masks),
        mask_decoded_sha256=None if masks is None else m.native.array_sha(masks),
        packed_frame_mask_sha256=bytes(packed["frame_mask_sha256"][0]).decode(),
        detection_shape=list(detections.shape),
        detection_dtype=str(detections.dtype),
        detection_sha256=m.native.array_sha(detections),
        selected_packed_arrays=m.signatures(packed),
    )
    return row, dict(image_bgr=image, detections=detections, masks=masks, packed=packed)


def create(root, mode="one"):
    frames = (m.FIRST, m.FIRST + 1, m.FIRST + 2)
    writer = m._SpoolWriter(root, {"source": "synthetic"}, frames, frames[1:], m.VIEWS)
    before = []
    for v in m.VIEWS:
        for n in frames:
            row, data = sample(n, v, mode)
            saved = (
                copy.deepcopy(row),
                m.signatures(data["packed"]),
                m.signature(data["image_bgr"]),
                m.signature(data["detections"]),
            )
            writer(row, data)
            assert saved == (
                row,
                m.signatures(data["packed"]),
                m.signature(data["image_bgr"]),
                m.signature(data["detections"]),
            )
            before.append((row, data))
    writer.finish()
    return writer, before


def reader(root):
    return m.SpoolReader(
        root,
        expected_manifest_sha256=m.sha(root / "spool.manifest.json"),
        expected_bindings={"source": "synthetic"},
    )


def test_fixed_sampling_and_179_359_1077_scope():
    assert len(m.boundaries()) == 179 and len(m.retained_frames()) == 359
    assert len(m.native.fixed_sample_frames()) == 543
    assert set(m.retained_frames()) <= set(m.native.fixed_sample_frames())
    assert m.retained_frames()[-1] == 348599


@pytest.mark.parametrize("mode", ["one", "empty", "missing"])
def test_lossless_roundtrip_none_empty_all13_signedzero_nan_bits_and_nonmutation(tmp_path, mode):
    root = tmp_path / "spool"
    writer, inputs = create(root, mode)
    r = reader(root)
    assert len(r.rows) == 9 and len(r.index) == 6
    assert len(list((root / "samples").glob("*.npz"))) == 6
    for row, data in inputs:
        if row["frame_idx"] == m.FIRST:
            with pytest.raises(KeyError):
                r.read_view(row["frame_idx"], row["view"])
            continue
        actual = r.read_view(row["frame_idx"], row["view"])
        assert m.signatures(actual["packed"]) == m.signatures(data["packed"])
        for k in ("image_bgr", "detections"):
            assert m.signature(actual[k]) == m.signature(data[k])
            assert not np.shares_memory(actual[k], data[k])
        assert (actual["masks"] is None) == (mode == "missing")
        if mode == "empty":
            assert actual["masks"].shape == (0, 4, 5)
        actual["image_bgr"][:] = 0
        assert r.read_view(row["frame_idx"], row["view"])["image_bgr"].sum() > 0
    assert writer.closed
    assert r.manifest["final_audit_complete"] is False
    with pytest.raises(ValueError):
        r.require_final_scope()


@pytest.mark.parametrize("fault", ["order", "native_offset", "mask", "BGR", "SL", "packed_dtype"])
def test_bad_callback_retains_failure_without_success(tmp_path, fault):
    root = tmp_path / "spool"
    w = m._SpoolWriter(root, {}, (m.FIRST,), (m.FIRST,), m.VIEWS)
    row, data = sample(m.FIRST, m.VIEWS[0])
    if fault == "order":
        row["view"] = m.VIEWS[1]
    elif fault == "native_offset":
        row["video_offset"] += 1
    elif fault == "mask":
        data["masks"][0, 0, 0] = True
    elif fault == "BGR":
        data["image_bgr"][0, 0, 0] += 1
    elif fault == "SL":
        data["detections"].flat[0] = 1
    else:
        data["packed"]["frame_idx"] = data["packed"]["frame_idx"].astype(np.int32)
    with pytest.raises(ValueError):
        w(row, data)
    assert (root / "failure.json").is_file() and not (root / "spool.manifest.json").exists()
    with pytest.raises(ValueError):
        w.finish()
    with pytest.raises(ValueError):
        m._SpoolWriter(root, {}, (m.FIRST,), (m.FIRST,), m.VIEWS)


def test_incomplete_callback_failure_preserved(tmp_path):
    w = m._SpoolWriter(tmp_path / "spool", {}, (m.FIRST,), (m.FIRST,), m.VIEWS)
    w(*sample(m.FIRST, m.VIEWS[0]))
    with pytest.raises(ValueError):
        w.finish()
    assert len(list((w.root / "samples").glob("*.npz"))) == 1
    assert not (w.root / "spool.manifest.json").exists()


@pytest.mark.parametrize("fault", ["file_bytes", "extra", "record_bytes", "bindings"])
def test_closed_spool_tamper_rejected(tmp_path, fault):
    root = tmp_path / "spool"
    create(root)
    if fault == "file_bytes":
        next((root / "samples").glob("*.npz")).write_bytes(b"bad")
    elif fault == "extra":
        (root / "extra").write_text("bad")
    elif fault == "record_bytes":
        rows = (root / "records.jsonl").read_text().splitlines()
        value = json.loads(rows[0])
        value["view"] = "wrong"
        rows[0] = json.dumps(value)
        (root / "records.jsonl").write_text("\n".join(rows) + "\n")
    else:
        with pytest.raises(ValueError):
            m.SpoolReader(
                root,
                expected_manifest_sha256=m.sha(root / "spool.manifest.json"),
                expected_bindings={"source": "other"},
            )
        return
    with pytest.raises(ValueError):
        reader(root)


def test_success_spool_cannot_be_changed_by_late_callbacks_finish_or_abort(tmp_path):
    root = tmp_path / "spool"
    writer, _ = create(root)
    before = {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        writer(*sample(m.FIRST, m.VIEWS[0]))
    with pytest.raises(ValueError):
        writer.finish()
    writer.abort(RuntimeError("later unrelated audit failure"))
    assert before == {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", ["registration.json", "records.jsonl"])
def test_precommit_metadata_changes_are_not_resealed(tmp_path, name):
    root = tmp_path / "spool"
    writer = m._SpoolWriter(root, {}, (m.FIRST,), (m.FIRST,), m.VIEWS)
    for v in m.VIEWS:
        writer(*sample(m.FIRST, v))
    with (root / name).open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError):
        writer.finish()
    assert (root / "failure.json").exists() and not (root / "spool.manifest.json").exists()
