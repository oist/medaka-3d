"""Short synthetic native-window streaming; no actual tracking outputs are read."""

import hashlib
import io
from types import SimpleNamespace

import evaluate_endpoint_mask_final_continuation as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    work = root / "work" / "three_body"
    work.mkdir(parents=True)
    for module in (m, m.base):
        monkeypatch.setattr(module, "REPO", root)
        monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(m, "ROOTS", tuple(work / name for name in m.ARMS))
    monkeypatch.setattr(m, "FIRST", 1000)
    monkeypatch.setattr(m, "START", 1001)
    monkeypatch.setattr(m, "STOP", 1004)
    for module, names in (
        (m, ("load_headers",)),
        (m.base, ("load_headers", "source_code", "load_rig")),
    ):
        for name in names:
            monkeypatch.setattr(
                module,
                name,
                lambda *args, **kwargs: pytest.fail("real metadata/rig entry forbidden"),
            )
    monkeypatch.setattr(
        m.prepare_sequence.cv2,
        "VideoCapture",
        lambda *args, **kwargs: pytest.fail("video forbidden"),
    )
    real_load = np.load

    def only_captured_bytes(path, *args, **kwargs):
        assert isinstance(path, io.BytesIO), "tests decode captured temporary bytes only"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", only_captured_bytes)
    return work


def candidate():
    count = m.STOP - m.FIRST
    return dict(
        frame_idx=np.arange(m.FIRST, m.STOP, dtype=np.int64),
        pose_xyz=np.full((count, 3, 3, 3), 2.0),
        identity_review=np.zeros((count, 3), bool),
    )


def packed():
    # Three source frames, with the middle camera frame unavailable. Every
    # nonempty mask is exactly one pixel; original instance order is 0,1.
    image = np.zeros((2, 1200, 1200), bool)
    image[0, 2, 2] = image[1, 2, 25] = True
    empty = np.empty((0, 1200, 1200), bool)
    digest = hashlib.sha256(image.tobytes()).hexdigest().encode()
    data = dict(
        image_shape_hw=np.array([1200, 1200], np.int32),
        frame_idx=np.arange(m.START, m.STOP, dtype=np.int64),
        frame_offsets=np.array([0, 2, 2, 4], np.int64),
        mask_boxes=np.array([[2, 2, 3, 3], [25, 2, 26, 3]] * 2, np.int32),
        mask_byte_offsets=np.arange(5, dtype=np.int64),
        mask_bits=np.ones(4, np.uint8),
        scores=np.ones(4, np.float32),
        pred_boxes=np.zeros((4, 4), np.float32),
        classes=np.zeros(4, np.int64),
        instance_id=np.array([0, 1, 0, 1], np.int64),
        source_track_id=np.full(4, -1, np.int64),
        view_available=np.array([True, False, True]),
        frame_mask_sha256=np.array(
            [digest, hashlib.sha256(empty.tobytes()).hexdigest().encode(), digest], dtype="S64"
        ),
    )
    return data


def test_native_offset_keeps_the_requested_interval_and_original_row_order():
    source = candidate()
    source["pose_xyz"][:, 0, 0, 0] = source["frame_idx"]
    selected = m.candidate_rows(source)
    assert selected["frame_idx"].tolist() == [1001, 1002, 1003]
    assert selected["pose_xyz"][:, 0, 0, 0].tolist() == [1001, 1002, 1003]
    assert not any(a.flags.writeable for a in selected.values())
    assert all(not np.shares_memory(selected[k], source[k]) for k in source)


@pytest.mark.parametrize("mode", ["duplicate", "offset", "shape", "dtype"])
def test_invalid_candidate_native_or_field_shape_rejected(mode):
    source = candidate()
    if mode == "duplicate":
        source["frame_idx"][2] = source["frame_idx"][1]
    if mode == "offset":
        source["frame_idx"] += 1
    if mode == "shape":
        source["pose_xyz"] = source["pose_xyz"][:, :2]
    if mode == "dtype":
        source["identity_review"] = source["identity_review"].astype(np.int8)
    with pytest.raises(ValueError):
        m.candidate_rows(source)


def test_candidate_archive_decodes_only_three_fields_and_checks_signatures(isolated):
    selected = candidate()
    payload = io.BytesIO()
    np.savez_compressed(
        payload, **selected, **{f"unused{i}": np.array([object()], object) for i in range(18)}
    )
    data = payload.getvalue()
    path = isolated / "candidate.npz"
    path.write_bytes(data)
    sigs = {k: m.signature(v) for k, v in selected.items()}
    sigs.update({f"unused{i}": {} for i in range(18)})
    reference = dict(
        candidate=dict(
            path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
        ),
        arrays=sigs,
    )
    inputs, guards = {}, {}
    result = m.load_candidate(reference, inputs, guards)
    assert result["frame_idx"].tolist() == [1001, 1002, 1003]
    assert len(inputs) == len(guards) == 1


@pytest.mark.parametrize("mode", ["wrong_native", "offset_terminal", "track_ids", "wrong_dtype"])
def test_packed_structure_rejects_wrong_source_window_and_schema(mode):
    p = packed()
    if mode == "wrong_native":
        p["frame_idx"][0] -= 1
    if mode == "offset_terminal":
        p["frame_offsets"][-1] += 1
    if mode == "track_ids":
        p["source_track_id"][0] = 8
    if mode == "wrong_dtype":
        p["mask_bits"] = p["mask_bits"].astype(np.int8)
    with pytest.raises(ValueError):
        m.validate_packed_structure(p)


def test_decoder_native_offsets_availability_and_hash_are_enforced():
    p = packed()
    m.validate_packed_structure(p)
    first, available = m.decoded_frame(p, 1001)
    assert available and first.shape == (2, 1200, 1200) and first.sum() == 2
    missing, available = m.decoded_frame(p, 1002)
    assert not available and missing.shape == (0, 1200, 1200)
    with pytest.raises(ValueError, match="offset"):
        m.decoded_frame(p, 1000)
    p["frame_mask_sha256"][2] = b"0" * 64
    with pytest.raises(ValueError, match="SHA"):
        m.decoded_frame(p, 1003)


def test_three_arm_stream_reuses_each_native_mask_once_and_preserves_unknown_breaks(monkeypatch):
    candidates = [m.candidate_rows(candidate()) for _ in range(3)]
    # A saved final trajectory is a copy; adjust synthetic fixture before freezing.
    candidates = [{k: v.copy() for k, v in c.items()} for c in candidates]
    candidates[1]["pose_xyz"][:, 0, 2, 0] = 25
    candidates[2]["pose_xyz"][2, 2, 2, 0] = 25
    candidates[2]["identity_review"][2, 2] = True
    for c in candidates:
        for a in c.values():
            a.flags.writeable = False
    fields = {v: packed() for v in m.VIEWS}
    decoder = m.prepare_sequence.decode_frame_masks
    decoder_calls, projector_calls = [], []

    def counted_decode(data, index):
        decoder_calls.append((id(data), index))
        return decoder(data, index)

    def project(xyz):
        projector_calls.append(xyz.shape)
        assert xyz.shape == (9, 3, 3)
        return xyz[..., :2]

    monkeypatch.setattr(m.prepare_sequence, "decode_frame_masks", counted_decode)
    rig = SimpleNamespace(camera=lambda view: SimpleNamespace(project=project))
    result = m.evaluate_frames(candidates, fields, rig)
    assert len(decoder_calls) == 9 and len(set(decoder_calls)) == 9
    assert len(projector_calls) == 6  # unavailable middle skips all three projections
    assert result["classification"].shape == (3, 3, 3, 3)
    assert (result["classification"][0, 0] == "same_instance").all()
    assert (result["classification"][0, 1, 0] == "split_instances").all()
    assert (result["classification"][1] == "unavailable").all()
    assert (result["nearby_mask_count"][1] == -1).all()
    rows = m.summarize(result)
    assert rows[0]["split_body_frames"] == 0 and rows[1]["split_body_frames"] == 2
    assert rows[2]["split_body_frames"] == rows[2]["split_body_reviewed"] == 1
    assert [
        (r["start_native"], r["stop_native_exclusive"]) for r in rows[1]["scene_split_runs"]
    ] == [(1001, 1002), (1003, 1004)]
    assert rows[1]["unknown_body_frames"] == 3
    result["identity_review"][2, 2, 2] = False
    with pytest.raises(ValueError, match="mandatory identity review"):
        m.summarize(result)


def test_run_boundaries_use_source166fps_and_do_not_join_across_unknown_gap():
    flags = np.array([True, True, False, True, False])
    rows = m.run_intervals(flags, np.arange(90000, 90005))
    assert rows == [
        dict(
            start_native=90000,
            stop_native_exclusive=90002,
            native_frames=2,
            duration_seconds=2 / 166,
        ),
        dict(
            start_native=90003,
            stop_native_exclusive=90004,
            native_frames=1,
            duration_seconds=1 / 166,
        ),
    ]
    assert m.run_intervals(np.zeros(5, bool), np.arange(5)) == []


def test_output_rejection_precedes_all_real_input_entries(isolated):
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(isolated.parent / "outside", "0" * 64, "1" * 64)
    existing = isolated / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        m.run(existing, "0" * 64, "1" * 64)
