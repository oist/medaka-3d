"""Synthetic fixed-window preparation contracts; no source-video/model execution."""

import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import prepare_temporal_point64 as p  # noqa: E402
from prepare_sequence import PackedMasks  # noqa: E402


class Capture:
    def __init__(self, *, bad_position=None, fail_at=None):
        self.position = 0
        self.reads = 0
        self.seeks = []
        self.released = False
        self.bad_position = bad_position
        self.fail_at = fail_at

    def isOpened(self):
        return True

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.position = value
        self.seeks.append(value)
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position + (1 if self.reads == self.bad_position else 0)
        return {
            cv2.CAP_PROP_FPS: 166,
            cv2.CAP_PROP_FRAME_COUNT: 348600,
            cv2.CAP_PROP_FRAME_HEIGHT: 3,
            cv2.CAP_PROP_FRAME_WIDTH: 4,
        }[prop]

    def read(self):
        self.reads += 1
        self.position += 1
        return self.reads != self.fail_at, np.full((3, 4, 3), self.reads - 1, np.uint8)

    def release(self):
        self.released = True


def frame_hashes():
    return [p.signature(np.full((3, 4, 3), i, np.uint8))["sha256"] for i in range(64)]


def test_exact64_one_seek_no65():
    cap = Capture()
    rows = list(p.read64(cap, frame_hashes(), shape=(3, 4)))
    assert [n for n, _ in rows] == list(range(66483, 66547))
    assert cap.reads == 64 and cap.seeks == [66483]
    assert rows[-1][1][0, 0, 0] == 63


@pytest.mark.parametrize("fault", ["sha", "position", "eof", "length"])
def test_native_decode_rejects_wrong_evidence(fault):
    cap = Capture(
        bad_position=5 if fault == "position" else None, fail_at=5 if fault == "eof" else None
    )
    hashes = frame_hashes()
    if fault == "sha":
        hashes[4] = "0" * 64
    if fault == "length":
        hashes.pop()
    with pytest.raises(ValueError):
        list(p.read64(cap, hashes, shape=(3, 4)))
    assert cap.reads <= 5


def records():
    result = [dict(index=i, start=49800 + 1660 * i, stop=49800 + 1660 * (i + 1)) for i in range(16)]
    result[10].update(manifest_sha256=p.MANIFEST_SHA, provenance_sha256=p.CHUNK_SHA)
    return result


@pytest.mark.parametrize("fault", ["range", "bool", "manifest", "seal", "length"])
def test_selected_fixed_chunk_rejects_changes(fault):
    values = records()
    if fault == "range":
        values[10]["start"] += 1
    elif fault == "bool":
        values[0]["index"] = False
    elif fault == "manifest":
        values[10]["manifest_sha256"] = "0" * 64
    elif fault == "seal":
        values[10]["provenance_sha256"] = "0" * 64
    else:
        values.pop()
    with pytest.raises(ValueError):
        p.selected_record(values)


def test_selected_record_no_other_payload_required():
    values = records()
    assert p.selected_record(values) is values[10]


def test_members_mixed_actual_schemas_and_conflict(tmp_path):
    path = tmp_path / "small.json"
    path.write_text("{}")
    digest = p.sha(path)
    assert p.member(p.members(dict(inputs={str(path): digest}), "inputs"), path)["size"] is None
    row = dict(path=str(path), sha256=digest, size=2)
    table = p.members(dict(outputs={"name": row, "alias": row.copy()}), "outputs")
    assert p.member(table, path, digest)["size"] == 2
    bad = dict(row, sha256="0" * 64)
    with pytest.raises(ValueError, match="conflicting"):
        p.members(dict(outputs={"name": row, "alias": bad}), "outputs")


def packed(mask_count, *, available=True):
    encoder = PackedMasks(shape=(1200, 1200))
    for frame in range(p.CHUNK_START, p.START + 1):
        masks = np.zeros((mask_count if frame == p.START else 0, 1200, 1200), bool)
        n = len(masks)
        encoder.append(frame, masks, np.ones(n, np.float32), np.zeros((n, 4)), np.zeros(n, int))
    result = encoder.arrays()
    result["view_available"][-1] = available
    return result


@pytest.mark.parametrize("count,available", [(0, False), (0, True), (1, True)])
def test_observation_preserves_missing_empty_and_empty_instance(count, available):
    data = packed(count, available=available)
    old = {k: p.signature(v) for k, v in data.items()}
    obs = p.observation(data, p.START, count)
    assert obs["mask_available"].shape == () and bool(obs["mask_available"]) == available
    assert obs["masks"].shape == (count, 1200, 1200)
    assert obs["frame_idx"].dtype == np.int64
    assert set(obs) == {"frame_idx", "masks", "mask_available"} | {
        "packed_" + k for k in p.native.SCHEMA_DTYPES
    }
    assert old == {k: p.signature(v) for k, v in data.items()}


def test_observation_rejects_source_sha_and_count():
    data = packed(1)
    with pytest.raises(ValueError, match="count"):
        p.observation(data, p.START, 0)
    data["frame_mask_sha256"][-1] = b"0" * 64
    with pytest.raises(ValueError, match="SHA"):
        p.observation(data, p.START, 1)


def query_fixture():
    image = np.zeros((32, 32, 3), np.uint8)
    image[8:24, 8:24] = 255
    masks = np.ones((2, 32, 32), bool)
    return image, dict(masks=masks, mask_available=np.asarray(True), frame_idx=np.asarray(p.START))


def test_query_npz_roundtrip_preserves_core_bytes_rows_and_duplicates(tmp_path):
    image, obs = query_fixture()
    expected = p.core.select_queries(image, obs["masks"])
    arrays, info = p.make_queries(image, obs)
    assert p.signature(arrays["xy"]) == p.signature(expected["xy"])
    assert p.signature(arrays["mask_index"]) == p.signature(expected["mask_index"])
    assert info["rows"] == expected["rows"] and info["mask_count"] == 2
    assert info["repeated_xy_count"] > 0
    path = tmp_path / "query.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as z:
        assert {k: p.signature(z[k]) for k in z.files} == {
            k: p.signature(v) for k, v in arrays.items()
        }
    for row in info["rows"]:
        assert arrays["corner_index"][row["query_start"] : row["query_stop"]].tolist() == list(
            range(row["query_count"])
        )


def test_query_input_mutation_is_rejected(monkeypatch):
    image, obs = query_fixture()
    original = p.core.select_queries

    def corrupt(image, masks):
        result = original(image, masks)
        image[0, 0, 0] += 1
        return result

    monkeypatch.setattr(p.core, "select_queries", corrupt)
    with pytest.raises(ValueError, match="mutated"):
        p.make_queries(image, obs)


def test_query_unavailable_differs_from_available_zero():
    image = np.zeros((32, 32, 3), np.uint8)
    for available, count in ((False, None), (True, 0)):
        arrays, row = p.make_queries(
            image, dict(masks=np.empty((0, 32, 32), bool), mask_available=np.asarray(available))
        )
        assert row["mask_count"] == count and row["available"] == available
        assert arrays["xy"].shape == (0, 2) and arrays["corner_index"].dtype == np.int64


def test_first_reference_checks_without_loading_detections(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "PREP", tmp_path)
    image = np.zeros((3, 4, 3), np.uint8)
    obs = dict(
        frame_idx=np.asarray(p.START, np.int64),
        masks=np.zeros((0, 3, 4), bool),
        mask_available=np.asarray(True),
        packed_frame_idx=np.asarray([p.START], np.int64),
    )
    cv2.imwrite(str(tmp_path / "first.png"), image)
    stored = {k: v for k, v in obs.items() if k != "frame_idx"}
    # An object field raises if deserialized with allow_pickle=False.
    np.savez(tmp_path / "first.npz", **stored, detections=np.asarray([object()], object))
    row = dict(
        png_path="first.png",
        observation_path="first.npz",
        observation_array_signatures={k: p.signature(v) for k, v in stored.items()},
    )
    p.verify_first(image, obs, row)
    obs["mask_available"] = np.asarray(False)
    with pytest.raises(ValueError, match="observation"):
        p.verify_first(image, obs, row)


def test_new_output_membership_and_change_guard(tmp_path):
    path = tmp_path / "file.json"
    path.write_text("{}")
    rows = dict(file=dict(path=str(path.resolve()), size_bytes=2, sha256=p.sha(path)))
    p.output_guard(tmp_path, rows)
    extra = tmp_path / "extra"
    extra.write_text("unexpected")
    with pytest.raises(ValueError, match="membership"):
        p.output_guard(tmp_path, rows)
    extra.unlink()
    path.write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        p.output_guard(tmp_path, rows)


def test_failed_decode_seals_partial_and_closes_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "WORK", tmp_path)
    monkeypatch.setattr(p, "source_code", lambda: {})
    monkeypatch.setattr(p, "no_model_imports", lambda: None)
    data = packed(0)
    # Run's full-header check needs all1660 native entries, no instances.
    count = 1660
    data.update(
        frame_idx=np.arange(p.CHUNK_START, p.CHUNK_STOP, dtype=np.int64),
        frame_offsets=np.zeros(count + 1, np.int64),
        view_available=np.ones(count, bool),
        frame_mask_sha256=np.repeat(data["frame_mask_sha256"][:1], count),
    )
    np.savez(tmp_path / "mask.npz", **data)
    inputs = dict(
        root=tmp_path,
        shared={},
        videos={v: tmp_path / "video" for v in p.VIEWS},
        manifest=dict(
            masks={v: dict(path="mask.npz") for v in p.VIEWS},
            raw={v: dict(frame_bgr_sha256=["0" * 64] * 1660) for v in p.VIEWS},
        ),
    )
    monkeypatch.setattr(p, "metadata_inputs", lambda evidence: inputs)
    monkeypatch.setattr(p, "verify_payloads", lambda inputs, evidence: None)
    cap = Capture(fail_at=1)
    monkeypatch.setattr(p.cv2, "VideoCapture", lambda path: cap)

    def failing(*args):
        raise RuntimeError("synthetic decode failure")

    monkeypatch.setattr(p, "read64", failing)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="synthetic"):
        p.run(out)
    seal = json.loads((out / "preparation.provenance.json").read_text())
    assert seal["status"] == "failed" and cap.released
    assert "plan.json" in seal["outputs"] and not (out / "summary.json").exists()
    with pytest.raises(ValueError, match="new workspace"):
        p.run(out)


def test_actual_small_json_contract_without_payload_reads():
    if not (p.PREP / "preparation.provenance.json").is_file():
        pytest.skip("workspace evidence not present")
    evidence = {}
    info = p.metadata_inputs(evidence)
    assert len(info["references"]) == 19
    assert [info["first"][v]["mask_count"] for v in p.VIEWS] == [4, 5, 4]
    assert not any(Path(path).suffix in {".npz", ".mp4", ".png", ".pth"} for path in evidence)
    assert all("candidate" not in path and "arrays.npz" not in path for path in evidence)


def test_initial_rows_all60_order_and_counts():
    rows = [
        dict(frame_idx=66483 + 166 * i, view=v, mask_count=(4, 5, 4)[j])
        for i in range(60)
        for j, v in enumerate(p.VIEWS)
    ]
    assert list(p.first_rows(rows)) == list(p.VIEWS)
    changed = copy.deepcopy(rows)
    changed[0]["mask_count"] = 3
    with pytest.raises(ValueError, match="13"):
        p.first_rows(changed)
    rows[3], rows[4] = rows[4], rows[3]
    with pytest.raises(ValueError, match="native"):
        p.first_rows(rows)
