"""Anonymous334-query boundary, frozen-loop parity and saved64-prefix binding."""

import ast
import hashlib
import sys
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import mask_transport_stream as streaming  # noqa: E402
import semantic_node_stream as original_stream  # noqa: E402
import track_mask_transport200 as m  # noqa: E402


def test_online_numeric_loop_only_changes_finite_query_bound():
    old = ast.parse((WORK / "semantic_node_stream.py").read_text(encoding="utf-8"))
    new = ast.parse((WORK / "mask_transport_stream.py").read_text(encoding="utf-8"))
    before, after = [next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name == "run_online200") for tree in (old, new)]
    for node in ast.walk(after):
        if isinstance(node, ast.Constant) and node.value == 157:
            node.value = 9
        elif isinstance(node, ast.Constant) and node.value == (
                "at most157 finite float32 auxiliary queries per view required"):
            node.value = "at most9 finite float32 semantic queries per view required"
    assert ast.dump(before) == ast.dump(after)
    assert streaming.StreamingLK is original_stream.StreamingLK


def query_fixture():
    sizes = [30, 64, 63]
    query = dict(
        xy=np.arange(314, dtype=np.float32).reshape(157, 2) + 100,
        mask_index=np.repeat(np.arange(3, dtype=np.int64), sizes),
        corner_index=np.concatenate([np.arange(n, dtype=np.int64) for n in sizes]),
        frame_idx=np.asarray(m.FIRST, np.int64), available=np.asarray(True, bool),
    )
    rows, cursor = [], 0
    for i, size in enumerate(sizes):
        rows.append(dict(mask_index=i, query_start=cursor, query_stop=cursor + size,
                         query_count=size))
        cursor += size
    row = dict(view="top_cam", query_count=157, mask_count=3, rows=rows,
               array_signatures=m.delta.signatures(query))
    return query, row


@pytest.mark.parametrize("corruption", ["count", "group", "corner", "missing_group", "bytes"])
def test_query_bytes_and_all_group_boundaries(corruption):
    query, row = query_fixture()
    m.validate_queries(query, row, "top_cam")
    if corruption == "count":
        query["xy"] = query["xy"][:-1]
    elif corruption == "group":
        query["mask_index"][30] = 0
        row["array_signatures"] = m.delta.signatures(query)
    elif corruption == "corner":
        query["corner_index"][30] = 1
        row["array_signatures"] = m.delta.signatures(query)
    elif corruption == "missing_group":
        row["rows"] = row["rows"][:-1]
    else:
        query["xy"][3, 0] += 1
    with pytest.raises(ValueError):
        m.validate_queries(query, row, "top_cam")


def test_duplicate_pixels_across_groups_are_preserved():
    query, row = query_fixture()
    query["xy"][30] = query["xy"][0]
    row["array_signatures"] = m.delta.signatures(query)
    m.validate_queries(query, row, "top_cam")
    assert query["mask_index"][0] != query["mask_index"][30]


def test_158_queries_rejected_before_iterator_or_backend():
    with pytest.raises(ValueError, match="at most157"):
        streaming.run_online200(None, np.zeros((158, 2), np.float32), None, None, first=0)


def test_prefix_guard_rejects_signed_zero_and_field_omission():
    current = dict(xy=np.zeros((200, 2), np.float32))
    saved = dict(xy=current["xy"][:64].copy())
    m.match_prefix(current, saved, count=64, name="control")
    saved["xy"][0, 0] = -0.0
    with pytest.raises(ValueError, match="bytes"):
        m.match_prefix(current, saved, count=64, name="control")
    with pytest.raises(ValueError, match="member"):
        m.match_prefix(current, {}, count=64, name="control")


def test_thread_policy_matches_old_LK_execution_not_new_PNG_transport():
    m.validate_thread_policy(dict(opencv_threads=32), dict(opencv_threads=32))
    with pytest.raises(ValueError, match="original64"):
        m.validate_thread_policy(dict(opencv_threads=32), dict(opencv_threads=2))


def test_recorded_thread_metadata_contract():
    import json

    prep = json.loads((WORK / "temporal_known_gap64_preparation_v1"
                       / "preparation.provenance.json").read_text(encoding="utf-8"))
    old = json.loads((WORK / "temporal_known_gap64_comparison_v1"
                      / "summary.json").read_text(encoding="utf-8"))
    new = json.loads((WORK / "semantic_node200_preparation_v2"
                      / "preparation.provenance.json").read_text(encoding="utf-8"))
    m.validate_thread_policy(prep["environment"], old["environment"])
    assert new["environment"]["opencv_threads"] == 2


class FakeBackend:
    instances: ClassVar[list] = []

    def __init__(self, device, closure):
        assert device == "cpu"
        self.initializations = self.forwards = self.closed = 0
        self.load_seconds = 0.0
        self.instances.append(self)

    def initialize(self, images, queries):
        self.initializations += 1
        self.queries = queries.copy()
        return None, None

    @staticmethod
    def arrays(queries, hi):
        xy = np.broadcast_to(queries, (hi, len(queries), 2)).copy()
        scores = np.full((hi, len(queries)), 0.8, np.float32)
        return dict(xy=xy, model_xy=xy.copy(), reported=scores > np.float32(0.6),
                    sigmoid_visibility=scores.copy(), sigmoid_confidence=np.ones_like(scores),
                    product=scores)

    def predict(self, images):
        assert len(images) == 16
        hi = 16 + self.forwards * 8
        self.forwards += 1
        return self.arrays(self.queries, hi), dict(forward_seconds=0.0)

    def close(self):
        self.closed += 1
        return dict(max_allocated_bytes=None, max_reserved_bytes=None)


def view_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PREPARATION", tmp_path)
    old_root = tmp_path / "original64"
    monkeypatch.setattr(m, "OLD_COMPARISON", old_root)
    image = np.zeros((1200, 1200, 3), np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = tmp_path / "frame.png"
    png.write_bytes(encoded.tobytes())
    members = {str(png): dict(size_bytes=png.stat().st_size, sha256=m.base.sha(png))}
    query, row = query_fixture()
    old_output = m.base.Output(old_root)
    for i in range(7):
        old_output.write(f"top_cam/online/call_{i}.npz",
                         FakeBackend.arrays(query["xy"], 16 + i * 8))
    old_output.write("top_cam/lk_raw.npz", m.core.lk_chain([image] * 64, query["xy"]))
    old_members = {record["path"]: record for record in old_output.records.values()}
    native_rows = [dict(view="top_cam", frame_idx=n, png_path=png.name,
                        bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                   for n in range(m.FIRST, m.STOP)]
    context = dict(natives=native_rows, queries={"top_cam": query}, query_rows=[row],
                   members=members, old_members=old_members)
    monkeypatch.setattr(m, "load_masks", lambda *_: np.zeros((0, 1200, 1200), bool))
    monkeypatch.setattr(m.base, "TorchOnline", FakeBackend)
    return context


def test_full200_157query_native_stream_saved64_parity_and_no_reset(tmp_path, monkeypatch):
    context = view_fixture(tmp_path, monkeypatch)
    output = m.base.Output(tmp_path / "output")
    result = m.run_view(context, "top_cam", {}, output, "cpu", [])
    backend = FakeBackend.instances[-1]
    assert (backend.initializations, backend.forwards, backend.closed) == (1, 24, 1)
    assert result["ct_first_reported"] == result["tracking199_query_denominator"] == 31243
    assert result["old64_all7_calls6_fields_byte_exact"]
    assert result["old64_LK_all_fields_byte_exact"]
    assert result["initial_group_count"] == 3
    assert len([p for p in output.records if "/memberships/" in p]) == 200
    assert len([p for p in output.records if "/calls/" in p and p.endswith(".npz")]) == 24
    output.final_guard()
    emission = m.arrays_from_bytes((output.root / "top_cam/emissions.npz").read_bytes())
    assert emission["first_xy"].shape == (200, 157, 2)
    assert emission["first_release_native"][-1] == m.STOP - 1


def test_prefix_mismatch_preserves_partial_output_and_closes(tmp_path, monkeypatch):
    context = view_fixture(tmp_path, monkeypatch)
    original = m.read_old

    def corrupt(ctx, name):
        arrays = original(ctx, name)
        if name.endswith("call_0.npz"):
            arrays["xy"][0, 0, 0] += 1
        return arrays

    monkeypatch.setattr(m, "read_old", corrupt)
    output = m.base.Output(tmp_path / "failed")
    with pytest.raises(ValueError, match="original64:call0"):
        m.run_view(context, "top_cam", {}, output, "cpu", [])
    assert FakeBackend.instances[-1].closed == 1
    assert "top_cam/calls/000.npz" in output.records
    assert "top_cam/calls/000.json" in output.records
