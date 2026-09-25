"""Synthetic diagnostic persistence only; no tracker state or recording inputs."""

import copy
from pathlib import Path
from types import SimpleNamespace

import nodewise_diagnostic_batch as module
import numpy as np
import pytest


def diagnostics(frame):
    """Asymmetric candidate arrays and unknown values expose codec aliasing/loss."""
    instance = dict(
        schema="instance_rescore_frame_diagnostics_v1",
        frame_idx=frame,
        candidate_index=np.arange(3, dtype=np.int32),
        gate_preparation_seconds=0.125,
        candidate_scoring_seconds=0.25,
        views={"side2_cam": {"eligible": False}, "top_cam": {"eligible": True}},
        nested={
            "permutation": np.array([2, 0, 1], dtype=np.int32),
            "z_first": [None, np.float64(np.inf), np.float32(-np.inf), float("nan")],
            "a_second": np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1],
            "empty": np.empty((0, 3), dtype=np.int8),
            "bools": np.array([True, False, True]),
            "names": np.array(["body_row", "original", "末尾"], dtype="U10"),
            "scalar": np.array(4.25, dtype=np.float64),
        },
    )
    body = dict(
        schema="body_row_frame_diagnostics_v1",
        frame_idx=frame,
        frame_completed=True,
        incoming_proposal_count=2,
        outgoing_proposal_count=3,
        decoder_candidate_cap=8,
        headroom=6,
        zero_headroom=False,
        generated_scene_count=2,
        added_proposal_count=1,
        excluded_by_capacity=1,
        new_proposal_indices=np.array([2], dtype=np.int64),
        new_proposal_kind="body_row",
        new_proposal_plans=np.array([[[0, -1, 2], [1, 0, -1]]], dtype=np.int8),
        nested=copy.deepcopy(instance["nested"]),
        **dict(zip(module.TIMING_FIELDS, [0.01, 0.02, 0.03, 0.06], strict=True)),
    )
    return instance, body


def tracker_for(frame):
    instance, body = diagnostics(frame)
    calls = []

    def exported(value, label):
        calls.append(label)
        return copy.deepcopy(value)

    tracker = SimpleNamespace(
        previous_frame=frame,
        export_instance_diagnostics=lambda: exported(instance, "instance"),
        export_body_row_diagnostics=lambda: exported(body, "body"),
    )
    return tracker, instance, body, calls


def assert_exact(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes(order="C") == expected.tobytes(order="C")
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_exact(left, right)
    elif isinstance(expected, (float, np.floating)) and not np.isfinite(expected):
        value = np.asarray(actual)
        assert value.shape == ()
        assert np.isnan(value) if np.isnan(expected) else value == expected
    else:
        assert actual == expected


@pytest.fixture(autouse=True)
def isolated_io(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real input or scientific execution is forbidden")

    for target in (module.frozen, module.codec, module.proof):
        for key in ("WORK", "REPO"):
            if hasattr(target, key):
                monkeypatch.setattr(target, key, tmp_path)
    monkeypatch.setattr(module.frozen, "PARENT", tmp_path / "missing_parent")
    for target, key in (
        (module.frozen, "load_parent_headers"),
        (module.frozen, "load_parent_payload"),
        (module.frozen.original, "verified_inputs"),
        (module.frozen.original, "SourceSequenceReader"),
        (module.frozen.torch.cuda, "synchronize"),
        (module.frozen.prior, "read_checkpoint"),
    ):
        monkeypatch.setattr(target, key, forbidden)
    load = np.load
    read_json = module.proof.read_json

    def local_load(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        assert kwargs.get("allow_pickle") is False
        return load(path, *args, **kwargs)

    def local_json(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        return read_json(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", local_load)
    monkeypatch.setattr(module.proof, "read_json", local_json)


def restore_batch(record):
    metadata = module.proof.read_json(record["metadata"]["path"])
    with np.load(record["archive"]["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return module.codec.restore_diagnostic(metadata["tree"], arrays)["frames"]


def test_100_frame_flush_then_chunk_tail_preserves_every_export(tmp_path):
    out = module.proof.Output(tmp_path / "synthetic_batch")
    writer = module.NodewiseDiagnosticBatch(out)
    expected = []
    for i in range(102):
        native = 66400 + i
        tracker, instance, body, calls = tracker_for(native)
        expected.append(
            dict(frame_idx=native, instance=copy.deepcopy(instance), body_row=copy.deepcopy(body))
        )
        row = writer.append(tracker, native)
        assert calls == ["instance", "body"]
        assert row["frame_idx"] == native
        assert writer.totalframes == i + 1
        if i == 98:
            assert writer.pending_frames == 99 and writer.saved_frames == 0
            assert out.records == {} and writer.batches == []
        elif i == 99:
            assert writer.pending_frames == 0 and writer.saved_frames == 100
            assert len(writer.batches) == 1 and len(out.records) == 2
        # Real exporters give owned deep copies. Destroy the live tracker data.
        instance["nested"]["a_second"][:] = -999
        body["new_proposal_indices"][:] = -999
    assert writer.pending_frames == 2 and writer.saved_frames == 100
    tail = writer.flush()
    assert tail["first_frame"] == 66500 and tail["stop_frame_exclusive"] == 66502
    assert writer.saved_frames == writer.totalframes == len(writer.records) == 102
    assert writer.pending_frames == 0 and len(out.records) == 4
    assert writer.flush() is None and len(out.records) == 4
    actual = [row for record in writer.batches for row in restore_batch(record)]
    assert_exact(actual, expected)
    out.final_guard()


def test_compact_counts_and_timings_match_old_per_frame_writer(tmp_path):
    tracker, instance, body, _ = tracker_for(81340)
    old_out = module.proof.Output(tmp_path / "old_synthetic")
    old_row = module.frozen.save_frame_diagnostics(old_out, tracker, 81340)
    new_out = module.proof.Output(tmp_path / "new_synthetic")
    writer = module.NodewiseDiagnosticBatch(new_out, batch_size=1)
    actual = writer.append(tracker, 81340)
    assert {k: v for k, v in actual.items() if k != "body_row_timing_seconds"} == old_row
    assert actual["body_row_timing_seconds"] == {key: body[key] for key in module.TIMING_FIELDS}
    saved = restore_batch(writer.batches[0])[0]
    assert_exact(saved["instance"], instance)
    assert_exact(saved["body_row"], body)
    assert list(saved["instance"]["views"]) == ["side2_cam", "top_cam"]


@pytest.mark.parametrize("size", [0, 101, -1, True, 1.0, None])
def test_invalid_batch_bound_rejected(tmp_path, size):
    out = module.proof.Output(tmp_path / "bad_size")
    with pytest.raises(ValueError, match="batch size"):
        module.NodewiseDiagnosticBatch(out, batch_size=size)
    assert out.records == {}


@pytest.mark.parametrize("native", [66400, 66402, -1, True, 66401.0])
def test_wrong_native_order_poisoned_before_export(tmp_path, native):
    writer = module.NodewiseDiagnosticBatch(module.proof.Output(tmp_path / "order"))
    writer.append(tracker_for(66400)[0], 66400)
    tracker, _, _, calls = tracker_for(native)
    with pytest.raises(ValueError, match="consecutive"):
        writer.append(tracker, native)
    assert calls == [] and writer.totalframes == writer.pending_frames == 1
    assert writer.saved_frames == 0
    with pytest.raises(ValueError, match="previously failed"):
        writer.append(tracker_for(66401)[0], 66401)
    with pytest.raises(ValueError, match="previously failed"):
        writer.flush()


@pytest.mark.parametrize(
    "damage", ["tracker", "instance", "body", "incomplete", "count", "indices", "timing"]
)
def test_stale_foreign_incomplete_and_inconsistent_exports_rejected(tmp_path, damage):
    writer = module.NodewiseDiagnosticBatch(module.proof.Output(tmp_path / "invalid_export"))
    tracker, instance, body, calls = tracker_for(90258)
    if damage == "tracker":
        tracker.previous_frame -= 1
    elif damage == "instance":
        instance["frame_idx"] -= 1
    elif damage == "body":
        body["frame_idx"] += 1
    elif damage == "incomplete":
        body["frame_completed"] = False
    elif damage == "count":
        instance["candidate_index"] = np.arange(4)
    elif damage == "indices":
        instance["candidate_index"] = np.array([0, 2, 1])
    else:
        body["generation_seconds"] = np.nan
    with pytest.raises(ValueError):
        writer.append(tracker, 90258)
    assert writer.totalframes == writer.saved_frames == writer.pending_frames == 0
    assert writer.records == writer.batches == []
    if damage == "tracker":
        assert calls == []
    with pytest.raises(ValueError, match="previously failed"):
        writer.flush()


@pytest.mark.parametrize("failure", ["write_json", "corrupt_readback"])
def test_failed_flush_retains_pending_without_success_or_retry(tmp_path, monkeypatch, failure):
    out = module.proof.Output(tmp_path / "failed_batch")
    writer = module.NodewiseDiagnosticBatch(out, batch_size=1)
    if failure == "write_json":
        write = out.write

        def broken_write(name, value):
            if name.endswith(".json"):
                raise OSError("synthetic interrupted metadata write")
            return write(name, value)

        monkeypatch.setattr(out, "write", broken_write)
    else:
        read_json = module.proof.read_json

        def corrupt(path):
            value = read_json(path)
            value["frame_count"] = 99
            return value

        monkeypatch.setattr(module.proof, "read_json", corrupt)
    with pytest.raises((ValueError, OSError)):
        writer.append(tracker_for(90141)[0], 90141)
    assert writer.totalframes == writer.pending_frames == len(writer.records) == 1
    assert writer.saved_frames == 0 and writer.batches == []
    assert len(out.records) == (1 if failure == "write_json" else 2)
    with pytest.raises(ValueError, match="previously failed"):
        writer.flush()
