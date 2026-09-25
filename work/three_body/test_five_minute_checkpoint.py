"""CPU tests of dynamic history preservation and the real lossless storage codec.

Synthetic arrays exercise storage and caller invariants, not tracker semantic
decoding or fish identity. No recorded inputs, image decoder or GPU is used.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import continue_fast_tail_five_minutes as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def forbid_gpu(monkeypatch):
    assert not m.torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU forbidden in checkpoint CPU tests")

    for name in ("is_available", "synchronize", "set_per_process_memory_fraction"):
        monkeypatch.setattr(m.torch.cuda, name, forbidden)
    monkeypatch.setattr(m.prior, "FastTailRecoveryContinuousTracker", forbidden)
    monkeypatch.setattr(m.original, "SourceSequenceReader", forbidden)
    yield
    assert not m.torch.cuda.is_initialized()


def fixture(parent_stop=m.RESUME, checkpoint_stop=m.RESUME + m.CHUNK):
    """Use real native lengths and variable beam counts; all39 logical fields."""
    count = checkpoint_stop - m.FIRST
    parent_count = parent_stop - m.FIRST
    counts = np.resize(np.arange(1, 7, dtype=np.int64), count)
    offsets = np.r_[0, np.cumsum(counts)]
    rows, parent_rows = int(offsets[-1]), int(offsets[parent_count])
    arrays = {"history_offsets": offsets}
    for index, field in enumerate(m.qualified.RECORD_FIELDS):
        key = "history_" + field
        if field == "proposal_kind":
            arrays[key] = np.full(rows, "flow", dtype="U24")
            arrays[key][-1] = "new_future_proposal"
        elif field == "pose_xyz":
            value = (np.arange(rows, dtype=np.float64) % 97) / 13
            value.view(np.uint64)[1] = 0x7FF80000000000AB
            value.view(np.uint64)[2] = 0x8000000000000000
            arrays[key] = value
        else:
            arrays[key] = ((np.arange(rows) + index) % 251).astype(np.int16)
    for index, field in enumerate(m.original.STATE_FIELDS):
        arrays["state_" + field] = np.array([index, -0.0, np.nan], dtype=np.float64)
    for field in ("detections", "images_bgr", "masks", "backgrounds_bgr"):
        for view in m.original.VIEWS:
            arrays["previous_" + field + "_" + view] = np.arange(12, dtype=np.uint8)
    assert len(arrays) == 39
    parent_arrays = {}
    for key, value in arrays.items():
        if key == "history_offsets":
            parent_arrays[key] = value[: parent_count + 1].copy()
        elif key.startswith("history_"):
            parent_arrays[key] = value[:parent_rows].copy()
        else:
            parent_arrays[key] = value.copy()
    # The current history may need wider Unicode storage than the old prefix.
    parent_arrays["history_proposal_kind"] = parent_arrays["history_proposal_kind"].astype("U4")
    metadata = dict(
        previous_frame=checkpoint_stop - 1,
        processed_frames=count,
        initialize_count=1,
        restore_count=5,
    )
    parent_metadata = dict(
        previous_frame=parent_stop - 1,
        processed_frames=parent_count,
        initialize_count=1,
        restore_count=4,
    )
    parent = dict(arrays=parent_arrays, metadata=parent_metadata)
    tracker = SimpleNamespace(export_state=lambda: (arrays, metadata))
    return tracker, parent, arrays, metadata


@pytest.mark.parametrize(
    ("parent_stop", "checkpoint_stop"),
    [(66400, 68060), (66400, 116200), (76360, 78020), (76360, 116200)],
)
def test_dynamic_native_prefix_and_all39_real_codec(tmp_path, parent_stop, checkpoint_stop):
    tracker, parent, arrays, metadata = fixture(parent_stop, checkpoint_stop)
    before = m.array_signature(arrays)
    old_before = m.array_signature(parent["arrays"])
    old_metadata = copy.deepcopy(parent["metadata"])
    binding = dict(schema="synthetic_dynamic_history", parent_stop=parent_stop)
    root = tmp_path / "checkpoint"
    loaded, restored_metadata, row = m.preserve_checkpoint(tracker, parent, root, binding)
    assert set(loaded) == set(arrays) and len(loaded) == 39
    assert m.array_signature(loaded) == before == m.array_signature(arrays)
    assert m.array_signature(parent["arrays"]) == old_before
    assert parent["metadata"] == old_metadata
    assert restored_metadata == metadata
    assert row["previous_frame"] == checkpoint_stop - 1
    assert row["processed_frames"] == checkpoint_stop - m.FIRST
    assert row["initialize_count"] == 1 and row["restore_count"] == 5
    assert row["manifest_sha256"] == m.prior.checks.sha(root / "manifest.json")
    assert row["payload_bytes"] == (root / "arrays.npz").stat().st_size
    assert {p.name for p in root.iterdir()} == {"manifest.json", "metadata.json", "arrays.npz"}
    assert loaded["history_pose_xyz"].view(np.uint64)[1] == 0x7FF80000000000AB
    assert loaded["history_pose_xyz"].view(np.uint64)[2] == 0x8000000000000000
    assert loaded["history_proposal_kind"].dtype == np.dtype("U24")
    assert parent["arrays"]["history_proposal_kind"].dtype == np.dtype("U4")


@pytest.mark.parametrize("field", ("offsets", *m.qualified.RECORD_FIELDS))
def test_any_retained_field_tamper_is_rejected_before_write(tmp_path, field):
    tracker, parent, arrays, _ = fixture()
    key = "history_" + field
    value = arrays[key]
    index = len(parent["arrays"][key]) - 1
    if value.dtype.kind == "U":
        value[index] = "changed"
    else:
        value[index] += 1
    root = tmp_path / "must_not_write"
    with pytest.raises(ValueError, match="retained history prefix changed: " + key):
        m.preserve_checkpoint(tracker, parent, root, {})
    assert not root.exists()


@pytest.mark.parametrize("field", ("pose_xyz", "costs", "parent"))
def test_retained_middle_beam_changes_are_rejected(tmp_path, field):
    tracker, parent, arrays, _ = fixture()
    key = "history_" + field
    middle_frame = 20003
    index = int(parent["arrays"]["history_offsets"][middle_frame]) + 2
    arrays[key][index] += 1
    with pytest.raises(ValueError, match="retained history prefix changed: " + key):
        m.preserve_checkpoint(tracker, parent, tmp_path / "must_not_write", {})


@pytest.mark.parametrize("target", ("parent", "checkpoint"))
@pytest.mark.parametrize("delta", (-1, 1))
def test_processed_frame_metadata_disagreement_is_rejected(tmp_path, target, delta):
    tracker, parent, _, metadata = fixture()
    changed = parent["metadata"] if target == "parent" else metadata
    changed["processed_frames"] += delta
    root = tmp_path / "must_not_write"
    with pytest.raises(ValueError, match="retained frame extent differs"):
        m.preserve_checkpoint(tracker, parent, root, {})
    assert not root.exists()


def test_truncated_checkpoint_cannot_replace_larger_parent(tmp_path):
    tracker, parent, arrays, metadata = fixture()
    arrays["history_offsets"] = arrays["history_offsets"][:100]
    metadata["processed_frames"] = 99
    with pytest.raises(ValueError, match="retained frame extent differs"):
        m.preserve_checkpoint(tracker, parent, tmp_path / "must_not_write", {})


def test_missing_nonhistory_state_field_is_rejected(tmp_path):
    tracker, parent, arrays, _ = fixture()
    del arrays["previous_images_bgr_top_cam"]
    root = tmp_path / "must_not_write"
    with pytest.raises(ValueError, match="complete39 checkpoint required"):
        m.preserve_checkpoint(tracker, parent, root, {})
    assert not root.exists()


def test_existing_checkpoint_is_never_overwritten(tmp_path):
    tracker, parent, _, _ = fixture()
    root = tmp_path / "checkpoint"
    m.preserve_checkpoint(tracker, parent, root, {})
    manifest = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        m.preserve_checkpoint(tracker, parent, root, {})
    assert (root / "manifest.json").read_bytes() == manifest
