"""Mask staging preserves independent proposals and cannot silently omit inputs."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "labs/20260908_mask_volume/stage_mask_inputs.py"
SPEC = importlib.util.spec_from_file_location("_mask_staging_test", MODULE)
staging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(staging)


def make_cache(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(schema_version="sam2_id_masks_v1", image_height=6, image_width=7)
        for frame in frames:
            group = handle.create_group(f"frame_{frame:06d}")
            # An explicit empty frame is different from an absent group.
            if frame == 2:
                continue
            for index in range(frame % 4 + 1):
                instance = group.create_group(f"instance_{index:03d}")
                mask = np.zeros((6, 7), dtype=np.uint8)
                mask[1:3, index : index + 2] = 1
                instance.create_dataset("mask", data=mask)
                instance.create_dataset("bbox", data=[index, 1, index + 2, 3])
                instance.attrs.update(
                    frame=frame, instance_id=index, sam2_track_id=index, score=0.51 + 0.1 * index
                )


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / "source"
    for relative in staging.RELATIVE_MASKS.values():
        make_cache(root / relative, range(10))
    return root


def test_selection_uses_shared_index_extent_and_earlier_tie():
    chosen, targets, shared = staging.select_frames([[0, 4, 6, 10], [0, 4, 6, 9, 10]], 3)
    np.testing.assert_array_equal(chosen, [0, 4, 10])
    np.testing.assert_array_equal(targets, [0, 5, 10])
    np.testing.assert_array_equal(shared, [0, 4, 6, 10])


def test_stage_preserves_all_masks_empty_frame_and_checksums(sources, tmp_path):
    out = tmp_path / "staged"
    manifest = staging.stage_inputs(sources, out)
    assert manifest["frames"] == list(range(10))
    assert json.loads((out / "staging.provenance.json").read_text())["status"] == "ok"
    for entry in manifest["views"].values():
        assert entry["source_sha256"] == staging.sha256(sources / entry["source_relative"])
        assert entry["output_sha256"] == staging.sha256(out / entry["output_npz"])
        with np.load(out / entry["output_npz"], allow_pickle=False) as data:
            assert data["masks"].dtype == bool
            assert data["view_available"].all()
            assert data["frame_offsets"][2] == data["frame_offsets"][3]
            # Four proposals remain four; no exactly-three truncation.
            assert data["frame_offsets"][4] - data["frame_offsets"][3] == 4
            np.testing.assert_array_equal(data["boxes"][0], [0, 1, 2, 3])
            assert data["scores"][0] == 0.51
            assert data["masks"][0].sum() == 4
            np.testing.assert_array_equal(data["instance_id"], data["source_track_id"])


def test_missing_source_fails_before_creating_destination(sources, tmp_path):
    missing = sources / list(staging.RELATIVE_MASKS.values())[-1]
    missing.rename(missing.with_suffix(".unavailable"))
    out = tmp_path / "new_parent" / "stage"
    with pytest.raises(FileNotFoundError):
        staging.stage_inputs(sources, out)
    assert not out.parent.exists()


def test_no_overwrite_and_bad_mask_fail_before_output(sources, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite"):
        staging.stage_inputs(sources, out)
    assert sentinel.read_bytes() == b"keep"
    source = sources / next(iter(staging.RELATIVE_MASKS.values()))
    with h5py.File(source, "r+") as handle:
        handle["frame_000000/instance_000/mask"][0, 0] = 2
    with pytest.raises(ValueError, match="invalid binary mask"):
        staging.stage_inputs(sources, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
