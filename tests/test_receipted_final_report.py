"""Synthetic report arithmetic only; no real checkpoint, residual or GPU inputs."""

import copy
import hashlib
import json
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import receipted_final_report as m  # noqa: E402


def candidate(count=5, first=100):
    pose = np.zeros((count, 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(count)[:, None, None]
    result = dict(
        frame_idx=np.arange(first, first + count, dtype=np.int64),
        pose_xyz=pose,
        observed=np.ones((count, 3, 3), bool),
        identity_review=np.zeros((count, 3), bool),
    )
    refresh_coordinates(result)
    return result


def refresh_coordinates(data):
    direct = data["observed"][:, :, 1]
    accepted = direct & ~data["identity_review"]
    data["xyz"] = np.where(direct[..., None], data["pose_xyz"][:, :, 1], np.nan)
    data["accepted_xyz"] = np.where(accepted[..., None], data["pose_xyz"][:, :, 1], np.nan)


def test_support_review_acceptance_counts_reuse_frozen_contract_and_preserve_bytes():
    data = candidate()
    data["observed"][1:3, 0, 1] = False
    data["observed"][:, 1, 2] = False
    data["identity_review"][[0, 4], 0] = True
    data["identity_review"][:, 2] = True
    refresh_coordinates(data)
    before = {key: (a.dtype, a.shape, a.tobytes()) for key, a in data.items()}
    result = m.candidate_metrics(data)
    assert result["direct_counts"] == m.audit_long_final.SUPPORT.direct_counts(data)
    assert result["direct_counts"]["direct_pectoral_per_slot"] == [3, 5, 5]
    assert result["direct_counts"]["accepted_pectoral_per_slot"] == [1, 5, 0]
    assert result["per_slot"][0]["accepted_pectoral_missing_intervals"] == [[100, 103], [104, 105]]
    assert result["per_slot"][2]["identity_review_count"] == 5
    assert result["node_denominator"] == 45 and result["pectoral_denominator"] == 15
    assert not result["biological_ID_truth"] and not result["final_audit_complete"]
    assert before == {key: (a.dtype, a.shape, a.tobytes()) for key, a in data.items()}
    json.dumps(result, allow_nan=False)


def test_missing_runs_span_native_1660_boundary_and_keep_interval_edges():
    data = candidate(1670, 49800)
    data["observed"][1658:1663, 0, 0] = False
    data["observed"][:2, 2, 2] = False
    data["observed"][-3:, 2, 2] = False
    result = m.candidate_metrics(data)
    node = result["per_slot"][0]["nodes"][0]
    assert node["missing_intervals"] == [[51458, 51463]]
    assert node["longest_missing_frames"] == 5 and node["missing_count"] == 5
    assert result["per_slot"][2]["nodes"][2]["missing_intervals"] == [
        [49800, 49802],
        [51467, 51470],
    ]
    assert result["frame_interval"] == [49800, 51470]


@pytest.mark.parametrize("supported", [False, True])
def test_entire_missing_or_entire_supported_interval(supported):
    data = candidate(3)
    data["observed"][:] = supported
    refresh_coordinates(data)
    result = m.candidate_metrics(data)
    for slot in result["per_slot"]:
        for node in slot["nodes"]:
            assert node["missing_intervals"] == ([] if supported else [[100, 103]])
            assert node["supported_count"] == (3 if supported else 0)
            assert node["latent_adjacent_steps"] == dict(count=2, max_mm=1.0)
            assert node["both_supported_adjacent_steps"] == dict(
                count=2 if supported else 0, max_mm=1.0 if supported else None
            )


def test_step_statistics_use_adjacent_support_and_do_not_bridge_gap():
    data = candidate()
    data["pose_xyz"][:, 0, 0, 0] = [0, 1, 100, 102, 105]
    data["observed"][2, 0, 0] = False
    node = m.candidate_metrics(data)["per_slot"][0]["nodes"][0]
    assert node["latent_adjacent_steps"] == dict(count=4, max_mm=99.0)
    assert node["both_supported_adjacent_steps"] == dict(count=2, max_mm=3.0)
    assert node["missing_intervals"] == [[102, 103]]


def test_one_frame_has_no_fictitious_zero_length_step():
    data = candidate(1)
    node = m.candidate_metrics(data)["per_slot"][0]["nodes"][0]
    assert node["latent_adjacent_steps"] == dict(count=0, max_mm=None)
    assert node["both_supported_adjacent_steps"] == dict(count=0, max_mm=None)


@pytest.mark.parametrize(
    "drift",
    [
        lambda a: a.update(frame_idx=a["frame_idx"].astype(np.int32)),
        lambda a: a["frame_idx"].__setitem__(2, 103),
        lambda a: a.update(frame_idx=a["frame_idx"][::-1]),
        lambda a: a.update(observed=a["observed"].astype(np.uint8)),
        lambda a: a.update(identity_review=a["identity_review"].astype(np.int64)),
        lambda a: a["pose_xyz"].__setitem__((0, 0, 0, 0), np.inf),
        lambda a: a["xyz"].__setitem__((0, 0, 0), np.nan),
        lambda a: a["accepted_xyz"].__setitem__((0, 0), np.inf),
        lambda a: a["xyz"].__setitem__((0, 0), np.nan),
        lambda a: a["identity_review"].__setitem__((0, 0), True),
    ],
)
def test_candidate_schema_and_direct_support_drift_rejected(drift):
    data = candidate()
    drift(data)
    with pytest.raises(ValueError):
        m.candidate_metrics(data)


def test_empty_candidate_is_not_an_audited_native_interval():
    with pytest.raises(ValueError, match="nonempty contiguous"):
        m.candidate_metrics(candidate(0))


def block_arrays(first=100, count=2):
    shape = (count, 3, 3, 3)
    return dict(
        frame_idx=np.arange(first, first + count, dtype=np.int64),
        valid_source=np.zeros(shape, bool),
        ray_distance_mm=np.full(shape, 1e200, np.float64),
        native_pixel_error=np.full(shape, -1e200, np.float64),
    )


def save_block(root, name, arrays):
    path = root / name
    np.savez_compressed(path, **arrays)
    return dict(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )


def test_exact_cross_block_quantiles_ignore_invalid_extremes_and_count_nonfinite(tmp_path):
    a, b = block_arrays(), block_arrays(102)
    for item in (a, b):
        item["valid_source"][:, 0, 0, 0] = True
    a["ray_distance_mm"][:, 0, 0, 0] = [1, 2]
    b["ray_distance_mm"][:, 0, 0, 0] = [3, 4]
    a["native_pixel_error"][:, 0, 0, 0] = [10, 20]
    b["native_pixel_error"][:, 0, 0, 0] = [np.nan, np.inf]
    records = [save_block(tmp_path, "a.npz", a), save_block(tmp_path, "b.npz", b)]
    before = copy.deepcopy(records)
    result = m.residual_metrics(records, tmp_path)
    assert result["frame_interval"] == [100, 104] and result["native_frames"] == 4
    assert result["view_node_denominator"] == 108 and result["valid_source_count"] == 4
    ray, pixel = (result["metrics"][name] for name in m.METRICS)
    assert ray["finite_supported_count"] == 4 and ray["nonfinite_supported_count"] == 0
    np.testing.assert_allclose(ray["quantiles"], [1, 2.5, 3.7, 3.85, 3.97, 4], rtol=0, atol=1e-14)
    assert ray["max"] == 4.0
    assert pixel["finite_supported_count"] == 2 and pixel["nonfinite_supported_count"] == 2
    assert pixel["max"] == 20.0
    np.testing.assert_array_equal(
        pixel["quantiles"], np.quantile([10.0, 20.0], [0, 0.5, 0.9, 0.95, 0.99, 1])
    )
    combined = {key: np.concatenate([a[key], b[key]]) for key in a}
    merged = save_block(tmp_path, "merged.npz", combined)
    assert m.residual_metrics([merged], tmp_path)["metrics"] == result["metrics"]
    assert records == before
    for record in records:
        assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("valid", [False, True])
def test_no_finite_supported_residual_has_null_statistics(tmp_path, valid):
    arrays = block_arrays()
    arrays["valid_source"][:] = valid
    if valid:
        arrays["ray_distance_mm"][:] = np.nan
        arrays["native_pixel_error"][:] = -np.inf
    record = save_block(tmp_path, "empty.npz", arrays)
    result = m.residual_metrics([record], tmp_path)
    for metric in result["metrics"].values():
        assert metric["finite_supported_count"] == 0
        assert metric["nonfinite_supported_count"] == (54 if valid else 0)
        assert metric["max"] is None and metric["quantiles"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("next_first", [99, 100, 101, 103])
def test_residual_duplicate_overlap_gap_and_reverse_blocks_rejected(tmp_path, next_first):
    records = [
        save_block(tmp_path, "a.npz", block_arrays()),
        save_block(tmp_path, "b.npz", block_arrays(next_first)),
    ]
    with pytest.raises(ValueError, match="order/gap"):
        m.residual_metrics(records, tmp_path)


@pytest.mark.parametrize(
    "drift",
    [
        lambda a: a.update(extra=np.zeros(1)),
        lambda a: a.pop("native_pixel_error"),
        lambda a: a.update(frame_idx=np.array([100, 102], np.int64)),
        lambda a: a.update(frame_idx=a["frame_idx"].astype(np.int32)),
        lambda a: a.update(valid_source=a["valid_source"].astype(np.uint8)),
        lambda a: a.update(ray_distance_mm=a["ray_distance_mm"].astype(np.float32)),
        lambda a: a.update(native_pixel_error=a["native_pixel_error"][:, :, 0]),
    ],
)
def test_residual_archive_schema_rejected_before_statistics(tmp_path, drift):
    arrays = block_arrays()
    with np.errstate(over="ignore"):
        drift(arrays)
    record = save_block(tmp_path, "bad.npz", arrays)
    with pytest.raises(ValueError):
        m.residual_metrics([record], tmp_path)


def test_duplicate_npz_member_rejected_even_when_hash_is_current(tmp_path):
    record = save_block(tmp_path, "duplicate.npz", block_arrays())
    path = Path(record["path"])
    with zipfile.ZipFile(path, "a") as archive:
        data = archive.read("frame_idx.npy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr("frame_idx.npy", data)
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["size_bytes"] = path.stat().st_size
    with pytest.raises(ValueError, match="array schema"):
        m.residual_metrics([record], tmp_path)


def test_source_sha_rejects_changed_file_and_reports_do_not_rewrite_it(tmp_path):
    record = save_block(tmp_path, "source.npz", block_arrays())
    path = Path(record["path"])
    altered = path.read_bytes() + b"changed"
    path.write_bytes(altered)
    with pytest.raises(ValueError, match="SHA differs"):
        m.residual_metrics([record], tmp_path)
    assert path.read_bytes() == altered


def test_residual_paths_stay_within_output_and_relative_paths_work(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    record = save_block(root, "source.npz", block_arrays())
    record["path"] = "source.npz"
    assert m.residual_metrics([record], root)["native_frames"] == 2
    outside = save_block(tmp_path, "outside.npz", block_arrays())
    outside["path"] = "../outside.npz"
    with pytest.raises(ValueError, match="outside output"):
        m.residual_metrics([outside], root)


def test_empty_residual_sequence_and_global_cap_are_explicit(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="nonempty ordered"):
        m.residual_metrics([], tmp_path)
    record = save_block(tmp_path, "source.npz", block_arrays())
    monkeypatch.setattr(m, "MAX_SUPPORTED_VALUES", 27)
    with pytest.raises(ValueError, match="denominator exceeds cap"):
        m.residual_metrics([record], tmp_path)
