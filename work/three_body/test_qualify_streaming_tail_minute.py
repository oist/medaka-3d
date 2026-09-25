"""Small CPU checks of the real-minute caller's strict comparison and source gates."""

import dataclasses
import io
import json

import numpy as np
import pytest
import qualify_streaming_tail_minute as q


@pytest.fixture
def compared():
    frames = np.arange(4, dtype=np.int64)
    xyz = np.arange(72, dtype=np.float64).reshape(4, 3, 2, 3) / 4
    direct = np.ones((4, 3, 2), bool)
    direct[2, 0, 0] = False
    xyz[2, 0, 0] = np.nan
    result = q.streaming.frozen.solve_identity_node_memory_dp(
        frames, np.zeros((4, 6)), xyz, direct, node_indices=(1, 0)
    )
    arrays, scalars, readonly = {}, {}, {}
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            arrays[field.name] = value.copy()
            readonly[field.name] = value.copy()
            readonly[field.name].flags.writeable = False
        else:
            scalars[field.name] = list(value) if isinstance(value, tuple) else value
    result = dataclasses.replace(result, **readonly)
    return result, arrays, scalars, (frames, xyz, direct)


def test_all_fields_exact(compared):
    result, arrays, scalars, _ = compared
    report = q.compare_result(result, arrays, scalars)
    assert report["all_exact"]
    assert report["array_count"] == len(report["arrays"]) == 29
    assert report["scalar_count"] == len(report["scalars"]) == 19


@pytest.mark.parametrize("field", ["arrays", "scalars"])
def test_rejects_missing_field(compared, field):
    result, arrays, scalars, _ = compared
    container = arrays if field == "arrays" else scalars
    container.pop(next(iter(container)))
    with pytest.raises(ValueError, match="all 29 arrays and 19 scalars"):
        q.compare_result(result, arrays, scalars)


def test_rejects_dtype_even_with_equal_values(compared):
    result, arrays, scalars, _ = compared
    arrays["frame_idx"] = arrays["frame_idx"].astype(np.int32)
    with pytest.raises(ValueError, match="frame_idx logical bytes"):
        q.compare_result(result, arrays, scalars)


def test_rejects_nan_payload_change(compared):
    result, arrays, scalars, _ = compared
    value = arrays["edge_squared_distance_mm2"]
    where = np.flatnonzero(np.isnan(value.reshape(-1)))[0]
    value.reshape(-1).view(np.uint64)[where] ^= np.uint64(1)
    assert np.array_equal(value, result.edge_squared_distance_mm2, equal_nan=True)
    with pytest.raises(ValueError, match="edge_squared_distance_mm2 logical bytes"):
        q.compare_result(result, arrays, scalars)


def test_rejects_scalar_single_ulp_difference(compared):
    result, arrays, scalars, _ = compared
    scalars["path_cost"] = float(np.nextafter(scalars["path_cost"], np.inf))
    with pytest.raises(ValueError, match="path_cost float64 bytes"):
        q.compare_result(result, arrays, scalars)


def test_rejects_bool_as_integer(compared):
    result, arrays, scalars, _ = compared
    scalars["initial_state"] = False
    with pytest.raises(ValueError, match="initial_state scalar type"):
        q.compare_result(result, arrays, scalars)


def test_rejects_writable_result(compared):
    result, arrays, scalars, _ = compared
    result = dataclasses.replace(result, frame_idx=result.frame_idx.copy())
    with pytest.raises(ValueError, match="frame_idx result must be read-only"):
        q.compare_result(result, arrays, scalars)


def test_rejects_noncontiguous_layout():
    with pytest.raises(ValueError, match="C-contiguous"):
        q.array_signature(np.arange(12).reshape(3, 4).T)


def test_refuses_object_archives():
    stream = io.BytesIO()
    np.savez(stream, sample=np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        q.read_arrays(stream.getvalue())


@pytest.fixture
def metadata():
    plan = dict(
        schema="tail_minute_mouth_memory_motion_v1",
        native_interval=[q.START, q.STOP],
        node_indices=[1, 0],
        geometry="tail",
        dp_config=q.proof.DP_CONFIG.copy(),
        initial_P=0,
        unary="float64 exact zero T,6",
    )
    summary = dict(
        schema=plan["schema"],
        status="ok",
        native_interval=plan["native_interval"].copy(),
        native_frames=q.STOP - q.START,
        node_indices=[1, 0],
        geometry="tail",
        raw_geometry_sources_review_modified=False,
        RGB_fit_GPU_executed=False,
        report=dict(all_three_ID_truth=False),
    )
    return plan, summary


def test_fixed_metadata(metadata):
    q.validate_metadata(*metadata)


@pytest.mark.parametrize(
    "key,value", [("max_total_states", 2_000_000), ("weight", 0.3), ("tie_rtol", 1e-8)]
)
def test_rejects_changed_caps_or_objective(metadata, key, value):
    plan, summary = metadata
    plan["dp_config"][key] = value
    with pytest.raises(ValueError, match="unchanged full-DAG caps"):
        q.validate_metadata(plan, summary)


def test_rejects_shortened_interval(metadata):
    plan, summary = metadata
    summary["native_interval"][1] -= 1
    with pytest.raises(ValueError, match="same successful saved Tail minute"):
        q.validate_metadata(plan, summary)


@pytest.mark.parametrize("damage", [None, "layout", "payload", "extra"])
def test_spool_verification(tmp_path, monkeypatch, compared, damage):
    frames, xyz, direct = compared[3]
    monkeypatch.setattr(q, "TARGET_BLOCK_FRAMES", 1)
    result = q.streaming.solve_streaming_identity_node_memory_dp(
        frames,
        np.zeros((4, 6)),
        xyz,
        direct,
        spool_root=tmp_path / "spool",
        target_block_frames=1,
        node_indices=(1, 0),
    )
    if damage == "layout":
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        manifest["logical_arrays"]["frame_idx"]["dtype"] = "<i4"
        result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "payload":
        with (tmp_path / "spool/blocks/b000000.npz").open("ab") as stream:
            stream.write(b"x")
    elif damage == "extra":
        (tmp_path / "spool/unregistered.txt").write_text("extra", encoding="utf-8")
    if damage:
        with pytest.raises(ValueError, match=r"layout differs|member changed|membership differs"):
            q.verify_spool(tmp_path / "spool", result)
    else:
        metadata, files = q.verify_spool(tmp_path / "spool", result)
        assert metadata["block_count"] > 1
        assert result.manifest_path in files
