"""Tiny CPU graphs only: independent enumeration and frozen full-DAG bytes."""

import dataclasses
import hashlib
import json

import identity_streaming_node_memory as m
import numpy as np
import pytest
from identity_node_memory_dp import solve_identity_node_memory_dp as old_solve
from test_identity_terminal_conditioned import enumerate_conditions, snapshot, tiny_case


def old_result(xyz, direct, unary):
    return old_solve(
        np.arange(100, 100 + len(xyz), dtype=np.int64),
        unary,
        xyz,
        direct,
        node_indices=(1,) if direct.shape[-1] == 1 else (1, 0),
    )


def solve(tmp_path, xyz, direct, unary, **kwargs):
    return m.solve_streaming_identity_node_memory_dp(
        np.arange(100, 100 + len(xyz), dtype=np.int64),
        unary,
        xyz,
        direct,
        spool_root=tmp_path / "stream",
        node_indices=(1,) if direct.shape[-1] == 1 else (1, 0),
        **kwargs,
    )


def exact_result(actual, expected):
    assert type(actual) is type(expected) is m.frozen.IdentityNodeMemoryDPResult
    array_count = scalar_count = 0
    for field in dataclasses.fields(expected):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(right, np.ndarray):
            array_count += 1
            assert left.shape == right.shape and left.dtype == right.dtype, field.name
            assert left.tobytes(order="C") == right.tobytes(order="C"), field.name
            assert not left.flags.writeable, field.name
        else:
            scalar_count += 1
            assert type(left) is type(right), field.name
            if isinstance(right, float):
                assert np.asarray(left).tobytes() == np.asarray(right).tobytes(), field.name
            else:
                assert left == right, field.name
    assert (array_count, scalar_count) == (29, 19)


@pytest.mark.parametrize("seed,k", [(0, 1), (7, 1), (0, 2), (7, 2)])
@pytest.mark.parametrize("target", [1, 3, 999])
def test_every_output_and_independent_exhaustive_costs(tmp_path, monkeypatch, seed, k, target):
    xyz, direct, unary = tiny_case(seed, k, length=5)
    # One true separator and a terminal partial-memory suffix.
    direct[2] = True
    xyz[2] = np.arange(3 * k * 3).reshape(3, k, 3) / 5
    direct[-1, 0, 0] = False
    xyz[-1, 0, 0] = np.nan
    unary[1, 2] = np.inf
    unary[-1, 4] = np.inf
    expected = old_result(xyz, direct, unary)
    before = [value.tobytes() for value in (xyz, direct, unary)]

    def forbidden(*args, **kwargs):
        raise AssertionError("streaming helper cannot call the complete original solver")

    monkeypatch.setattr(m.frozen, "solve_identity_node_memory_dp", forbidden)
    streamed = solve(tmp_path, xyz, direct, unary, target_block_frames=target)
    exact_result(streamed.result, expected)
    transfer, constrained, backward = enumerate_conditions(expected, xyz, direct, unary)
    np.testing.assert_allclose(streamed.result.path_cost, transfer.min(), atol=1e-12, rtol=1e-13)
    np.testing.assert_allclose(
        streamed.result.min_marginal_cost, constrained.min(axis=2), atol=1e-12, rtol=1e-13
    )
    np.testing.assert_allclose(
        streamed.result.backward_cost, backward.min(axis=1), atol=1e-12, rtol=1e-13
    )
    assert before == [value.tobytes() for value in (xyz, direct, unary)]
    assert sum(b["owned_states"] for b in streamed.metadata["blocks"]) == len(
        expected.state_permutation
    )
    assert streamed.metadata["global_edges"] == len(expected.edge_source)
    assert streamed.metadata["intermediate_ID_reset"] is False
    assert streamed.metadata["global_caps_reinterpreted"] is False
    assert streamed.metadata["actual_medaka_qualified"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "ties",
        "huge_offset",
        "initial_forbidden",
        "middle_forbidden",
        "terminal_forbidden",
        "no_separator",
    ],
)
def test_centering_ties_forbidden_and_terminal_memory(tmp_path, kind):
    xyz, direct, unary = tiny_case(3, length=4)
    if kind == "ties":
        xyz[:] = 0
        direct[:] = True
        unary[:] = 0
    elif kind == "huge_offset":
        unary += 1e14
    elif kind == "initial_forbidden":
        unary[0, 0] = np.inf
    elif kind == "middle_forbidden":
        unary[1] = np.inf
        direct[1] = True
        xyz[1] = 0
    elif kind == "terminal_forbidden":
        unary[-1] = np.inf
    else:
        direct[1:] = False
        xyz[1:] = np.nan
    expected = old_result(xyz, direct, unary)
    streamed = solve(tmp_path, xyz, direct, unary, target_block_frames=1)
    exact_result(streamed.result, expected)
    if kind == "huge_offset":
        assert np.any(
            expected.forward_cost + expected.backward_cost != expected.node_min_marginal_cost
        )
    if kind == "ties":
        assert np.all(streamed.result.path_state == 0)
        assert np.all(streamed.result.optimal_state_mask[1:])
    if kind == "no_separator":
        assert streamed.metadata["block_count"] == 1
        assert streamed.result.terminal_unsettled_gap.all()
    if "forbidden" in kind:
        assert streamed.result.status == "infeasible"
        assert np.all(streamed.result.path_state == -1)


@pytest.mark.parametrize("initial", [0.0, 3.75, np.inf])
def test_one_frame_has_one_initial_state_no_edges(tmp_path, initial):
    xyz, direct, unary = np.zeros((1, 3, 2, 3)), np.ones((1, 3, 2), bool), np.ones((1, 6))
    unary[0, 0] = initial
    expected = old_result(xyz, direct, unary)
    streamed = solve(tmp_path, xyz, direct, unary)
    exact_result(streamed.result, expected)
    assert streamed.metadata["global_states"] == 1 and streamed.metadata["global_edges"] == 0
    assert streamed.result.edge_delta_frames.shape == (0, 3, 2)


def test_all_six_actual_frontier_costs_and_stable_global_edges_are_kept(tmp_path):
    xyz = np.zeros((4, 3, 2, 3))
    xyz[..., 0] = np.arange(3)[None, :, None] * 2
    direct, unary = np.ones((4, 3, 2), bool), np.zeros((4, 6))
    unary[0, 0], unary[1, 3] = 9.0, np.inf
    streamed = solve(tmp_path, xyz, direct, unary, target_block_frames=1)
    exact_result(streamed.result, old_result(xyz, direct, unary))
    blocks = streamed.metadata["blocks"]
    assert len(blocks) == 3 and blocks[0]["owned_states"] == 7
    assert [b["owned_states"] for b in blocks[1:]] == [6, 6]
    with np.load(tmp_path / "stream" / blocks[0]["path"], allow_pickle=False) as z:
        frontier = z["centered_forward"][-6:]
        assert np.isposinf(frontier[3]) and len(np.unique(frontier)) > 2
    with np.load(tmp_path / "stream" / blocks[1]["path"], allow_pickle=False) as z:
        np.testing.assert_array_equal(z["source"], np.repeat(np.arange(1, 7), 6))
        np.testing.assert_array_equal(z["state"][:, 0], np.arange(6))
        assert z["state"].shape == (6, 7)
    assert blocks[1]["imported_boundary"] == 1
    # Centering includes all six unary columns even though only P0 exists at
    # t0; the other five zeros make the offset zero, and F itself retains 9.
    assert blocks[0]["boundary_cumulative_offset"] == 0.0
    assert streamed.result.forward_cost[0] == 9.0


def test_manifest_all_logical_arrays_and_fresh_readonly_files(tmp_path):
    xyz, direct, unary = tiny_case()
    streamed = solve(tmp_path, xyz, direct, unary, target_block_frames=1)
    manifest = json.loads(streamed.manifest_path.read_text())
    assert len(manifest["logical_arrays"]) == 29 and len(manifest["scalars"]) == 19
    for name, row in manifest["files"].items():
        path = tmp_path / "stream" / name
        assert path.stat().st_size == row["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    for name, layout in manifest["logical_arrays"].items():
        path = tmp_path / "stream" / layout["path"]
        if layout["size_bytes"]:
            fresh = np.memmap(path, dtype=layout["dtype"], mode="r", shape=tuple(layout["shape"]))
            assert fresh.tobytes() == getattr(streamed.result, name).tobytes()
    total = sum(p.stat().st_size for p in (tmp_path / "stream").rglob("*") if p.is_file())
    assert total == streamed.metadata["storage_bytes_including_manifest"]
    assert not (tmp_path / "stream/manifest.pending.json").exists()
    assert not (tmp_path / "stream/failure.json").exists()


@pytest.mark.parametrize(
    "field,limit,resource",
    [
        ("global_state_limit", 12, "total_states"),
        ("global_edge_limit", 41, "total_edges"),
        ("max_block_states", 6, "block_states"),
        ("max_block_array_bytes", 71, "block_array_bytes"),
        ("max_storage_bytes", 1024, "storage_bytes"),
    ],
)
def test_explicit_resource_limits_stop_without_result_or_hidden_reset(
    tmp_path, field, limit, resource
):
    xyz, direct, unary = np.zeros((4, 3, 2, 3)), np.ones((4, 3, 2), bool), np.zeros((4, 6))
    limits = dataclasses.replace(m.DEFAULT_RESOURCES, **{field: limit})
    with pytest.raises(m.StreamingCapacityError) as caught:
        solve(tmp_path, xyz, direct, unary, resources=limits, target_block_frames=1)
    assert caught.value.resource == resource
    if field == "max_storage_bytes":
        assert caught.value.limit == limits.max_storage_bytes
    assert not (tmp_path / "stream/manifest.json").exists()
    if field == "global_state_limit":
        assert caught.value.attempted == 13  # all earlier blocks still count
    if field == "global_edge_limit":
        assert caught.value.attempted == 42
    total = sum(p.stat().st_size for p in (tmp_path / "stream").rglob("*") if p.is_file())
    assert total <= limits.max_storage_bytes


def test_fixed_layer_limit_is_not_a_per_block_override(tmp_path, monkeypatch):
    xyz, direct, unary = np.zeros((2, 3, 2, 3)), np.ones((2, 3, 2), bool), np.zeros((2, 6))

    def too_many(*args):
        return (np.zeros((100_001, 7), np.int64), *(np.empty(0) for _ in range(5)))

    monkeypatch.setattr(m.frozen, "_next_layer", too_many)
    with pytest.raises(m.StreamingCapacityError) as caught:
        solve(tmp_path, xyz, direct, unary)
    assert caught.value.resource == "layer_states" and caught.value.limit == 100_000


@pytest.mark.parametrize(
    "fault", ["frames", "initial", "nodes", "unary", "direct", "xyz", "resource", "target"]
)
def test_input_schema_refusal_precedes_root_creation(tmp_path, fault):
    xyz, direct, unary = tiny_case()
    frames, options = np.arange(100, 104), {}
    if fault == "frames":
        frames[-1] += 1
    elif fault == "initial":
        options["initial_state"] = 1
    elif fault == "nodes":
        options["node_indices"] = (0, 1)
    elif fault == "unary":
        unary[1, 2] = -np.inf
    elif fault == "direct":
        direct[0, 1, 0] = False
    elif fault == "xyz":
        xyz[0, 0, 0] = np.nan
    elif fault == "resource":
        options["resources"] = {}
    else:
        options["target_block_frames"] = True
    with pytest.raises(ValueError):
        m.solve_streaming_identity_node_memory_dp(
            frames, unary, xyz, direct, spool_root=tmp_path / "stream", **options
        )
    assert not (tmp_path / "stream").exists()


def test_readonly_strided_inputs_preserved_and_existing_root_refused(tmp_path):
    xyz, direct, unary = tiny_case()
    backing = np.empty((4, 12))
    backing[:, ::2] = unary
    unary = backing[:, ::2]
    for value in (xyz, direct, unary):
        value.flags.writeable = False
    expected = old_result(xyz, direct, unary)
    before = snapshot(expected)
    streamed = solve(tmp_path, xyz, direct, unary)
    exact_result(streamed.result, expected)
    assert snapshot(expected) == before
    with pytest.raises(ValueError, match="new spool"):
        solve(tmp_path, xyz, direct, unary)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, np.int64(1)])
def test_resource_contract_values_are_explicit_positive_python_integers(value):
    with pytest.raises(ValueError):
        m.StreamingResources(global_state_limit=value)


def test_changed_forward_block_rejected_before_global_result(tmp_path, monkeypatch):
    xyz, direct, unary = tiny_case()
    original = m._assemble

    def changed(storage, blocks, *args):
        path = storage.root / blocks[0]["path"]
        with path.open("r+b") as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(bytes([first[0] ^ 1]))
        return original(storage, blocks, *args)

    monkeypatch.setattr(m, "_assemble", changed)
    with pytest.raises(ValueError, match="block bytes changed"):
        solve(tmp_path, xyz, direct, unary)
    root = tmp_path / "stream"
    assert not (root / "manifest.json").exists()
    assert json.loads((root / "failure.json").read_text())["completed_result_available"] is False


def test_late_source_drift_after_readonly_mapping_retains_failed_status(tmp_path, monkeypatch):
    xyz, direct, unary = tiny_case()
    original, calls = m._sha, 0

    def drift(path):
        nonlocal calls
        if str(path) == m.__file__:
            calls += 1
            if calls > 1:
                return "0" * 64
        return original(path)

    monkeypatch.setattr(m, "_sha", drift)
    with pytest.raises(ValueError, match="source changed during"):
        solve(tmp_path, xyz, direct, unary)
    root = tmp_path / "stream"
    assert not (root / "manifest.json").exists()
    assert json.loads((root / "failure.json").read_text())["status"] == "failed"


def test_explicit_global_contract_is_stored_without_per_block_reset(tmp_path):
    xyz, direct, unary = np.zeros((5, 3, 2, 3)), np.ones((5, 3, 2), bool), np.zeros((5, 6))
    resources = dataclasses.replace(
        m.DEFAULT_RESOURCES, global_state_limit=25, global_edge_limit=114
    )
    actual = solve(tmp_path, xyz, direct, unary, resources=resources, target_block_frames=1)
    expected = old_solve(
        np.arange(100, 105), unary, xyz, direct, max_total_states=25, max_total_edges=114
    )
    exact_result(actual.result, expected)
    assert actual.metadata["global_states"] == 25 and actual.metadata["global_edges"] == 114
