"""Exact memory-DAG arithmetic with separator commits and disk-backed output.

This is a CPU memory architecture experiment, not online final identity. Later
blocks import the actual six centered forward states, including infinite ones.
All logical states/edges remain counted globally. Array/disk budgets are not an
OS RSS guarantee. Only tests may compare against the frozen whole-DAG solver.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from contextlib import suppress
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[2] / "labs/20260909_three_body"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
import identity_node_memory_dp as frozen  # noqa: E402

SCHEMA = "identity_streaming_node_memory_v1"
MAX_LAYER_STATES = 100_000
FROZEN_SHA = "e5331f706137cc361154cc065caaa58f02686a53019c1971795adb5702e94dd6"
CORE_SHA = "2e848b6c4b14eebcd3d4d3791a4f61aae3e8f4e9fd0413588bd6a1548121e354"


@dataclass(frozen=True)
class StreamingResources:
    global_state_limit: int = 1_000_000
    global_edge_limit: int = 6_000_000
    max_block_states: int = 1_000_000
    max_block_array_bytes: int = 512 * 1024**2
    max_storage_bytes: int = 4 * 1024**3

    def __post_init__(self):
        for field in fields(self):
            if type(getattr(self, field.name)) is not int or getattr(self, field.name) < 1:
                raise ValueError("positive Python integer required: " + field.name)


@dataclass(frozen=True)
class StreamingResult:
    result: frozen.IdentityNodeMemoryDPResult
    metadata: dict
    manifest_path: Path


DEFAULT_RESOURCES = StreamingResources()


class StreamingCapacityError(frozen.IdentityMemoryCapacityError):
    """A registered global, block-array or disk limit stopped exact execution."""


def _limit(time, resource, value, maximum):
    if value > maximum:
        raise StreamingCapacityError(time, resource, value, maximum)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class _LimitedWriter:
    def __init__(self, stream, allowed, native, already_stored):
        self.stream, self.allowed, self.native = stream, allowed, native
        self.already_stored = already_stored

    def write(self, data):
        _limit(
            self.native,
            "storage_bytes",
            self.already_stored + self.stream.tell() + len(data),
            self.allowed,
        )
        return self.stream.write(data)

    def tell(self):
        return self.stream.tell()

    def seek(self, *args):
        return self.stream.seek(*args)

    def flush(self):
        return self.stream.flush()

    def read(self, *args):
        # NumPy recognizes file-like objects by the presence of read; the
        # exclusive output stream itself remains write-only.
        return self.stream.read(*args)


class _Storage:
    """Logical file-size quota; exclusive files under one newly created root."""

    def __init__(self, root, limit):
        self.root, self.limit = root, limit
        self.used, self.maps, self.layouts, self.files = 0, {}, {}, {}

    def write(self, relative, writer, native=-1):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            try:
                writer(_LimitedWriter(stream, self.limit, native, self.used))
            finally:
                stream.flush()
                size = path.stat().st_size
                self.used += size
                self.files[relative] = size
        return path

    def json(self, relative, value):
        data = json.dumps(value, sort_keys=True, allow_nan=False, indent=2).encode() + b"\n"
        return self.write(relative, lambda stream: stream.write(data))

    def allocate(self, name, shape, dtype):
        shape, dtype = tuple(int(x) for x in shape), np.dtype(dtype)
        size = math.prod(shape) * dtype.itemsize
        _limit(-1, "storage_bytes", self.used + size, self.limit)
        relative = "arrays/" + name + ".dat"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.truncate(size)
        value = (
            np.memmap(path, dtype=dtype, mode="r+", shape=shape) if size else np.empty(shape, dtype)
        )
        self.used += size
        self.files[relative] = size
        self.maps[name] = value
        self.layouts[name] = dict(
            path=relative, shape=list(shape), dtype=dtype.str, size_bytes=size
        )
        return value

    def array(self, name, value):
        result = self.allocate(name, value.shape, value.dtype)
        result[...] = value
        return result

    def flush(self):
        for value in self.maps.values():
            if isinstance(value, np.memmap):
                value.flush()

    def readonly(self):
        self.flush()
        result = {}
        for name, value in self.maps.items():
            layout = self.layouts[name]
            if isinstance(value, np.memmap):
                value._mmap.close()
                value = np.memmap(
                    self.root / layout["path"],
                    dtype=layout["dtype"],
                    mode="r",
                    shape=tuple(layout["shape"]),
                )
            else:
                value.flags.writeable = False
            result[name] = value
        self.maps = result
        return result


def _inputs(frame_idx, unary, node_xyz, direct_nodes, node_indices, initial_state):
    frames = frozen._frames(frame_idx)
    if (
        isinstance(initial_state, (bool, np.bool_))
        or not isinstance(initial_state, (int, np.integer))
        or initial_state != 0
    ):
        raise ValueError("fixed initial P012 required")
    if (
        type(node_indices) is not tuple
        or node_indices not in ((1,), (1, 0))
        or any(type(x) is not int for x in node_indices)
    ):
        raise ValueError("original chest/mouth node order required")
    n, k = len(frames), len(node_indices)
    xyz, observed = np.asarray(node_xyz), np.asarray(direct_nodes)
    if xyz.shape != (n, 3, k, 3) or xyz.dtype.kind not in "fiu":
        raise ValueError("real T,3,K,3 coordinates required")
    if observed.shape != (n, 3, k) or observed.dtype != bool or not observed[0].all():
        raise ValueError("boolean T,3,K with initially direct nodes required")
    xyz = xyz.astype(np.float64, copy=True)
    if not np.isfinite(xyz[observed]).all():
        raise ValueError("every directly observed coordinate must be finite")
    emission = frozen._costs(unary, (n, 6), "unary")
    return frames, emission, xyz, observed.copy()


def _forward(frames, emission, observed, xyz, scale, factor, resources, target_frames, storage):
    n, k = len(frames), observed.shape[2]
    previous = np.r_[0, np.repeat(np.arange(3), k)][None].astype(np.int64)
    forward = np.array([emission[0, 0]])
    layers, edges = [0, 1], [0]
    minima, spans = np.zeros(n), np.zeros(n)
    blocks, pending = [], {}
    names = (
        "state",
        "centered_forward",
        "parent",
        "source",
        "target",
        "motion",
        "delta",
        "distance",
    )
    empty = {
        "source": ((0,), np.int64),
        "target": ((0,), np.int64),
        "motion": ((0,), np.float64),
        "delta": ((0, 3, k), np.int64),
        "distance": ((0, 3, k), np.float64),
    }
    start, first_node, first_edge, pending_states, pending_bytes = 0, 0, 0, 0, 0
    peak_states = peak_bytes = 0

    def append(time, values):
        nonlocal pending_states, pending_bytes, peak_states, peak_bytes
        states = len(values["state"])
        count = pending_states + states
        size = pending_bytes + sum(value.nbytes for value in values.values())
        _limit(time, "block_states", count, resources.max_block_states)
        _limit(time, "block_array_bytes", size, resources.max_block_array_bytes)
        for name, value in values.items():
            pending.setdefault(name, []).append(value)
        pending_states, pending_bytes = count, size
        peak_states, peak_bytes = max(peak_states, count), max(peak_bytes, size)

    def commit(time):
        nonlocal start, first_node, first_edge, pending_states, pending_bytes, pending
        arrays = {
            name: np.concatenate(pending[name]) if name in pending else np.empty(*empty[name])
            for name in names
        }
        relative = f"blocks/b{len(blocks):06d}.npz"
        path = storage.write(relative, lambda stream: np.savez(stream, **arrays), time)
        blocks.append(
            dict(
                path=relative,
                sha256=_sha(path),
                first_time=start,
                stop_time=time + 1,
                first_node=first_node,
                stop_node=layers[-1],
                first_edge=first_edge,
                stop_edge=edges[-1],
                array_bytes=pending_bytes,
                owned_states=pending_states,
                starts_with_global_initial=start == 0,
                imported_boundary=None if start == 0 else start - 1,
                terminal_fully_direct=bool(observed[time].all()),
            )
        )
        start, first_node, first_edge = time + 1, layers[-1], edges[-1]
        pending_states = pending_bytes = 0
        pending = {}

    _limit(0, "total_states", 1, resources.global_state_limit)
    append(0, dict(state=previous, centered_forward=forward, parent=np.array([-1], np.int64)))
    for t in range(1, n):
        attempted = edges[-1] + 6 * len(previous)
        _limit(t, "total_edges", attempted, resources.global_edge_limit)
        current, source, target, motion, delta, distance = frozen._next_layer(
            previous, t, observed, xyz, scale, factor
        )
        _limit(t, "layer_states", len(current), MAX_LAYER_STATES)
        _limit(t, "total_states", layers[-1] + len(current), resources.global_state_limit)
        if observed[t].all() and len(current) != 6:
            raise AssertionError("fully direct layer must contain all six structural states")
        minima[t] = motion.min()
        centered = motion - minima[t]
        spans[t] = centered.max()
        candidates = forward[source] + centered
        best_prefix = np.full(len(current), np.inf)
        np.minimum.at(best_prefix, target, candidates)
        parents = np.full(len(current), len(candidates), np.int64)
        is_best = candidates == best_prefix[target]
        np.minimum.at(parents, target[is_best], np.flatnonzero(is_best))
        parents = source[parents] + layers[-2]
        forward = best_prefix + emission[t, current[:, 0]]
        append(
            t,
            dict(
                state=current,
                centered_forward=forward,
                parent=parents,
                source=source + layers[-2],
                target=target + layers[-1],
                motion=motion,
                delta=delta,
                distance=distance,
            ),
        )
        layers.append(layers[-1] + len(current))
        edges.append(attempted)
        previous = current
        if observed[t].all() and t - start + 1 >= target_frames:
            commit(t)
    if pending:
        commit(n - 1)
    return (
        np.asarray(layers, np.int64),
        np.asarray(edges, np.int64),
        minima,
        spans,
        blocks,
        peak_states,
        peak_bytes,
    )


def _assemble(storage, blocks, layers, edges, k, offsets):
    n, e = int(layers[-1]), int(edges[-1])
    specifications = {
        "state_permutation": ((n,), "i8"),
        "state_last_event": ((n, 3, k), "i8"),
        "edge_source": ((e,), "i8"),
        "edge_target": ((e,), "i8"),
        "edge_cost": ((e,), "f8"),
        "edge_delta_frames": ((e, 3, k), "i8"),
        "edge_squared_distance_mm2": ((e, 3, k), "f8"),
        "forward_cost": ((n,), "f8"),
        "backward_cost": ((n,), "f8"),
        "node_min_marginal_cost": ((n,), "f8"),
        "_centered_forward": ((n,), "f8"),
        "_parent": ((n,), "i8"),
    }
    values = {name: storage.allocate(name, *spec) for name, spec in specifications.items()}
    values["backward_cost"][:] = np.inf
    for block in blocks:
        path = storage.root / block["path"]
        if _sha(path) != block["sha256"]:
            raise ValueError("forward block bytes changed")
        sl, el = (
            slice(block["first_node"], block["stop_node"]),
            slice(block["first_edge"], block["stop_edge"]),
        )
        with np.load(path, allow_pickle=False) as archive:
            nodes = archive["state"]
            values["state_permutation"][sl] = nodes[:, 0]
            values["state_last_event"][sl] = nodes[:, 1:].reshape(-1, 3, k)
            centered = archive["centered_forward"]
            values["_centered_forward"][sl] = centered
            times = np.repeat(
                np.arange(block["first_time"], block["stop_time"]),
                np.diff(layers)[block["first_time"] : block["stop_time"]],
            )
            values["forward_cost"][sl] = centered + offsets[times]
            values["_parent"][sl] = archive["parent"]
            for target, source in (
                ("edge_source", "source"),
                ("edge_target", "target"),
                ("edge_cost", "motion"),
                ("edge_delta_frames", "delta"),
                ("edge_squared_distance_mm2", "distance"),
            ):
                values[target][el] = archive[source]
    return values


def _reverse(values, layers, edges, emission, minima, offsets):
    backward, forward = values["backward_cost"], values["_centered_forward"]
    p, n = values["state_permutation"], len(emission)
    backward[layers[-2] :] = 0
    for t in range(n - 1, 0, -1):
        span = slice(edges[t - 1], edges[t])
        source, target = values["edge_source"][span], values["edge_target"][span]
        centered = values["edge_cost"][span] - minima[t]
        cost = centered + emission[t, p[target]] + backward[target]
        np.minimum.at(backward, source, cost)
    best = float(forward[layers[-2] :].min())
    total = float(offsets[-1])
    marginal = np.full((n, 6), np.inf)
    for t in range(n):
        span = slice(layers[t], layers[t + 1])
        node = forward[span] + backward[span]
        np.minimum.at(marginal[t], p[span], node)
        values["node_min_marginal_cost"][span] = node + total
        backward[span] = backward[span] + (total - offsets[t])
    return best, marginal


def _derived(frames, k, layers, values, marginal, best, tolerance):
    """Original final path/tie arithmetic; only node columns come from mmap."""
    n = len(frames)
    path, path_nodes = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    mapping = np.full((n, 3), -1, np.int64)
    terminal_frame, terminal_slot, terminal_age = (np.full((3, k), -1, np.int64) for _ in range(3))
    unsettled, clamped = np.zeros((3, k), bool), 0
    if not np.isfinite(best):
        status = "infeasible"
        deltas, optimum = np.full((n, 6), np.nan), np.zeros((n, 6), bool)
        id_deltas = np.full((n, 3, 3), np.nan)
        state_gap, id_gap = np.full(n, np.nan), np.full((n, 3), np.nan)
        state_status, id_status = (
            np.full(n, "infeasible", "U32"),
            np.full((n, 3), "infeasible", "U32"),
        )
        global_gap, global_status = np.nan, "infeasible"
    else:
        status = "optimal"
        path_nodes[-1] = layers[-2] + np.argmin(values["_centered_forward"][layers[-2] :])
        for t in range(n - 1, 0, -1):
            path_nodes[t - 1] = values["_parent"][path_nodes[t]]
        path[:] = values["state_permutation"][path_nodes]
        mapping[:] = frozen.PERMUTATIONS[path]
        deltas = np.full((n, 6), np.inf)
        finite = np.isfinite(marginal)
        deltas[finite] = marginal[finite] - best
        if np.any(deltas < -tolerance):
            raise FloatingPointError("min-marginal lies below optimal cost beyond tolerance")
        negative = deltas < 0
        clamped = int(negative.sum())
        deltas[negative] = 0.0
        if np.any(deltas[np.arange(n), path] > tolerance):
            raise FloatingPointError("chosen path disagrees with forward/backward costs")
        optimum = deltas <= tolerance
        alternative = deltas.copy()
        alternative[np.arange(n), path] = np.inf
        state_gap = alternative.min(axis=1)
        id_deltas = np.empty((n, 3, 3))
        for identity, slot in np.ndindex(3, 3):
            id_deltas[:, identity, slot] = deltas[:, frozen.PERMUTATIONS[:, identity] == slot].min(
                axis=1
            )
        id_alt = id_deltas.copy()
        id_alt[np.arange(n)[:, None], np.arange(3)[None], mapping] = np.inf
        id_gap = id_alt.min(axis=2)
        state_status, id_status = (
            frozen._alternative_status(state_gap, tolerance),
            frozen._alternative_status(id_gap, tolerance),
        )
        state_status[0], id_status[0] = "fixed_initial_constraint", "fixed_initial_constraint"
        global_gap = float(state_gap.min())
        global_status = str(frozen._alternative_status(np.array(global_gap), tolerance))
        terminal_event = values["state_last_event"][path_nodes[-1]]
        terminal_frame, terminal_slot = frames[terminal_event // 3], terminal_event % 3
        terminal_age = frames[-1] - terminal_frame
        unsettled = terminal_age > 0
    return dict(
        status=status,
        path_state=path,
        permutation_by_id=mapping,
        path_node=path_nodes,
        min_marginal_delta=deltas,
        optimal_state_mask=optimum,
        id_slot_min_marginal_delta=id_deltas,
        state_alternative_gap=state_gap,
        identity_alternative_gap=id_gap,
        state_alternative_status=state_status,
        identity_alternative_status=id_status,
        global_alternative_gap=global_gap,
        global_alternative_status=global_status,
        terminal_last_event_frame=terminal_frame,
        terminal_last_event_slot=terminal_slot,
        terminal_age_frames=terminal_age,
        terminal_unsettled_gap=unsettled,
        roundoff_clamped_deltas=clamped,
    )


def _scalar_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return dict(float64_bytes=np.asarray(value, np.float64).tobytes().hex())
    return value


def solve_streaming_identity_node_memory_dp(
    frame_idx,
    unary,
    node_xyz,
    direct_nodes,
    *,
    spool_root,
    resources=DEFAULT_RESOURCES,
    target_block_frames=256,
    node_indices=(1, 0),
    scale_mm=0.8,
    weight=0.2,
    initial_state=0,
    tie_atol=1e-10,
    tie_rtol=1e-12,
):
    """Return the real frozen result dataclass plus separate storage metadata.

    Successful arrays are owned by read-only files in spool_root, whose lifetime
    must outlive the result. No temporary cleanup or original-file modification
    is performed. A fresh root is required. Limits count the global logical DAG,
    current block payload arrays (not RSS/NumPy temporaries), and all file sizes.
    Whole-problem infeasibility is a completed exact result, not execution failure.
    """
    if type(resources) is not StreamingResources:
        raise ValueError("explicit StreamingResources required")
    if type(target_block_frames) is not int or target_block_frames < 1:
        raise ValueError("positive target_block_frames required")
    frames, unary_raw, xyz, observed = _inputs(
        frame_idx, unary, node_xyz, direct_nodes, node_indices, initial_state
    )
    scale, factor = (
        frozen._scalar(scale_mm, "scale_mm", positive=True),
        frozen._scalar(weight, "weight"),
    )
    atol, rtol = frozen._scalar(tie_atol, "tie_atol"), frozen._scalar(tie_rtol, "tie_rtol")
    if _sha(frozen.__file__) != FROZEN_SHA:
        raise ValueError("frozen memory solver source changed")
    if _sha(LAB / "identity_dp.py") != CORE_SHA:
        raise ValueError("frozen centering/tie source changed")
    code = {
        str(Path(__file__).resolve()): _sha(__file__),
        str(Path(frozen.__file__).resolve()): FROZEN_SHA,
        str(LAB / "identity_dp.py"): CORE_SHA,
    }
    root = Path(spool_root).resolve()
    if root.exists():
        raise ValueError("new spool root required")
    root.mkdir(parents=True)
    storage = _Storage(root, resources.max_storage_bytes)
    try:
        with np.errstate(over="raise", invalid="raise"):
            emission, unary_offsets, unary_spans = frozen._center_cost_blocks(unary_raw, (1,))
            layers, edges, minima, spans, blocks, peak_states, peak_bytes = _forward(
                frames,
                emission,
                observed,
                xyz,
                scale,
                factor,
                resources,
                target_block_frames,
                storage,
            )
            offsets = np.cumsum(unary_offsets + minima)
            total_offset = float(offsets[-1])
            cost_span = float(unary_spans.sum() + spans.sum())
            tolerance = float(atol + rtol * cost_span)
            for block in blocks:
                boundary = block["stop_time"] - 1
                block.update(
                    boundary_native=int(frames[boundary]),
                    boundary_cumulative_offset=float(offsets[boundary]),
                    terminal_global_nodes=[int(layers[boundary]), int(layers[boundary + 1])],
                )
            values = _assemble(storage, blocks, layers, edges, len(node_indices), offsets)
            best, marginal = _reverse(values, layers, edges, emission, minima, offsets)
            logical = _derived(frames, len(node_indices), layers, values, marginal, best, tolerance)
            logical.update(
                {name: value for name, value in values.items() if not name.startswith("_")}
            )
            logical.update(
                frame_idx=frames,
                permutations=frozen.PERMUTATIONS.copy(),
                layer_offsets=layers,
                edge_offsets=edges,
                min_marginal_cost=marginal + total_offset,
                path_cost=float(np.float64(best) + total_offset),
                numerical_tie_tolerance=tolerance,
                common_cost_offset=total_offset,
                numerical_cost_span=cost_span,
                initial_state=0,
                scale_mm=scale,
                weight=factor,
                max_layer_states=MAX_LAYER_STATES,
                max_total_states=resources.global_state_limit,
                max_total_edges=resources.global_edge_limit,
                node_indices=node_indices,
            )
        for name, value in logical.items():
            if isinstance(value, np.ndarray) and name not in storage.maps:
                storage.array(name, value)
        for name, value in dict(
            _unary_offsets=unary_offsets,
            _unary_spans=unary_spans,
            _edge_minima=minima,
            _edge_spans=spans,
            _global_offsets=offsets,
        ).items():
            storage.array(name, value)
        metadata = dict(
            schema=SCHEMA,
            execution_status="complete",
            mathematical_status=logical["status"],
            blocks=blocks,
            block_count=len(blocks),
            global_states=int(layers[-1]),
            global_edges=int(edges[-1]),
            maximum_layer_states=int(np.diff(layers).max()),
            peak_block_states=peak_states,
            peak_block_array_bytes=peak_bytes,
            resources=asdict(resources),
            max_layer_states=MAX_LAYER_STATES,
            target_block_frames=target_block_frames,
            initial_states=1,
            separator_frontier_states=6,
            intermediate_ID_reset=False,
            global_caps_reinterpreted=False,
            resource_scope="logical block arrays and file sizes; not OS RSS",
            offsets="one full global np.cumsum",
            spans="two full global np.sum reductions",
            final_ID_online=False,
            actual_medaka_qualified=False,
            python=platform.python_version(),
            numpy=np.__version__,
            sources=code,
            inputs={
                name: dict(
                    shape=list(value.shape),
                    dtype=value.dtype.str,
                    sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                )
                for name, value in dict(
                    frame_idx=frames, unary=unary_raw, node_xyz=xyz, direct_nodes=observed
                ).items()
            },
            storage_bytes_before_manifest=storage.used,
        )
        readonly = storage.readonly()
        logical.update(
            {
                name: readonly[name]
                for name, value in logical.items()
                if isinstance(value, np.ndarray)
            }
        )
        result = frozen.IdentityNodeMemoryDPResult(**logical)
        array_names = {
            field.name
            for field in fields(result)
            if isinstance(getattr(result, field.name), np.ndarray)
        }
        scalars = {
            field.name: _scalar_json(getattr(result, field.name))
            for field in fields(result)
            if field.name not in array_names
        }
        manifest = dict(
            metadata=metadata,
            logical_arrays={
                name: layout for name, layout in storage.layouts.items() if name in array_names
            },
            auxiliary_arrays={
                name: layout for name, layout in storage.layouts.items() if name not in array_names
            },
            scalars=scalars,
            files={
                name: dict(size_bytes=size, sha256=_sha(root / name))
                for name, size in storage.files.items()
            },
        )
        if any(_sha(path) != digest for path, digest in code.items()):
            raise ValueError("source changed during streaming solve")
        pending = storage.json("manifest.pending.json", manifest)
        final = pending.rename(root / "manifest.json")
        metadata["storage_bytes_including_manifest"] = storage.used
        return StreamingResult(result, metadata, final)
    except BaseException as error:
        storage.flush()
        with suppress(OSError, StreamingCapacityError):
            storage.json(
                "failure.json",
                dict(
                    schema=SCHEMA,
                    status="failed",
                    error_type=type(error).__name__,
                    error=str(error),
                    storage_bytes_before_failure_record=storage.used,
                    completed_result_available=False,
                ),
            )
        raise
