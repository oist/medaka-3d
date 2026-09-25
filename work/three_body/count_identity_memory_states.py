"""Count exact support-only memory-DP states without scoring or pruning identities.

State=(current permutation index, last directly observed event for each ID).
An event is 3*zero-based frame offset+raw slot. Current permutations remain
distinct even when all last-event memories are equal. This independent counter
does not import the proposed memory-DP implementation or read positions/colors.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from medaka3d.provenance import provenance  # noqa: E402

PERMUTATIONS = tuple(itertools.permutations(range(3)))


@dataclass(frozen=True)
class Caps:
    maximum_layer_states: int = 100_000
    maximum_total_states: int = 1_000_000
    maximum_total_edges: int = 6_000_000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ReachableCount:
    completed: bool
    frame_idx: np.ndarray
    state_counts: np.ndarray
    edge_counts: np.ndarray
    total_states: int
    total_edges: int
    stop: dict | None
    layers: tuple | None  # Tiny independent tests only; production retains two layers.


def _validate(frame_idx, direct):
    frames, supported = np.asarray(frame_idx), np.asarray(direct)
    if (
        frames.ndim != 1
        or not len(frames)
        or frames.dtype != np.int64
        or np.any(frames < 0)
        or np.any(np.diff(frames) != 1)
    ):
        raise ValueError("frame_idx must retain every consecutive nonnegative native int64 frame")
    if supported.shape != (len(frames), 3) or supported.dtype != bool:
        raise ValueError("direct pectoral support must be boolean T,3")
    if not supported[0].all():
        raise ValueError("all three first-frame pectorals must be directly supported")
    if len(frames) > np.iinfo(np.int64).max // 3:
        raise ValueError("event encoding exceeds int64")
    return frames, supported


def reachable_counts(frame_idx, direct, caps=None, *, retain_layers=False):
    """Enumerate full reachable state sets, or stop explicitly at the first cap.

    Each complete transition has exactly 6*N_previous edges, including distinct
    previous permutations with the same memory. Those equal memories may share
    successor-set construction; edge counts are never reduced by that shortcut.
    No state is dropped in order to continue. A cap during next-layer enumeration
    reports its witnessed lower bound and does not call that layer complete.
    """
    frames, supported = _validate(frame_idx, direct)
    caps = caps or Caps()
    if retain_layers and len(frames) > 8:
        raise ValueError("retaining state sets is restricted to tiny <=8-frame tests")
    current = {(0, (0, 1, 2))}
    counts, edge_counts, total_states, total_edges = [1], [], 1, 0
    layers = [frozenset(current)] if retain_layers else None
    stop = None
    for offset in range(1, len(frames)):
        full_edges = 6 * len(current)
        if total_edges + full_edges > caps.maximum_total_edges:
            stop = dict(
                reason="maximum_total_edges",
                attempted_frame=int(frames[offset]),
                next_layer_edges=full_edges,
                required_total_edges=total_edges + full_edges,
                attempted_layer_states_lower_bound=None,
            )
            break
        limit = min(caps.maximum_layer_states, caps.maximum_total_states - total_states)
        candidate = set()
        # At an all-direct frame every old event is overwritten by current evidence.
        # The resulting six states merge histories, not their path costs or seeds.
        memories = (None,) if supported[offset].all() else {state[1] for state in current}
        for memory in memories:
            for p, slots in enumerate(PERMUTATIONS):
                events = tuple(
                    3 * offset + slot if supported[offset, slot] else memory[identity]
                    for identity, slot in enumerate(slots)
                )
                candidate.add((p, events))
                if len(candidate) > limit:
                    reason = (
                        "maximum_layer_states"
                        if len(candidate) > caps.maximum_layer_states
                        else "maximum_total_states"
                    )
                    stop = dict(
                        reason=reason,
                        attempted_frame=int(frames[offset]),
                        next_layer_edges=full_edges,
                        required_total_edges=total_edges + full_edges,
                        attempted_layer_states_lower_bound=len(candidate),
                    )
                    break
            if stop is not None:
                break
        if stop is not None:
            break
        current = candidate
        counts.append(len(current))
        edge_counts.append(full_edges)
        total_states += len(current)
        total_edges += full_edges
        if layers is not None:
            layers.append(frozenset(current))
    return ReachableCount(
        stop is None,
        frames[: len(counts)].copy(),
        np.asarray(counts, np.int64),
        np.asarray(edge_counts, np.int64),
        total_states,
        total_edges,
        stop,
        None if layers is None else tuple(layers),
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_supported_geometry(root):
    """Check the completed producer seal and deserialize only native frames/support."""
    root = Path(root).resolve(strict=True)
    if not root.is_relative_to(REPO):
        raise ValueError("completed geometry must be preserved in the workspace")
    record_path, candidate_path = root / "experiment.provenance.json", root / "candidate.npz"
    hashes = {str(p): sha256(p) for p in (record_path, candidate_path)}
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("status") != "ok"
        or record.get("stage") != "three_body_persistent_node_recovery_algorithm_arm"
    ):
        raise ValueError("counting requires completed persistent geometry")
    params = record["params"]
    output = Path(params["output"])
    output = (output if output.is_absolute() else REPO / output).resolve(strict=True)
    if (
        output != root
        or params.get("algorithm_arm") != "v1_plus_persistent_mouth_recovery_graph"
        or params.get("inputs_unchanged_during_run") is not True
        or params.get("code_unchanged_during_run") is not True
    ):
        raise ValueError("geometry producer params are not consistently sealed")
    item = record["outputs"]["candidate"]
    bound_path = Path(item["path"])
    bound_path = (bound_path if bound_path.is_absolute() else REPO / bound_path).resolve(
        strict=True
    )
    if bound_path != candidate_path or item["sha256"] != hashes[str(candidate_path)]:
        raise ValueError("geometry support payload is not bound to the completed output")
    with np.load(candidate_path, allow_pickle=False) as archive:
        if len(archive.files) != 21 or len(set(archive.files)) != 21:
            raise ValueError("persistent geometry must preserve its 21-array schema")
        frames, observed = archive["frame_idx"], archive["observed"]
    if observed.shape != (len(frames), 3, 3) or observed.dtype != bool:
        raise ValueError("invalid direct node support array")
    direct = observed[:, :, 1].copy()
    _validate(frames, direct)
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise RuntimeError("geometry evidence changed while loading support")
    return frames, direct, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(REPO) or output.exists() or output.is_symlink():
        raise ValueError("a new workspace output directory is required")
    paths = [
        Path(__file__),
        REPO / "src/medaka3d/provenance.py",
        REPO / "tests/test_three_body_identity_memory_count.py",
    ]
    codes = {str(path): sha256(path) for path in paths}
    output.mkdir(parents=True)
    caps = Caps()
    with provenance(
        "three_body_memory_identity_support_reachability", output, name="count"
    ) as record:
        record.param("caps", asdict(caps))
        record.param("device", "cpu")
        record.param("geometry_positions_color_gt_or_reference_arrays_deserialized", False)
        record.param("cost_evaluation_or_approximate_pruning", False)
        record.param("initial_state", [0, [0, 1, 2]])
        record.param("full_transition_edges", "6 * previous complete layer states")
        output_hashes, inputs, rows = {}, {}, []
        for path, digest in codes.items():
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record.input(path, path, sha256=digest)
            record.output("code:" + str(Path(path).relative_to(REPO)), target, sha256=digest)
            output_hashes[str(target)] = digest
        for label, directory in (
            ("transient_200", "persistent_mouth_transient_200_v1"),
            ("interval_666", "persistent_mouth_interval_666_v1"),
            ("continuous_minute", "persistent_mouth_continuous_one_minute_v1"),
        ):
            frames, direct, verified = load_supported_geometry(REPO / "work/three_body" / directory)
            inputs.update(verified)
            start = time.perf_counter()
            result = reachable_counts(frames, direct, caps)
            elapsed = time.perf_counter() - start
            payload = output / f"{label}.npz"
            np.savez_compressed(
                payload,
                completed_frame_idx=result.frame_idx,
                state_counts=result.state_counts,
                edge_counts=result.edge_counts,
                full_frame_idx=frames,
                direct_pectoral=direct,
            )
            output_hashes[str(payload)] = sha256(payload)
            record.output(label, payload, sha256=output_hashes[str(payload)])
            peak = int(result.state_counts.max())
            peak_indices = np.flatnonzero(result.state_counts == peak)
            rows.append(
                dict(
                    case=label,
                    geometry_run=directory,
                    requested_native_frames=len(frames),
                    completed_native_frames=len(result.frame_idx),
                    completed=result.completed,
                    stop=result.stop,
                    peak_complete_layer_states=peak,
                    first_peak_frame=int(result.frame_idx[peak_indices[0]]),
                    number_of_peak_frames=len(peak_indices),
                    total_complete_states=result.total_states,
                    total_complete_edges=result.total_edges,
                    all_three_direct_frames=int(direct.all(axis=1).sum()),
                    support_count_histogram=np.bincount(direct.sum(axis=1), minlength=4).tolist(),
                    elapsed_count_seconds=elapsed,
                )
            )
            print(json.dumps(rows[-1]), flush=True)
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        summary = dict(
            status="complete_support_reachability_preflight",
            all_cases_completed=all(row["completed"] for row in rows),
            caps=asdict(caps),
            cases=rows,
            input_sha256=inputs,
            code_sha256=codes,
            supports_only=True,
            approximation_or_pruning=False,
            initial_permutation_fixed=0,
            tests="tiny full-path enumeration",
            limitations=[
                "Counts apply to all support-reachable states with no cost-based constraints.",
                "All-direct frames merge event histories; they do not reset identity or path cost.",
                "Runtime measures set enumeration only, not forward/backward weighted DP.",
                "Candidate/provenance seals were rechecked; upstream images were not reread.",
            ],
        )
        path = output / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        output_hashes[str(path)] = sha256(path)
        record.output("summary", path, sha256=output_hashes[str(path)])
        for path, digest in {**inputs, **codes, **output_hashes}.items():
            if sha256(path) != digest:
                raise RuntimeError("input/source/output changed during reachable-state count")
        record.param("inputs_sources_outputs_unchanged", True)
    return summary


if __name__ == "__main__":
    main()
