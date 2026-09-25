"""Bind three review anchors to a saved five-minute automatic identity mapping.

Reads only sealed JSON and the small mapping archive, never the streaming DAG,
video, GPU state or human review. Numeric IDs retain their initial-slot scope.
The caller must already have verified the raw candidate into its evidence map.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
START, STOP = 66400, 116200
IDENTITY_STAGE = "fast_tail_continuation_streaming_identity_v1"
COMPARISON_STAGE = "fast_tail_continuation_comparison_v1"
IDENTITY_SOURCE = WORK / "run_fast_tail_continuation_identity.py"
IDENTITY_SOURCE_SHA = "87154f972479ac71ce44ab3d859fb5fb4e8ee81947f335e81d4f5b49781cf9fa"
COMPARISON_SOURCE = WORK / "compare_fast_tail_continuation.py"
COMPARISON_SOURCE_SHA = "2e4a749d5ea13116f05a622afd47478af311525fbd856b32a116bd9ac765acba"
MILESTONE_PATH = (
    WORK / "fast_tail_five_minute_continuation_v1/milestones/through_116200/summary.json"
)
PERMUTATIONS = np.array(list(itertools.permutations(range(3))), dtype=np.int64)
PERMUTATIONS.flags.writeable = False
ANCHOR_SCHEMA = {
    "frame_idx": ((3,), "iu"),
    "pose_xyz": ((3, 3, 3, 3), "f"),
    "sources": ((3, 3, 3, 3), "iu"),
    "observed": ((3, 3, 3), "b"),
    "identity_review": ((3, 3), "b"),
}
MAPPING_FIELDS = {
    "frame_idx",
    "identity_to_raw_slot",
    "permutation_state",
    "state_alternative_gap",
    "identity_alternative_gap",
    "state_alternative_status",
    "identity_alternative_status",
    "optimal_state_mask",
    "min_marginal_delta",
    "id_slot_min_marginal_delta",
    "mapping_changed_from_initial_slots",
    "mapping_transition",
    "tied_identity_slots",
    "terminal_age_frames",
    "terminal_unsettled_gap",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def local(path):
    result = Path(path).resolve(strict=True)
    require(result.is_relative_to(WORK.resolve()), "identity evidence must be in workspace")
    return result


def read_bound(path, expected, evidence):
    path = local(path)
    require(valid_sha(expected), "explicit SHA256 required")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "bound input SHA differs: " + str(path))
    evidence.add(path, expected)
    return data


def json_document(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)

    def number(value):
        result = float(value)
        require(np.isfinite(result), "nonfinite JSON number")
        return result

    result = json.loads(
        data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid, parse_float=number
    )
    require(isinstance(result, dict), "JSON object required")
    return result


def member(root, seal, name, evidence, *, json_value=True):
    path = local(root / name)
    row = seal["outputs"].get(name)
    require(
        isinstance(row, dict)
        and path.is_relative_to(root)
        and Path(row["path"]).resolve() == path
        and type(row["size_bytes"]) is int
        and row["size_bytes"] == path.stat().st_size,
        "sealed output path or size differs: " + name,
    )
    data = read_bound(path, row["sha256"], evidence)
    return json_document(data) if json_value else data


def edge(seal, path, digest):
    require(
        seal["inputs"].get(str(Path(path).resolve())) == digest,
        "required identity/comparison input edge differs: " + str(path),
    )


def flags(document, expected, label):
    require(
        all(type(document.get(k)) is type(v) and document[k] == v for k, v in expected.items()),
        label,
    )


def validate_permutations(mapping, length):
    require(
        isinstance(mapping, np.ndarray)
        and mapping.shape == (length, 3)
        and mapping.dtype.kind in "iu"
        and np.array_equal(np.sort(mapping, axis=1), np.tile([0, 1, 2], (length, 1))),
        "one bijective whole-fish permutation per native frame required",
    )


def validate_mapping(arrays, report):
    """Validate the complete native mapping before selecting any review rows."""
    require(set(arrays) == MAPPING_FIELDS, "saved mapping archive field membership differs")
    frames, mapping, state = (
        arrays[k] for k in ("frame_idx", "identity_to_raw_slot", "permutation_state")
    )
    n = STOP - START
    require(
        frames.shape == (n,)
        and frames.dtype.kind in "iu"
        and np.array_equal(frames, np.arange(START, STOP)),
        "complete ordered native five-minute frame interval required",
    )
    validate_permutations(mapping, n)
    require(np.array_equal(mapping[0], [0, 1, 2]), "initial numeric identity must be P012")
    require(
        state.shape == (n,)
        and state.dtype.kind in "iu"
        and np.all(state < 6)
        and np.all(state >= 0)
        and np.array_equal(PERMUTATIONS[state], mapping),
        "permutation state differs",
    )
    tol = report["numerical_tie_tolerance"]
    require(type(tol) in (int, float) and np.isfinite(tol) and tol >= 0, "invalid DP tie tolerance")
    for prefix, shape in (("state", (n,)), ("identity", (n, 3))):
        gap, status = (arrays[prefix + "_alternative_" + suffix] for suffix in ("gap", "status"))
        require(
            gap.shape == shape
            and gap.dtype.kind == "f"
            and not np.isnan(gap).any()
            and np.all(gap >= 0)
            and status.shape == shape
            and status.dtype.kind == "U",
            "invalid saved alternative gap/status",
        )
        expected = np.full(shape, "finite_gap", dtype="U32")
        expected[np.isposinf(gap)] = "no_feasible_alternative"
        expected[np.isfinite(gap) & (gap <= tol)] = "tie_within_tolerance"
        require(np.all(np.isposinf(gap[0])), "fixed initial identity has no alternative gap")
        expected[0] = "fixed_initial_constraint"
        require(np.array_equal(status, expected), "saved ambiguity status disagrees with gap")
    changed = mapping != [0, 1, 2]
    transition = np.zeros((n, 3), bool)
    transition[1:] = mapping[1:] != mapping[:-1]
    tied = arrays["identity_alternative_status"] == "tie_within_tolerance"
    for key, value in (
        ("mapping_changed_from_initial_slots", changed),
        ("mapping_transition", transition),
        ("tied_identity_slots", tied),
    ):
        require(
            arrays[key].dtype.kind == "b" and np.array_equal(arrays[key], value),
            "saved derived mapping flag differs: " + key,
        )
    optimal = arrays["optimal_state_mask"]
    require(
        optimal.shape == (n, 6)
        and optimal.dtype.kind == "b"
        and np.all(optimal[np.arange(n), state]),
        "chosen state must be optimal",
    )
    for key, shape in (("min_marginal_delta", (n, 6)), ("id_slot_min_marginal_delta", (n, 3, 3))):
        value = arrays[key]
        require(
            value.shape == shape
            and value.dtype.kind == "f"
            and not np.isnan(value).any()
            and np.all(value >= 0),
            "invalid saved marginal delta",
        )
    for key, kind in (("terminal_age_frames", "iu"), ("terminal_unsettled_gap", "b")):
        value = arrays[key]
        require(value.shape == (3, 2) and value.dtype.kind in kind, "terminal node memory schema")
        require(value.tolist() == report[key], "terminal memory report differs")
    require(
        np.all(arrays["terminal_age_frames"] >= 0)
        and np.array_equal(arrays["terminal_unsettled_gap"], arrays["terminal_age_frames"] > 0),
        "terminal missing-memory flags disagree with saved ages",
    )
    flags(
        report,
        dict(
            native_frames=n,
            changed_mapping_native_frames=int(changed.any(1).sum()),
            mapping_transition_native_frames=int(transition.any(1).sum()),
            multiple_optimal_permutation_frames=int((optimal.sum(1) > 1).sum()),
            all_three_ID_truth=False,
            biological_swap_zero_verified=False,
            review_flags_cleared=False,
        ),
        "identity report scope/counts differ",
    )


def anchor_indices(anchor_frames):
    frames = np.asarray(anchor_frames)
    require(
        frames.shape == (3,) and frames.dtype.kind in "iu" and np.all(frames[1:] > frames[:-1]),
        "three strictly ordered native anchors required",
    )
    return frames


def load_identity_anchors(
    identity_root,
    identity_sha,
    *,
    comparison_root,
    comparison_sha,
    milestone_sha,
    raw_candidate_reference,
    anchor_frames,
    evidence,
):
    """Return a separate DP mapping; any outside anchor makes all three unavailable.

    evidence must already bind the raw candidate loaded by the caller. A verified
    successful five-minute comparison and identity result are required even when
    requested anchors lie outside the scientific interval. No extrapolation occurs.
    """
    anchors = anchor_indices(anchor_frames)
    root, body_root = local(identity_root), local(comparison_root)
    seal = json_document(read_bound(root / "identity.provenance.json", identity_sha, evidence))
    body = json_document(
        read_bound(body_root / "comparison.provenance.json", comparison_sha, evidence)
    )
    flags(seal, dict(stage=IDENTITY_STAGE, status="ok"), "successful identity seal required")
    flags(body, dict(stage=COMPARISON_STAGE, status="ok"), "successful comparison seal required")
    reference = raw_candidate_reference
    require(
        isinstance(reference, dict)
        and set(reference) == {"path", "sha256", "arrays"}
        and valid_sha(reference["sha256"]),
        "exact raw candidate reference required",
    )
    candidate_path = local(reference["path"])
    require(
        str(candidate_path) == reference["path"]
        and evidence.paths.get(str(candidate_path)) == reference["sha256"],
        "caller must first bind the actual raw candidate",
    )
    for source, digest in (
        (IDENTITY_SOURCE, IDENTITY_SOURCE_SHA),
        (COMPARISON_SOURCE, COMPARISON_SOURCE_SHA),
    ):
        read_bound(source, digest, evidence)
        edge(seal, source, digest)
    edge(body, COMPARISON_SOURCE, COMPARISON_SOURCE_SHA)
    edge(seal, body_root / "comparison.provenance.json", comparison_sha)
    for value in (seal, body):
        edge(value, MILESTONE_PATH, milestone_sha)
        edge(value, candidate_path, reference["sha256"])
    milestone = json_document(read_bound(MILESTONE_PATH, milestone_sha, evidence))
    flags(
        milestone,
        dict(
            status="ok",
            horizon_stop=STOP,
            scientific_interval=[START, STOP],
            five_minute_candidate=True,
        ),
        "matching five-minute milestone required",
    )
    require(
        milestone["candidate"]["path"] == reference["path"]
        and milestone["candidate"]["sha256"] == reference["sha256"],
        "milestone candidate reference differs",
    )
    body_plan = member(body_root, body, "plan.json", evidence)
    body_summary = member(body_root, body, "summary.json", evidence)
    for document in (body_plan, body_summary):
        flags(
            document,
            dict(schema=COMPARISON_STAGE, horizon_stop=STOP),
            "five-minute comparison scope",
        )
        require(
            document["candidate_references"]["fast_tail"] == reference,
            "body comparison raw candidate differs",
        )
    require(body_plan["milestone_sha256"] == milestone_sha, "body milestone binding differs")
    flags(
        body_summary,
        dict(
            status="ok",
            native_interval=[START, STOP],
            native_frames=STOP - START,
            candidate_all21_and_source_arrays_unchanged=True,
            review_modified=False,
            all_three_ID_truth=False,
            external_pose_resets=0,
            logical_initialization_count=1,
            same_parent_and_horizon=True,
        ),
        "body comparison preservation/scope differs",
    )
    plan, summary, report = (
        member(root, seal, name, evidence)
        for name in ("plan.json", "summary.json", "identity_report.json")
    )
    for document in (plan, summary):
        flags(
            document,
            dict(
                schema=IDENTITY_STAGE,
                native_interval=[START, STOP],
                intermediate_ID_reset=False,
                cap_increase=False,
            ),
            "identity scope differs",
        )
        require(document["raw_candidate_reference"] == reference, "identity raw candidate differs")
    flags(
        plan,
        dict(
            comparison_sha256=comparison_sha,
            milestone_sha256=milestone_sha,
            one_solver_call=True,
            initial_P=[0, 1, 2],
            node_indices=[1, 0],
            raw_geometry_sources_support_review_modified=False,
            physical_ID_truth=False,
        ),
        "identity plan binding or scope differs",
    )
    flags(
        summary,
        dict(
            status="ok",
            mathematical_status="optimal",
            native_frames=STOP - START,
            solver_calls=1,
            node_indices=[1, 0],
            raw21_unchanged=True,
            all29_arrays_and19_scalars_saved=True,
            spool_all_files_SHA_layout_membership_verified=True,
            all_three_ID_truth=False,
            biological_swap_zero_verified=False,
        ),
        "successful exact identity result required",
    )
    require(summary["report"] == report, "identity report summary binding differs")
    mapping_bytes = member(
        root, seal, "identity_mapping_and_ambiguity.npz", evidence, json_value=False
    )
    with np.load(io.BytesIO(mapping_bytes), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate mapping archive member")
        arrays = {key: archive[key] for key in archive.files}
    validate_mapping(arrays, report)
    outside = [int(f) for f in anchors if not START <= int(f) < STOP]
    binding = dict(
        identity_provenance_sha256=identity_sha,
        comparison_sha256=comparison_sha,
        milestone_sha256=milestone_sha,
        raw_candidate_reference=reference,
        mapping_output=seal["outputs"]["identity_mapping_and_ambiguity.npz"],
        native_interval=[START, STOP],
        anchor_frames=anchors.tolist(),
        initial_P=[0, 1, 2],
        all_three_ID_truth=False,
        biological_swap_zero_verified=False,
        review_flags_cleared=False,
        scope="Automatic numeric IDs initialized from raw slots; not sex or physical truth",
    )
    if outside:
        return dict(
            available=False,
            reason="anchors_outside_identity_interval",
            outside_frames=outside,
            permutation_by_id=None,
            ambiguity=None,
            binding=binding,
        )
    indices = anchors.astype(np.int64) - START
    permutation = arrays["identity_to_raw_slot"][indices].copy()
    permutation.flags.writeable = False
    ambiguity = {
        key: arrays[key][indices].copy()
        for key in (
            "state_alternative_gap",
            "identity_alternative_gap",
            "state_alternative_status",
            "identity_alternative_status",
            "optimal_state_mask",
            "tied_identity_slots",
        )
    }
    ambiguity.update(
        terminal_age_frames=arrays["terminal_age_frames"].copy(),
        terminal_unsettled_gap=arrays["terminal_unsettled_gap"].copy(),
    )
    for value in ambiguity.values():
        value.flags.writeable = False
    return dict(
        available=True,
        reason=None,
        outside_frames=[],
        permutation_by_id=permutation,
        ambiguity=ambiguity,
        binding=binding,
    )


def reorder_anchor_rows(rows, permutation_by_id):
    """Copy exactly five fields; one shared whole-fish row permutation per frame."""
    require(
        isinstance(rows, dict) and set(rows) == set(ANCHOR_SCHEMA),
        "exact five anchor fields required",
    )
    for key, (shape, kind) in ANCHOR_SCHEMA.items():
        value = rows[key]
        require(
            isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in kind,
            "anchor field schema differs: " + key,
        )
    anchor_indices(rows["frame_idx"])
    validate_permutations(permutation_by_id, 3)
    result = {"frame_idx": rows["frame_idx"].copy()}
    for key in ANCHOR_SCHEMA:
        if key != "frame_idx":
            result[key] = rows[key][np.arange(3)[:, None], permutation_by_id].copy()
    for value in result.values():
        value.flags.writeable = False
    return result
