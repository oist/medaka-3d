"""Saved Sparse motion before/after whole-body ID reordering; no new tracking.

Both endpoints of the primary raw/ID comparison lie inside the scientific
interval. The old raw latent maximum includes its preceding boundary and is
reproduced separately. Distances are estimated-coordinate diagnostics, not
fish speed, physical identity truth, or certification of the approximate IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
FIRST, START, STOP = 49800, 66400, 116200
ARM = "sparse_instance_rescore"
STAGE = "sparse_id_aligned_motion_v1"
COMPARISON = WORK / "sparse_instance_five_minute_comparison_v1"
IDENTITY = WORK / "sparse_instance_five_minute_identity_beam256_v1"
COMPARISON_SHA = "0cf955b7a2d15457d958c72ff647b1a2bc843e51039e63263e8788081454e909"
IDENTITY_SHA = "3573c02a05f844505622aa546c61f8bdf78e31606bda89fd06a7313403e85d96"
CANDIDATE_SHA = "acdd65b5f3717e73bf1b75ab7d0bfc775fc6ad60b4435d53efae5aa3a4d31d6b"
MAPPING_SHA = "745f5f50a75560f4dba4de6e80713ef711becae711092e928dd17e2ddeedec24"
EXECUTION_READY = True
NODES = ("mouth", "chest", "tail")
THRESHOLD_MM = 10.0
# Historical definitions and mapping producer are bound as evidence, never imported.
SOURCE_PINS = {
    "compare_sparse_instance_five_minutes.py": "1ec0893d7dd2bbc746004f71d9b81abc60096cc0173bf44871f56be948e1c8dc",  # noqa: E501
    "execute_sparse_instance_five_minute_identity_beam.py": "07d9937488057e620f4478a6458e949118d28c8a4ad8e9e75d41eb84e4f12609",  # noqa: E501
    "identity_node_memory_beam.py": "019791d34fda3292aa35371ba3aa8297c3f92ace1e19e8ffe514cd39fd4c92bd",  # noqa: E501
    "compare_bounded_one_minute.py": "5e1a4f570afadf204f96f3c6e83d55a9089f6fa3a14cf1b4c54cc84e4fe76cd0",  # noqa: E501
}
CANDIDATE_FIELDS = {
    "frame_idx",
    "pose_xyz",
    "observed",
    "sources",
    "identity_review",
    "mask_supported",
    "body_supported",
    "missing",
    "costs",
    "path_costs",
    "loss_parts",
    "score_margin",
    "proposal_kind",
    "hypotheses",
    "proposal_flow_nodes",
    "proposal_flow_parent",
    "node_missing",
    "direct_pectoral_age",
    "xyz",
    "accepted_xyz",
    "body_xyz",
}
MAPPING_FIELDS = {
    "frame_idx",
    "path_state",
    "permutation_by_id",
    "conditional_min_marginal_delta",
    "conditional_id_slot_min_marginal_delta",
    "conditional_optimal_state_mask",
    "conditional_state_alternative_gap",
    "conditional_identity_alternative_gap",
    "conditional_state_alternative_status",
    "conditional_identity_alternative_status",
    "terminal_last_event_frame",
    "terminal_last_event_slot",
    "terminal_age_frames",
    "terminal_unsettled_gap",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def signature(array):
    return dict(
        dtype=array.dtype.str,
        shape=list(array.shape),
        sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    )


def capture(path, evidence, expected=None, size=None):
    path = path.resolve()
    require(path.is_relative_to(WORK.resolve()), "read outside diagnostic workspace")
    before = path.stat()
    digest = sha(path)
    require(expected is None or digest == expected, "input SHA differs: " + path.name)
    require(size is None or before.st_size == size, "input size differs")
    require(
        (before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns),
        "input changed during capture",
    )
    evidence[str(path)] = dict(
        sha256=digest, size_bytes=before.st_size, mtime_ns=before.st_mtime_ns
    )
    return path


def document(path, evidence, expected=None):
    return json.loads(capture(path, evidence, expected).read_text(encoding="utf-8"))


def member(root, seal, name, evidence):
    row = seal["outputs"][name]
    path = (root / name).resolve()
    require(
        path.is_relative_to(root.resolve()) and Path(row["path"]).resolve() == path,
        "sealed member path differs",
    )
    return capture(path, evidence, row["sha256"], row["size_bytes"])


def member_json(root, seal, name, evidence):
    return json.loads(member(root, seal, name, evidence).read_text(encoding="utf-8"))


def load_headers(evidence):
    for name, digest in SOURCE_PINS.items():
        capture(WORK / name, evidence, digest)
    comp = document(COMPARISON / "comparison.provenance.json", evidence, COMPARISON_SHA)
    ids = document(IDENTITY / "identity.provenance.json", evidence, IDENTITY_SHA)
    for root, seal in ((COMPARISON, comp), (IDENTITY, ids)):
        require(
            seal["status"] == "ok" and seal["stage"] == root.name,
            "successful expected stage required",
        )
    for name in ("compare_sparse_instance_five_minutes.py", "compare_bounded_one_minute.py"):
        require(
            comp["inputs"].get(str((WORK / name).resolve())) == SOURCE_PINS[name],
            "comparison metric source differs",
        )
    for name in (
        "execute_sparse_instance_five_minute_identity_beam.py",
        "identity_node_memory_beam.py",
    ):
        require(
            ids["inputs"].get(str((WORK / name).resolve())) == SOURCE_PINS[name],
            "mapping implementation differs",
        )
    cs, cp = (
        member_json(COMPARISON, comp, name, evidence) for name in ("summary.json", "plan.json")
    )
    ins, inp = (
        member_json(IDENTITY, ids, name, evidence) for name in ("summary.json", "plan.json")
    )
    for root, docs in ((COMPARISON, (cs, cp)), (IDENTITY, (ins, inp))):
        for doc in docs:
            require(
                doc["schema"] == root.name
                and doc["native_interval"] == [START, STOP]
                and doc["native_frames"] == STOP - START,
                "scientific interval differs",
            )
    require(cs["status"] == ins["status"] == "ok", "successful summaries required")
    require(cs["candidate_references"] == cp["candidate_references"], "candidate binding differs")
    reference = cs["candidate_references"][ARM]
    copy_row = comp["outputs"][f"candidates/{ARM}.npz"]
    require(
        reference["terminal_horizon"] == STOP and set(reference["arrays"]) == CANDIDATE_FIELDS,
        "same-horizon candidate21 reference required",
    )
    require(copy_row["sha256"] == CANDIDATE_SHA, "registered comparison copy differs")
    for key in ("path", "milestone_path"):
        hash_key = "sha256" if key == "path" else "milestone_sha256"
        require(
            comp["inputs"].get(reference[key]) == reference[hash_key],
            "original final/milestone input edge missing",
        )
    for doc in (ins, inp):
        require(
            doc["comparison_seal_sha256"] == COMPARISON_SHA
            and doc["raw_candidate_reference"] == reference
            and doc["consumed_comparison_copy"] == copy_row
            and doc["terminal_horizon"] == STOP
            and doc["retained_raw_interval"] == [FIRST, STOP],
            "ID candidate binding differs",
        )
    require(ins["raw21_signatures"] == reference["arrays"], "ID raw signatures differ")
    require(
        ids["inputs"].get(str((COMPARISON / "comparison.provenance.json").resolve()))
        == COMPARISON_SHA
        and ids["inputs"].get(copy_row["path"]) == CANDIDATE_SHA,
        "ID lacks consumed comparison/copy edges",
    )
    require(
        inp["initial_P"] == [0, 1, 2]
        and inp["initialization_frame"] == START
        and ins["intermediate_ID_resets"] == 0
        and inp["node_indices"] == [1, 0]
        and inp["beam_width"] == 256,
        "initialization/objective convention differs",
    )
    require(
        ins["gaps_conditional_on_retained_graph"]
        and not ins["global_optimality_certified"]
        and not ins["physical_ID_truth"]
        and not ins["review_flags_cleared"]
        and ins["raw_sources_review_unchanged"],
        "ID uncertainty/raw contract differs",
    )
    require(
        ids["outputs"]["identity_mapping_and_ambiguity.npz"]["sha256"] == MAPPING_SHA,
        "registered mapping differs",
    )
    motion = member_json(COMPARISON, comp, f"motion/{ARM}_full.json", evidence)
    require(
        motion["by_slot_node"] == cs["methods"][ARM]["intervals"]["full"]["node_metrics"],
        "motion and comparison summary disagree",
    )
    return comp, ids, reference, motion, ins


def selected_archive(path, fields, selected):
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(fields) and set(archive.files) == fields,
            "archive membership differs",
        )
        arrays = {key: archive[key] for key in selected}
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def validate_geometry(frames, pose, observed):
    require(
        frames.dtype == np.int64
        and frames.ndim == 1
        and len(frames) >= 2
        and np.all(np.diff(frames) == 1),
        "contiguous native frames required",
    )
    require(
        pose.dtype == np.float64
        and pose.shape == (len(frames), 3, 3, 3)
        and np.isfinite(pose).all(),
        "finite FP64 latent poses required",
    )
    require(
        observed.dtype == np.bool_ and observed.shape == pose.shape[:-1],
        "per-node boolean direct support required",
    )


def reorder_whole_body(pose, observed, permutation):
    """permutation[t, ID] is the raw slot, not the inverse mapping."""
    require(
        permutation.shape == (len(pose), 3)
        and permutation.dtype.kind in "iu"
        and np.array_equal(
            np.sort(permutation, axis=1), np.broadcast_to(np.arange(3), permutation.shape)
        ),
        "bijective per-frame ID-to-raw-slot permutation required",
    )
    time = np.arange(len(pose))[:, None]
    return pose[time, permutation].copy(), observed[time, permutation].copy()


def motion(frames, pose, observed):
    validate_geometry(frames, pose, observed)
    steps = np.linalg.norm(np.diff(pose, axis=0), axis=-1)
    direct = observed[:-1] & observed[1:]
    rows, events = [], []
    for body in range(3):
        for node, name in enumerate(NODES):
            values, keep = steps[:, body, node], direct[:, body, node]
            row = dict(
                body=body,
                node=name,
                supported_frames=int(observed[:, body, node].sum()),
                missing_frames=int((~observed[:, body, node]).sum()),
            )
            for mode, mask in (
                ("all_estimated", np.ones(len(values), bool)),
                ("both_direct", keep),
            ):
                selected = values[mask]
                large = mask & (values > THRESHOLD_MM)
                row[mode] = dict(
                    adjacent_steps=int(mask.sum()),
                    steps_gt10mm=int(large.sum()),
                    maximum_step_mm=float(selected.max()) if len(selected) else None,
                    maximum_ending_frames=frames[1:][mask & (values == selected.max())].tolist()
                    if len(selected)
                    else [],
                )
                for t in np.flatnonzero(large):
                    events.append(
                        dict(
                            mode=mode,
                            body=body,
                            node=name,
                            previous_frame=int(frames[t]),
                            frame_idx=int(frames[t + 1]),
                            distance_mm=float(values[t]),
                            previous_direct=bool(observed[t, body, node]),
                            current_direct=bool(observed[t + 1, body, node]),
                        )
                    )
            rows.append(row)
    return dict(by_body_node=rows, large_events=events), steps, direct


def verify_raw_reference(raw_result, saved, full_frames, full_pose, start, stop):
    """Integer counts and FP64 maxima must equal the sealed historical report."""
    require(len(saved["by_slot_node"]) == 9, "nine saved slot/node rows required")
    before = np.flatnonzero((full_frames[1:] >= start) & (full_frames[1:] < stop))
    legacy = np.linalg.norm(np.diff(full_pose, axis=0), axis=-1)[before].max(axis=0)
    for row, old in zip(raw_result["by_body_node"], saved["by_slot_node"], strict=True):
        body, node = row["body"], NODES.index(row["node"])
        require((old["slot"], old["node"]) == (body, row["node"]), "saved metric order differs")
        for key in ("supported_frames", "missing_frames"):
            require(row[key] == old[key], "raw support count differs")
        for key, previous in (
            ("adjacent_steps", "directly_supported_adjacent_steps"),
            ("steps_gt10mm", "directly_supported_adjacent_steps_gt10mm"),
            ("maximum_step_mm", "maximum_directly_supported_adjacent_step_mm"),
        ):
            require(row["both_direct"][key] == old[previous], "raw direct motion differs")
        require(
            float(legacy[body, node]) == old["maximum_latent_adjacent_step_mm"],
            "historical latent boundary scope differs",
        )
    return dict(
        raw_direct_counts_and_maxima_equal=True,
        raw_support_counts_equal=True,
        old_latent_maxima_reproduced_in_separate_boundary_scope=True,
        old_latent_scope="Ending native inside interval; includes start-minus-one.",
        primary_scope="Both endpoints inside interval for raw and mapped.",
        equality="Exact integer and float64 scalar equality; no fitted tolerance.",
    )


def compare_motion(frames, pose, observed, permutation):
    raw, raw_steps, raw_direct = motion(frames, pose, observed)
    aligned_pose, aligned_observed = reorder_whole_body(pose, observed, permutation)
    aligned, id_steps, id_direct = motion(frames, aligned_pose, aligned_observed)
    boundaries = np.flatnonzero(np.any(permutation[1:] != permutation[:-1], axis=1))
    unchanged = np.ones(len(frames) - 1, bool)
    unchanged[boundaries] = False
    require(
        np.array_equal(np.sort(raw_steps[unchanged], axis=1), np.sort(id_steps[unchanged], axis=1))
        and np.array_equal(
            np.sort(np.where(raw_direct, raw_steps, -1)[unchanged], axis=1),
            np.sort(np.where(id_direct, id_steps, -1)[unchanged], axis=1),
        ),
        "constant-permutation distance/support multisets differ",
    )
    boundary_records = [
        dict(
            previous_frame=int(frames[t]),
            frame_idx=int(frames[t + 1]),
            previous_permutation=permutation[t].tolist(),
            current_permutation=permutation[t + 1].tolist(),
            raw_step_mm=raw_steps[t].tolist(),
            id_step_mm=id_steps[t].tolist(),
            raw_both_direct=raw_direct[t].tolist(),
            id_both_direct=id_direct[t].tolist(),
        )
        for t in boundaries
    ]
    followed = []
    # Follow the ID owning the previous raw endpoint; do not compare same-number
    # rows after a permutation change or confuse lost support with a small step.
    for event in raw["large_events"]:
        if event["mode"] != "both_direct":
            continue
        t = event["previous_frame"] - int(frames[0])
        body, node = event["body"], NODES.index(event["node"])
        identity = int(np.flatnonzero(permutation[t] == body)[0])
        supported = bool(id_direct[t, identity, node])
        distance = float(id_steps[t, identity, node])
        followed.append(
            dict(
                **event,
                ID_at_previous_endpoint=identity,
                mapped_current_raw_slot=int(permutation[t + 1, identity]),
                mapped_distance_mm=distance,
                mapped_both_direct=supported,
                permutation_boundary=bool(t in boundaries),
                mapped_outcome="not_both_direct"
                if not supported
                else "persists_gt10mm"
                if distance > THRESHOLD_MM
                else "at_most_10mm",
            )
        )
    return dict(
        raw=raw,
        saved_beam_ID_aligned=aligned,
        permutation_boundaries=boundary_records,
        unchanged_permutation_adjacent_pairs=int(unchanged.sum()),
        per_node_distance_and_direct_multisets_preserved_off_boundaries=True,
        raw_direct_large_event_ID_followup=followed,
    )


def focused_samples(result, frames, pose, observed, permutation):
    wanted = set()
    for record in (
        result["raw"]["large_events"]
        + result["saved_beam_ID_aligned"]["large_events"]
        + result["permutation_boundaries"]
    ):
        wanted.update((record["previous_frame"], record["frame_idx"]))
    selected = np.searchsorted(frames, sorted(wanted))
    return dict(
        frame_idx=frames[selected].copy(),
        pose_xyz=pose[selected].copy(),
        observed=observed[selected].copy(),
        permutation_by_id=permutation[selected].copy(),
    )


def run(output):
    output = Path(output).resolve()
    require(EXECUTION_READY, "source/CPU preparation only; execution disabled")
    require(
        output.is_relative_to(WORK.resolve()) and output != WORK.resolve() and not output.exists(),
        "fresh output inside workspace required",
    )
    require("torch" not in sys.modules, "CPU diagnostic must not import tracking/GPU modules")
    evidence = {}
    comp, ids, reference, saved, id_summary = load_headers(evidence)
    candidate_path = member(COMPARISON, comp, f"candidates/{ARM}.npz", evidence)
    mapping_path = member(IDENTITY, ids, "identity_mapping_and_ambiguity.npz", evidence)
    raw = selected_archive(candidate_path, CANDIDATE_FIELDS, ("frame_idx", "pose_xyz", "observed"))
    mapping = selected_archive(mapping_path, MAPPING_FIELDS, ("frame_idx", "permutation_by_id"))
    signatures = {k: signature(v) for k, v in raw.items()}
    mapping_signatures = {k: signature(v) for k, v in mapping.items()}
    require(
        all(signatures[k] == reference["arrays"][k] for k in raw), "decoded candidate field differs"
    )
    require(
        np.array_equal(raw["frame_idx"], np.arange(FIRST, STOP, dtype=np.int64))
        and np.array_equal(mapping["frame_idx"], np.arange(START, STOP, dtype=np.int64))
        and mapping["frame_idx"].dtype == np.int64,
        "exact archive frame ranges required",
    )
    keep = slice(START - FIRST, STOP - FIRST)
    frames, pose, observed = (raw[k][keep] for k in ("frame_idx", "pose_xyz", "observed"))
    permutation = mapping["permutation_by_id"]
    require(np.array_equal(permutation[0], [0, 1, 2]), "scientific initial P012 required")
    result = compare_motion(frames, pose, observed, permutation)
    raw_check = verify_raw_reference(
        result["raw"], saved, raw["frame_idx"], raw["pose_xyz"], START, STOP
    )
    require(all(signature(v) == signatures[k] for k, v in raw.items()), "raw input mutated")
    require(
        all(signature(v) == mapping_signatures[k] for k, v in mapping.items()), "ID input mutated"
    )
    for name, row in evidence.items():
        stat = Path(name).stat()
        require(
            (stat.st_size, stat.st_mtime_ns) == (row["size_bytes"], row["mtime_ns"]),
            "captured input changed",
        )
    summary = dict(
        schema=STAGE,
        status="ok",
        native_interval=[START, STOP],
        native_frames=STOP - START,
        adjacent_frame_pairs=STOP - START - 1,
        threshold_mm=THRESHOLD_MM,
        raw_metric_qualification=raw_check,
        **result,
        decoded_candidate_signatures=signatures,
        decoded_mapping_signatures=mapping_signatures,
        nonfinite_estimated_coordinates=0,
        beam_uncertainty={
            k: id_summary[k]
            for k in (
                "pruning_occurred",
                "finite_prefixes_discarded",
                "gaps_conditional_on_retained_graph",
                "conditional_tied_frames",
                "unknown_ID_alternative_cells",
                "global_optimality_certified",
            )
        },
        raw_geometry_changed=False,
        support_changed=False,
        review_read_or_cleared=False,
        physical_ID_truth=False,
        solver_or_fitting_called=False,
        GPU_used=False,
        interpretation=[
            "One whole-body permutation is applied to coordinates and every node's support.",
            "All-estimated includes unsupported predictions; both-direct requires both endpoints.",
            "Steps in mm/native frame are estimated geometry, not measured animal speed.",
            "The saved beam objective uses chest and mouth [1,0], not a direct tail-motion term; "
            "fitted body nodes remain coupled.",
            "A reduced step under saved approximate IDs does not establish physical continuity.",
            "Only three candidate and two mapping fields were decoded; "
            "other fields inherit sealed archive integrity.",
        ],
    )
    output.mkdir()
    samples = focused_samples(result, frames, pose, observed, permutation)
    samples_path = output / "event_boundary_samples.npz"
    np.savez_compressed(samples_path, **samples)
    summary["focused_samples"] = dict(
        native_frames=len(samples["frame_idx"]),
        fields={k: signature(v) for k, v in samples.items()},
        scope="Raw poses/support and saved P at all large-event and boundary endpoints; "
        "no geometry correction.",
    )
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    seal = dict(
        stage=STAGE,
        status="ok",
        created_at_utc=datetime.now(UTC).isoformat(),
        inputs=evidence,
        source_sha256=sha(Path(__file__)),
        numpy_version=np.__version__,
        outputs={
            "summary.json": dict(
                path=str(summary_path),
                sha256=sha(summary_path),
                size_bytes=summary_path.stat().st_size,
            ),
            "event_boundary_samples.npz": dict(
                path=str(samples_path),
                sha256=sha(samples_path),
                size_bytes=samples_path.stat().st_size,
            ),
        },
    )
    (output / "diagnostic.provenance.json").write_text(
        json.dumps(seal, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    run(args.output)
