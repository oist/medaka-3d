"""Three fixed adjacent pairs: saved Sparse samples versus final Depth raw/ID motion.

No full-interval rescoring, fitting, biological-ID alignment, or review changes.
The inherited helpers select saved data only; their workflow entrypoints are unused.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import diagnose_sparse_id_aligned_motion as motion
import numpy as np
import score_variant_female_five_minute_anchors as female

depth, proof = female.depth, female.proof
WORK = Path(__file__).resolve().parent
FIRST, START, STOP = 49800, 66400, 116200
PAIRS = ((80424, 80425), (82719, 82720), (88372, 88373))
FRAMES = np.array([frame for pair in PAIRS for frame in pair], dtype=np.int64)
SPARSE, DEPTH = "sparse_instance_rescore", "expanded_depth_instance_rescore"
STAGE = "sparse_depth_three_pair_motion_v1"
SPARSE_ROOT = WORK / motion.STAGE
EXECUTION_READY = True
MOTION_SHA = "abb5bbf8ce97782a03bf3297675e3141b5eedae36d7bf3c655c63a639290299b"
FEMALE_FALSE_SHA = "59c989a179576049796d495038044c962e6c4b83c79b0ecd6d5def0823ac072c"
FEMALE_TRUE_SHA = "e2922be13f11966d883bdd74ab96955d3d8ed7d755d4105f895a2d6695b7c30a"
DEPTH_TRUE_SHA = "a521bab6e074403ed5fccb73c8929814d72f2734d186ac13fdead129c2ae119e"
SAMPLE_FIELDS = {"frame_idx", "pose_xyz", "observed", "permutation_by_id"}


def normalized_flag_digest(data):
    """Normalize only the executable top-level assignment; quoted examples stay intact."""
    matches = list(re.finditer(rb"(?m)^EXECUTION_READY = (True|False)(?=\r?$)", data))
    motion.require(len(matches) == 1, "one exact execution assignment required")
    flag = matches[0]
    normalized = data[: flag.start(1)] + b"False" + data[flag.end(1) :]
    return hashlib.sha256(normalized).hexdigest()


def bind_sources(evidence):
    data = Path(female.__file__).read_bytes()
    motion.require(
        normalized_flag_digest(data) == FEMALE_FALSE_SHA
        and hashlib.sha256(data).hexdigest() in {FEMALE_FALSE_SHA, FEMALE_TRUE_SHA},
        "female helper changed beyond its executable flag",
    )
    named = (
        (Path(motion.__file__), MOTION_SHA),
        (Path(female.__file__), hashlib.sha256(data).hexdigest()),
        (Path(depth.__file__), DEPTH_TRUE_SHA),
        (Path(depth.comparison.__file__), depth.COMPARISON_SHA),
        (Path(depth.frozen.__file__), depth.FROZEN_SHA),
        (Path(depth.frozen.beam.__file__), depth.BEAM_SHA),
    )
    for path, digest in named:
        evidence.add(path, digest)
    return {str(path.resolve()): digest for path, digest in named}


def load_headers(comparison_sha, depth_id_sha, sparse_motion_sha, evidence):
    sources = bind_sources(evidence)
    common_seal, binding = depth.comparison_headers(comparison_sha, evidence)
    common_plan = depth.document(depth.COMPARISON_ROOT, common_seal, "plan.json", evidence)
    references = common_plan["candidate_references"]
    identity = female.identity_headers(
        WORK / depth.STAGE,
        depth_id_sha,
        Path(depth.__file__),
        DEPTH_TRUE_SHA,
        DEPTH,
        references[DEPTH],
        depth.COMPARISON_ROOT,
        comparison_sha,
        depth.COMPARISON_SHA,
        evidence,
    )
    # Explicit direct helper dependencies, not a dynamic sys.modules inventory.
    for module in (proof, depth.comparison.common):
        path = Path(module.__file__).resolve()
        digest = common_seal["inputs"][str(path)]
        evidence.add(path, digest)
        sources[str(path)] = digest
    sparse = female.read_seal(
        SPARSE_ROOT, "diagnostic.provenance.json", sparse_motion_sha, motion.STAGE, evidence
    )
    motion.require(
        sparse["source_sha256"] == MOTION_SHA, "Sparse diagnostic implementation differs"
    )
    saved = depth.document(SPARSE_ROOT, sparse, "summary.json", evidence)
    motion.require(
        saved["schema"] == motion.STAGE
        and saved["status"] == "ok"
        and saved["native_interval"] == [START, STOP]
        and saved["native_frames"] == STOP - START
        and saved["focused_samples"]["native_frames"] == 273
        and set(saved["focused_samples"]["fields"]) == SAMPLE_FIELDS,
        "complete saved Sparse sample contract required",
    )
    motion.require(
        saved["raw_metric_qualification"]["raw_direct_counts_and_maxima_equal"]
        and not saved["raw_geometry_changed"]
        and not saved["support_changed"]
        and not saved["review_read_or_cleared"]
        and not saved["physical_ID_truth"],
        "Sparse diagnostic preservation/qualification differs",
    )
    full_signatures = saved["decoded_candidate_signatures"]
    motion.require(
        set(full_signatures) == {"frame_idx", "pose_xyz", "observed"}
        and all(v == references[SPARSE]["arrays"][k] for k, v in full_signatures.items()),
        "Sparse samples do not derive from same common final candidate",
    )
    for path, digest in (
        (motion.COMPARISON / "comparison.provenance.json", motion.COMPARISON_SHA),
        (motion.IDENTITY / "identity.provenance.json", motion.IDENTITY_SHA),
        (motion.COMPARISON / f"candidates/{SPARSE}.npz", motion.CANDIDATE_SHA),
        (motion.IDENTITY / "identity_mapping_and_ambiguity.npz", motion.MAPPING_SHA),
    ):
        motion.require(
            sparse["inputs"].get(str(path.resolve()), {}).get("sha256") == digest,
            "Sparse diagnostic consumed input edge differs",
        )
    return dict(
        common_seal=common_seal,
        reference=binding["raw_candidate_reference"],
        identity=identity,
        sparse_seal=sparse,
        sparse_summary=saved,
        sources=sources,
    )


def select_native(arrays, requested=FRAMES):
    frames = arrays["frame_idx"]
    motion.require(
        frames.dtype == np.int64
        and frames.ndim == 1
        and len(frames) > 0
        and np.all(np.diff(frames) > 0),
        "unique ordered native index required",
    )
    rows = np.searchsorted(frames, requested)
    motion.require(
        np.all(rows < len(frames)) and np.array_equal(frames[rows], requested),
        "every requested native endpoint must exist",
    )
    motion.require(
        all(len(value) == len(frames) for value in arrays.values()),
        "selected arrays must share the frame axis",
    )
    return {key: value[rows].copy() for key, value in arrays.items()}


def check_mapping(frames, permutation, *, full_science):
    motion.require(
        frames.dtype == np.int64
        and frames.ndim == 1
        and permutation.shape == (len(frames), 3)
        and permutation.dtype.kind in "iu"
        and np.array_equal(
            np.sort(permutation, axis=1), np.broadcast_to(np.arange(3), permutation.shape)
        ),
        "whole-body ID-to-raw-slot permutation required",
    )
    if full_science:
        motion.require(
            np.array_equal(frames, np.arange(START, STOP, dtype=np.int64))
            and np.array_equal(permutation[0], [0, 1, 2]),
            "full scientific native range and original P012 required",
        )


def load_selected(headers, evidence):
    sample_path = depth.comparison.common.bound_member(
        SPARSE_ROOT, headers["sparse_seal"], "event_boundary_samples.npz", evidence
    )
    samples = motion.selected_archive(sample_path, SAMPLE_FIELDS, SAMPLE_FIELDS)
    motion.require(
        len(samples["frame_idx"]) == 273
        and {k: motion.signature(v) for k, v in samples.items()}
        == headers["sparse_summary"]["focused_samples"]["fields"],
        "saved Sparse sample signatures differ",
    )
    sparse = select_native(samples)
    candidate_path = depth.comparison.common.bound_member(
        depth.COMPARISON_ROOT, headers["common_seal"], f"candidates/{DEPTH}.npz", evidence
    )
    raw = motion.selected_archive(
        candidate_path, motion.CANDIDATE_FIELDS, ("frame_idx", "pose_xyz", "observed")
    )
    motion.require(
        all(motion.signature(v) == headers["reference"]["arrays"][k] for k, v in raw.items()),
        "Depth candidate three-field signatures differ",
    )
    motion.require(
        np.array_equal(raw["frame_idx"], np.arange(FIRST, STOP, dtype=np.int64))
        and raw["frame_idx"].dtype == np.int64
        and raw["pose_xyz"].shape == (STOP - FIRST, 3, 3, 3)
        and raw["pose_xyz"].dtype == np.float64
        and raw["observed"].shape == (STOP - FIRST, 3, 3)
        and raw["observed"].dtype == np.bool_,
        "Depth retained geometry extent differs",
    )
    ids = headers["identity"]
    mapping_path = depth.comparison.common.bound_member(
        ids["root"], ids["seal"], "identity_mapping_and_ambiguity.npz", evidence
    )
    mapping = motion.selected_archive(
        mapping_path, motion.MAPPING_FIELDS, ("frame_idx", "permutation_by_id")
    )
    check_mapping(mapping["frame_idx"], mapping["permutation_by_id"], full_science=True)
    selected = select_native(raw)
    chosen_mapping = select_native(mapping)
    selected["permutation_by_id"] = chosen_mapping["permutation_by_id"]
    for arm in (sparse, selected):
        check_mapping(arm["frame_idx"], arm["permutation_by_id"], full_science=False)
        for value in arm.values():
            value.setflags(write=False)
    return {SPARSE: sparse, DEPTH: selected}


def evaluate_pairs(arms):
    results = []
    for first, second in PAIRS:
        entry = dict(native_pair=[first, second], arms={})
        for name, arrays in arms.items():
            pair = select_native(arrays, np.array([first, second], np.int64))
            frames, pose, support, permutation = (
                pair[k] for k in ("frame_idx", "pose_xyz", "observed", "permutation_by_id")
            )
            motion.validate_geometry(frames, pose, support)
            mapped_pose, mapped_support = motion.reorder_whole_body(pose, support, permutation)
            # Each 2-frame pair is contiguous. The six-frame selection is not.
            checked = motion.compare_motion(frames, pose, support, permutation)
            row = dict(
                permutation_by_id=permutation.tolist(),
                permutation_changed=bool(np.any(permutation[0] != permutation[1])),
                raw_direct_large_event_ID_followup=checked["raw_direct_large_event_ID_followup"],
            )
            for mode, geometry, observed in (
                ("raw", pose, support),
                ("saved_beam_ID", mapped_pose, mapped_support),
            ):
                distances = np.linalg.norm(geometry[1] - geometry[0], axis=-1)
                direct = observed[0] & observed[1]
                row[mode] = dict(
                    distance_mm=distances.tolist(),
                    previous_direct=observed[0].tolist(),
                    current_direct=observed[1].tolist(),
                    both_direct=direct.tolist(),
                    gt10mm_all_estimated=(distances > motion.THRESHOLD_MM).tolist(),
                    gt10mm_both_direct=((distances > motion.THRESHOLD_MM) & direct).tolist(),
                )
            entry["arms"][name] = row
        results.append(entry)
    return results


def run(comparison_sha, depth_id_sha, sparse_motion_sha, output):
    motion.require(EXECUTION_READY, "source/CPU preparation only")
    output = Path(output).resolve()
    motion.require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output required",
    )
    evidence = proof.Evidence()
    headers = load_headers(comparison_sha, depth_id_sha, sparse_motion_sha, evidence)
    arms = load_selected(headers, evidence)
    before = {
        arm: {k: motion.signature(v) for k, v in arrays.items()} for arm, arrays in arms.items()
    }
    results = evaluate_pairs(arms)
    motion.require(
        before
        == {
            arm: {k: motion.signature(v) for k, v in arrays.items()} for arm, arrays in arms.items()
        },
        "selected source data mutated",
    )
    depth.cpu_only()
    out = proof.Output(output)
    for arm, arrays in arms.items():
        out.write(f"selected/{arm}.npz", arrays)
    summary = dict(
        schema=STAGE,
        status="ok",
        native_pairs=[list(pair) for pair in PAIRS],
        nodes=list(motion.NODES),
        selected_array_signatures=before,
        pairs=results,
        comparison_seal_sha256=comparison_sha,
        depth_ID_seal_sha256=depth_id_sha,
        sparse_motion_seal_sha256=sparse_motion_sha,
        depth_mapping_uncertainty={
            k: headers["identity"]["summary"][k]
            for k in (
                "pruning_occurred",
                "gaps_conditional_on_retained_graph",
                "conditional_tied_frames",
                "unknown_ID_alternative_cells",
                "global_optimality_certified",
            )
        },
        sparse_mapping_uncertainty=headers["sparse_summary"]["beam_uncertainty"],
        sparse_original_full_candidate_or_mapping_decoded=False,
        Depth_candidate_fields_decoded=["frame_idx", "pose_xyz", "observed"],
        Depth_mapping_fields_decoded=["frame_idx", "permutation_by_id"],
        whole_five_minute_rescoring=False,
        candidate_modified=False,
        support_modified=False,
        review_read_or_cleared=False,
        physical_ID_truth=False,
        GPU_used=False,
        limitations=[
            "These pairs were fixed from two known Sparse large steps and a permutation boundary, "
            "not sampled independently for Depth.",
            "Numeric raw slots and initial ID labels are not independently matched biological "
            "fish across arms.",
            "All estimated poses include latent unsupported nodes; only both_direct has saved "
            "direct support at both endpoints.",
            "Saved beam paths use mouth and chest, with gaps conditional on retained hypotheses; "
            "tails remain coupled through body fitting.",
            "These three selected pairs do not measure whole-interval performance or validate "
            "physical identity.",
            "No human confirmation is supplied by this coordinate/support diagnostic.",
        ],
    )
    out.write("summary.json", summary)
    evidence.final_guard()
    out.final_guard()
    with (output / "comparison.provenance.json").open("xb") as stream:
        stream.write(
            proof.json_bytes(
                dict(
                    stage=STAGE,
                    status="ok",
                    inputs=evidence.paths,
                    outputs=out.records,
                    source_sha256=motion.sha(Path(__file__)),
                    helper_sources=headers["sources"],
                    numpy_version=np.__version__,
                )
            )
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--depth-id-sha", required=True)
    parser.add_argument("--sparse-motion-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.comparison_sha, args.depth_id_sha, args.sparse_motion_sha, args.output)
