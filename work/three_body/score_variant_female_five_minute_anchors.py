"""Disabled CPU female57 evaluation of successful Sparse -6 / Depth -12 results.

Reuse the sealed female annotations and their saved SLEAP-row correspondence.
No SL, mask, video, original candidate, checkpoint39, or identity graph decoding.
Only the two comparison copies (five fields) and two small mappings are decoded.
Sparse's separately reviewed execution caller is an explicit input: d433 stays frozen.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import pairwise
from pathlib import Path

import numpy as np
import run_expanded_depth_five_minute_identity_beam as depth
import score_female_five_minute_anchors as prior

proof, require, comparison = depth.proof, depth.require, depth.comparison
WORK, REPO, START, STOP = depth.WORK, depth.REPO, depth.START, depth.STOP
ARMS = comparison.ARMS
STAGE = "variant_female_57_five_minute_endpoint_evaluation_v1"
EXECUTION_READY = True
PRIOR_ROOT = WORK / prior.STAGE
PRIOR_SEAL_SHA = "122b66af931510b9950709a07397f71a70ef31898870de85af3a6ce047137539"
PRIOR_SHA = "b50e7668b91c6474edfe9d5d2150deeda8a823c11c7a08d5c451e6709f616165"
DEPTH_REVIEWED_FALSE_SHA = "fbb61641846cf53676b8960ecfd7a356f70cc908ba2764c18ebf97cd86f0f6b5"
DEPTH_EXECUTION_SHA = "a521bab6e074403ed5fccb73c8929814d72f2734d186ac13fdead129c2ae119e"
SPARSE_SOURCE = WORK / "execute_sparse_instance_five_minute_identity_beam.py"
SPARSE_EXECUTION_SHA = "07d9937488057e620f4478a6458e949118d28c8a4ad8e9e75d41eb84e4f12609"
COMPARISON_SHA = "db30f8fd32a58a60b77ef1daa50dbf42da7a0d181eb4b43da47cdf2b2ee2694e"
MAPPING_FIELDS = (
    "frame_idx",
    "path_state",
    "permutation_by_id",
    "conditional_identity_alternative_gap",
    "conditional_identity_alternative_status",
)


def reviewed_flag_bytes(data):
    """Validate only the permitted flag transition; never rewrite a source."""
    old, new = b"EXECUTION_READY = False", b"EXECUTION_READY = True"
    require((data.count(old), data.count(new)) in ((1, 0), (0, 1)), "one execution flag required")
    return data.replace(new, old)


def source_code():
    # Do not call prior.control.source_code(): it pins unused execution entrypoints.
    code = comparison.source_code()
    expected = {
        Path(prior.__file__).resolve(): PRIOR_SHA,
        Path(comparison.__file__).resolve(): COMPARISON_SHA,
        Path(depth.frozen.__file__).resolve(): depth.FROZEN_SHA,
        Path(prior.generative.__file__).resolve(): prior.GENERATIVE_SHA,
    }
    for path, digest in expected.items():
        require(
            proof.sha(path) == digest and code.get(str(path)) == digest,
            "frozen reusable source differs: " + path.name,
        )
    require(
        hashlib.sha256(reviewed_flag_bytes(Path(depth.__file__).read_bytes())).hexdigest()
        == DEPTH_REVIEWED_FALSE_SHA,
        "Depth ID source changed beyond its execution flag",
    )
    require(str(Path(__file__).resolve()) in code, "new source absent from actual closure")
    return code


def read_seal(root, filename, digest, stage, evidence):
    root = Path(root).resolve()
    require(
        root.is_relative_to(WORK) and proof.valid_sha(digest), "explicit workspace seal required"
    )
    path = root / filename
    evidence.add(path, digest)
    seal = proof.read_json(path)
    require(seal.get("stage") == stage and seal.get("status") == "ok", "successful stage required")
    return seal


def document(root, seal, name, evidence):
    return depth.document(root, seal, name, evidence)


def validate_saved_anchors(fixed, observations):
    require(
        fixed["frames"] == list(prior.FRAMES)
        and fixed["CSV_sha256"] == prior.GT_SHA
        and fixed["source_interval"] == [START, STOP]
        and fixed["labels_created"] is False,
        "fixed57 external annotation contract differs",
    )
    rows = observations["matches"]
    require(
        [a["frame"] for a in rows] == list(prior.FRAMES)
        and [a["frame"] for a in observations["diagnostics"]] == list(prior.FRAMES),
        "saved match frame denominator/order differs",
    )
    for row in rows:
        require(
            set(row) == {"frame", "detection_rows", "pixels", "match_errors_px"},
            "saved MouthAnchor schema differs",
        )
        require(
            set(row["detection_rows"]) == set(row["match_errors_px"])
            and set(row["detection_rows"]) <= set(row["pixels"]) <= set(prior.SIDES),
            "saved match view scope differs",
        )
        require(
            all(type(v) is int and v >= 0 for v in row["detection_rows"].values()),
            "saved detection row must be nonnegative integer",
        )
        require(
            all(np.isfinite(v) and 0 <= v <= 5 for v in row["match_errors_px"].values()),
            "saved match exceeds frozen 5px rule",
        )
        for xy in row["pixels"].values():
            require(np.asarray(xy).shape == (2,) and np.isfinite(xy).all(), "saved mouth pixels")
    return [prior.generative.MouthAnchor(**row) for row in rows]


def load_saved_anchors(new_comparison, evidence):
    seal = read_seal(
        PRIOR_ROOT, "evaluation.provenance.json", PRIOR_SEAL_SHA, prior.STAGE, evidence
    )
    depth.edge(seal, Path(prior.__file__), PRIOR_SHA)
    summary = document(PRIOR_ROOT, seal, "summary.json", evidence)
    fixed = document(PRIOR_ROOT, seal, "fixed_female_anchor_set.json", evidence)
    observations = document(PRIOR_ROOT, seal, "anchor_observations.json", evidence)
    require(
        summary["requested_anchors"] == 57
        and summary["requested_adjacent_intervals"] == 56
        and summary["annotation_CSV_sha256"] == prior.GT_SHA
        and summary["female_endpoint_evaluation_only"] is True
        and summary["same_horizon_first11_control_exact"] is True,
        "successful female57 scope differs",
    )
    old_root = prior.control.COMPARISON_ROOT
    old_sha = prior.control.COMPARISON_SEAL_SHA
    depth.edge(seal, old_root / "comparison.provenance.json", old_sha)
    old_seal = read_seal(
        old_root, "comparison.provenance.json", old_sha, prior.control.comparison.STAGE, evidence
    )
    old_plan = document(old_root, old_seal, "plan.json", evidence)
    shared = old_plan["shared_SL_calibration_sha256"]
    require(len(shared) == 7, "three SL and four calibration references required")
    for path, digest in shared.items():
        depth.edge(old_seal, Path(path), digest)
        depth.edge(new_comparison, Path(path), digest)
    # These are sealed input-edge checks only; no SL/calibration payload is reread.
    for view in prior.SIDES:
        paths = [p for p in shared if Path(p).name == f"20251225_K_1_2_FCMC__{view}.npz"]
        require(len(paths) == 1, "one matching side SLEAP input")
        depth.edge(seal, Path(paths[0]), shared[paths[0]])
    return validate_saved_anchors(fixed, observations), dict(
        evaluation_seal_sha256=PRIOR_SEAL_SHA,
        fixed_anchor_set=fixed,
        fixed_anchor_set_output=seal["outputs"]["fixed_female_anchor_set.json"],
        matches_output=seal["outputs"]["anchor_observations.json"],
        shared_SL_calibration_sha256=shared,
        SLEAP_rematching_performed=False,
        original_annotations_and_saved_matching_unchanged=True,
    )


def same_candidate_reference(actual, expected, *, legacy_sparse=False, tracking_sha=None):
    common = {"path", "sha256", "milestone_path", "milestone_sha256", "terminal_horizon", "arrays"}
    expected_keys = common | {"tracking_seal_sha256"}
    actual_keys = common if legacy_sparse else expected_keys
    require(
        set(actual) == actual_keys
        and set(expected) == expected_keys
        and all(actual[k] == expected[k] for k in common)
        and (tracking_sha if legacy_sparse else actual["tracking_seal_sha256"])
        == expected["tracking_seal_sha256"]
        and actual["terminal_horizon"] == STOP
        and len(actual["arrays"]) == 21,
        "identity and common comparison do not bind the same original final/milestone21",
    )


def identity_headers(
    root,
    digest,
    source,
    source_sha,
    arm,
    reference,
    expected_comparison_root,
    expected_comparison_sha,
    expected_comparison_source_sha,
    evidence,
):
    """Actual caller source is explicit; no old-stage fabrication or auto-discovery."""
    root, source = Path(root).resolve(), Path(source).resolve()
    require(source.is_relative_to(WORK) and source.suffix == ".py", "workspace caller source")
    evidence.add(source, source_sha)
    seal = read_seal(root, "identity.provenance.json", digest, root.name, evidence)
    depth.edge(seal, source, source_sha)
    legacy_sparse = arm == ARMS[0]
    if not legacy_sparse:
        depth.edge(seal, Path(depth.frozen.__file__), depth.FROZEN_SHA)
    depth.edge(seal, Path(depth.frozen.beam.__file__), depth.BEAM_SHA)
    depth.edge(
        seal, expected_comparison_root / "comparison.provenance.json", expected_comparison_sha
    )
    plan = document(root, seal, "plan.json", evidence)
    summary = document(root, seal, "summary.json", evidence)
    scalars = document(root, seal, "beam_scalars.json", evidence)
    body_seal = read_seal(
        expected_comparison_root,
        "comparison.provenance.json",
        expected_comparison_sha,
        expected_comparison_root.name,
        evidence,
    )
    depth.edge(
        body_seal,
        Path(depth.frozen.comparison.__file__) if legacy_sparse else Path(comparison.__file__),
        expected_comparison_source_sha,
    )
    body_plan = document(expected_comparison_root, body_seal, "plan.json", evidence)
    tracking_sha = (
        body_plan["tracking_seals"][arm] if legacy_sparse else reference["tracking_seal_sha256"]
    )
    depth.edge(
        body_seal,
        Path(reference["path"]).parent / "tracking.provenance.json",
        reference["tracking_seal_sha256"],
    )
    for doc in (plan, summary):
        require(
            doc["schema"] == root.name
            and doc["native_interval"] == [START, STOP]
            and doc["native_frames"] == STOP - START
            and doc["terminal_horizon"] == STOP
            and doc["comparison_seal_sha256"] == expected_comparison_sha
            and doc["comparison_source_sha256"] == expected_comparison_source_sha
            and doc["raw21_signatures"] == reference["arrays"]
            and doc["global_optimality_certified"] is False
            and doc["physical_ID_truth"] is False
            and doc["review_flags_cleared"] is False,
            "identity scope/source/truth contract differs",
        )
        same_candidate_reference(
            doc["raw_candidate_reference"],
            reference,
            legacy_sparse=legacy_sparse,
            tracking_sha=tracking_sha,
        )
    require(
        plan["source_sha256"].get(str(source)) == source_sha
        and plan["dp_config"] == depth.CONFIG
        and plan["initial_P"] == [0, 1, 2]
        and plan["initialization_frame"] == START
        and plan["node_indices"] == [1, 0]
        and summary["solve_calls"] == 1
        and summary["beam_width"] == 256
        and summary["intermediate_ID_resets"] == 0
        and summary["original_raw21_unchanged"] is True
        and summary["raw_sources_review_unchanged"] is True
        and summary["gaps_conditional_on_retained_graph"] is True,
        "identity algorithm or preservation contract differs",
    )
    require(
        body_plan["candidate_references"][arm] == plan["raw_candidate_reference"],
        "actual identity comparison arm/reference differs",
    )
    copy_name = f"candidates/{arm}.npz"
    copy_descriptor = body_seal["outputs"][copy_name]
    require(
        plan["consumed_comparison_copy"] == summary["consumed_comparison_copy"] == copy_descriptor
        and Path(copy_descriptor["path"]).resolve()
        == (expected_comparison_root / copy_name).resolve()
        and type(copy_descriptor["size_bytes"]) is int
        and copy_descriptor["size_bytes"] > 0,
        "identity consumed comparison copy differs",
    )
    depth.edge(seal, Path(copy_descriptor["path"]), copy_descriptor["sha256"])
    return dict(
        root=root,
        seal=seal,
        scalars=scalars,
        source_sha256=source_sha,
        seal_sha256=digest,
        summary=summary,
    )


def load_mapping(bundle):
    root, seal, evidence = bundle["root"], bundle["seal"], bundle["evidence"]
    path = comparison.common.bound_member(
        root, seal, "identity_mapping_and_ambiguity.npz", evidence
    )
    with np.load(path, allow_pickle=False) as z:
        require(
            len(z.files) == len(set(z.files)) == 14
            and set(z.files) == set(depth.frozen.MAPPING_FIELDS),
            "mapping14 schema",
        )
        arrays = {key: z[key] for key in MAPPING_FIELDS}
    prior.validate_mapping(arrays, tolerance=bundle["scalars"]["numerical_tie_tolerance"])
    return {k: v[np.asarray(prior.FRAMES) - START].copy() for k, v in arrays.items()}


def label_change_brackets(score):
    """Secondary endpoint observation; never a localized swap event/count."""
    rows = score["anchors"]
    available = [(i, row) for i, row in enumerate(rows) if row["identity"] is not None]
    return [
        dict(
            left_available=a,
            right_available=b,
            unavailable_native_frames=[r["frame"] for r in rows[i + 1 : j]],
            event_time_localized=False,
            physical_swap_count_inferred=False,
        )
        for (i, a), (j, b) in pairwise(available)
        if j > i + 1 and a["identity"] != b["identity"]
    ]


def evaluate(rows, anchors, mapping):
    raw = prior.score_rows(rows, anchors)
    automatic = prior.score_rows(prior.reorder_rows(rows, mapping["permutation_by_id"]), anchors)
    for t, item in enumerate(automatic["endpoint_details"]):
        identity = item["raw_result"]["identity"]
        item.update(
            raw_slot_by_ID=mapping["permutation_by_id"][t].tolist(),
            conditional_ID_alternative_status=mapping["conditional_identity_alternative_status"][
                t
            ].tolist(),
            conditional_ID_alternative_gap=depth.explicit_json(
                mapping["conditional_identity_alternative_gap"][t]
            ),
            same_as_initial_numeric_ID1=None if identity is None else identity == 1,
            initial_ID1_sex_certified=False,
        )
    return dict(
        raw_slots=raw,
        automatic_IDs=automatic,
        secondary_label_change_brackets={
            name: {
                mode: label_change_brackets(value[mode]) for mode in ("raw", "with_original_review")
            }
            for name, value in (("raw_slots", raw), ("automatic_IDs", automatic))
        },
    )


def run(args):
    require(EXECUTION_READY, "source-only: ROOT review required before actual variant scoring")
    depth.cpu_only()
    output = args.output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        code = source_code()
        for path, digest in code.items():
            evidence.add(path, digest)
        common_seal, _ = depth.comparison_headers(args.comparison_sha, evidence)
        common_plan = document(depth.COMPARISON_ROOT, common_seal, "plan.json", evidence)
        references = common_plan["candidate_references"]
        anchors, label_binding = load_saved_anchors(common_seal, evidence)
        sparse_root = WORK / depth.frozen.STAGE
        sparse_seal = read_seal(
            sparse_root,
            "identity.provenance.json",
            args.sparse_id_sha,
            depth.frozen.STAGE,
            evidence,
        )
        sparse_summary = document(sparse_root, sparse_seal, "summary.json", evidence)
        sparse_comparison_sha = sparse_summary["comparison_seal_sha256"]
        # All success/schema/source gates precede candidate/mapping decoding.
        bundles = {
            ARMS[0]: identity_headers(
                sparse_root,
                args.sparse_id_sha,
                SPARSE_SOURCE,
                SPARSE_EXECUTION_SHA,
                ARMS[0],
                references[ARMS[0]],
                depth.frozen.COMPARISON_ROOT,
                sparse_comparison_sha,
                depth.frozen.COMPARISON_SHA,
                evidence,
            ),
            ARMS[1]: identity_headers(
                WORK / depth.STAGE,
                args.depth_id_sha,
                Path(depth.__file__),
                DEPTH_EXECUTION_SHA,
                ARMS[1],
                references[ARMS[1]],
                depth.COMPARISON_ROOT,
                args.comparison_sha,
                COMPARISON_SHA,
                evidence,
            ),
        }
        results = {}
        for arm in ARMS:
            path = comparison.common.bound_member(
                depth.COMPARISON_ROOT, common_seal, f"candidates/{arm}.npz", evidence
            )
            descriptor = common_seal["outputs"][f"candidates/{arm}.npz"]
            selected = prior.load_rows(
                dict(path=str(path), sha256=descriptor["sha256"], arrays=references[arm]["arrays"]),
                evidence,
            )
            bundles[arm]["evidence"] = evidence
            results[arm] = evaluate(selected, anchors, load_mapping(bundles[arm]))
        out.write("fixed_female_anchor_set.json", label_binding["fixed_anchor_set"])
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                requested_anchors=57,
                requested_adjacent_intervals=56,
                native_interval=[START, STOP],
                results=results,
                candidate_references=references,
                comparison_seal_sha256=args.comparison_sha,
                label_binding=label_binding,
                identity_bindings={
                    a: dict(seal_sha256=b["seal_sha256"], source_sha256=b["source_sha256"])
                    for a, b in bundles.items()
                },
                source_sha256=code,
                female_endpoint_evaluation_only=True,
                all57_external_annotations_and_SL_matches_reused=True,
                new_training_or_threshold_tuning=False,
                all_three_physical_ID_truth=False,
                dense_swap_zero_certified=False,
                reviews_cleared=False,
                geometry_modified=False,
                conditional_gaps_are_probabilities=False,
                SL_payload_reads=0,
                raw_video_mask_reads=0,
                checkpoint39_reads=0,
                original_final_milestone_payload_reads=0,
                comparison_candidate_decodes=2,
                candidate_fields_decoded=5,
                mapping_decodes=2,
                identity_graph_reads=0,
                identity_solves=0,
                GPU_used=False,
                primary_adjacent_interval_metric_unchanged=True,
                no_bridging_for_primary=True,
            ),
        )
        require(source_code() == code, "source closure changed")
        depth.cpu_only()
        evidence.final_guard()
        out.final_guard()
        status = dict(status="ok")
    except BaseException as error:
        status = dict(status="failed", error_type=type(error).__name__, error=str(error))
        out.write("failure.json", status)
        raise
    finally:
        with (output / "evaluation.provenance.json").open("xb") as f:
            f.write(
                proof.json_bytes(
                    dict(
                        stage=STAGE,
                        started_at=started,
                        finished_at=proof.utc(),
                        inputs=evidence.paths,
                        outputs=out.records,
                        **status,
                    )
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("comparison-sha", "sparse-id-sha", "depth-id-sha"):
        parser.add_argument("--" + option, required=True)
    for option in ("output",):
        parser.add_argument("--" + option, type=Path, required=True)
    run(parser.parse_args())
