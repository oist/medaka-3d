"""Reuse the fixed female57 labels on successful Nodewise raw and beam-ID results.

CPU endpoint evaluation only: one five-field candidate read and one five-field
mapping read, after successful comparison/ID and registered-seven-input gates.
No SL/CSV rematching, other method evaluation, solver, video, mask or checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import run_nodewise_body_row_five_minute_identity_beam as identity
import score_variant_female_five_minute_anchors as metrics

prior = metrics.prior
proof, common, require = identity.proof, identity.common, identity.require
WORK, START, STOP = identity.WORK, identity.START, identity.STOP
STAGE = "nodewise_female_57_five_minute_endpoint_evaluation_v1"
IDENTITY_SHA = "dbb289f4fc154309e0aa3a372a693363da40d7ed4ae23c54288ae1d197fbe0a0"
METRICS_SHA = "e2922be13f11966d883bdd74ab96955d3d8ed7d755d4105f895a2d6695b7c30a"
MAPPING_FIELDS = metrics.MAPPING_FIELDS
document, edge = identity.document, identity.edge


def source_code(evidence):
    # Explicit frozen helper pins; no legacy entrypoint's dynamic inventory.
    code = identity.source_code(evidence)
    for path, digest in (
        (Path(identity.__file__), IDENTITY_SHA),
        (Path(metrics.__file__), METRICS_SHA),
        (Path(prior.__file__), metrics.PRIOR_SHA),
        (Path(prior.generative.__file__), prior.GENERATIVE_SHA),
        (Path(prior.control.__file__), prior.CONTROL_SHA),
        (Path(prior.control.comparison.__file__), prior.control.COMPARISON_CODE_SHA),
        (Path(metrics.depth.__file__), metrics.DEPTH_EXECUTION_SHA),
    ):
        code[str(path.resolve())] = evidence.add(path, digest)
    code[str(Path(__file__).resolve())] = evidence.add(Path(__file__))
    return code


def read_seal(root, filename, digest, stage, evidence):
    root = Path(root).resolve()
    require(root.is_relative_to(WORK) and proof.valid_sha(digest), "explicit workspace seal")
    evidence.add(root / filename, digest)
    seal = proof.read_json(root / filename)
    common.exact_fields(seal, dict(stage=stage, status="ok"), "successful stage required")
    return seal


def saved_anchors(headers, evidence):
    """Reuse sealed matching and verify all three SL plus four calibration edges."""
    root = WORK / prior.STAGE
    seal = read_seal(
        root, "evaluation.provenance.json", metrics.PRIOR_SEAL_SHA, prior.STAGE, evidence
    )
    edge(seal, Path(prior.__file__), metrics.PRIOR_SHA)
    summary = document(root, seal, "summary.json", evidence)
    fixed = document(root, seal, "fixed_female_anchor_set.json", evidence)
    observations = document(root, seal, "anchor_observations.json", evidence)
    common.exact_fields(
        summary,
        dict(
            requested_anchors=57,
            requested_adjacent_intervals=56,
            annotation_CSV_sha256=prior.GT_SHA,
            female_endpoint_evaluation_only=True,
            same_horizon_first11_control_exact=True,
        ),
        "saved female57 evaluation scope differs",
    )
    old_root, old_sha = prior.control.COMPARISON_ROOT, prior.control.COMPARISON_SEAL_SHA
    edge(seal, old_root / "comparison.provenance.json", old_sha)
    old_seal = read_seal(
        old_root, "comparison.provenance.json", old_sha, prior.control.comparison.STAGE, evidence
    )
    old_plan = document(old_root, old_seal, "plan.json", evidence)
    shared = old_plan["shared_SL_calibration_sha256"]
    require(len(shared) == 7, "registered three SL and four calibration inputs required")
    expected_names = {
        *(f"20251225_K_1_2_FCMC__{v}.npz" for v in prior.generative.VIEWS),
        "calibration_manifest.json",
        "cam0_bundle_refractive.json",
        "cam1_bundle_refractive.json",
        "cam2_bundle_refractive.json",
    }
    require({Path(p).name for p in shared} == expected_names, "registered seven-input roles")
    index_path = common.bound_member(
        headers["producer_root"], headers["producer_seal"], "input_index.json", evidence
    )
    edge(headers["comparison_seal"], index_path, proof.sha(index_path))
    index = proof.read_json(index_path)
    require(
        set(index) == {"original_binding", "prefix_chunks"}
        and len(index["prefix_chunks"]) == 40
        and index["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"] == shared,
        "Nodewise registered observations differ from saved matching",
    )
    for path, digest in shared.items():
        edge(old_seal, Path(path), digest)
        edge(headers["producer_seal"], Path(path), digest)
        if any(Path(path).name == f"20251225_K_1_2_FCMC__{v}.npz" for v in prior.SIDES):
            edge(seal, Path(path), digest)
    anchors = metrics.validate_saved_anchors(fixed, observations)
    return (
        anchors,
        fixed,
        observations,
        dict(
            evaluation_seal_sha256=metrics.PRIOR_SEAL_SHA,
            fixed_anchor_set_output=seal["outputs"]["fixed_female_anchor_set.json"],
            matches_output=seal["outputs"]["anchor_observations.json"],
            shared_SL_calibration_sha256=shared,
            producer_input_index=headers["producer_seal"]["outputs"]["input_index.json"],
            matching_reused_without_SL_or_CSV_read=True,
            all57_denominator_preserved=True,
        ),
    )


def identity_headers(digest, headers, evidence):
    root = WORK / identity.STAGE
    seal = read_seal(root, "identity.provenance.json", digest, identity.STAGE, evidence)
    edge(seal, Path(identity.__file__), IDENTITY_SHA)
    edge(seal, Path(identity.frozen.__file__), identity.FROZEN_SHA)
    edge(seal, Path(identity.frozen.beam.__file__), identity.frozen.BEAM_SHA)
    edge(
        seal,
        WORK / identity.comparison.STAGE / "comparison.provenance.json",
        headers["comparison_sha256"],
    )
    ref = headers["reference"]
    edge(seal, headers["producer_root"] / "tracking.provenance.json", ref["tracking_seal_sha256"])
    edge(seal, Path(ref["candidate"]["path"]), ref["candidate"]["sha256"])
    plan, summary, scalars = (
        document(root, seal, name + ".json", evidence)
        for name in ("plan", "summary", "beam_scalars")
    )
    expected = dict(
        schema=identity.STAGE,
        native_interval=[START, STOP],
        native_frames=STOP - START,
        retained_raw_interval=[identity.FIRST, STOP],
        terminal_horizon=STOP,
        comparison_seal_sha256=headers["comparison_sha256"],
        comparison_source_sha256=identity.COMPARISON_SHA,
        tracking_seal_sha256=ref["tracking_seal_sha256"],
        tracking_source_sha256=identity.comparison.NODEWISE_RUNNER_SHA,
        raw_candidate_reference=ref,
        decoded_field_signatures={k: ref["decoded_field_signatures"][k] for k in identity.FIELDS},
        decoded_candidate_fields=list(identity.FIELDS),
        undecoded18_numerically_verified=False,
        beam_source_sha256=identity.frozen.BEAM_SHA,
        beam_width=256,
        node_indices=[1, 0],
        dp_config=identity.frozen.CONFIG,
        initial_P=[0, 1, 2],
        initialization_frame=START,
        intermediate_ID_resets=0,
        global_optimality_certified=False,
        gaps_conditional_on_retained_graph=True,
        gaps_are_probabilities=False,
        physical_ID_truth=False,
        biological_naming_applied=False,
        review_flags_cleared=False,
        raw_candidate_archive_modified=False,
    )
    for doc in (plan, summary):
        common.exact_fields(doc, expected, "successful Nodewise ID scope/reference differs")
        require(
            doc["source_sha256"].get(str(Path(identity.__file__).resolve())) == IDENTITY_SHA,
            "executed ID caller source differs",
        )
    common.exact_fields(
        summary,
        dict(
            status="ok",
            solve_calls=1,
            decoded_fields_unchanged=True,
            support_and_geometry_reestimated=False,
            exact_DP_rerun=False,
        ),
        "ID preservation or single solve contract differs",
    )
    tolerance = scalars["numerical_tie_tolerance"]
    require(
        np.isfinite(tolerance)
        and tolerance >= 0
        and tolerance == summary["numerical_tie_tolerance"],
        "conditional mapping tolerance differs",
    )
    return dict(root=root, seal=seal, seal_sha256=digest, tolerance=tolerance)


def load_rows(headers, evidence):
    ref = headers["reference"]
    path = common.bound_member(
        headers["producer_root"], headers["producer_seal"], "candidate.npz", evidence
    )
    return prior.load_rows(
        dict(
            path=str(path),
            sha256=ref["candidate"]["sha256"],
            arrays=ref["all21_declared_signatures"],
        ),
        evidence,
    )


def load_mapping(bundle, evidence):
    path = common.bound_member(
        bundle["root"], bundle["seal"], "identity_mapping_and_ambiguity.npz", evidence
    )
    with np.load(path, allow_pickle=False) as z:
        require(
            len(z.files) == 14 and set(z.files) == set(identity.frozen.MAPPING_FIELDS),
            "mapping14 membership",
        )
        arrays = {k: z[k] for k in MAPPING_FIELDS}
    # Validate original science start P012, not the first selected female anchor.
    prior.validate_mapping(arrays, n=STOP - START, tolerance=bundle["tolerance"])
    selected = {k: v[np.asarray(prior.FRAMES) - START].copy() for k, v in arrays.items()}
    for v in selected.values():
        v.flags.writeable = False
    return selected


def run(comparison_sha, nodewise_id_sha, output):
    identity.frozen.cpu_only()
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        code = source_code(evidence)
        headers = identity.comparison_headers(comparison_sha, evidence)
        bundle = identity_headers(nodewise_id_sha, headers, evidence)
        anchors, fixed, observations, label_binding = saved_anchors(headers, evidence)
        # All successful seals, matching and identity scope gates precede payload reads.
        rows = load_rows(headers, evidence)
        mapping = load_mapping(bundle, evidence)
        before = common.signature(rows), common.signature(mapping)
        results = metrics.evaluate(rows, anchors, mapping)
        require(
            before == (common.signature(rows), common.signature(mapping)), "saved rows modified"
        )
        out.write("fixed_female_anchor_set.json", fixed)
        out.write("anchor_observations.json", observations)
        out.write("selected_candidate_rows.npz", rows)
        out.write("selected_mapping_rows.npz", mapping)
        out.write("selected_signatures.json", dict(candidate=before[0], mapping=before[1]))
        for stream in ("raw_slots", "automatic_IDs"):
            for mode in ("raw", "with_original_review"):
                out.write(f"scores/{stream}_{mode}.json", results[stream][mode])
        out.write(
            "endpoint_details.json",
            {s: results[s]["endpoint_details"] for s in ("raw_slots", "automatic_IDs")},
        )
        out.write(
            "secondary_label_change_brackets.json", results["secondary_label_change_brackets"]
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                requested_anchors=57,
                requested_adjacent_intervals=56,
                comparison_seal_sha256=comparison_sha,
                identity_seal_sha256=nodewise_id_sha,
                identity_source_sha256=IDENTITY_SHA,
                candidate_reference=headers["reference"],
                label_binding=label_binding,
                source_sha256=code,
                results=results,
                female_endpoint_evaluation_only=True,
                initial_ID1_sex_certified=False,
                all_three_physical_ID_truth=False,
                primary_adjacent_interval_metric_unchanged=True,
                no_bridging_for_primary=True,
                conditional_gaps_are_probabilities=False,
                reviews_cleared=False,
                geometry_modified=False,
                candidate_npz_decodes=1,
                candidate_fields_decoded=list(prior.FIELDS),
                undecoded_candidate16_numerically_verified=False,
                mapping_npz_decodes=1,
                mapping_fields_decoded=list(MAPPING_FIELDS),
                undecoded_mapping9_numerically_verified=False,
                mapping_direction="permutation_by_id[frame, ID] = raw_slot; whole body gathered",
                other_methods_reevaluated=False,
                SL_CSV_video_mask_checkpoint39_reads=0,
                identity_graph_reads=0,
                identity_solves=0,
                GPU_used=False,
            ),
        )
        identity.frozen.cpu_only()
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
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--nodewise-id-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.comparison_sha, args.nodewise_id_sha, args.output)
