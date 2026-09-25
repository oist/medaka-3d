"""Add seven completed bounded C34 streams to a preserved common13 human review.

Only saved three-frame SL mouth caches and seven final native C34 archives are
decoded. This is an endpoint comparison, not a certificate of interior identity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path

import compare_transition_review_source_mouths as common
import numpy as np
import score_classical_window_female_anchors as window

proof, require = common.proof, common.require
REPO, WORK, VIEWS = common.REPO, common.WORK, common.VIEWS
STAGE = "transition_review_with_bounded_classical_v1"
EXECUTION_READY = False
COMMON_FALSE_SHA = "d7ae6837db20476e27a9539e6e05d3443da63552ca7b09b792e578d137a8f56e"
WINDOW_FALSE_SHA = "8f3caf52246d4b4d1595c814bb260575b8e1d17b4e20713bbad276a9a9aec001"
CACHE_FIELDS = {"frame_idx", "offsets", "mouth_uv", "original_flat_row_ranges"}
ORIGINAL_KEYS = tuple("C34_lambda_" + v for v in window.LAMBDAS) + tuple(
    arm + "/" + space
    for arm in common.upstream.ARMS
    for space in ("raw_slots", "automatic_initial_slot_IDs")
)


def flag_digest(data, assignment, disabled_sha, *, require_enabled=False):
    """Normalize exactly one executable assignment; quoted examples stay intact."""
    false, true = (assignment + " = False").encode(), (assignment + " = True").encode()
    lines = data.splitlines(keepends=True)
    indices = [i for i, line in enumerate(lines) if line.rstrip(b"\r\n") in (false, true)]
    require(len(indices) == 1, "one execution assignment required")
    i = indices[0]
    require(
        not require_enabled or lines[i].rstrip(b"\r\n") == true,
        "successful common producer requires enabled assignment",
    )
    lines[i] = lines[i].replace(true, false)
    require(
        hashlib.sha256(b"".join(lines)).hexdigest() == disabled_sha,
        "source differs beyond execution assignment",
    )
    return hashlib.sha256(data).hexdigest()


def source_code(seal, summary, evidence):
    """Bind used helpers to the successful common source map; no producer import."""
    code = summary["source_sha256"]
    for path, digest in code.items():
        require(
            proof.valid_sha(digest) and seal["inputs"].get(path) == digest, "common source edge"
        )
    normalized = {}
    for module, name, expected, executed in (
        (common, "EXECUTION_ENABLED", COMMON_FALSE_SHA, True),
        (window, "EXECUTION_READY", WINDOW_FALSE_SHA, False),
    ):
        path = Path(module.__file__).resolve()
        actual = flag_digest(path.read_bytes(), name, expected, require_enabled=executed)
        evidence.add(path, actual)
        normalized[str(path)] = actual
    require(
        code.get(str(Path(common.__file__).resolve()))
        == normalized[str(Path(common.__file__).resolve())],
        "common executed source differs",
    )
    # The common writer has an explicit source map, not a dynamic module inventory.
    for path, digest in code.items():
        evidence.add(path, digest)
    for module in (
        common.upstream,
        common.classical,
        proof,
        common.upstream.human,
        common.upstream.encounter_review,
    ):
        require(
            str(Path(module.__file__).resolve()) in code,
            "used pure helper absent from common source map",
        )
    evidence.add(Path(__file__).resolve())
    evidence.add(window.PRODUCER, window.PRODUCER_SHA)
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
        "CUDA initialized",
    )


def validate_common_summary(summary, parsed):
    require(
        summary["schema"] == common.STAGE
        and summary["status"] == "ok"
        and summary["layer"] == common.LAYER
        and summary["streams"] == 13
        and summary["requested_endpoints_per_stream"] == 9
        and summary["requested_links_per_stream"] == 6
        and summary["anchor_frames"] == parsed.anchor_frames.tolist()
        and summary["original_review_sha256"] == hashlib.sha256(parsed.review_bytes).hexdigest()
        and summary["all7_lambdas_scored"] is True
        and summary["lambda_selected"] is False
        and tuple(summary["results"]) == ORIGINAL_KEYS,
        "common13 review/stream/denominator differs",
    )
    require(
        summary["all_three_physical_ID_truth"] is False
        and summary["dense_swap_zero_certified"] is False,
        "common endpoint scope differs",
    )
    for result in summary["results"].values():
        for score in result["scores"].values():
            require(
                score["requested_anchors"] == 9
                and score["requested_animal_intervals"] == 6
                and set(score["interval_counts"]) == {"preserved", "changed", "unscored"}
                and sum(score["interval_counts"].values()) == 6,
                "inherited9/6 denominator differs",
            )


def sl_binding(summary, captures, seal):
    refs = summary["SL3_references"]
    require(set(refs) == set(VIEWS), "three saved SL references required")
    shared = summary["inherited_lineage"]["shared_SL_calibration_sha256"]
    result = {}
    for view in VIEWS:
        ref, captured = refs[view], captures["SL/" + view]
        require(
            all(captured[k] == ref[k] for k in ("path", "sha256", "size_bytes"))
            and proof.valid_sha(ref["sha256"])
            and seal["inputs"].get(ref["path"]) == ref["sha256"]
            and shared.get(ref["path"]) == ref["sha256"]
            and captured["archive_reads"] == 1
            and captured["numeric_fields_decoded"] == ["frame_idx", "points"]
            and captured["score_decoded"] is False
            and set(captured["selected_signatures"]) == CACHE_FIELDS,
            "saved SL source/capture binding differs",
        )
        result[ref["path"]] = ref["sha256"]
    require(len(result) == 3, "SL view paths must be distinct")
    return result


def validate_cache(cache, captured, frames):
    require(
        set(cache) == CACHE_FIELDS
        and {k: proof.signature(v) for k, v in cache.items()} == captured["selected_signatures"],
        "saved mouth cache signature differs",
    )
    idx, offsets, mouths, ranges = (
        cache[k] for k in ("frame_idx", "offsets", "mouth_uv", "original_flat_row_ranges")
    )
    require(
        idx.dtype.kind in "iu"
        and np.array_equal(idx, frames)
        and offsets.dtype.kind in "iu"
        and offsets.shape == (4,)
        and offsets[0] == 0
        and np.all(offsets[1:] >= offsets[:-1])
        and mouths.dtype.kind == "f"
        and mouths.shape == (int(offsets[-1]), 2)
        and ranges.dtype.kind in "iu"
        and ranges.shape == (3, 2)
        and np.all(ranges >= 0)
        and np.all(ranges[:, 1] >= ranges[:, 0])
        and np.all(ranges[1:, 0] >= ranges[:-1, 1])
        and np.array_equal(np.diff(offsets), ranges[:, 1] - ranges[:, 0]),
        "saved mouth native order/row ranges differ",
    )


def validate_inherited_assignments(summary, parsed, caches):
    """Verify literal saved row-to-pixel lookup, without rescoring the original13."""
    for result in summary["results"].values():
        source = np.asarray(result["source_rows"])
        pixels, reasons = common.assigned_mouths(source, parsed.anchor_frames, caches)
        require(
            proof.safe_json(pixels.tolist()) == result["assigned_SL_mouth_pixels"]
            and reasons.tolist() == result["source_lookup_status"],
            "inherited assigned source/pixel row differs",
        )


def load_context(common_root, common_sha, producer_root, replay_sha, evidence):
    common_root, producer_root = window.local(common_root), window.local(producer_root)
    sealed = window.read_json(common_root / "comparison.provenance.json", common_sha, evidence)
    require(
        sealed.get("stage") == common.STAGE and sealed.get("status") == "ok",
        "successful common13 required",
    )
    expected = {
        "summary.json",
        "original_review.json",
        "source_manifest.json",
        "captured_new_inputs.json",
    }
    expected.update("SL_anchors/" + v + ".npz" for v in VIEWS)
    expected.update("C34_anchors/lambda_" + v + ".npz" for v in window.LAMBDAS)
    require(set(sealed["outputs"]) == expected, "common14 output schema differs")
    summary = window.document(common_root, sealed, "summary.json", evidence)
    source_code(sealed, summary, evidence)
    raw = common.upstream.member(common_root, sealed, "original_review.json", evidence, binary=True)
    manifest = common.upstream.member(
        common_root, sealed, "source_manifest.json", evidence, binary=True
    )
    parsed = common.upstream.human.parse_review(raw, manifest, clip_range=common.upstream.CLIP)
    validate_common_summary(summary, parsed)
    captured = window.document(common_root, sealed, "captured_new_inputs.json", evidence)
    sl = sl_binding(summary, captured, sealed)
    replay = window.read_json(producer_root / "replay.provenance.json", replay_sha, evidence)
    require(
        replay.get("stage") == window.PRODUCER_STAGE and replay.get("status") == "ok",
        "successful final bounded replay required",
    )
    expected = {"plan.json", "stage_a.json", "stage_a.npz", "summary.json"}
    expected.update(
        f"lam{v}/{name}"
        for v in window.LAMBDAS
        for name in ("c28.npz", "c33.npz", "c30.npz", "C34-label.npz", "diagnostics.json")
    )
    require(set(replay["outputs"]) == expected, "all7 final replay outputs required")
    plan = window.document(producer_root, replay, "plan.json", evidence)
    final = window.document(producer_root, replay, "summary.json", evidence)
    window.validate_producer_headers(
        plan, final, replay, {"binding": {"matched_SL_references": sl}}
    )
    arms = {
        v: window.member(producer_root, replay, f"lam{v}/C34-label.npz") for v in window.LAMBDAS
    }
    # Both successful stages and every lambda are qualified before any NPZ decode.
    caches = {}
    for view in VIEWS:
        caches[view] = common.bounded_anchor_archive(
            common_root, sealed, f"SL_anchors/{view}.npz", CACHE_FIELDS, evidence
        )
        validate_cache(caches[view], captured["SL/" + view], parsed.anchor_frames)
    validate_inherited_assignments(summary, parsed, caches)
    return dict(
        parsed=parsed,
        common_summary=summary,
        common_seal=sealed,
        producer_plan=plan,
        producer_summary=final,
        producer_seal=replay,
        arms=arms,
        caches=caches,
        shared_SL=sl,
    )


def execute(args):
    require(EXECUTION_READY and args.execute, "bounded transition extension is disabled")
    output = Path(args.output).resolve()
    require(
        output.is_relative_to(WORK.resolve()) and not output.exists(),
        "fresh workspace output required",
    )
    out, evidence, captures = proof.Output(output), proof.Evidence(), {}
    status = dict(status="failed")
    started = proof.utc()
    try:
        context = load_context(
            args.common_root, args.common_sha, args.producer_root, args.replay_sha, evidence
        )
        parsed, inherited = context["parsed"], context["common_summary"]
        results = copy.deepcopy(inherited["results"])
        for lam in window.LAMBDAS:
            rows, captures[lam] = window.load_rows(
                context["arms"][lam],
                context["producer_summary"]["results"]["lam" + lam]["array_signatures"],
                parsed.anchor_frames,
            )
            results["bounded_C34_lambda_" + lam] = common.score_stream(
                parsed,
                rows["source"],
                context["caches"],
                rows["slot_track"] >= 0,
                eligibility_name="C34_saved_label_available",
            )
            out.write(f"bounded_C34_anchors/lambda_{lam}.npz", rows)
        require(
            len(results) == 20 and {k: results[k] for k in ORIGINAL_KEYS} == inherited["results"],
            "original13 results changed",
        )
        out.write("original_review.json", parsed.review_bytes)
        out.write("source_manifest.json", parsed.manifest_bytes)
        out.write("original_common_summary.json", inherited)
        out.write("captured_bounded_inputs.json", captures)
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                layer=common.LAYER,
                streams=20,
                requested_endpoints_per_stream=9,
                requested_links_per_stream=6,
                anchor_frames=parsed.anchor_frames.tolist(),
                results=results,
                common_comparison_root=str(Path(args.common_root).resolve()),
                common_comparison_sha256=args.common_sha,
                original_common_summary_reference=context["common_seal"]["outputs"]["summary.json"],
                original13_results_preserved=True,
                original13_rescored=False,
                unchanged_3D_primary_reference=inherited["unchanged_3D_primary_reference"],
                inherited_identity_bindings=inherited["inherited_identity_bindings"],
                inherited_conditional_ID_ambiguity=inherited["inherited_conditional_ID_ambiguity"],
                inherited_lineage=inherited["inherited_lineage"],
                original_review_sha256=hashlib.sha256(parsed.review_bytes).hexdigest(),
                biological_naming_preserved_not_applied=parsed.review.get("biological_naming"),
                producer_root=str(Path(args.producer_root).resolve()),
                producer_seal_sha256=args.replay_sha,
                producer_summary_reference=context["producer_seal"]["outputs"]["summary.json"],
                bounded_references=context["arms"],
                shared_SL_path_sha256=context["shared_SL"],
                process_interval=list(window.PROCESS),
                science_interval=list(window.SCIENCE),
                future_horizon_exclusive=116200,
                manual_reinitializations=0,
                shared_seed=False,
                initialization=context["producer_plan"]["initialization"],
                original_C25_objective_preserved=True,
                transition_cache_identity_bug_corrected=True,
                initial_condition_limitation=(
                    "Unconstrained classical initialization; "
                    "common seed equivalence is not established."
                ),
                old_C34_full_recording_future_information_remains_in_original13=True,
                all7_bounded_lambdas_scored=True,
                lambda_selected=False,
                saved_SL_anchor_archive_reads=3,
                bounded_C34_archive_reads=7,
                bounded_decoded_fields=list(window.DECODE_FIELDS),
                full_SL_old_C34_3D_beam_video_mask_calibration_checkpoint_reads=0,
                inference_or_solver_calls=0,
                GPU_used=False,
                track_or_review_modified=False,
                biological_identity_assigned=False,
                all_three_physical_ID_truth=False,
                dense_swap_zero_certified=False,
                endpoint_only_diagnostics_are_not_interior_certificates=True,
                capture_final_guard=(
                    "Bounded archives captured once with SHA; final original stat guard only."
                ),
            ),
        )
        source_code(context["common_seal"], inherited, evidence)
        evidence.final_guard()
        for record in captures.values():
            require(
                window.old.file_stat(Path(record["path"])) == record["file_stat"],
                "captured bounded input changed",
            )
        out.final_guard()
        status = dict(status="ok")
    except BaseException as error:
        status = dict(status="failed", error_type=type(error).__name__, error=str(error))
        out.write("failure.json", status)
        raise
    finally:
        with (output / "comparison.provenance.json").open("xb") as stream:
            stream.write(
                proof.json_bytes(
                    dict(
                        stage=STAGE,
                        started_at=started,
                        finished_at=proof.utc(),
                        inputs={
                            **evidence.paths,
                            **{r["path"]: r["sha256"] for r in captures.values()},
                        },
                        outputs=out.records,
                        **status,
                    )
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--common-sha", required=True)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--replay-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    execute(parser.parse_args())
