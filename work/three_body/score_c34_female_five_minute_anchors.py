"""Saved C34-label association at the same 57 female anchors, for all seven lambdas.

No mouth reconstruction or strict mouth-support claim is made. The primary
eligibility rule is the old method's slot_track >= 0. A source-only layer is
reported separately. Original frozen scorer output is retained, with an explicit
translation of its mouth-support reason to the C34 label-availability meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import score_female_five_minute_anchors as prior

proof, require, generative = prior.proof, prior.require, prior.generative
WORK, REPO = prior.WORK, prior.REPO

# Import the standalone adapter after the frozen prior establishes import roots.
import long_baseline_support as baseline  # noqa: E402

INPUT_ROOT = REPO / "work/generative_tracking_inputs"
PRIOR_ROOT = WORK / prior.STAGE
START, STOP, FRAMES = prior.START, prior.STOP, prior.FRAMES
LAMBDAS = ("0", "0.1", "0.25", "0.5", "0.767", "2", "8")
FRAME_COUNT = baseline.FRAME_COUNT
STAGE = "c34_female_57_five_minute_endpoint_evaluation_v1"
EXECUTION_READY = True
PRIOR_SEAL_SHA = "122b66af931510b9950709a07397f71a70ef31898870de85af3a6ce047137539"
PRIOR_SOURCE_SHA = "b50e7668b91c6474edfe9d5d2150deeda8a823c11c7a08d5c451e6709f616165"
BASELINE_SOURCE_SHA = "7e7ed92ee8c2879c1a09a3b504c9f5190d8f2b270aa78168ac184cfffc4cb5e4"
CATALOG_SHA = "7bcfcd453ad345d7a2ca6269554f0aae6f158c430f8ae8dd223382549325181f"
INPUT_FIELDS = ("source", "slot_track")
REFERENCE_METHODS = ("original", "fast_tail", "fast_tail_beam_ID")
AVAILABILITY_RULE = "unique original C34 instance-source association and slot_track >= 0"


def local(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO), "input escaped workspace")
    return path


def read_json(path, digest, evidence):
    path = local(path)
    evidence.add(path, digest)
    return proof.read_json(path)


def document(root, seal, name, evidence):
    normalized = baseline.relative(name)
    path = local(root / normalized)
    require(path.is_relative_to(root.resolve()), "sealed member escaped its root")
    row = seal["outputs"][name]
    require(
        path == local(row["path"]) and path.stat().st_size == row["size_bytes"],
        "sealed member path/size differs",
    )
    return read_json(path, row["sha256"], evidence)


def source_code(seal, evidence):
    expected = dict(seal["inputs"])
    expected[str(Path(baseline.__file__).resolve())] = BASELINE_SOURCE_SHA
    require(
        expected.get(str(Path(prior.__file__).resolve())) == PRIOR_SOURCE_SHA
        and expected.get(str(Path(generative.__file__).resolve())) == prior.GENERATIVE_SHA,
        "saved female evaluator source pins differ",
    )
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not any(
            path.is_relative_to(root) for root in (WORK, REPO / "src", REPO / "labs")
        ):
            continue
        if path == Path(__file__).resolve():
            evidence.add(path)
        else:
            require(proof.valid_sha(expected.get(str(path))), "unbound local helper: " + str(path))
            evidence.add(path, expected[str(path)])
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
        "CPU evaluator initialized CUDA",
    )


def saved_anchors(fixed, observations):
    require(
        fixed["frames"] == list(FRAMES)
        and fixed["CSV_sha256"] == prior.GT_SHA
        and fixed["source_interval"] == [START, STOP]
        and fixed["labels_created"] is False,
        "fixed female anchor denominator/source differs",
    )
    rows = observations["matches"]
    require(
        [r["frame"] for r in rows] == list(FRAMES)
        and [r["frame"] for r in observations["diagnostics"]] == list(FRAMES),
        "saved match frames/order differ",
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
            all(type(v) is int and v >= 0 for v in row["detection_rows"].values())
            and all(np.isfinite(v) and 0 <= v <= 5 for v in row["match_errors_px"].values()),
            "saved unambiguous 5px match malformed",
        )
        require(
            all(
                np.asarray(xy).shape == (2,) and np.isfinite(xy).all()
                for xy in row["pixels"].values()
            ),
            "saved mouth coordinates malformed",
        )
    return [generative.MouthAnchor(**row) for row in rows]


def catalog_entries(catalog):
    entries = {}
    for item in catalog["files"]:
        name = baseline.relative(item["relative"])
        require(name not in entries, "duplicate normalized catalog path")
        entries[name] = item
    return entries


def registered_file(entries, name):
    name = baseline.relative(name)
    row = entries[name]
    path = local(INPUT_ROOT / name)
    require(
        path.is_relative_to(INPUT_ROOT.resolve())
        and path == local(row["local"])
        and type(row["bytes"]) is int
        and path.stat().st_size == row["bytes"]
        and proof.valid_sha(row["sha256"]),
        "registered input path/size/SHA malformed",
    )
    return dict(path=str(path), sha256=row["sha256"], size_bytes=row["bytes"])


def load_headers(evidence):
    seal = read_json(PRIOR_ROOT / "evaluation.provenance.json", PRIOR_SEAL_SHA, evidence)
    require(
        seal["stage"] == prior.STAGE and seal["status"] == "ok",
        "successful fixed female57 evaluation required",
    )
    source_code(seal, evidence)
    fixed = document(PRIOR_ROOT, seal, "fixed_female_anchor_set.json", evidence)
    observations = document(PRIOR_ROOT, seal, "anchor_observations.json", evidence)
    summary = document(PRIOR_ROOT, seal, "summary.json", evidence)
    require(
        summary["status"] == "ok"
        and summary["native_interval"] == [START, STOP]
        and summary["requested_anchors"] == 57
        and summary["requested_adjacent_intervals"] == 56
        and summary["annotation_CSV_sha256"] == prior.GT_SHA
        and summary["female_endpoint_evaluation_only"] is True
        and tuple(summary["results"]) == REFERENCE_METHODS,
        "saved generative reference evaluation scope differs",
    )
    anchors = saved_anchors(fixed, observations)
    catalog = read_json(INPUT_ROOT / "staging_manifest.json", CATALOG_SHA, evidence)
    entries = catalog_entries(catalog)
    # Reuse sealed SL-row matches only when they refer to this registered cache.
    shared = {}
    for view in prior.SIDES:
        name = f"work/c01_s16_predictions/bottomup/{baseline.EXPERIMENT}__{view}.npz"
        row = entries[name]
        path = local(INPUT_ROOT / name)
        require(
            path == local(row["local"]) and seal["inputs"].get(str(path)) == row["sha256"],
            "saved mouth matching used another SLEAP cache",
        )
        shared[str(path)] = row["sha256"]  # No SL file payload read/hash.
    gt_name = f"assets/groundtruth/fish_sex_gt/{baseline.EXPERIMENT}/female_sex_gt.csv"
    require(entries[gt_name]["sha256"] == prior.GT_SHA, "registered female annotation SHA differs")
    references = {}
    for value in LAMBDAS:
        name = f"{baseline.BASELINE}/C34-label__{baseline.EXPERIMENT}__lam{value}.npz"
        references[value] = registered_file(entries, name)
    expected_names = {
        f"{baseline.BASELINE}/C34-label__{baseline.EXPERIMENT}__lam{v}.npz" for v in LAMBDAS
    }
    require(
        {n for n in entries if n.startswith(baseline.BASELINE + "/C34-label__")} == expected_names,
        "registered C34 lambda set differs",
    )
    return dict(
        anchors=anchors,
        c34_references=references,
        binding=dict(
            prior_evaluation_seal_sha256=PRIOR_SEAL_SHA,
            fixed_anchor_output=seal["outputs"]["fixed_female_anchor_set.json"],
            matched_anchor_output=seal["outputs"]["anchor_observations.json"],
            reference_3D_summary=seal["outputs"]["summary.json"],
            annotation_CSV_sha256=prior.GT_SHA,
            matched_SL_references=shared,
            registration_catalog_sha256=CATALOG_SHA,
        ),
        reference_results=summary["results"],
    )


def validate_rows(rows, frames):
    require(
        isinstance(frames, np.ndarray)
        and frames.ndim == 1
        and frames.dtype == np.int64
        and len(frames) > 0
        and np.all(frames >= 0)
        and np.all(np.diff(frames) > 0),
        "unique increasing native anchor frames required",
    )
    require(set(rows) == {"frame_idx", *INPUT_FIELDS}, "exact C34 row fields required")
    require(
        np.array_equal(rows["frame_idx"], frames) and rows["frame_idx"].dtype == np.int64,
        "C34 anchor frame order differs",
    )
    count = len(frames)
    require(
        rows["source"].shape == (count, 3, 3)
        and rows["source"].dtype == np.int8
        and np.all(rows["source"] >= -1),
        "C34 source shape/type/negative index differs",
    )
    labels = rows["slot_track"]
    require(
        labels.shape == (count, 3)
        and labels.dtype == np.int32
        and np.all((labels == -1) | (labels == np.arange(3))),
        "C34-label columns required; gated track IDs are not accepted",
    )


def gap_change_brackets(frozen):
    """Diagnostic endpoint changes across U anchors; never alter adjacent counts."""
    previous, brackets = None, []
    anchors = frozen["anchors"]
    for index, anchor in enumerate(anchors):
        if anchor["identity"] is None:
            continue
        if previous is not None:
            prior_index, prior_anchor = previous
            if index > prior_index + 1 and anchor["identity"] != prior_anchor["identity"]:
                brackets.append(
                    dict(
                        previous_available_frame=prior_anchor["frame"],
                        next_available_frame=anchor["frame"],
                        previous_identity=prior_anchor["identity"],
                        next_identity=anchor["identity"],
                        skipped_requested_frames=[
                            a["frame"] for a in anchors[prior_index + 1 : index]
                        ],
                    )
                )
        previous = index, anchor
    return brackets


def score_rows(rows, anchors):
    frames = np.array([a.frame for a in anchors], dtype=np.int64)
    validate_rows(rows, frames)
    before = {k: proof.signature(v) for k, v in rows.items()}
    eligibility = rows["slot_track"] >= 0
    results = {}
    for mode, available in (
        ("source_association_only", np.ones(eligibility.shape, bool)),
        ("c34_label_available", eligibility),
    ):
        frozen = generative.score_female_identity(frames, rows["source"], available, anchors)
        # Do not rewrite the frozen function's original result/reason strings.
        interpretation = []
        for anchor in frozen["anchors"]:
            reason = anchor["reason"]
            interpretation.append(
                dict(
                    frame=anchor["frame"],
                    identity=anchor["identity"],
                    frozen_scorer_reason=reason,
                    C34_reason="C34 label unavailable"
                    if reason == "mouth not supported by two views"
                    else reason,
                )
            )
        results[mode] = dict(
            frozen_scorer_output=frozen,
            C34_anchor_interpretation=interpretation,
            label_change_brackets_over_unscored_anchors=gap_change_brackets(frozen),
            bracket_scope=(
                "endpoint label changes across unavailable anchors; "
                "not event counts or interior truth"
            ),
        )
    require(before == {k: proof.signature(v) for k, v in rows.items()}, "C34 input rows changed")
    return dict(
        results=results,
        original_label_available_by_slot=eligibility.tolist(),
        primary_layer="c34_label_available",
        source_filtered_before_conflict_check=False,
        eligibility_rule=AVAILABILITY_RULE,
        frozen_observed_argument_means="original C34 label availability, not mouth support",
        equivalent_review_flag_available=False,
        strict_mouth_support_computed=False,
        saved_chest_finiteness_used_for_eligibility=False,
    )


def file_stat(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def load_anchor_rows(reference, frames):
    require(
        isinstance(frames, np.ndarray)
        and frames.ndim == 1
        and frames.dtype == np.int64
        and np.all((frames >= 0) & (frames < FRAME_COUNT)),
        "C34 native selection out of bounds",
    )
    path = local(reference["path"])
    before = file_stat(path)
    payload = path.read_bytes()
    require(
        len(payload) == reference["size_bytes"]
        and hashlib.sha256(payload).hexdigest() == reference["sha256"]
        and before == file_stat(path),
        "captured C34 archive SHA/size/stat differs",
    )
    expected = {
        "slot_track",
        "xyz",
        "residual_mm",
        "margin_mm",
        "n_views_used",
        "source",
        *(f"track__{v}" for v in generative.VIEWS),
    }
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(
            len(archive.files) == len(expected) and set(archive.files) == expected,
            "C34-label nine-array schema differs",
        )
        arrays = {k: archive[k] for k in INPUT_FIELDS}
    require(
        arrays["source"].shape == (FRAME_COUNT, 3, 3)
        and arrays["source"].dtype == np.int8
        and arrays["slot_track"].shape == (FRAME_COUNT, 3)
        and arrays["slot_track"].dtype == np.int32,
        "C34 full native row count/dtypes differ",
    )
    for array in arrays.values():
        array.flags.writeable = False
    signatures = {k: proof.signature(v) for k, v in arrays.items()}
    rows = dict(
        frame_idx=np.asarray(frames, dtype=np.int64).copy(),
        **{k: v[frames].copy() for k, v in arrays.items()},
    )
    validate_rows(rows, np.asarray(frames, dtype=np.int64))
    require(
        signatures == {k: proof.signature(v) for k, v in arrays.items()},
        "captured original C34 arrays changed",
    )
    for array in rows.values():
        array.flags.writeable = False
    return rows, dict(
        **reference,
        file_stat=before,
        archive_reads=1,
        decoded_fields=list(INPUT_FIELDS),
        full_array_signatures=signatures,
        selected_array_signatures={k: proof.signature(v) for k, v in rows.items()},
    )


def compact(frozen):
    return {
        k: frozen[k]
        for k in (
            "requested_anchors",
            "available_anchors",
            "requested_intervals",
            "interval_counts",
        )
    }


def execute(args):
    require(EXECUTION_READY and args.execute, "reviewed C34 evaluation is not enabled")
    output = args.output.resolve()
    require(
        output.is_relative_to(WORK.resolve()) and not output.exists(),
        "fresh workspace output required",
    )
    out, evidence, captures = proof.Output(output), proof.Evidence(), {}
    started = proof.utc()
    try:
        context = load_headers(evidence)
        plan = dict(
            schema=STAGE,
            native_interval=[START, STOP],
            requested_anchors=57,
            requested_adjacent_intervals=56,
            fixed_frames=list(FRAMES),
            lambdas=list(LAMBDAS),
            primary_eligibility=AVAILABILITY_RULE,
            source_only_reference_layer=True,
            candidate_references=context["c34_references"],
            evidence_binding=context["binding"],
            all_registered_lambdas_reported=True,
            lambda_selected_from_results=False,
            SLEAP_rematching=False,
            mouth_or_tail_reconstruction=False,
            source_rows_are_whole_instance_assignments=True,
            C34_saved_geometry_is_chest_only=True,
            saved_chest_finiteness_used_for_eligibility=False,
            old_C34_uses_full_recording_offline_future_information=True,
            same_seed_or_initialization_comparison=False,
            new_seed_borrowed_C34_lambda="0.767",
            native_index_basis=(
                "registered 348600 rows and frozen C34 producer/adapter; "
                "no frame_idx array in C34 NPZ"
            ),
            frozen_scorer_reason_translation={
                "mouth not supported by two views": "C34 label unavailable"
            },
            existing_3D_reference_names={
                "original": "previous generative Original, not classical C34",
                "fast_tail": "FastTail raw slots",
                "fast_tail_beam_ID": "FastTail approximate motion IDs",
            },
        )
        out.write("plan.json", plan)
        results, table = {}, []
        frames = np.array(FRAMES, dtype=np.int64)
        for value in LAMBDAS:
            rows, capture = load_anchor_rows(context["c34_references"][value], frames)
            captures[value] = capture
            scored = score_rows(rows, context["anchors"])
            result = dict(lambda_value=value, **scored)
            out.write(f"arms/lambda_{value}.json", result)
            row_record = out.write(f"arms/lambda_{value}_anchor_rows.npz", rows)
            with np.load(row_record["path"], allow_pickle=False) as saved:
                require(
                    set(saved.files) == set(rows)
                    and all(
                        proof.signature(saved[k]) == proof.signature(v) for k, v in rows.items()
                    ),
                    "saved C34 anchor rows changed",
                )
            results[value] = result
            table.append(
                dict(
                    lambda_value=value,
                    **{
                        k: dict(
                            **compact(v["frozen_scorer_output"]),
                            label_change_brackets_over_unscored_anchors=v[
                                "label_change_brackets_over_unscored_anchors"
                            ],
                        )
                        for k, v in scored["results"].items()
                    },
                )
            )
            print(json.dumps(table[-1]), flush=True)
        require(tuple(results) == LAMBDAS, "one or more registered lambda results missing")
        for record in captures.values():
            require(
                file_stat(Path(record["path"])) == record["file_stat"],
                "captured baseline changed after read",
            )
        out.write("captured_baseline_inputs.json", captures)
        references = {
            method: {
                mode: dict(
                    **compact(context["reference_results"][method][mode]),
                    label_change_brackets_over_unscored_anchors=gap_change_brackets(
                        context["reference_results"][method][mode]
                    ),
                )
                for mode in ("raw", "with_original_review")
            }
            for method in REFERENCE_METHODS
        }
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                requested_anchors=57,
                requested_adjacent_intervals=56,
                results_table=table,
                existing_3D_reference_counts=references,
                existing_3D_reference_summary=context["binding"]["reference_3D_summary"],
                existing_3D_acceptance=(
                    "saved strict mouth support, and separately original review flags; "
                    "different from C34 saved-label eligibility"
                ),
                evidence_binding=context["binding"],
                C34_primary_eligibility=AVAILABILITY_RULE,
                all7_lambdas_scored=True,
                lambda_selected=False,
                C34_archive_reads=7,
                decoded_C34_fields=list(INPUT_FIELDS),
                SLEAP_payload_reads=0,
                annotation_CSV_payload_reads=0,
                video_mask_calibration_payload_reads=0,
                checkpoint39_reads=0,
                GPU_used=False,
                new_identity_solver_calls=0,
                generated_geometry=False,
                whole_three_physical_ID_accuracy=False,
                dense_swap_zero_certified=False,
                male_identity_truth_evaluated=False,
                score_scope=(
                    "female endpoints only; unscored anchors/56 adjacent intervals "
                    "retained without bridging gaps"
                ),
                identical_information_or_seed_conditions=False,
                old_C34_full_recording_offline_future_information=True,
                original_C34_producer_provenance_not_requalified=True,
                capture_final_guard=(
                    "same file stat; captured bytes SHA checked once, "
                    "original decoded arrays unchanged at selection"
                ),
            ),
        )
        evidence.final_guard()
        out.final_guard()
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        status, error = "ok", None
    except BaseException as exc:
        status, error = "failed", repr(exc)
        raise
    finally:
        seal = dict(
            stage=STAGE,
            status=status,
            error=error,
            started_at=started,
            finished_at=proof.utc(),
            inputs={**evidence.paths, **{r["path"]: r["sha256"] for r in captures.values()}},
            outputs=out.records,
        )
        (output / "evaluation.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    execute(parser.parse_args())
