"""Female57 endpoint evaluation of a successful bounded classical replay.

Uses saved, unambiguous 5px SL-row matches and the frozen C34 label-availability
scorer. Native frame_idx is authoritative; no full-record array is fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import score_c34_female_five_minute_anchors as old

proof, require = old.proof, old.require
WORK, REPO = old.WORK, old.REPO
STAGE = "classical_window_female_57_endpoint_evaluation_v1"
EXECUTION_READY = True
PRODUCER_STAGE = "classical_same_interval_unseeded_v1"
PRODUCER = WORK / "run_classical_same_interval.py"
PRODUCER_SHA = "f7153147f3e138b2d97253b439941a55862b7f3fd0bd3881bda2bd594a7515f7"
OLD_SOURCE_SHA = "9a46a7dabec305d082db4cdd766e193815577815b0d6027cfa773c8a8f1078b2"
OLD_ROOT = WORK / old.STAGE
OLD_SEAL_SHA = "6af69e2cee756d7d65ad04de6cdb48a42968d2910e10b7ab42c9a5eab88efede"
ANCHOR_ROOT = WORK / old.prior.STAGE
ANCHOR_SEAL_SHA = old.PRIOR_SEAL_SHA
PROCESS = (49800, 116200)
SCIENCE = (66400, 116200)
VIEWS = ("top_cam", "side1_cam", "side2_cam")
LAMBDAS = old.LAMBDAS
FIELDS = {
    "frame_idx",
    "xyz",
    "residual_mm",
    "source",
    "slot_rays",
    "margin_mm",
    "n_views_used",
    "slot_track",
    *("track__" + v for v in VIEWS),
    *("instance_frame_idx__" + v for v in VIEWS),
}
DECODE_FIELDS = ("frame_idx", "source", "slot_track")


def local(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO.resolve()), "input outside workspace")
    return path


def read_json(path, digest, evidence):
    path = local(path)
    evidence.add(path, digest)
    return proof.read_json(path)


def member(root, seal, name):
    require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe member")
    row = seal["outputs"][name]
    path = local(root / name)
    require(
        path.is_relative_to(root.resolve())
        and path == local(row["path"])
        and type(row["size_bytes"]) is int
        and path.stat().st_size == row["size_bytes"]
        and proof.valid_sha(row["sha256"]),
        "sealed member path/size/SHA differs",
    )
    return row


def document(root, seal, name, evidence):
    row = member(root, seal, name)
    return read_json(row["path"], row["sha256"], evidence)


def source_code(old_seal, evidence):
    expected = dict(old_seal["inputs"])
    require(
        expected.get(str(Path(old.__file__).resolve())) == OLD_SOURCE_SHA,
        "frozen C34 scorer pin differs",
    )
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix != ".py" or not any(
            path.is_relative_to(p) for p in (WORK, REPO / "src", REPO / "labs")
        ):
            continue
        if path == Path(__file__).resolve():
            evidence.add(path)
        else:
            require(proof.valid_sha(expected.get(str(path))), "unbound helper: " + str(path))
            evidence.add(path, expected[str(path)])
    evidence.add(PRODUCER, PRODUCER_SHA)  # Source only; do not import/run producer.
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
        "CUDA initialized",
    )


def validate_endpoint_table(table):
    require(
        [row["lambda_value"] for row in table] == list(LAMBDAS),
        "reference lambda set/order differs",
    )
    for row in table:
        for mode in ("source_association_only", "c34_label_available"):
            result = row[mode]
            require(
                result["requested_anchors"] == 57
                and result["requested_intervals"] == 56
                and 0 <= result["available_anchors"] <= 57
                and set(result["interval_counts"]) == {"preserved", "swap", "unscored"}
                and sum(result["interval_counts"].values()) == 56,
                "reference endpoint denominator differs",
            )


def load_reference_headers(evidence):
    seal = read_json(OLD_ROOT / "evaluation.provenance.json", OLD_SEAL_SHA, evidence)
    require(
        seal["stage"] == old.STAGE and seal["status"] == "ok", "successful old evaluation required"
    )
    source_code(seal, evidence)
    summary = document(OLD_ROOT, seal, "summary.json", evidence)
    require(
        summary["schema"] == old.STAGE
        and summary["status"] == "ok"
        and summary["native_interval"] == list(SCIENCE)
        and summary["requested_anchors"] == 57
        and summary["requested_adjacent_intervals"] == 56
        and summary["all7_lambdas_scored"] is True
        and summary["lambda_selected"] is False,
        "old female reference scope differs",
    )
    validate_endpoint_table(summary["results_table"])
    anchor_seal = read_json(ANCHOR_ROOT / "evaluation.provenance.json", ANCHOR_SEAL_SHA, evidence)
    require(
        anchor_seal["stage"] == old.prior.STAGE and anchor_seal["status"] == "ok",
        "successful fixed anchor evaluation required",
    )
    fixed = document(ANCHOR_ROOT, anchor_seal, "fixed_female_anchor_set.json", evidence)
    observations = document(ANCHOR_ROOT, anchor_seal, "anchor_observations.json", evidence)
    anchors = old.saved_anchors(fixed, observations)
    binding = summary["evidence_binding"]
    require(
        binding["prior_evaluation_seal_sha256"] == ANCHOR_SEAL_SHA
        and binding["fixed_anchor_output"] == anchor_seal["outputs"]["fixed_female_anchor_set.json"]
        and binding["matched_anchor_output"] == anchor_seal["outputs"]["anchor_observations.json"]
        and binding["annotation_CSV_sha256"] == old.prior.GT_SHA,
        "old/new fixed anchor source differs",
    )
    require(
        tuple(summary["existing_3D_reference_counts"]) == old.REFERENCE_METHODS,
        "generative reference methods differ",
    )
    return dict(
        anchors=anchors,
        binding=binding,
        old_summary=summary,
        old_summary_record=seal["outputs"]["summary.json"],
    )


def signature(array):
    """Producer uses dtype names, unlike the frozen evaluator's dtype.str."""
    array = np.asarray(array)
    return dict(
        dtype=str(array.dtype),
        shape=list(array.shape),
        sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    )


def validate_descriptors(descriptors):
    require(set(descriptors) == FIELDS, "all14 producer array descriptors required")
    count = PROCESS[1] - PROCESS[0]
    fixed = dict(
        frame_idx=("int64", [count]),
        xyz=("float32", [count, 3, 3]),
        residual_mm=("float32", [count, 3]),
        source=("int8", [count, 3, 3]),
        slot_rays=("int8", [count, 3]),
        margin_mm=("float32", [count]),
        n_views_used=("int8", [count]),
        slot_track=("int32", [count, 3]),
    )
    for key, row in descriptors.items():
        require(
            set(row) == {"dtype", "shape", "sha256"} and proof.valid_sha(row["sha256"]),
            "malformed producer array signature",
        )
        if key in fixed:
            require(
                (row["dtype"], row["shape"]) == fixed[key], "producer array schema differs: " + key
            )
        else:
            require(
                row["dtype"] == "int32"
                and len(row["shape"]) == 1
                and type(row["shape"][0]) is int
                and row["shape"][0] >= 0,
                "producer flat instance schema differs",
            )
    for view in VIEWS:
        require(
            descriptors["track__" + view]["shape"]
            == descriptors["instance_frame_idx__" + view]["shape"],
            "per-view instance lengths differ",
        )


def validate_producer_headers(plan, summary, seal, references):
    require(
        seal["stage"] == PRODUCER_STAGE and seal["status"] == "ok",
        "successful final classical replay required",
    )
    require(
        plan["schema"] == PRODUCER_STAGE
        and summary["stage"] == PRODUCER_STAGE
        and summary["status"] == "ok",
        "producer stage differs",
    )
    for doc in (plan, summary):
        require(
            doc["process_interval"] == list(PROCESS) and doc["science_interval"] == list(SCIENCE),
            "producer interval/horizon differs",
        )
    require(
        plan["process_frames"] == PROCESS[1] - PROCESS[0]
        and plan["lambdas"] == [float(x) for x in LAMBDAS]
        and plan["initial_seed_shared"] is False
        and plan["initialization"] == "unconstrained C25 first-frame costs"
        and plan["future_horizon_exclusive"] == PROCESS[1]
        and plan["time_chunking"] is False
        and plan["outside_observations_used"] is False
        and plan["original_C25_objective_preserved"] is True
        and plan["transition_cache_identity_bug_corrected"] is True,
        "producer processing or initial-condition contract differs",
    )
    require(
        summary["all7_lambdas"] is True
        and summary["stage_a_calls"] == 3
        and summary["c25_whole_window_calls"] == 7
        and summary["manual_reinitializations"] == 0
        and summary["shared_seed"] is False
        and summary["final_future_horizon"] == PROCESS[1]
        and list(summary["results"]) == ["lam" + v for v in LAMBDAS],
        "incomplete or reset classical replay",
    )
    require(
        seal["inputs"].get(str(PRODUCER.resolve())) == PRODUCER_SHA
        and plan["source_binding"]["sources"].get(str(PRODUCER.resolve())) == PRODUCER_SHA,
        "producer source edge differs",
    )
    records = plan["inputs"]["records"]
    record_map = {r["path"]: r for r in records}
    require(len(record_map) == len(records) == 9, "expected catalog/seed/SL3/calibration4 records")
    for path, row in record_map.items():
        require(
            proof.valid_sha(row["sha256"]) and seal["inputs"].get(path) == row["sha256"],
            "captured input not bound to producer seal",
        )
    for path, digest in plan["source_binding"]["sources"].items():
        require(
            proof.valid_sha(digest) and seal["inputs"].get(path) == digest,
            "producer source not bound to final seal",
        )
    for path, digest in references["binding"]["matched_SL_references"].items():
        require(
            seal["inputs"].get(path) == digest and record_map.get(path, {}).get("sha256") == digest,
            "saved anchor rows refer to different SLEAP detections",
        )
    require(set(plan["inputs"]["selected"]) == set(VIEWS), "producer views incomplete")
    for selected in plan["inputs"]["selected"].values():
        require(
            selected["native_interval"] == list(PROCESS)
            and selected["outside_points_scores_deserialized"] is False
            and selected["full_native_index_read"] is True,
            "selected SL window differs",
        )
    require(
        plan["inputs"]["seed_reference"]["frame"] == PROCESS[0], "initial diagnostic frame differs"
    )
    for result in summary["results"].values():
        validate_descriptors(result["array_signatures"])


def load_context(root, replay_sha, evidence):
    root = local(root)
    require(root.is_dir() and proof.valid_sha(replay_sha), "explicit final replay SHA required")
    seal = read_json(root / "replay.provenance.json", replay_sha, evidence)
    # Check final success before opening any candidate or auxiliary producer result.
    require(
        seal["stage"] == PRODUCER_STAGE and seal["status"] == "ok",
        "successful final classical replay required",
    )
    expected = {"plan.json", "stage_a.json", "stage_a.npz", "summary.json"}
    expected.update(
        f"lam{v}/{name}"
        for v in LAMBDAS
        for name in ("c28.npz", "c33.npz", "c30.npz", "C34-label.npz", "diagnostics.json")
    )
    require(set(seal["outputs"]) == expected, "incomplete all7 replay output set")
    references = load_reference_headers(evidence)
    plan = document(root, seal, "plan.json", evidence)
    summary = document(root, seal, "summary.json", evidence)
    validate_producer_headers(plan, summary, seal, references)
    arms = {v: member(root, seal, f"lam{v}/C34-label.npz") for v in LAMBDAS}
    return dict(
        arms=arms,
        producer_summary=summary,
        producer_plan=plan,
        references=references,
        producer_seal_sha256=replay_sha,
    )


def select_rows(arrays, requested):
    frames, source, labels = (arrays[k] for k in DECODE_FIELDS)
    require(
        frames.dtype == np.int64
        and frames.ndim == 1
        and np.array_equal(frames, np.arange(*PROCESS, dtype=np.int64)),
        "missing/duplicate/out-of-order native producer frames",
    )
    count = len(frames)
    require(
        source.dtype == np.int8 and source.shape == (count, 3, 3) and np.all(source >= -1),
        "source rows must preserve int8 detector offsets",
    )
    require(
        labels.dtype == np.int32
        and labels.shape == (count, 3)
        and np.all((labels == -1) | (labels == np.arange(3))),
        "invalid C34-label availability",
    )
    requested = np.asarray(requested)
    require(
        requested.dtype == np.int64
        and requested.ndim == 1
        and len(requested) > 0
        and np.all(np.diff(requested) > 0)
        and np.all((requested >= SCIENCE[0]) & (requested < SCIENCE[1])),
        "invalid scientific anchor frames",
    )
    indices = np.searchsorted(frames, requested)
    require(
        np.all(indices < count) and np.array_equal(frames[indices], requested),
        "requested anchor missing from producer",
    )
    rows = {k: arrays[k][indices].copy() for k in DECODE_FIELDS}
    old.validate_rows(rows, requested)
    for array in rows.values():
        array.flags.writeable = False
    return rows, indices


def load_rows(reference, descriptors, requested):
    path = local(reference["path"])
    before = old.file_stat(path)
    payload = path.read_bytes()
    require(
        len(payload) == reference["size_bytes"]
        and hashlib.sha256(payload).hexdigest() == reference["sha256"]
        and old.file_stat(path) == before,
        "captured bounded archive differs",
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(
            len(archive.files) == len(FIELDS) and set(archive.files) == FIELDS,
            "bounded C34 requires genuine14-field archive",
        )
        arrays = {key: archive[key] for key in DECODE_FIELDS}
    for key, array in arrays.items():
        require(
            signature(array) == descriptors[key], "producer saved-array signature differs: " + key
        )
        array.flags.writeable = False
    rows, indices = select_rows(arrays, requested)
    return rows, dict(
        **reference,
        file_stat=before,
        decoded_fields=list(DECODE_FIELDS),
        selected_native_frames=requested.tolist(),
        selected_array_rows=indices.tolist(),
        independently_checked_signatures={k: signature(v) for k, v in arrays.items()},
        other11_fields_bound_by_archive_and_success_seal_only=True,
    )


def execute(args):
    require(EXECUTION_READY and args.execute, "bounded classical female evaluation is disabled")
    output = args.output.resolve()
    require(
        output.is_relative_to(WORK.resolve()) and not output.exists(),
        "fresh workspace output required",
    )
    out, evidence, captures = proof.Output(output), proof.Evidence(), {}
    status, error = "failed", None
    try:
        context = load_context(args.producer_root, args.replay_sha, evidence)
        references = context["references"]
        requested = np.asarray([a.frame for a in references["anchors"]], np.int64)
        require(requested.tolist() == list(old.FRAMES), "fixed57 anchor denominator differs")
        out.write(
            "plan.json",
            dict(
                stage=STAGE,
                native_interval=list(SCIENCE),
                producer_interval=list(PROCESS),
                producer_seal_sha256=args.replay_sha,
                producer_source_sha256=PRODUCER_SHA,
                arms=context["arms"],
                fixed_anchor_binding=references["binding"],
                requested_anchors=57,
                requested_adjacent_intervals=56,
                all7_lambdas=list(LAMBDAS),
                initial_condition=context["producer_plan"]["initialization"],
                shared_seed=False,
                initial_seed_diagnostic_reference=context["producer_plan"]["inputs"][
                    "seed_reference"
                ],
                old_reference_summary=references["old_summary_record"],
                source_rows_mean="original within-native-frame SL instance offsets",
                source_filtered_before_conflict_check=False,
                acceptance="saved C34 slot_track label availability; not strict mouth support",
            ),
        )
        table = []
        for value in LAMBDAS:
            descriptors = context["producer_summary"]["results"]["lam" + value]["array_signatures"]
            rows, capture = load_rows(context["arms"][value], descriptors, requested)
            captures[value] = capture
            scored = old.score_rows(rows, references["anchors"])
            out.write(f"lam{value}/evaluation.json", scored)
            out.write(f"lam{value}/anchor_rows.npz", rows)
            table.append(
                dict(
                    lambda_value=value,
                    **{
                        mode: dict(
                            **old.compact(result["frozen_scorer_output"]),
                            label_change_brackets_over_unscored_anchors=result[
                                "label_change_brackets_over_unscored_anchors"
                            ],
                        )
                        for mode, result in scored["results"].items()
                    },
                )
            )
        validate_endpoint_table(table)
        out.write("captured_inputs.json", captures)
        out.write(
            "summary.json",
            dict(
                stage=STAGE,
                status="ok",
                native_interval=list(SCIENCE),
                requested_anchors=57,
                requested_adjacent_intervals=56,
                results_table=table,
                old_full_record_C34_table=references["old_summary"]["results_table"],
                existing_3D_reference_counts=references["old_summary"][
                    "existing_3D_reference_counts"
                ],
                old_reference_summary=references["old_summary_record"],
                producer_seal_sha256=args.replay_sha,
                producer_interval=list(PROCESS),
                bounded_future_horizon=PROCESS[1],
                shared_seed=False,
                all7_lambdas=True,
                lambda_selected=False,
                strict_mouth_support_recomputed=False,
                original_label_eligibility_preserved=True,
                SLEAP_rematched=False,
                old_C34_or_3D_payloads_read=False,
                raw_video_calibration_mask_reads=0,
                checkpoint_reads=0,
                identity_solver_calls=0,
                GPU_used=False,
                physical_all3_ID_truth_validated=False,
                dense_swap_zero_certified=False,
                limitations=[
                    "female57 endpoints only;56 adjacent requested intervals retain U",
                    "changes across unscored anchors are endpoint brackets, not event counts",
                    "bounded C34 uses unconstrained initialization; new tracker seed is not shared",
                    "old C34 uses full-record future information; "
                    "cache-corrected replay is a new run",
                    "generative support/review acceptance differs "
                    "from saved C34 label availability",
                ],
            ),
        )
        for capture in captures.values():
            require(
                old.file_stat(Path(capture["path"])) == capture["file_stat"], "input stat changed"
            )
        evidence.final_guard()
        out.final_guard()
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        status = "ok"
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        (output / "evaluation.provenance.json").write_bytes(
            proof.json_bytes(
                dict(
                    stage=STAGE,
                    status=status,
                    error=error,
                    inputs={
                        **evidence.paths,
                        **{r["path"]: r["sha256"] for r in captures.values()},
                    },
                    outputs=out.records,
                )
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-root", type=Path, default=WORK / PRODUCER_STAGE)
    parser.add_argument("--replay-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    execute(parser.parse_args())
