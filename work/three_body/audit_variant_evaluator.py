"""Exercise new evidence binding on a completed 200-frame run, not minute scoring."""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
import evaluate_variant_minute as evaluation  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

OUT = REPO / "work/three_body/variant_evaluator_cpu_qa_v1"


def main():
    if OUT.exists():
        raise FileExistsError("CPU evidence QA is immutable")
    run = REPO / "work/three_body/persistent_mouth_transient_200_v1"
    v1 = REPO / "work/three_body/v1_transient_detectron2"
    source = [
        Path(__file__),
        LAB / "evaluate_variant_minute.py",
        LAB / "evaluate_minute.py",
        LAB / "run_experiment.py",
        LAB / "run_node_recovery.py",
        LAB / "run_persistent_node_recovery.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        LAB / "diagnostics.py",
        LAB / "review_events.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/eval/generative.py",
        REPO / "src/medaka3d/eval/reacquisition.py",
        REPO / "tests/test_three_body_variant_minute_evaluation.py",
        REPO / "tests/test_three_body_minute_evaluation.py",
    ]
    source_hashes = {str(path): evaluation.base.sha256(path) for path in source}
    summary = evaluation.load_json(run / "summary.json")
    v1_summary = evaluation.load_json(v1 / "summary.json")
    frames = np.arange(333519, 333719, dtype=np.int64)
    started = time.perf_counter()
    hashes = evaluation.verify_variant_evidence(run, summary, expected_frames=frames)
    v1_hashes = evaluation.original.verify_run_evidence(v1, v1_summary)
    parity = evaluation.same_v1_inputs(run, summary, v1, v1_summary, hashes, v1_hashes)
    with np.load(run / "candidate.npz", allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    evaluation.validate_variant_arrays(arrays, frames, summary)
    rejection = None
    try:
        evaluation.original.load_completed(run)
    except ValueError as error:
        rejection = str(error)
    if rejection != "expected the complete single-initialization registered minute":
        raise ValueError("The full-minute gate did not reject this 200-frame input as expected")
    elapsed = time.perf_counter() - started
    hashes.update(v1_hashes)
    for path in (v1 / "summary.json", v1 / "experiment.provenance.json"):
        hashes[str(path)] = evaluation.base.sha256(path)
    hashes.update(source_hashes)
    result = dict(
        status="completed_CPU_evidence_QA_not_minute_evaluation",
        variant_run=str(run),
        original_run=str(v1),
        frames=[333519, 333719],
        array_count=len(arrays),
        algorithm_arm=summary["algorithm_arm"],
        source_code_sha256=source_hashes,
        actual_absolute_path_input_binding=True,
        original_evaluator_role_verification_reused_for_v1=True,
        manifest_raw_masks_backgrounds_calibration_nodes_seed_snapshots_verified=True,
        inference_parity=parity,
        persistent_native_ages_validated=True,
        full_minute_gate_rejection=rejection,
        gpu_used=False,
        tracker_called=False,
        common_metrics_called=False,
        minute_candidate_read=False,
        elapsed_seconds=elapsed,
        tests_executed_before_this_audit=dict(
            new_tests=30,
            existing_tests=30,
            passed=60,
            command=(
                ".venv/Scripts/python.exe -m pytest "
                "tests/test_three_body_variant_minute_evaluation.py "
                "tests/test_three_body_minute_evaluation.py -q -p no:cacheprovider "
                "--basetemp work/qv2 --tb=short"
            ),
            note="Default temporary directory was inaccessible; fresh workspace basetemp passed.",
        ),
        independent_source_review=dict(
            reviewer="forward_projection agent",
            source_changed=False,
            tests_rerun=False,
            result="No blocker found in verify_variant_evidence, same_v1_inputs and CLI ordering.",
        ),
        limitations=[
            "This only verifies a completed 200-frame development artifact and input contracts.",
            "No scientific tracking result or individual identity correctness is established.",
            "The separate full-minute comparison still requires a successfully sealed full run.",
        ],
    )
    if any(evaluation.base.sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("Evidence or evaluation source changed during CPU QA")
    OUT.mkdir(parents=True)
    with provenance("variant_evaluator_CPU_evidence_QA", OUT, name="audit") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path in source:
            target = OUT / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record.output(
                str(target.relative_to(OUT)), target, sha256=evaluation.base.sha256(target)
            )
        path = OUT / "summary.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=evaluation.base.sha256(path))
        if any(evaluation.base.sha256(path) != digest for path, digest in hashes.items()):
            raise ValueError("Evidence changed while sealing CPU QA")
    print(
        json.dumps(
            dict(
                output=str(OUT),
                evaluator_sha256=source_hashes[str(LAB / "evaluate_variant_minute.py")],
                summary_sha256=evaluation.base.sha256(OUT / "summary.json"),
                provenance_sha256=evaluation.base.sha256(OUT / "audit.provenance.json"),
            )
        )
    )


if __name__ == "__main__":
    main()
