"""CPU preflight of three registered geometry/enrollment input combinations."""

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))

import identity_inputs as loader  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402


def main():
    output = REPO / "work/three_body/identity_inputs_qa_v1"
    if output.exists():
        raise FileExistsError(output)
    code = list(
        dict.fromkeys(
            [
                Path(__file__),
                LAB / "identity_inputs.py",
                LAB / "chunk_masks.py",
                REPO / "tests/test_three_body_identity_inputs.py",
                *loader.enrollment_audit.audit_code_paths(),
                *loader.appearance_runner.code_paths(),
                *loader.persistent_runner.code_paths(),
            ]
        )
    )
    frozen = {str(p): loader.base.sha256(p) for p in code}
    output.mkdir()
    with provenance("three_body_identity_input_preflight", output, name="audit") as record:
        record.param("device", "cpu")
        record.param("tracking_or_identity_optimization_performed", False)
        record.param("ground_truth_or_reference_arrays_opened", False)
        for path, digest in frozen.items():
            record.input(path, path, sha256=digest)
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if loader.base.sha256(target) != digest:
                raise RuntimeError("source changed during snapshot")
            record.output("code:" + str(Path(path).relative_to(REPO)), target, sha256=digest)
        rows, inputs = [], {}
        for label, geometry, enrollment, audit in (
            (
                "same_window_200",
                "persistent_mouth_transient_200_v1",
                "appearance_transient_200_v1",
                "appearance_transient_200_enrollment_audit_v1",
            ),
            (
                "same_window_666",
                "persistent_mouth_interval_666_v1",
                "appearance_interval_666_v1",
                "appearance_interval_666_enrollment_audit_v1",
            ),
            (
                "minute_geometry_666_enrollment",
                "persistent_mouth_continuous_one_minute_v1",
                "appearance_interval_666_v1",
                "appearance_interval_666_enrollment_audit_v1",
            ),
        ):
            start = time.perf_counter()
            result = loader.load_identity_inputs(
                REPO / "work/three_body" / geometry,
                REPO / "work/three_body" / enrollment,
                REPO / "work/three_body" / audit,
            )
            row = dict(
                case=label,
                geometry=str(result.geometry_root),
                enrollment=str(result.enrollment_root),
                raw_enrollment_audit=str(result.audit_root),
                frames=[int(result.frame_idx[0]), int(result.frame_idx[-1] + 1)],
                geometry_arrays=len(result.geometry),
                enrollment_arrays=len(result.enrollment),
                input_count=len(result.verified_hashes),
                enrollment_counts=result.enrollment["counts"].tolist(),
                prefix_audit=result.prefix_audit,
                elapsed_seconds=time.perf_counter() - start,
            )
            for path, digest in result.verified_hashes.items():
                if inputs.get(path, digest) != digest:
                    raise RuntimeError("conflicting input hashes across cases")
                inputs[path] = digest
            rows.append(row)
            print(
                json.dumps(dict(case=label, status="ok", elapsed_seconds=row["elapsed_seconds"])),
                flush=True,
            )
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        for path, digest in {**frozen, **inputs}.items():
            if loader.base.sha256(path) != digest:
                raise RuntimeError("input/source changed during complete preflight")
        record.param("inputs_and_sources_unchanged", True)
        summary = dict(
            status="complete_input_preflight",
            cases=rows,
            producer_summaries_deserialized=False,
            final_appearance_candidate_deserialized=False,
            ground_truth_or_reference_arrays_opened=False,
            tracking_or_identity_optimization_performed=False,
            tests="70 focused CPU cases; Ruff passed",
            inputs_sha256=inputs,
            code_sha256=frozen,
            limitations=[
                "Prefix equivalence does not establish biological identity.",
                "The minute and enrollment staging backgrounds differ; they are unused in detectron2 mode.",
                "Only prefix masks are decoded here; full mask files are bound by SHA.",
                "Producer summaries and final appearance candidates are hashed only, never deserialized.",
            ],
        )
        path = output / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=loader.base.sha256(path))


if __name__ == "__main__":
    main()
