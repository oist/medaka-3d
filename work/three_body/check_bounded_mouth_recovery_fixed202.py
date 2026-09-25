"""Completed fixed202 CPU equivalence check; no tracking, fitting or GPU."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import audit_bounded_mouth_seeds as audited
import bounded_mouth_recovery as helper
import numpy as np
from node_geometry import RayObservations, supported_nodes
from node_recovery import NodeRecoveryConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

checks, prior, REPO = audited.checks, audited.prior, audited.REPO
STAGE = "bounded_mouth_recovery_fixed202_equivalence_v1"
AUDIT_ROOT = REPO / "work/three_body/bounded_mouth_seed_fixed202_independent_audit_v1"
AUDIT_SHA = "8b07bd1efcc7251e8c1614e344bbbc0d4aa6c2a26ff2da006cce98d8717842a0"
HELPER_SHA = "b128932013196d2f54cd99e69763f88ef492057222bd62800bf8e0f58e30f9df"
ATOL = 1e-10
FAILED_ROOT = REPO / "work/three_body/bounded_mouth_recovery_fixed202_equivalence_v1"


def bind_member(path, record, hashes):
    path = Path(path).resolve()
    item = record["inputs"].get(str(path))
    checks.require(
        item is not None and Path(item["path"]).resolve() == path,
        "missing source-bound input: " + str(path),
    )
    checks.require(checks.sha(path) == item["sha256"], "bound input changed: " + str(path))
    hashes[str(path)] = item["sha256"]
    return path


def compare(actual, expected, name, differences, *, exact=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    checks.require(actual.shape == expected.shape, name + " shape")
    if exact:
        checks.require(checks.equal_bytes(actual, expected), name + " dtype/bytes")
    else:
        checks.require(np.isfinite(actual).all() and np.isfinite(expected).all(), name + " finite")
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        checks.require(error <= ATOL, name + " numeric tolerance")
        differences[name] = max(differences.get(name, 0.0), error)


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = prior.generated.comparison.source_registry.local_code()
    checks.require(checks.sha(helper.__file__) == HELPER_SHA, "frozen helper changed")
    hashes = dict(code)
    for name in ("check.provenance.json", "failed_driver.py", "FAILURE.md"):
        path = FAILED_ROOT / name
        hashes[str(path)] = checks.sha(path)
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="check") as record:
        audit = audited.sealed(
            AUDIT_ROOT, "audit.provenance.json", audited.STAGE, AUDIT_SHA, hashes
        )
        audit_summary = checks.read(audited.required(AUDIT_ROOT / "summary.json", hashes))
        checks.require(
            audit_summary["every_rank_and_candidate_array_matches"]
            and audit_summary["all202_actual_calibration_support_reprojected"],
            "successful prior independent numerical audit required",
        )
        eval_seal_path = audited.ROOT / "evaluation.provenance.json"
        bind_member(eval_seal_path, audit, hashes)
        evaluation = audited.sealed(
            audited.ROOT,
            eval_seal_path.name,
            "bounded_mouth_seed_fixed202_evaluation",
            audit_summary["evaluation_seal_sha256"],
            hashes,
        )
        bind_member(prior.ROOT / "generation.provenance.json", audit, hashes)
        generation = audited.sealed(
            prior.ROOT, "generation.provenance.json", prior.generated.STAGE, prior.SHA, hashes
        )
        for path, digest in code.items():
            if path in audit["inputs"]:
                checks.require(
                    audit["inputs"][path]["sha256"] == digest,
                    "loaded historical code differs: " + path,
                )
        for module in (helper.original, helper.bounded_ray_seed):
            source = str(Path(module.__file__).resolve())
            checks.require(
                evaluation["inputs"][source]["sha256"] == code[source],
                "original helper/solver differs from evaluated version",
            )
        summary = checks.read(audited.required(audited.ROOT / "summary.json", hashes))
        expected_frames = checks.read(audited.required(audited.ROOT / "frames.json", hashes))
        expected_rows = [
            json.loads(line)
            for line in audited.required(audited.ROOT / "all202.jsonl", hashes)
            .read_text()
            .splitlines()
        ]
        checks.require(
            summary["native_interval"] == [67663, 67696]
            and summary["combinations"] == len(expected_rows) == 202
            and summary["admitted"] == 85
            and summary["returned"] == 62
            and [row["frame_idx"] for row in expected_frames] == list(range(67663, 67696)),
            "fixed202 scope changed",
        )
        checkpoint = prior.prior.BRANCH / "checkpoints/through_068060"
        for name in ("manifest.json", "metadata.json"):
            bind_member(checkpoint / name, audit, hashes)
            bind_member(checkpoint / name, generation, hashes)
        context = checks.read(checkpoint / "metadata.json")["context"]
        checks.require(
            summary["original_recovery_config"] == context["recovery_config"],
            "recovery configuration changed",
        )
        cfg = NodeRecoveryConfig(**context["recovery_config"])
        binding = checks.read(checkpoint / "manifest.json")["bindings"]["original_binding"]
        calibration = None
        for path, digest in binding["inputs"]["seed"]["SL_and_calibration_sha256"].items():
            if Path(path).suffix == ".json":
                bind_member(path, audit, hashes)
                checks.require(hashes[str(Path(path).resolve())] == digest, "calibration binding")
                if Path(path).name == "calibration_manifest.json":
                    calibration = Path(path).parent
        checks.require(calibration is not None, "calibration missing")
        rig = load_rig(calibration)
        normalize = prior.generated.comparison.support_audit.COMMON.normalize
        checks.require(
            normalize([dataclasses.asdict(camera) for camera in rig.cameras]) == context["cameras"],
            "actual rig differs from saved context",
        )
        fields = ["predicted_pose", "predicted_plan", "initial_lengths"] + [
            view + suffix
            for view in prior.prior.VIEWS
            for suffix in ("_points", "_ray_origins", "_ray_directions")
        ]
        differences, frames, diagnostics = {}, [], []
        elapsed = 0.0
        for frame, expected_frame in zip(range(67663, 67696), expected_frames, strict=True):
            path = bind_member(prior.ROOT / f"f{frame}_generation.npz", audit, hashes)
            audited.required(path, hashes)
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key].copy() for key in fields}
            with np.load(
                audited.required(audited.ROOT / f"f{frame}_bounded.npz", hashes), allow_pickle=False
            ) as archive:
                expected = {key: archive[key].copy() for key in archive.files}
            for key in ("predicted_pose", "predicted_plan", "initial_lengths"):
                compare(data[key], expected[key], key, differences, exact=True)
            frozen = {key: value.tobytes() for key, value in data.items()}
            scene, plan = data["predicted_pose"], data["predicted_plan"]
            rays = RayObservations(
                {view: data[view + "_points"] for view in prior.prior.VIEWS},
                tuple(data[view + "_ray_origins"] for view in prior.prior.VIEWS),
                tuple(data[view + "_ray_directions"] for view in prior.prior.VIEWS),
            )
            started = time.perf_counter()
            result = helper.bounded_mouth_recovery_proposals(
                scene, plan, rays, rig, data["initial_lengths"], cfg
            )
            elapsed += time.perf_counter() - started
            checks.require(
                frozen == {key: value.tobytes() for key, value in data.items()},
                "helper mutated original frame input",
            )
            checks.require(all(p.node == 0 for p in result.proposals), "mouth-only helper required")
            target_proposals = [p for p in result.proposals if p.fish == 0]
            rows = [row for row in expected_rows if row["frame_idx"] == frame]
            expected_outside = {
                tuple(row["source_rows"]): row
                for row in rows
                if row["original_first_gate"] == "outside_box"
            }
            attempts = result.diagnostics["fish"][0]["attempts"]
            checks.require(
                [tuple(a["source_rows"]) for a in attempts] == list(expected_outside),
                "outside-only combination order differs",
            )
            checks.require(
                all(
                    not row["admitted"]
                    for row in rows
                    if row["original_first_gate"] != "outside_box"
                ),
                "fixture contains an accepted original inside candidate",
            )
            admitted = []
            for attempt in attempts:
                expected_row = expected_outside[tuple(attempt["source_rows"])]
                checks.require(
                    (attempt["status"] == "admitted") == expected_row["admitted"],
                    "admission differs",
                )
                compare(attempt["point_xyz"], expected_row["xyz"], "bounded_point", differences)
                compare(attempt["ray_rms_mm"], expected_row["ray_rms_mm"], "RMS", differences)
                if attempt["status"] == "admitted":
                    candidate, assignment = scene.copy(), plan.copy()
                    candidate[0, 0] = attempt["point_xyz"]
                    assignment[0, :, 0] = attempt["source_rows"]
                    sources, observed, distance = supported_nodes(candidate, assignment, rays, rig)
                    compare(
                        sources,
                        np.asarray(expected_row["strict_sources"], np.int32),
                        "strict_sources",
                        differences,
                        exact=True,
                    )
                    compare(
                        observed,
                        np.asarray(expected_row["observed"], bool),
                        "observed",
                        differences,
                        exact=True,
                    )
                    compare(
                        distance,
                        expected_row["strict_float32_ray_distance_mm"],
                        "strict_ray_distances",
                        differences,
                    )
                    compare(
                        attempt["mouth_pectoral_mm"],
                        expected_row["mouth_chest_length_mm"],
                        "mouth_pectoral",
                        differences,
                    )
                    compare(
                        attempt["length_ratio"], expected_row["length_ratio"], "ratio", differences
                    )
                    displacement = float(np.linalg.norm(candidate[0, 0] - scene[0, 0]))
                    key = (
                        -int((sources[0, :, 0] >= 0).sum()),
                        attempt["ray_rms_mm"],
                        displacement,
                        tuple(assignment[0, :, 0]),
                    )
                    admitted.append((key, candidate, assignment, expected_row))
            admitted.sort(key=lambda item: item[0])
            compare(
                np.stack([a[1] for a in admitted]),
                expected["admitted_pose"],
                "ranked_admitted_pose",
                differences,
            )
            compare(
                np.stack([a[2] for a in admitted]),
                expected["admitted_plan"],
                "ranked_admitted_plan",
                differences,
                exact=True,
            )
            compare(
                np.array([a[3]["index_in_frame"] for a in admitted], np.int64),
                expected["admitted_combination_indices"],
                "admitted_ranking",
                differences,
                exact=True,
            )
            checks.require(
                len(target_proposals) == expected_frame["returned"]
                and len(admitted) == expected_frame["admitted"],
                "per-frame count",
            )
            for proposal, (_, candidate, assignment, row) in zip(
                target_proposals, admitted[: cfg.proposals_per_fish], strict=True
            ):
                compare(proposal.pose_xyz, candidate, "returned_pose", differences, exact=True)
                compare(proposal.plan, assignment, "returned_plan", differences, exact=True)
                checks.require(
                    proposal.supported_views == row["strict_mouth_views"], "support count"
                )
                compare(
                    proposal.displacement_mm,
                    np.linalg.norm(candidate[0, 0] - scene[0, 0]),
                    "displacement",
                    differences,
                )
                compare(
                    proposal.mouth_pectoral_mm,
                    row["mouth_chest_length_mm"],
                    "returned_length",
                    differences,
                )
                compare(proposal.length_ratio, row["length_ratio"], "returned_ratio", differences)
                compare(proposal.ray_rms_mm, row["ray_rms_mm"], "returned_RMS", differences)
            np.savez_compressed(
                output / f"f{frame}_helper.npz",
                returned_pose=np.stack([p.pose_xyz for p in result.proposals]),
                returned_plan=np.stack([p.plan for p in result.proposals]),
                returned_fish=np.array([p.fish for p in result.proposals], np.int64),
            )
            frames.append(
                dict(
                    frame_idx=frame,
                    combinations=len(rows),
                    bounded_attempts=len(attempts),
                    admitted=len(admitted),
                    returned=len(target_proposals),
                    all_fish_counts=[
                        {
                            key: item[key]
                            for key in ("fish", "bounded_attempts", "admitted", "returned")
                        }
                        for item in result.diagnostics["fish"]
                    ],
                )
            )
            diagnostics.append(dict(frame_idx=frame, **result.diagnostics))
        report = dict(
            schema=STAGE,
            status="ok",
            native_interval=[67663, 67696],
            frames=33,
            compared_raw_slot=0,
            combinations=202,
            bounded_attempts=sum(f["bounded_attempts"] for f in frames),
            admitted=sum(f["admitted"] for f in frames),
            returned=sum(f["returned"] for f in frames),
            all_fish_admitted=sum(
                item["admitted"] for f in frames for item in f["all_fish_counts"]
            ),
            all_fish_returned=sum(
                item["returned"] for f in frames for item in f["all_fish_counts"]
            ),
            other_slots_reference_parity_claim=False,
            helper_seconds=elapsed,
            numeric_atol=ATOL,
            numeric_rtol=0,
            max_differences=differences,
            discrete_dtype_bytes_exact=True,
            per_frame_counts_and_ranking_match=True,
            input_arrays_unchanged=True,
            original_inside_candidates_returned=False,
            prior_independent_audit_sha256=AUDIT_SHA,
            historical_source_bindings_checked=True,
            helper_sha256=HELPER_SHA,
            actual_raw_video_or_SL_redecoded=False,
            old_winner_coordinates_loaded=False,
            age_gate_applied=False,
            fit_performed=False,
            tracking_performed=False,
            GPU_used=False,
            identity_truth_or_held_out_improvement_claim=False,
            adopted=False,
            prior_failed_check=str(FAILED_ROOT),
            prior_failure_reason="driver assumed all3-fish helper scope equals S0-only reference",
        )
        checks.require(report["admitted"] == 85 and report["returned"] == 62, "total counts")
        for name, value in (
            ("summary.json", report),
            ("frames.json", frames),
            ("diagnostics.json", diagnostics),
        ):
            (output / name).write_text(
                json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
            )
        (output / "RESULT.md").write_text(
            "# Bounded mouth helper: fixed202 CPU equivalence\n\n"
            f"S0 across 33 frames; {report['bounded_attempts']} outside-box combinations solved; "
            "85 admitted and 62 returned candidates match the independently audited evaluation. "
            "Every frame's rank, source plan, direct-support flags and proposal fields passed; "
            f"maximum numeric difference {max(differences.values()):.3g} (allowed 1e-10).\n\n"
            "Only the current predicted scene/plan, current calibrated rays and initial lengths "
            "were read. The remaining original inside-box combinations are deliberately not "
            "returned by this additional helper; a future caller must preserve the original pool. "
            f"The full three-fish helper returned {report['all_fish_returned']} proposals; "
            "Other slots are retained separately and have no counterpart in this S0-only reference. "
            "The first check failed on an incorrect whole-helper-equals-S0 scope assumption; "
            "its failed seal and executed driver are preserved and bound as inputs here. "
            "The helper does not apply ages. No old winner coordinates, fit, tracking, raw-video "
            "decode, GPU or identity truth is used. This is a post-hoc equivalence test, not "
            "evidence of fit/beam retention or adoption.\n",
            encoding="utf-8",
        )
        for source, digest in code.items():
            path = output / "code_snapshot" / Path(source).relative_to(REPO)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(source).read_bytes())
            checks.require(checks.sha(path) == digest, "source snapshot changed")
        checks.require(
            code == prior.generated.comparison.source_registry.local_code(), "source set changed"
        )
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "bound input/source changed")
            record.input(path, path, sha256=digest)
        outputs = {}
        for path in output.rglob("*"):
            if path.is_file() and not path.name.endswith("provenance.json"):
                outputs[str(path)] = checks.sha(path)
                record.output(str(path.relative_to(output)), path, sha256=outputs[str(path)])
        for path, digest in {**hashes, **outputs}.items():
            checks.require(checks.sha(path) == digest, "final source/input/output changed")
        record.param("all_bound_files_unchanged", True)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
