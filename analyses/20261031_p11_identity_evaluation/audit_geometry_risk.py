"""Audit completed risk outputs and export paired connection evidence for review."""

import json
import os
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from run_pipeline import HERE, REPO, settings, sha256, write_json

from medaka3d.eval.flip_risk import connection_changes
from medaka3d.provenance import provenance


def main():
    policy = tomllib.loads((HERE / "config_geometry_risk_v3.toml").read_text())
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / policy["source_config"])
    cfg, _, _, source = settings()
    run = source.parent / policy["run_id"]
    if json.loads((run / "status.json").read_text())["status"] != "complete":
        raise ValueError("The comparison has not completed")
    out = run / "audit_v1"
    out.mkdir(exist_ok=False)
    checks, totals = {}, []
    manifest = json.loads((run / "artifact_manifest.json").read_text())
    verified = 0
    for name, info in manifest.items():
        if sha256(run / name) != info["sha256"]:
            raise ValueError(f"Artifact changed: {name}")
        verified += 1
    with provenance("p11_geometry_risk_audit", out) as rec:
        for path in (Path(__file__), REPO / "src/medaka3d/eval/flip_risk.py"):
            shutil.copy2(path, out / path.name)
            rec.input(path.name, path, sha256=sha256(path))
        rec.input(
            "manifest",
            run / "artifact_manifest.json",
            sha256=sha256(run / "artifact_manifest.json"),
        )
        summaries = pd.read_csv(run / "warning_summary.csv")
        scores = pd.read_csv(run / "correction_summary.csv")
        exp = cfg["experiment"]
        for arm in cfg["pipeline"]["lambdas"]:
            tag = f"{arm:g}"
            dest = run / f"lam{tag}"
            with np.load(source / "scoring_rerun_v1" / f"scoring_arrays_lam{tag}.npz") as z:
                times, weights, before = z["recording_time_s"], z["duration_s"], z["matched_slot"]
            with np.load(dest / "c30_scoring.npz") as z:
                after = z["matched_slot"]
            changed, count = connection_changes(before, after, times, weights)
            changed.to_csv(out / f"c30_connection_changes_lam{tag}.csv", index=False)
            count["lambda_mm"] = arm
            totals.extend(count.to_dict("records"))
            c33 = (
                source
                / "pipeline/c33_ray_rejection"
                / f"{exp}__C28-gated"
                / f"C33-gated__{exp}__lam{tag}.npz"
            )
            with np.load(c33) as origin, np.load(dest / "c30_ablation.npz") as new:
                perm = new["permutation_to_c33"]
                if not np.array_equal(
                    np.sort(perm, axis=1), np.tile(np.arange(3), (len(times), 1))
                ):
                    raise AssertionError("Invalid C30/C34 permutation")
                for field in ("xyz", "source"):
                    expected = np.take_along_axis(origin[field], perm[:, :, None], axis=1)
                    if not np.array_equal(expected, new[field], equal_nan=True):
                        raise AssertionError("Geometry or contributing sources were modified")
                if not np.array_equal(new["label_available"], origin["n_views_used"] >= 2):
                    raise AssertionError(
                        "Frame label mask differs from the original stage contract"
                    )
            intervals = pd.read_csv(dest / "warning_intervals_identity_core.csv")
            capture = pd.read_csv(dest / "group_capture_identity_core.csv")
            raw = pd.read_csv(dest / "raw_capture_identity_core.csv")
            mask = np.zeros(len(times), bool)
            for row in intervals.itertuples():
                mask[row.start_frame : row.stop_frame_exclusive] = True
                if not np.isclose(
                    weights[row.start_frame : row.stop_frame_exclusive].sum(), row.duration_s
                ):
                    raise AssertionError("Interval duration mismatch")
            with np.load(dest / "warning_mask.npz") as z:
                np.testing.assert_array_equal(mask, z["warning"])
            summary = summaries[
                summaries.lambda_mm.eq(arm) & summaries.rule.eq("identity_core")
            ].iloc[0]
            if not np.isclose(weights[mask].sum(), summary.warning_seconds):
                raise AssertionError("Warning union duration mismatch")
            # Independent endpoint checks against the union mask; broad missing
            # brackets cannot earn containment from merely touching a warning.
            fully_covered = []
            for row in capture.itertuples():
                fully_covered.append(bool(mask[row.before_frame : row.after_frame + 1].all()))
            if fully_covered != capture.warning_contains_bracket.tolist():
                raise AssertionError("Warning bracket containment mismatch")
            selected = scores[scores.lambda_mm.eq(arm) & scores.block.eq("all")]
            baseline = selected[selected.method.eq("baseline")].iloc[0]
            candidate = selected[selected.method.eq("c30_permissive_velocity")].iloc[0]
            if not np.isclose(
                candidate.wrong_seconds,
                baseline.wrong_seconds
                - candidate.corrected_seconds
                + candidate.newly_wrong_seconds,
            ):
                raise AssertionError("Paired error-time accounting mismatch")
            intervals["reference_group_ids"] = [
                ",".join(
                    map(
                        str,
                        capture.loc[
                            (capture.start_s >= row.start_s) & (capture.end_s < row.end_s),
                            "group_id",
                        ],
                    )
                )
                for row in intervals.itertuples()
            ]
            intervals["reference_group_count"] = intervals.reference_group_ids.map(
                lambda s: len(s.split(",")) if s else 0
            )
            intervals["review_priority"] = np.select(
                [
                    intervals.reasons.str.contains("nonidentity_nearest"),
                    intervals.reasons.str.contains("large_step"),
                ],
                ["1_assignment_disagreement", "2_large_step"],
                default="3_proximity_or_missing",
            )
            intervals.to_csv(out / f"review_queue_lam{tag}.csv", index=False)
            clean = pd.read_csv(dest / "lower_risk_intervals.csv")
            checks[tag] = {
                "geometry_and_sources_exact_permutations": True,
                "matched_availability_identical": bool(np.array_equal(before >= 0, after >= 0)),
                "warning_duration_and_containment_independently_verified": True,
                "warning_interval_count": len(intervals),
                "warning_seconds": float(weights[mask].sum()),
                "sustained_groups": len(capture),
                "sustained_groups_contained": sum(fully_covered),
                "all_observed_changes": len(raw),
                "all_observed_changes_contained": int(raw.warning_contains_bracket.sum()),
                "lower_risk_intervals": len(clean),
                "lower_risk_seconds": float(clean.duration_s.sum()),
                "lower_risk_intervals_with_mixed_reference_slot": int(
                    (clean[[f"slot{i}_reference_labels_observed" for i in range(3)]] > 1)
                    .any(axis=1)
                    .sum()
                ),
                "longest_lower_risk_s": float(clean.duration_s.max()),
                "reference_guidance_is_not_deployable_accuracy": True,
            }
        pd.DataFrame(totals).to_csv(out / "c30_connection_summary.csv", index=False)
        write_json(
            out / "validation.json", {"manifest_artifacts_verified": verified, "arms": checks}
        )
        rec.output("validation", out / "validation.json", sha256=sha256(out / "validation.json"))
    print(json.dumps(checks["0.767"], indent=2), flush=True)


if __name__ == "__main__":
    main()
