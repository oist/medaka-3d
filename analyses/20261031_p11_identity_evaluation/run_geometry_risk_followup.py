"""Separate identity warnings from quality alerts and ablate one C30 condition."""

import importlib.util
import json
import os
import shutil
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate import load_reference
from run_geometry_risk import LAB, load_lab, lower_risk_table, reference_oracle
from run_pipeline import HERE, REPO, settings, sha256, write_json

from medaka3d.eval.flip_risk import (
    capture_brackets,
    corrected_slots,
    paired_identity_time,
    reference_change_brackets,
)
from medaka3d.eval.review_identity import fixed_anchor_mapping, match_positions
from medaka3d.provenance import provenance
from medaka3d.stages import register


def ablation_module():
    spec = importlib.util.spec_from_file_location("p11_c30_ablation", LAB / "c30_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_ablation(path, dest, degraded, prerequisite):
    """Worker receives geometry and detector quality only, never reference labels."""
    mod = ablation_module()
    with np.load(path) as arrays:
        xyz = arrays["xyz"].astype(float)
        resolved = np.flatnonzero((arrays["n_views_used"] >= 2) & np.isfinite(xyz).any(axis=(1, 2)))
        linker = mod.PermissiveVelocity(len(xyz), resolved, degraded, prerequisite)
        register("linker", "permissive_" + Path(dest).name, linker)
        perm, audit = linker.permutation(xyz)
        np.savez_compressed(
            Path(dest) / "c30_ablation.npz",
            permutation_to_c33=perm,
            xyz=np.take_along_axis(xyz, perm[:, :, None], axis=1),
            source=np.take_along_axis(arrays["source"], perm[:, :, None], axis=1),
            label_available=arrays["n_views_used"] >= 2,
        )
    write_json(Path(dest) / "c30_ablation_decisions.json", audit)
    return str(dest)


def main():
    policy_path = HERE / "config_geometry_risk_v3.toml"
    policy = tomllib.loads(policy_path.read_text())
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / policy["source_config"])
    cfg, _, _, source = settings()
    previous = source.parent / policy["first_experiment"]
    out = source.parent / policy["run_id"]
    out.mkdir(exist_ok=False)
    write_json(out / "status.json", {"status": "running"})
    module = load_lab()
    manifest = json.loads((source / "artifact_manifest.json").read_text())
    previous_manifest = json.loads((previous / "artifact_manifest.json").read_text())
    inputs = json.loads((source / "inputs.json").read_text())
    exp = cfg["experiment"]
    summaries, corrections = [], []
    with provenance("p11_geometry_motion_risk_followup", out) as rec:

        def checked(path, root=source, hashes=manifest):
            digest = sha256(path)
            if hashes[path.relative_to(root).as_posix()]["sha256"] != digest:
                raise ValueError(f"Frozen artifact changed: {path}")
            rec.input(str(path.relative_to(root)), path, sha256=digest)
            return path

        for path in [
            Path(__file__),
            HERE / "run_geometry_risk.py",
            policy_path,
            LAB / "README.md",
            LAB / "linker.py",
            LAB / "c30_ablation.py",
            REPO / "src/medaka3d/eval/flip_risk.py",
            REPO / "src/medaka3d/eval/review_identity.py",
            REPO / "tools/c30_slot_stitch.py",
            REPO / "tools/c34_microflip.py",
            REPO / "tools/c31_doubt_channel.py",
        ]:
            target = out / "source" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            rec.input(path.name, path, sha256=sha256(path))
        rec.param("configuration", policy)
        rec.param("scope", "Retrospective P-1-1 comparison; no appearance and no promotion")
        with np.load(checked(source / "native_timing.npz")) as native:
            times = native["top_cam"]
        ref, ref_valid, _ = load_reference(inputs, len(times), cfg["evaluation"])
        rec.input("reference", Path(inputs["review"]["backup"]), sha256=inputs["review"]["sha256"])
        # Prepare the same detector-quality mask used by the original C30.
        import c01_s16_predictions as c01
        import c30_slot_stitch as c30

        degraded = np.zeros(len(times), bool)
        for view in ("top_cam", "side1_cam", "side2_cam"):
            path = Path(inputs[view]["cache"])
            if sha256(path) != inputs[view]["cache_sha256"]:
                raise ValueError("Frozen detection cache changed")
            rec.input(view, path, sha256=inputs[view]["cache_sha256"])
            detections = c01.load(source / "pipeline", exp, view, "bottomup")
            best = c30.best_scores(detections, len(times))
            degraded |= (best >= 0) & (best < c30.SCORE_DEGRADED)
        for h in (1, 2):
            widened = degraded.copy()
            widened[h:] |= degraded[:-h]
            widened[:-h] |= degraded[h:]
            degraded = widened
        futures = {}
        with ProcessPoolExecutor(max_workers=3) as pool:
            for arm in cfg["pipeline"]["lambdas"]:
                tag = f"{arm:g}"
                dest = out / f"lam{tag}"
                dest.mkdir()
                c33_path = checked(
                    source
                    / "pipeline/c33_ray_rejection"
                    / f"{exp}__C28-gated"
                    / f"C33-gated__{exp}__lam{tag}.npz"
                )
                futures[tag] = pool.submit(
                    run_ablation,
                    c33_path,
                    dest,
                    degraded,
                    policy["c30_ablation"]["velocity_needs_median_mm"],
                )
                with np.load(
                    checked(source / "scoring_rerun_v1" / f"scoring_arrays_lam{tag}.npz")
                ) as score:
                    matched, anchor, weights = (
                        score["matched_slot"],
                        score["anchor_reference_to_slot"],
                        score["duration_s"],
                    )
                    base_codes = score["status_code"]
                primary_valid = None
                with np.load(
                    checked(previous / f"lam{tag}/warning_masks.npz", previous, previous_manifest)
                ) as stored:
                    flags = {k: stored[k] for k in policy["warning"]["rules"]["without_ray_churn"]}
                    nearest, valid = stored["nearest_fish_mm"], stored["position_valid"]
                    primary_valid = valid
                groups = pd.read_csv(
                    checked(
                        previous / f"lam{tag}/reference_change_groups.csv",
                        previous,
                        previous_manifest,
                    )
                )
                raw = pd.read_csv(
                    checked(
                        previous / f"lam{tag}/all_observed_reference_changes.csv",
                        previous,
                        previous_manifest,
                    )
                )
                duration = times[-1] - policy["evaluation"]["preparation_seconds"]
                for name, keys in policy["warning"]["rules"].items():
                    intervals, mask = module.warning_intervals(
                        {k: flags[k] for k in keys},
                        nearest,
                        times,
                        weights,
                        policy["warning"]["proximity_mm"],
                        policy["warning"]["padding_s"],
                    )
                    context = policy["evaluation"]["review_context_s"]
                    intervals["review_start_s"] = np.maximum(300, intervals.start_s - context)
                    intervals["review_end_s"] = np.minimum(times[-1], intervals.end_s + context)
                    intervals["review_start_frame"] = np.searchsorted(
                        times, intervals.review_start_s
                    )
                    intervals["review_stop_frame_exclusive"] = np.searchsorted(
                        times, intervals.review_end_s, side="right"
                    )
                    intervals.to_csv(dest / f"warning_intervals_{name}.csv", index=False)
                    capture = capture_brackets(groups, intervals)
                    raw_capture = capture_brackets(raw, intervals)
                    capture.to_csv(dest / f"group_capture_{name}.csv", index=False)
                    raw_capture.to_csv(dest / f"raw_capture_{name}.csv", index=False)
                    row = {
                        "lambda_mm": arm,
                        "rule": name,
                        "interval_count": len(intervals),
                        "warning_seconds": float(weights[mask].sum()),
                        "warning_fraction": float(weights[mask].sum() / duration),
                        "group_count": len(groups),
                        "groups_contained": int(capture.warning_contains_bracket.sum()),
                        "groups_overlap": int(capture.warning_overlap.sum()),
                        "raw_changes": len(raw),
                        "raw_contained": int(raw_capture.warning_contains_bracket.sum()),
                        "raw_overlap": int(raw_capture.warning_overlap.sum()),
                        "outside_warning_wrong_fish_seconds": float(
                            (weights[~mask, None] * (base_codes[~mask] == 1)).sum()
                        ),
                        "encounter_seconds": float(weights[nearest < 20].sum()),
                        "encounter_warning_seconds": float(weights[mask & (nearest < 20)].sum()),
                    }
                    for percentile in (50, 90, 95, 100):
                        row[f"duration_p{percentile}_s"] = float(
                            np.percentile(intervals.duration_s, percentile)
                        )
                    summaries.append(row)
                    if name == policy["warning"]["primary_rule"]:
                        lower_risk_table(
                            mask, primary_valid, times, weights, matched, anchor
                        ).to_csv(dest / "lower_risk_intervals.csv", index=False)
                        np.savez_compressed(dest / "warning_mask.npz", warning=mask)
                        perm, decisions = reference_oracle(intervals, matched, len(times))
                        oracle = paired_identity_time(
                            matched, corrected_slots(matched, perm), anchor, weights
                        )
                        oracle.update(
                            lambda_mm=arm, method="reference_endpoint_oracle", block="all"
                        )
                        corrections.append(oracle)
                        write_json(dest / "reference_endpoint_oracle.json", decisions)
                print(f"Warnings complete: lambda {tag}", flush=True)
            for arm in cfg["pipeline"]["lambdas"]:
                tag = f"{arm:g}"
                dest = Path(futures[tag].result())
                with np.load(source / "scoring_rerun_v1" / f"scoring_arrays_lam{tag}.npz") as score:
                    matched, anchor, weights = (
                        score["matched_slot"],
                        score["anchor_reference_to_slot"],
                        score["duration_s"],
                    )
                with np.load(dest / "c30_ablation.npz") as candidate:
                    valid = (
                        np.isfinite(candidate["xyz"]).all(axis=2)
                        & ((candidate["source"] >= 0).sum(axis=2) >= 2)
                        & candidate["label_available"][:, None]
                    )
                    matches = match_positions(
                        ref,
                        candidate["xyz"],
                        ref_valid,
                        valid,
                        max_distance_mm=cfg["evaluation"]["max_match_distance_mm"],
                        min_margin_mm=cfg["evaluation"]["min_match_margin_mm"],
                    )
                anchor_frames = (times >= 298) & (times < 300)
                mapping = fixed_anchor_mapping(
                    matches.slots, anchor_frames, minimum_matches=100, minimum_agreement=0.95
                )
                new_anchor = np.array(mapping["reference_to_slot"])
                if np.any(anchor < 0) or np.any(new_anchor < 0):
                    raise ValueError("Unresolved anchor in C30 comparison")
                rename = anchor[np.argsort(new_anchor)]
                aligned = np.where(matches.slots >= 0, rename[np.maximum(matches.slots, 0)], -1)
                if not np.array_equal(matched >= 0, aligned >= 0):
                    raise AssertionError("Permutation-only method changed match availability")
                np.savez_compressed(
                    dest / "c30_scoring.npz",
                    matched_slot=aligned,
                    anchor_reference_to_slot=anchor,
                    original_anchor=new_anchor,
                )
                for method, predicted in (
                    ("baseline", matched),
                    ("c30_permissive_velocity", aligned),
                ):
                    for start, end in (
                        (300, times[-1]),
                        (300, 900),
                        (900, 1500),
                        (1500, times[-1]),
                    ):
                        w = weights * ((times >= start) & (times < end))
                        row = paired_identity_time(matched, predicted, anchor, w)
                        row.update(
                            lambda_mm=arm,
                            method=method,
                            block="all"
                            if start == 300 and end == times[-1]
                            else f"{start:g}-{end:g}",
                        )
                        corrections.append(row)
                transitions = reference_change_brackets(aligned, times, weights)
                transitions.to_csv(dest / "c30_sustained_reference_changes.csv", index=False)
                print(f"C30 ablation scored: lambda {tag}", flush=True)
        pd.DataFrame(summaries).to_csv(out / "warning_summary.csv", index=False)
        pd.DataFrame(corrections).to_csv(out / "correction_summary.csv", index=False)
        write_json(
            out / "status.json",
            {
                "status": "complete",
                "appearance_used": False,
                "matching_availability_unchanged_all_arms": True,
            },
        )
        rec.output(
            "summary", out / "warning_summary.csv", sha256=sha256(out / "warning_summary.csv")
        )
    write_json(
        out / "artifact_manifest.json",
        {
            p.relative_to(out).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    print(out, flush=True)


if __name__ == "__main__":
    main()
