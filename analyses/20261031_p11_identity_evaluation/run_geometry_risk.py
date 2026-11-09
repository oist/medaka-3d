"""Run frozen geometry/motion hypotheses using the existing manual reference."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from run_pipeline import HERE, REPO, settings, sha256, write_json

from medaka3d.eval.flip_risk import (
    capture_brackets,
    corrected_slots,
    endpoint_outcomes,
    group_change_brackets,
    paired_identity_time,
    reference_change_brackets,
)
from medaka3d.eval.review_identity import contiguous_runs, interval_weights
from medaka3d.provenance import provenance
from medaka3d.stages import register

LAB = REPO / "labs/20261104_geometry_flip_risk"


def load_lab():
    spec = importlib.util.spec_from_file_location("p11_motion_risk", LAB / "linker.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference_oracle(intervals, matched, n):
    """Diagnostic only: perfect reference endpoint checks at detected warnings."""
    labels = np.full_like(matched, -1)
    for animal in range(3):
        frames = np.flatnonzero(matched[:, animal] >= 0)
        labels[frames, matched[frames, animal]] = animal
    perm = np.tile(np.arange(3, dtype=np.int8), (n, 1))
    current = np.arange(3)
    rows = []
    for interval in intervals.itertuples():
        a, b = interval.start_frame - 1, interval.stop_frame_exclusive
        if a < 0 or b >= n:
            continue
        row = {
            "interval_id": interval.interval_id,
            "before_frame": a,
            "after_frame": b,
            "permutation": [0, 1, 2],
            "decision": "refused",
            "reason": "unmatched_endpoint",
        }
        if (labels[[a, b]] >= 0).all():
            q = np.argsort(labels[b])[labels[a]]
            row.update(
                permutation=q.tolist(),
                reason="reference_only",
                decision="retained" if np.array_equal(q, np.arange(3)) else "corrected",
            )
            current = q[current]
            perm[b:] = current
        rows.append(row)
    return perm, rows


def lower_risk_table(mask, valid, times, weights, matched, anchor):
    rows = []
    selected = ~mask & valid.all(axis=1) & (weights > 0)
    for i, (lo, hi) in enumerate(contiguous_runs(selected)):
        duration = float(weights[lo:hi].sum())
        row = {
            "segment_id": i,
            "start_frame": int(lo),
            "stop_frame_exclusive": int(hi),
            "start_s": float(times[lo]),
            "end_s": float(times[min(hi, len(times) - 1)]),
            "duration_s": duration,
            "identity_status": "lower_risk_candidate_only",
        }
        for slot in range(3):
            support = np.array(
                [weights[lo:hi][matched[lo:hi, animal] == slot].sum() for animal in range(3)]
            )
            row[f"slot{slot}_reference_labels_observed"] = int((support > 0).sum())
            row[f"slot{slot}_reference_coverage"] = float(support.sum() / duration)
            row[f"slot{slot}_majority_label"] = (
                ["F", "M1", "M2"][support.argmax()] if support.sum() else "unknown"
            )
            row[f"slot{slot}_reference_purity"] = (
                float(support.max() / support.sum()) if support.sum() else np.nan
            )
        row.update(paired_identity_time(matched[lo:hi], matched[lo:hi], anchor, weights[lo:hi]))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    policy_path = HERE / "config_geometry_risk_v1.toml"
    policy = tomllib.loads(policy_path.read_text())
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / policy["source_config"])
    cfg, _, _, source = settings()
    out = source.parent / policy["run_id"]
    out.mkdir(exist_ok=False)
    write_json(out / "status.json", {"status": "running"})
    module = load_lab()
    manifest = json.loads((source / "artifact_manifest.json").read_text())
    inputs = json.loads((source / "inputs.json").read_text())
    exp = cfg["experiment"]
    chain = source / "pipeline/c34_microflip" / f"{exp}__{cfg['pipeline']['chain_tag']}"
    warning_summaries, corrections, endpoint_summaries, signal_summaries = [], [], [], []
    with provenance("p11_geometry_motion_risk", out) as rec:
        for filename in (inputs["review"]["path"], inputs["review"]["backup"]):
            path = Path(filename)
            if sha256(path) != inputs["review"]["sha256"]:
                raise ValueError("The original or backed-up manual reference changed")
            rec.input("manual_reference_" + path.parent.name, path, sha256=sha256(path))
        snapshot = out / "source"
        sources = [
            Path(__file__),
            policy_path,
            HERE / policy["source_config"],
            LAB / "README.md",
            LAB / "linker.py",
            REPO / "src/medaka3d/eval/flip_risk.py",
            REPO / "src/medaka3d/eval/review_identity.py",
            REPO / "uv.lock",
        ]
        for path in sources:
            dest = snapshot / path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            rec.input(path.relative_to(REPO).as_posix(), path, sha256=sha256(path))
        rec.param("configuration", policy)
        rec.param("source_run", source.name)
        rec.param("source_manifest_sha256", sha256(source / "artifact_manifest.json"))
        rec.param("reference_use", "scoring and explicitly named diagnostic oracle only")
        rec.param("physical_identity_certification", "not performed")
        for arm in cfg["pipeline"]["lambdas"]:
            tag = f"{arm:g}"
            dest = out / f"lam{tag}"
            dest.mkdir()
            paths = {
                "label": chain / f"C34-label__{exp}__lam{tag}.npz",
                "gated": chain / f"C34-gated__{exp}__lam{tag}.npz",
                "scoring": source / "scoring_rerun_v1" / f"scoring_arrays_lam{tag}.npz",
            }
            for name, path in paths.items():
                digest = sha256(path)
                if digest != manifest[path.relative_to(source).as_posix()]["sha256"]:
                    raise ValueError(f"Frozen source changed: {path}")
                rec.input(f"{name}_lam{tag}", path, sha256=digest)
            with (
                np.load(paths["label"]) as label,
                np.load(paths["gated"]) as gated,
                np.load(paths["scoring"]) as score,
            ):
                times, weights = score["recording_time_s"], score["duration_s"]
                expected_weights, duration = interval_weights(
                    times, policy["evaluation"]["preparation_seconds"]
                )
                if not np.array_equal(weights, expected_weights):
                    raise ValueError("Native interval weights changed")
                matched, anchor = score["matched_slot"], score["anchor_reference_to_slot"]
                xyz, local = label["xyz"], gated["slot_track"]
                if not np.array_equal(xyz, gated["xyz"], equal_nan=True):
                    raise ValueError("Label/gated geometry mismatch")
                flags, nearest, valid = module.signals(
                    xyz,
                    label["source"],
                    label["residual_mm"],
                    local,
                    label["margin_mm"],
                    policy["warning"],
                )
                baseline_codes = score["status_code"]
            base = paired_identity_time(matched, matched, anchor, weights)
            for key, code in (("correct_seconds", 0), ("wrong_seconds", 1)):
                if not np.isclose(base[key], (weights[:, None] * (baseline_codes == code)).sum()):
                    raise AssertionError("Shared scorer does not reproduce frozen baseline")
            ev = policy["evaluation"]
            changes = reference_change_brackets(
                matched,
                times,
                weights,
                minimum_seconds=ev["stable_seconds"],
                minimum_samples=ev["stable_samples"],
                split_gap_s=ev["split_reference_gap_s"],
            )
            raw_changes = reference_change_brackets(
                matched,
                times,
                weights,
                minimum_seconds=0,
                minimum_samples=1,
                split_gap_s=ev["split_reference_gap_s"],
            )
            groups = group_change_brackets(changes)
            changes.to_csv(dest / "sustained_reference_changes.csv", index=False)
            groups.to_csv(dest / "reference_change_groups.csv", index=False)
            raw_changes.to_csv(dest / "all_observed_reference_changes.csv", index=False)
            for name, mask in flags.items():
                signal_summaries.append(
                    {
                        "lambda_mm": arm,
                        "signal": name,
                        "seconds": float(weights[mask].sum()),
                        "fraction_expected_time": float(weights[mask].sum() / duration),
                    }
                )
            primary = None
            variants = [(p, False) for p in policy["warning"]["proximity_mm"]]
            variants.append((policy["warning"]["primary_proximity_mm"], True))
            for proximity, broad in variants:
                name = f"distance{proximity:g}" + ("_margin" if broad else "")
                intervals, mask = module.warning_intervals(
                    flags, nearest, times, weights, proximity, policy["warning"]["padding_s"], broad
                )
                intervals["review_start_s"] = np.maximum(
                    300, intervals.start_s - ev["review_context_s"]
                )
                intervals["review_end_s"] = np.minimum(
                    times[-1], intervals.end_s + ev["review_context_s"]
                )
                intervals["review_start_frame"] = np.searchsorted(times, intervals.review_start_s)
                intervals["review_stop_frame_exclusive"] = np.searchsorted(
                    times, intervals.review_end_s, side="right"
                )
                intervals.to_csv(dest / f"warning_intervals_{name}.csv", index=False)
                captured = capture_brackets(groups, intervals)
                captured.to_csv(dest / f"group_capture_{name}.csv", index=False)
                capture_brackets(raw_changes, intervals).to_csv(
                    dest / f"raw_capture_{name}.csv", index=False
                )
                all_seconds = float(weights[mask].sum())
                summary = {
                    "lambda_mm": arm,
                    "rule": name,
                    "expected_seconds": duration,
                    "warning_seconds": all_seconds,
                    "warning_fraction": all_seconds / duration,
                    "interval_count": len(intervals),
                    "reference_group_count": len(groups),
                    "groups_overlap": int(captured.warning_overlap.sum()),
                    "groups_contained": int(captured.warning_contains_bracket.sum()),
                    "raw_change_count": len(raw_changes),
                    "sustained_change_count": len(changes),
                    "missing_geometry_seconds": float(weights[flags["missing_position"]].sum()),
                    "encounter_seconds": float(weights[nearest < 20].sum()),
                    "encounter_warning_seconds": float(weights[mask & (nearest < 20)].sum()),
                    "outside_warning_wrong_fish_seconds": float(
                        (weights[~mask, None] * (baseline_codes[~mask] == 1)).sum()
                    ),
                }
                for percentile in (50, 90, 95, 100):
                    summary[f"duration_p{percentile}_s"] = (
                        float(np.percentile(intervals.duration_s, percentile))
                        if len(intervals)
                        else 0
                    )
                warning_summaries.append(summary)
                if proximity == policy["warning"]["primary_proximity_mm"] and not broad:
                    primary = intervals, mask
                    lower_risk_table(mask, valid, times, weights, matched, anchor).to_csv(
                        dest / "lower_risk_intervals.csv", index=False
                    )
                    np.savez_compressed(
                        dest / "warning_masks.npz",
                        time_s=times,
                        eligible=weights > 0,
                        warning=mask,
                        position_valid=valid,
                        nearest_fish_mm=nearest,
                        **flags,
                    )
            intervals, _warning_mask = primary
            methods = ["baseline", *policy["bridge"]["methods"], "reference_endpoint_oracle"]
            for method in methods:
                if method == "baseline":
                    perm = np.tile(np.arange(3, dtype=np.int8), (len(times), 1))
                    decisions = []
                elif method == "reference_endpoint_oracle":
                    perm, decisions = reference_oracle(intervals, matched, len(times))
                else:
                    linker = module.MotionBridge(times, intervals, local, policy["bridge"], method)
                    if arm == 0:
                        register("linker", "p11_" + method, linker)
                    perm, decisions = linker.permutation(xyz)
                predicted = corrected_slots(matched, perm)
                if not np.array_equal(
                    np.sort(perm, axis=1), np.tile(np.arange(3), (len(times), 1))
                ):
                    raise AssertionError("Invalid permutation")
                if np.any(perm[times < 300] != np.arange(3)):
                    raise AssertionError("Anchor/preparation window was modified")
                block_starts = [None, *ev["block_boundaries_s"]]
                for start in block_starts:
                    end = (
                        times[-1]
                        if start is None or start == block_starts[-1]
                        else block_starts[block_starts.index(start) + 1]
                    )
                    w = weights if start is None else weights * ((times >= start) & (times < end))
                    result = paired_identity_time(matched, predicted, anchor, w)
                    result.update(
                        lambda_mm=arm,
                        method=method,
                        block="all" if start is None else f"{start:g}-{end:g}",
                        expected_fish_seconds=3 * (duration if start is None else end - start),
                        corrected_intervals=sum(d["decision"] == "corrected" for d in decisions),
                        retained_intervals=sum(d["decision"] == "retained" for d in decisions),
                        refused_intervals=sum(d["decision"] == "refused" for d in decisions),
                    )
                    corrections.append(result)
                if decisions:
                    outcome = endpoint_outcomes(decisions, matched)
                    outcome.to_csv(dest / f"endpoint_outcomes_{method}.csv", index=False)
                    totals = outcome.groupby(["decision", "before_outcome", "after_outcome"]).size()
                    for (decision, old, new), count in totals.items():
                        endpoint_summaries.append(
                            {
                                "lambda_mm": arm,
                                "method": method,
                                "decision": decision,
                                "before_outcome": old,
                                "after_outcome": new,
                                "count": int(count),
                            }
                        )
                    write_json(dest / f"decisions_{method}.json", decisions)
                    np.savez_compressed(dest / f"permutation_{method}.npz", permutation=perm)
            print(
                f"Finished lambda {tag}: {len(groups)} reference groups, "
                f"{len(intervals)} primary warnings",
                flush=True,
            )
        for name, rows in (
            ("warning_summary", warning_summaries),
            ("correction_summary", corrections),
            ("endpoint_summary", endpoint_summaries),
            ("signal_summary", signal_summaries),
        ):
            pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
        write_json(
            out / "status.json",
            {
                "status": "complete",
                "source_run": source.name,
                "arms": cfg["pipeline"]["lambdas"],
                "appearance_used": False,
                "reference_only_in_evaluation_and_named_oracle": True,
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
