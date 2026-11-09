"""Build versioned reference-based join review tables for baseline or repair."""

import argparse
import importlib
import json
import os
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from audit_c30_trace import require_identical_replay, trace_stitch
from run_pipeline import HERE, REPO, settings, sha256, write_json

from medaka3d.eval import join_ledger
from medaka3d.provenance import provenance


def add_context(edges, xyz, sources, valid, bounds, context):
    """Context remains geometric evidence; no reflection or behavior inference."""
    result = edges.copy()
    for index, edge in edges.iterrows():
        slot = int(edge.raw_slot)
        for side in ("before", "after"):
            frame = int(edge[f"{side}_frame"])
            point = xyz[frame, slot]
            distances = np.linalg.norm(xyz[frame] - point, axis=1)
            distances[~valid[frame]] = np.nan
            distances[slot] = np.nan
            finite = distances[np.isfinite(distances)]
            result.loc[index, f"{side}_nearest_other_mm"] = (
                float(finite.min()) if len(finite) else np.nan
            )
            result.loc[index, f"{side}_contributing_cameras"] = int(
                (sources[frame, slot] >= 0).sum()
            )
            result.loc[index, f"{side}_wall_clearance_signed_mm"] = float(
                np.min(np.r_[point[:2] - bounds[0, :2], bounds[1, :2] - point[:2]])
            )
            result.loc[index, f"{side}_depth_below_surface_signed_mm"] = float(
                bounds[1, 2] - point[2]
            )
        result.loc[index, "reflection_status"] = context["reflection_status"]
        result.loc[index, "boris_status"] = context["boris_status"]
    return result


def associate_decisions(edges, decisions, times, tolerance):
    """Temporal association only; C34 and C30 slot columns need not coincide."""
    rows = []
    for edge in edges.itertuples():
        for decision in decisions:
            frame = decision.get("after_frame")
            if frame is None:
                continue
            when = float(times[frame])
            inside = edge.before_time_s < when <= edge.after_time_s
            nearby = abs(when - edge.after_time_s) <= tolerance
            if inside or nearby:
                rows.append(
                    {
                        "edge_id": edge.edge_id,
                        "c30_decision_id": decision["c30_decision_id"],
                        "association": "within_edge_interval" if inside else "nearby_only",
                        "time_offset_from_edge_after_s": when - edge.after_time_s,
                        "causal_attribution": "not_established",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "edge_id",
            "c30_decision_id",
            "association",
            "time_offset_from_edge_after_s",
            "causal_attribution",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "repair"], required=True)
    args = parser.parse_args()
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / "config_row_mapping_v1.toml")
    cfg, _, root, repair = settings()
    policy_path = HERE / "config_join_ledger_v2.toml"
    policy = tomllib.loads(policy_path.read_text())
    config = join_ledger.JoinLedgerConfig(**policy["support"])
    run = (
        repair
        if args.variant == "repair"
        else (root / cfg["output"]["relative_to_data_parent"] / cfg["comparison"]["baseline_run"])
    )
    if json.loads((run / "run_status.json").read_text())["status"] != "complete":
        raise ValueError("Join tables require a complete source run")
    score = run / cfg["evaluation"]["primary_scoring_directory"]
    validation = json.loads((score / "validation.json").read_text())
    if not validation["summary_reconciled"]:
        raise ValueError("Source scoring has not reconciled")
    inputs = json.loads((run / "inputs.json").read_text())
    out = repair / policy["version"] / args.variant
    out.mkdir(parents=True, exist_ok=False)
    pipeline, exp = run / "pipeline", cfg["experiment"]
    c30 = importlib.import_module("c30_slot_stitch")
    c01 = importlib.import_module("c01_s16_predictions")
    # Trace only the actual saved implementation. A later algorithm change is
    # a new comparison, not an explanation retroactively assigned to this run.
    for filename in ("c30_slot_stitch.py", "c31_doubt_channel.py"):
        if sha256(REPO / "tools" / filename) != sha256(run / "source" / "tools" / filename):
            raise ValueError(f"C30 replay source differs: {filename}")
    with np.load(run / "native_timing.npz", allow_pickle=False) as timing:
        times = timing["top_cam"]
    with provenance("p11_join_ledger", out) as rec:
        for source in (
            Path(__file__),
            policy_path,
            Path(join_ledger.__file__),
            HERE / "audit_c30_trace.py",
            run / "inputs.json",
            score / "validation.json",
        ):
            rec.input(source.name, source, sha256=sha256(source))
            shutil.copy2(source, out / source.name)
        rec.param("source_run", run.name)
        rec.param("policy", policy)
        rec.param("physical_identity_review", "Not performed; reference agreement only")
        manifest_path = (
            root
            / cfg["input"]["calibration_relative_to_data_parent"]
            / ("calibration_manifest.json")
        )
        expected = inputs["calibration_calibration_manifest.json"]["sha256"]
        if sha256(manifest_path) != expected:
            raise ValueError("Calibration manifest changed")
        rec.input("calibration_manifest", manifest_path, sha256=expected)
        physical = json.loads(manifest_path.read_text())["physical_geometry"]
        bounds = np.array(
            [
                [0, 0, 0],
                [
                    physical["inside_x_span_mm"],
                    physical["inside_y_span_mm"],
                    physical["water_height_from_inside_floor_mm"],
                ],
            ]
        )
        rec.param("tank_bounds_mm", bounds.tolist())
        degraded = np.zeros(len(times), bool)
        for view in ("top_cam", "side1_cam", "side2_cam"):
            cached = Path(inputs[view]["cache"])
            if sha256(cached) != inputs[view]["cache_sha256"]:
                raise ValueError(f"Frozen detections changed: {view}")
            data = c01.load(pipeline, exp, view, "bottomup")
            best = c30.best_scores(data, data.n_frames)
            degraded |= (best >= 0) & (best < c30.SCORE_DEGRADED)
            rec.input(view, cached, sha256=inputs[view]["cache_sha256"])
        # Reproduce C30's input-mask preparation; the stitcher itself is called
        # directly and its entire permutation and decision lists must match.
        for h in (1, 2):
            widened = degraded.copy()
            widened[h:] |= degraded[:-h]
            widened[:-h] |= degraded[h:]
            degraded = widened
        c30_root = pipeline / "c30_slot_stitch" / f"{exp}__C33-on-C28-v4"
        saved_c30 = json.loads((c30_root / "c30_summary.json").read_text())
        rec.input(
            "c30_summary",
            c30_root / "c30_summary.json",
            sha256=sha256(c30_root / "c30_summary.json"),
        )
        summaries = {}
        observed_changes = pd.read_csv(score / "assignment_change_events.csv")
        disagreements = pd.read_csv(score / "wrong_identity_intervals.csv")
        for name in ("assignment_change_events.csv", "wrong_identity_intervals.csv"):
            rec.input(name, score / name, sha256=sha256(score / name))
        for lam in cfg["pipeline"]["lambdas"]:
            tag = f"{lam:g}"
            dest = out / f"lam{tag}"
            dest.mkdir()
            scoring_path = score / f"scoring_arrays_lam{tag}.npz"
            c34_root = pipeline / "c34_microflip" / f"{exp}__{cfg['pipeline']['chain_tag']}"
            label_path = c34_root / f"C34-label__{exp}__lam{tag}.npz"
            gated_path = c34_root / f"C34-gated__{exp}__lam{tag}.npz"
            c33_path = (
                pipeline
                / "c33_ray_rejection"
                / f"{exp}__C28-gated"
                / (f"C33-gated__{exp}__lam{tag}.npz")
            )
            c30_path = c30_root / f"C30-gated__{exp}__lam{tag}.npz"
            for name, path in (
                ("score", scoring_path),
                ("label", label_path),
                ("gated", gated_path),
                ("c33", c33_path),
                ("c30", c30_path),
            ):
                rec.input(f"{name}_lam{tag}", path, sha256=sha256(path))
            with (
                np.load(scoring_path) as scoring,
                np.load(label_path) as label,
                np.load(gated_path) as gated,
            ):
                if not np.array_equal(times, scoring["recording_time_s"]):
                    raise ValueError("Native time grid changed")
                if not np.array_equal(label["xyz"], gated["xyz"], equal_nan=True):
                    raise ValueError("Label and local-track geometry differ")
                xyz, sources = label["xyz"], label["source"]
                valid = np.isfinite(xyz).all(axis=2) & ((sources >= 0).sum(axis=2) >= 2)
                tables = join_ledger.build_join_ledger(
                    gated["slot_track"],
                    scoring["matched_slot"],
                    times,
                    scoring["duration_s"],
                    scoring["eligible"],
                    valid,
                    reference_labels=tuple(scoring["reference_labels"]),
                    config=config,
                )
            tables["edges"] = add_context(
                tables["edges"], xyz, sources, valid, bounds, policy["context"]
            )
            for index, group in tables["event_candidates"].iterrows():
                member_ids = [int(value) for value in str(group.edge_ids).split(",")]
                members = tables["edges"].set_index("edge_id").loc[member_ids]
                context_seconds = policy["context"]["review_context_seconds"]
                start = max(0.0, members.before_time_s.min() - context_seconds)
                stop = min(times[-1], members.after_time_s.max() + context_seconds)
                tables["event_candidates"].loc[index, "review_start_frame"] = int(
                    np.searchsorted(times, start)
                )
                tables["event_candidates"].loc[index, "review_stop_frame_exclusive"] = int(
                    np.searchsorted(times, stop, side="right")
                )
                tables["event_candidates"].loc[index, "member_outcomes"] = json.dumps(
                    members.outcome.value_counts().to_dict(), sort_keys=True
                )
            with np.load(c33_path) as before, np.load(c30_path) as after:
                position = before["xyz"].astype(np.float64)
                resolved = (before["n_views_used"] >= 2) & np.isfinite(position).any(axis=(1, 2))
                replay = trace_stitch(
                    c30,
                    position,
                    np.flatnonzero(resolved),
                    degraded,
                    times,
                    window=saved_c30["window"],
                    margin_mm=saved_c30["margin_mm"],
                )
                require_identical_replay(replay, after["slot_perm"], saved_c30["arms"][tag])
            decisions = replay[3]
            tables["c30_decisions"] = pd.DataFrame(decisions)
            tables["c30_associations"] = associate_decisions(
                tables["edges"], decisions, times, policy["context"]["c30_nearby_seconds"]
            )
            summary = tables.pop("summary")
            summary["track_label_status_counts"] = (
                tables["tracks"].label_status.value_counts().to_dict()
            )
            summary["segment_label_status_counts"] = (
                tables["segments"].label_status.value_counts().to_dict()
            )
            summary["endpoint_outcomes"] = tables["edges"].endpoint_outcome.value_counts().to_dict()
            tables["observed_assignment_changes"] = observed_changes[
                observed_changes.lambda_mm.eq(lam)
            ].copy()
            tables["wrong_identity_intervals"] = disagreements[
                disagreements.lambda_mm.eq(lam)
            ].copy()
            summary["observed_assignment_change_count"] = len(tables["observed_assignment_changes"])
            summary["observed_assignment_change_scope"] = (
                "Includes changes inside mixed local tracks; "
                "gap endpoint changes are not swap counts"
            )
            summary["c30_replay_matches_saved_permutation_and_decisions"] = True
            summary["lambda_mm"] = lam
            summary["source_run"] = run.name
            summary["complete_expected_fish_seconds"] = json.loads(
                (score / "reference_inventory.json").read_text()
            )["expected_fish_seconds"]
            for name, table in tables.items():
                path = dest / f"{name}.csv"
                table.to_csv(path, index=False)
                rec.output(f"lam{tag}_{name}", path, sha256=sha256(path))
            write_json(dest / "summary.json", summary)
            summaries[tag] = summary
            print(f"{args.variant} lambda {tag}: {summary['edge_outcomes']}", flush=True)
        write_json(out / "summary.json", summaries)
        rec.output("summary", out / "summary.json", sha256=sha256(out / "summary.json"))
    print(out, flush=True)


if __name__ == "__main__":
    main()
