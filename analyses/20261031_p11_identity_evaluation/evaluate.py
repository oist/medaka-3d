"""Score frozen P-1-1 pipeline outputs against the user's reviewed identities."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from run_pipeline import REPO, configuration_path, settings, sha256, write_json

from medaka3d.eval.review_identity import (
    contiguous_runs,
    fixed_anchor_mapping,
    identity_status,
    interval_weights,
    match_positions,
    slot_change_events,
)
from medaka3d.provenance import provenance

LABELS = ["F", "M1", "M2"]
STATUS = [
    "correct_id",
    "wrong_id",
    "reference_unavailable",
    "prediction_missing",
    "distance_mismatch",
    "spatial_ambiguity",
    "anchor_unresolved",
]


def load_reference(inputs, n_frames, options):
    path = Path(inputs["review"]["backup"])
    if sha256(path) != inputs["review"]["sha256"]:
        raise RuntimeError("Frozen human reference checksum mismatch")
    cols = [
        "frame",
        "fish_id",
        "label",
        "source_track",
        "watched",
        "analysed",
        "uncertain",
        "bad3d",
        "nodes_bad",
        "is_interpolated",
        "pectoral_X",
        "pectoral_Y",
        "pectoral_Z",
    ]
    df = pd.read_csv(path, usecols=cols)
    if df.duplicated(["frame", "label"]).any() or df.duplicated(["frame", "fish_id"]).any():
        raise ValueError("Duplicate reference keys")
    if set(df.label.unique()) != set(LABELS):
        raise ValueError("Unexpected reference identities")
    if len(df) != n_frames * 3 or df.frame.min() != 0 or df.frame.max() != n_frames - 1:
        raise ValueError("Reference is not the expected complete frame grid")
    idx = pd.MultiIndex.from_product([range(n_frames), LABELS], names=["frame", "label"])
    df = df.set_index(["frame", "label"]).reindex(idx)
    if df.fish_id.isna().any():
        raise ValueError("Missing frame/identity rows")
    xyz = df[["pectoral_X", "pectoral_Y", "pectoral_Z"]].to_numpy().reshape(n_frames, 3, 3)
    flags = {
        c: df[c].to_numpy().reshape(n_frames, 3)
        for c in ["watched", "analysed", "uncertain", "bad3d", "nodes_bad", "is_interpolated"]
    }
    for name, values in flags.items():
        if values.dtype != bool:
            raise ValueError(f"Non-boolean or missing reference flag: {name}")
    valid = ~flags["uncertain"] & ~flags["bad3d"] & np.isfinite(xyz).all(axis=2)
    for option, flag, expected in [
        ("require_watched", "watched", True),
        ("require_analysed", "analysed", True),
        ("exclude_nodes_bad_reference", "nodes_bad", False),
        ("exclude_interpolated_reference", "is_interpolated", False),
    ]:
        if options[option]:
            valid &= flags[flag] == expected
    return xyz, valid, flags


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summarize(status, weights, eligible, total_duration, arm, distance):
    rows = []
    for i, label in enumerate(LABELS):
        row = {
            "lambda_mm": arm,
            "match_distance_mm": distance,
            "reference_label": label,
            "expected_frames": int(eligible.sum()),
            "expected_seconds": total_duration,
        }
        for key in STATUS:
            mask = eligible & (status[:, i] == key)
            row[f"{key}_frames"] = int(mask.sum())
            row[f"{key}_seconds"] = float(weights[mask].sum())
        correct, wrong = row["correct_id_seconds"], row["wrong_id_seconds"]
        row["id_agreement_on_scored_time"] = ratio(correct, correct + wrong)
        row["correct_id_time_fraction"] = ratio(correct, total_duration)
        row["scored_time_fraction"] = ratio(correct + wrong, total_duration)
        row["unprocessed_leading_seconds"] = float(total_duration - weights.sum())
        runs = contiguous_runs(eligible & (status[:, i] == "correct_id"))
        durations = [float(weights[lo:hi].sum()) for lo, hi in runs]
        row["correct_run_count"] = len(durations)
        row["correct_run_median_seconds"] = float(np.median(durations)) if durations else 0.0
        row["correct_run_max_seconds"] = max(durations, default=0.0)
        wrong_runs = contiguous_runs(eligible & (status[:, i] == "wrong_id"))
        row["wrong_run_count"] = len(wrong_runs)
        row["wrong_run_max_seconds"] = max(
            (float(weights[a:b].sum()) for a, b in wrong_runs), default=0.0
        )
        rows.append(row)
    return rows


def gated_diagnostics(path, matches, eligible, weights):
    """Retrospective local purity is diagnostic, not supported trial identity."""
    with np.load(path, allow_pickle=False) as data:
        tracks = data["slot_track"]
    safe = np.maximum(matches.slots, 0)
    track_at_ref = np.take_along_axis(tracks, safe, axis=1)
    track_at_ref[matches.slots < 0] = -1
    ids = np.unique(track_at_ref[(track_at_ref >= 0) & eligible[:, None]])
    rows = []
    count_table = np.zeros((int(ids.max()) + 1 if len(ids) else 0, 3), np.int64)
    time_table = np.zeros(count_table.shape, float)
    for i in range(3):
        mask = eligible & (track_at_ref[:, i] >= 0)
        count_table[:, i] = np.bincount(track_at_ref[mask, i], minlength=len(count_table))
        time_table[:, i] = np.bincount(
            track_at_ref[mask, i], weights=weights[mask], minlength=len(count_table)
        )
    for track in ids:
        counts = count_table[track]
        seconds = time_table[track]
        majority = int(np.argmax(seconds))
        rows.append(
            {
                "local_track_id": int(track),
                "majority_reference_label": LABELS[majority],
                "matched_frames": int(counts.sum()),
                "matched_seconds": float(seconds.sum()),
                "majority_seconds": float(seconds[majority]),
                "local_purity": ratio(seconds[majority], seconds.sum()),
                **{f"{label}_seconds": float(seconds[i]) for i, label in enumerate(LABELS)},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["previous", "rerun"], required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    cfg, paths, _root, run_dir = settings()
    ev = cfg["evaluation"]
    inputs = json.loads((run_dir / "inputs.json").read_text())
    exp = cfg["experiment"]
    pipeline_root = paths.work_dir if args.source == "previous" else run_dir / "pipeline"
    if args.source == "rerun":
        state = json.loads((run_dir / "run_status.json").read_text())
        if state["status"] != "complete":
            raise RuntimeError("Full rerun has not completed")
    output = run_dir / f"scoring_{args.source}_{args.version}"
    output.mkdir(exist_ok=False)
    chain = pipeline_root / "c34_microflip" / f"{exp}__{cfg['pipeline']['chain_tag']}"
    with np.load(run_dir / "native_timing.npz", allow_pickle=False) as native:
        times = native["top_cam"]
    ref, ref_valid, flags = load_reference(inputs, len(times), ev)
    eligible = times >= ev["preparation_seconds"]
    weights, duration = interval_weights(times, ev["preparation_seconds"])
    anchor = (times >= ev["anchor_start_seconds"]) & (
        times < ev["anchor_start_seconds"] + ev["anchor_duration_seconds"]
    )
    if np.any(anchor & eligible):
        raise ValueError("Anchor window must precede the scored analysis window")
    inventory = {
        "first_scored_frame": int(np.flatnonzero(eligible)[0]),
        "last_scored_frame": int(np.flatnonzero(eligible)[-1]),
        "first_scored_time_s": float(times[eligible][0]),
        "recording_end_s": float(times[-1]),
        "expected_fish_seconds": 3 * duration,
        "expected_fish_frames": int(eligible.sum()) * 3,
        "reference_eligible_fish_frames": int(ref_valid[eligible].sum()),
        "reference_eligible_fish_seconds": float((ref_valid * weights[:, None]).sum()),
        "unwatched_frames": np.flatnonzero(~flags["watched"].all(axis=1)).tolist(),
        "reference_flag_counts_post_preparation": {
            k: int(v[eligible].sum()) for k, v in flags.items()
        },
        "time_definition": (
            "Native top-camera acquisition seconds, zero at first acquired frame. "
            "Frame index i maps to acquisition counter i+1. No resampling."
        ),
        "duration_definition": (
            "Left-held intervals to the next native timestamp; last frame has zero duration. "
            "The partial interval before the first eligible acquisition remains unprocessed."
        ),
        "reference_limit": (
            "Human-reviewed IDs on derived geometry; watched means displayed, "
            "not a framewise confirmation. Geometry is shared, not independent 3D truth."
        ),
    }
    write_json(output / "reference_inventory.json", inventory)
    all_summary, all_events, wrong_intervals, context_rows, continuity_rows, gated_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    anchors, arm_results, input_hashes = {}, {}, {}
    with provenance("p11_review_identity_scoring", output) as rec:
        rec.input("review", Path(inputs["review"]["backup"]), sha256=inputs["review"]["sha256"])
        rec.param("configuration", cfg)
        rec.param("source", args.source)
        rec.param(
            "scope",
            "Agreement with supplied reviewed identity labels; "
            "no independent geometry accuracy or population-level certification",
        )
        for lam in cfg["pipeline"]["lambdas"]:
            tag = f"{lam:g}"
            path = chain / f"C34-label__{exp}__lam{tag}.npz"
            digest = sha256(path)
            input_hashes[tag] = {"path": str(path), "sha256": digest}
            rec.input(f"label_lam{tag}", path, sha256=digest)
            with np.load(path, allow_pickle=False) as z:
                pred = z["xyz"].copy()
                position_valid = np.isfinite(pred).all(axis=2) & (
                    (z["source"] >= 0).sum(axis=2) >= 2
                )
                output_labels = z["slot_track"].copy()
                slots = np.broadcast_to(np.arange(3), output_labels.shape)
                if np.any((output_labels >= 0) & (output_labels != slots)):
                    raise ValueError("C34-label IDs no longer equal the stored slot columns")
                pred_valid = position_valid & (output_labels >= 0)
                residual = z["residual_mm"].copy()
            if pred.shape != ref.shape:
                raise ValueError("Prediction and reference frame grids differ")
            primary = match_positions(
                ref,
                pred,
                ref_valid,
                pred_valid,
                max_distance_mm=ev["max_match_distance_mm"],
                min_margin_mm=ev["min_match_margin_mm"],
            )
            mapping = fixed_anchor_mapping(
                primary.slots,
                anchor,
                minimum_matches=ev["anchor_minimum_matches"],
                minimum_agreement=ev["anchor_minimum_agreement"],
            )
            anchors[tag] = mapping
            primary_status = identity_status(primary, mapping["reference_to_slot"])
            for distance in ev["sensitivity_distance_mm"]:
                matches = (
                    primary
                    if distance == ev["max_match_distance_mm"]
                    else match_positions(
                        ref,
                        pred,
                        ref_valid,
                        pred_valid,
                        max_distance_mm=distance,
                        min_margin_mm=ev["min_match_margin_mm"],
                    )
                )
                status = (
                    primary_status
                    if matches is primary
                    else identity_status(matches, mapping["reference_to_slot"])
                )
                all_summary.extend(summarize(status, weights, eligible, duration, lam, distance))
            codes = np.full(primary_status.shape, -1, np.int8)
            for i, name in enumerate(STATUS):
                codes[primary_status == name] = i
            np.savez_compressed(
                output / f"scoring_arrays_lam{tag}.npz",
                frame=np.arange(len(times)),
                recording_time_s=times,
                duration_s=weights,
                eligible=eligible,
                reference_labels=np.array(LABELS),
                status_names=np.array(STATUS),
                status_code=codes,
                matched_slot=primary.slots,
                distance_mm=primary.distance_mm.astype(np.float32),
                margin_mm=primary.margin_mm.astype(np.float32),
                anchor_reference_to_slot=np.array(mapping["reference_to_slot"]),
            )
            events = slot_change_events(primary.slots, eligible)
            for event in events:
                all_events.append(
                    {
                        "lambda_mm": lam,
                        **event,
                        "before_time_s": float(times[event["before_frame"]]),
                        "after_time_s": float(times[event["after_frame"]]),
                        "changes": json.dumps(event["changes"]),
                    }
                )
            for i, label in enumerate(LABELS):
                for first, stop in contiguous_runs(eligible & (primary_status[:, i] == "wrong_id")):
                    wrong_intervals.append(
                        {
                            "lambda_mm": lam,
                            "reference_label": label,
                            "start_frame": int(first),
                            "stop_frame_exclusive": int(stop),
                            "start_s": float(times[first]),
                            "stop_s": float(times[min(stop, len(times) - 1)]),
                            "duration_s": float(weights[first:stop].sum()),
                        }
                    )
            pair_distance = np.linalg.norm(ref[:, :, None, :] - ref[:, None, :, :], axis=3)
            for i in range(3):
                pair_distance[:, i, i] = np.inf
            close = np.min(pair_distance, axis=(1, 2)) <= ev["encounter_distance_mm"]
            context_known = ref_valid.all(axis=1)
            for context, mask in [
                ("close_20mm", close & context_known),
                ("separated", ~close & context_known),
                ("reference_incomplete", ~context_known),
            ]:
                mask &= eligible
                scorable = (
                    (primary_status == "correct_id") | (primary_status == "wrong_id")
                ) & mask[:, None]
                wrong = (primary_status == "wrong_id") & mask[:, None]
                total = float(weights[mask].sum() * 3)
                scored = float((scorable * weights[:, None]).sum())
                wrong_time = float((wrong * weights[:, None]).sum())
                context_rows.append(
                    {
                        "lambda_mm": lam,
                        "context": context,
                        "fish_seconds": total,
                        "scored_fish_seconds": scored,
                        "wrong_fish_seconds": wrong_time,
                        "wrong_fraction_of_scored": ratio(wrong_time, scored),
                        "unscored_fraction": ratio(total - scored, total),
                    }
                )
            joint = eligible & (primary_status == "correct_id").all(axis=1)
            arm_results[tag] = {
                "joint_correct_seconds": float(weights[joint].sum()),
                "joint_correct_fraction": ratio(weights[joint].sum(), duration),
                "adjacent_change_events": sum(e["kind"] == "adjacent_change" for e in events),
                "gap_endpoint_change_events": sum(
                    e["kind"] == "gap_endpoint_change" for e in events
                ),
                "prediction_position_fish_seconds": float(
                    (position_valid * weights[:, None]).sum()
                ),
                "labeled_position_fish_seconds": float((pred_valid * weights[:, None]).sum()),
            }
            gated_path = chain / f"C34-gated__{exp}__lam{tag}.npz"
            rec.input(f"gated_lam{tag}", gated_path, sha256=sha256(gated_path))
            gated_rows.extend(
                {"lambda_mm": lam, **r}
                for r in gated_diagnostics(gated_path, primary, eligible, weights)
            )
            bins = np.floor(times).astype(int)
            bin_duration = np.bincount(bins, weights=weights)
            bin_scores = {
                s: [
                    np.bincount(bins, weights=weights * (primary_status[:, i] == s))
                    for i in range(3)
                ]
                for s in ["correct_id", "wrong_id"]
            }
            for second in np.unique(bins[eligible]):
                for i, label in enumerate(LABELS):
                    entry = {
                        "lambda_mm": lam,
                        "recording_second": int(second),
                        "reference_label": label,
                    }
                    for s in ["correct_id", "wrong_id"]:
                        entry[s + "_seconds"] = float(bin_scores[s][i][second])
                    entry["observed_bin_seconds"] = float(bin_duration[second])
                    continuity_rows.append(entry)
            inverse = {
                slot: LABELS[i] for i, slot in enumerate(mapping["reference_to_slot"]) if slot >= 0
            }
            with np.load(gated_path, allow_pickle=False) as gated:
                local_tracks = gated["slot_track"][eligible].ravel()
            export = pd.DataFrame(
                {
                    "frame": np.repeat(np.flatnonzero(eligible), 3),
                    "recording_time_s": np.repeat(times[eligible], 3),
                    "raw_slot": np.tile(np.arange(3), eligible.sum()),
                    "predicted_anchor_label": np.tile(
                        [inverse.get(i, "") for i in range(3)], eligible.sum()
                    ),
                    "position_valid": position_valid[eligible].ravel(),
                    "label_available": pred_valid[eligible].ravel(),
                    "output_identity_label": output_labels[eligible].ravel(),
                    "local_track_id": local_tracks,
                    "identity_support": "automatic_unvalidated",
                    "X": pred[eligible, :, 0].ravel(),
                    "Y": pred[eligible, :, 1].ravel(),
                    "Z": pred[eligible, :, 2].ravel(),
                    "residual_mm": residual[eligible].ravel(),
                }
            )
            if args.source == "rerun":
                export.to_csv(
                    output / f"automatic_pectoral_3d_lam{tag}.csv.gz",
                    index=False,
                    float_format="%.8g",
                    compression={
                        "method": "gzip",
                        "compresslevel": cfg["output"]["csv_gzip_compresslevel"],
                        "mtime": 0,
                    },
                )
            print(
                f"Scored lambda {tag}: anchor {mapping['reference_to_slot']}; "
                f"joint correct {arm_results[tag]['joint_correct_fraction']:.3%}",
                flush=True,
            )
        pd.DataFrame(all_summary).to_csv(output / "identity_summary.csv", index=False)
        pd.DataFrame(all_events).to_csv(output / "assignment_change_events.csv", index=False)
        pd.DataFrame(wrong_intervals).to_csv(output / "wrong_identity_intervals.csv", index=False)
        pd.DataFrame(context_rows).to_csv(output / "proximity_context.csv", index=False)
        pd.DataFrame(continuity_rows).to_csv(output / "per_second_agreement.csv", index=False)
        pd.DataFrame(gated_rows).to_csv(output / "gated_local_track_purity.csv", index=False)
        write_json(output / "anchor_mappings.json", anchors)
        write_json(output / "arm_results.json", arm_results)
        write_json(output / "scoring_inputs.json", input_hashes)
        write_json(output / "scoring_configuration.json", cfg)
        for source in [
            Path(__file__),
            REPO / "src/medaka3d/eval/review_identity.py",
            configuration_path(),
        ]:
            shutil.copy2(source, output / source.name)
            rec.input(source.name, source, sha256=sha256(source))
        rec.output(
            "summary",
            output / "identity_summary.csv",
            sha256=sha256(output / "identity_summary.csv"),
        )
    print(output, flush=True)


if __name__ == "__main__":
    main()
