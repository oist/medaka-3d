"""Independently reconcile scoring outputs and compare rerun arrays to old runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_pipeline import settings, sha256, write_json


def audit_scoring(directory: Path, config: dict, reference_path: Path) -> dict:
    summary = pd.read_csv(directory / "identity_summary.csv")
    inventory = json.loads((directory / "reference_inventory.json").read_text())
    inputs = json.loads((directory / "scoring_inputs.json").read_text())
    reference = pd.read_csv(
        reference_path,
        usecols=[
            "frame",
            "label",
            "pectoral_X",
            "pectoral_Y",
            "pectoral_Z",
            "watched",
            "analysed",
            "uncertain",
            "bad3d",
            "nodes_bad",
            "is_interpolated",
        ],
    ).set_index(["frame", "label"])
    flags = reference[["watched", "analysed"]].all(axis=1)
    flags &= ~reference[["uncertain", "bad3d", "nodes_bad", "is_interpolated"]].any(axis=1)
    report = {
        "summary_reconciled": True,
        "arms": {},
        "independent_check": (
            "Scalar, identity-free nearest-position check on 1000 regularly spaced frames, "
            "restricted to exact shared coordinates with unique 2 mm margins. "
            "This validates scoring mechanics, not the correctness of human identities."
        ),
    }
    for lam in config["pipeline"]["lambdas"]:
        tag = f"{lam:g}"
        with np.load(directory / f"scoring_arrays_lam{tag}.npz") as z:
            times, weights, eligible = z["recording_time_s"], z["duration_s"], z["eligible"]
            status, names = z["status_code"], z["status_names"]
            slots, mapping, labels = (
                z["matched_slot"],
                z["anchor_reference_to_slot"],
                z["reference_labels"],
            )
        primary = summary[
            (summary.lambda_mm == lam)
            & (summary.match_distance_mm == config["evaluation"]["max_match_distance_mm"])
        ]
        for i, label in enumerate(labels):
            row = primary[primary.reference_label == label].iloc[0]
            total_frames, total_seconds = 0, 0.0
            for code, name in enumerate(names):
                mask = eligible & (status[:, i] == code)
                count, seconds = int(mask.sum()), float(weights[mask].sum())
                assert count == row[f"{name}_frames"]
                assert abs(seconds - row[f"{name}_seconds"]) < 1e-7
                total_frames += count
                total_seconds += seconds
            assert total_frames == row.expected_frames
            assert (
                abs(total_seconds + row.unprocessed_leading_seconds - row.expected_seconds) < 1e-6
            )
        source = Path(inputs[tag]["path"])
        assert sha256(source) == inputs[tag]["sha256"]
        with np.load(source) as z:
            xyz = z["xyz"]
            track = z["slot_track"]
        checked, discrepancies = 0, []
        sampled = np.linspace(inventory["first_scored_frame"], len(times) - 2, 1000, dtype=int)
        for frame in sampled:
            rows = reference.loc[frame].reindex(labels)
            ref_xyz = rows[["pectoral_X", "pectoral_Y", "pectoral_Z"]].to_numpy()
            distances = np.linalg.norm(ref_xyz[:, None, :] - xyz[frame][None, :, :], axis=2)
            distances[:, track[frame] < 0] = np.inf
            for i, label in enumerate(labels):
                if not flags.loc[(frame, label)] or not np.isfinite(ref_xyz[i]).all():
                    continue
                candidates = np.flatnonzero(distances[i] < 1e-4)
                if len(candidates) != 1:
                    continue
                slot = int(candidates[0])
                competitors = [distances[i, j] for j in range(3) if j != slot]
                competitors += [distances[j, slot] for j in range(3) if j != i]
                if not all((not np.isfinite(v)) or v > 2.0001 for v in competitors):
                    continue
                expected = 0 if slot == mapping[i] else 1
                if mapping[i] < 0:
                    continue
                checked += 1
                if status[frame, i] != expected or slots[frame, i] != slot:
                    discrepancies.append({"frame": int(frame), "reference_label": str(label)})
        if discrepancies:
            raise AssertionError(f"Independent scoring discrepancy: {discrepancies[:5]}")
        report["arms"][tag] = {
            "independent_exact_coordinate_checks": checked,
            "discrepancies": len(discrepancies),
        }
    return report


def array_difference(a, b):
    if a.shape != b.shape:
        raise AssertionError(f"Previous/rerun shape mismatch: {a.shape} versus {b.shape}")
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    finite = np.isfinite(a) & np.isfinite(b)
    difference = a[finite].astype(float) - b[finite].astype(float)
    return {
        "shape_equal": True,
        "different_elements": int((~same).sum()),
        "finite_pattern_changes": int((np.isfinite(a) != np.isfinite(b)).sum()),
        "max_absolute_difference": float(np.max(np.abs(difference), initial=0.0)),
    }


def compare_runs(config, paths, run_dir, score_dir, *, previous_score=None):
    exp = config["experiment"]
    relative = Path("c34_microflip") / f"{exp}__{config['pipeline']['chain_tag']}"
    comparison, aligned = {}, {}
    if previous_score is None:
        previous_score = run_dir / config["evaluation"]["previous_scoring_directory"]
    old_anchors = json.loads((previous_score / "anchor_mappings.json").read_text())
    new_anchors = json.loads((score_dir / "anchor_mappings.json").read_text())
    with np.load(run_dir / "native_timing.npz") as timing:
        eligible = timing["top_cam"] >= config["evaluation"]["preparation_seconds"]
    for lam in config["pipeline"]["lambdas"]:
        tag = f"{lam:g}"
        filename = f"C34-label__{exp}__lam{tag}.npz"
        with (
            np.load(paths.work_dir / relative / filename) as old,
            np.load(run_dir / "pipeline" / relative / filename) as new,
        ):
            fields, aligned_fields, post_fields = {}, {}, {}
            old_order = old_anchors[tag]["reference_to_slot"]
            new_order = new_anchors[tag]["reference_to_slot"]
            if min(old_order + new_order) < 0:
                raise ValueError("Cannot align runs with an unresolved identity anchor")
            for key in ["xyz", "source", "slot_track", "residual_mm", "n_views_used"]:
                a, b = old[key], new[key]
                fields[key] = array_difference(a, b)
                if key == "slot_track":
                    aligned_key = "label_available"
                    aligned_a, aligned_b = a[:, old_order] >= 0, b[:, new_order] >= 0
                elif key == "n_views_used":
                    aligned_key = key
                    aligned_a, aligned_b = a, b
                else:
                    aligned_key = key
                    aligned_a, aligned_b = a[:, old_order], b[:, new_order]
                aligned_fields[aligned_key] = array_difference(aligned_a, aligned_b)
                post_fields[aligned_key] = array_difference(
                    aligned_a[eligible], aligned_b[eligible]
                )
            comparison[tag] = fields
            with (
                np.load(previous_score / f"scoring_arrays_lam{tag}.npz") as old_score,
                np.load(score_dir / f"scoring_arrays_lam{tag}.npz") as new_score,
            ):
                assert np.array_equal(new_score["eligible"], eligible)
                weights = new_score["duration_s"]
                assert np.array_equal(old_score["eligible"], eligible)
                assert np.array_equal(old_score["duration_s"], weights)
                assert np.array_equal(old_score["status_names"], new_score["status_names"])
                changed = old_score["status_code"] != new_score["status_code"]
                identities = []
                for data, order in [(old_score, old_order), (new_score, new_order)]:
                    names = np.full(data["matched_slot"].shape, -1, np.int8)
                    for identity, slot in enumerate(order):
                        names[data["matched_slot"] == slot] = identity
                    identities.append(names)
                both = (identities[0] >= 0) & (identities[1] >= 0) & eligible[:, None]
                identity_changed = both & (identities[0] != identities[1])
                aligned[tag] = {
                    "previous_reference_to_slot": old_order,
                    "rerun_reference_to_slot": new_order,
                    "alignment": "The same fixed pre-window anchors used for scoring; no refitting",
                    "fields": aligned_fields,
                    "post_preparation_fields": post_fields,
                    "changed_score_status_observations": int(changed[eligible].sum()),
                    "changed_score_status_fish_seconds": float((changed * weights[:, None]).sum()),
                    "changed_identity_observations_both_matched": int(identity_changed.sum()),
                    "changed_identity_fish_seconds_both_matched": float(
                        (identity_changed * weights[:, None]).sum()
                    ),
                }
    return comparison, aligned


def compare_c27(config, paths, run_dir):
    relative = (
        Path("c27_uncapped_tracklets_any") / config["experiment"] / "mouth-pectoral_oks_serial"
    )
    comparison = {}
    for view in ["top_cam", "side1_cam", "side2_cam"]:
        old_path = paths.work_dir / relative / f"track2d__{view}.npy"
        new_path = run_dir / "pipeline" / relative / f"track2d__{view}.npy"
        old, new = np.load(old_path), np.load(new_path)
        assert old.shape == new.shape
        comparison[view] = {
            "shape": list(new.shape),
            "different_labels": int((old != new).sum()),
            "unique_labels": len(np.unique(new)),
            "previous_sha256": sha256(old_path),
            "rerun_sha256": sha256(new_path),
        }
    write_json(run_dir / "c27_previous_comparison.json", comparison)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["previous", "rerun"], required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    config, paths, _, run_dir = settings()
    directory = run_dir / f"scoring_{args.source}_{args.version}"
    reference = Path(json.loads((run_dir / "inputs.json").read_text())["review"]["backup"])
    report = audit_scoring(directory, config, reference)
    report["audit_source_sha256"] = sha256(Path(__file__))
    if args.source == "rerun":
        raw, aligned = compare_runs(config, paths, run_dir, directory)
        report["previous_run_array_comparison"] = raw
        report["anchor_aligned_comparison"] = aligned
        report["c27_previous_comparison"] = compare_c27(config, paths, run_dir)
    write_json(directory / "validation.json", report)
    print(json.dumps(report, indent=2))
