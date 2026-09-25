"""Compare frozen five-minute streams on the same 57 female mouth anchors.

This reads saved tracking and one saved beam result. It never calls a tracker,
solver, label matcher, or training code. Geometry is compared on paired human
frame/view observations; numeric fish slots are not treated as biological truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.eval.generative import MouthAnchor, score_female_identity  # noqa: E402

WORK = REPO / "work/three_body"
START, STOP, FPS = 66400, 116200, 166.0
STAGE = "resumed_nodewise_beam_female57_comparison_v1"
ANCHORS = WORK / "female_57_five_minute_endpoint_evaluation_v1"
COMPARISON = WORK / "resumed_nodewise_five_minute_comparison_v1"
PRIOR = WORK / "resumed_nodewise_female_57_comparison_v1"
RAW = WORK / "nodewise_body_row_five_minute_resume_099600_v1/candidate.npz"
SPARSE = WORK / "sparse_instance_five_minute_continuation_v1/candidate.npz"
DEFAULT_BEAM = WORK / "resumed_nodewise_five_minute_identity_beam256_v1"
FIELDS = ("frame_idx", "pose_xyz", "observed", "sources", "identity_review")
MODES = ("raw_slots", "review_gated")
FIXED_HASHES = {
    "fixed_female_anchor_set.json": "41993610151fc26a106a846e1ca5d9a535df6803d6a3c5ff4ba13503edaaf2c8",
    "anchor_observations.json": "f253e645359d29df85b6c008390967af2ca78c3424041841fc231994b66a27ac",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sealed_input(seal, path):
    matches = [digest for name, digest in seal["inputs"].items()
               if Path(name).resolve() == path.resolve()]
    require(len(matches) == 1, f"exactly one sealed input required: {path}")
    require(sha256(path) == matches[0], f"sealed input changed: {path}")
    return matches[0]


def sealed_output(seal, root, name):
    descriptor = seal["outputs"][name]
    path = root / name
    require(Path(descriptor["path"]).resolve() == path.resolve(), f"output location differs: {name}")
    require(sha256(path) == descriptor["sha256"], f"sealed output changed: {name}")
    return path


def load_stream(path):
    with np.load(path, allow_pickle=False) as payload:
        frames = payload["frame_idx"]
        keep = (frames >= START) & (frames < STOP)
        data = {key: payload[key][keep].copy() for key in FIELDS}
    require(np.array_equal(data["frame_idx"], np.arange(START, STOP)), "full native five-minute support required")
    n = STOP - START
    for key, shape in (("pose_xyz", (n, 3, 3, 3)), ("sources", (n, 3, 3, 3)),
                       ("observed", (n, 3, 3)), ("identity_review", (n, 3))):
        require(data[key].shape == shape, f"unexpected {key} shape")
    require(data["observed"].dtype == bool and data["identity_review"].dtype == bool, "support/review must be boolean")
    require(not np.any(data["observed"] & ~np.isfinite(data["pose_xyz"]).all(axis=-1)), "observed points must be finite")
    return data


def score_stream(data, anchors, review=None):
    return score_female_identity(data["frame_idx"], data["sources"][:, :, :, 0],
                                 data["observed"][:, :, 0], anchors, review=review)


def coverage(data, review=None):
    observed = data["observed"]
    flagged = data["identity_review"] if review is None else review
    accepted = ~flagged
    n = len(observed)
    slots = []
    for slot in range(3):
        slots.append({
            "slot": slot,
            "total_frames": n,
            "accepted_identity_frames": int(accepted[:, slot].sum()),
            "accepted_identity_fraction": float(accepted[:, slot].mean()),
            "accepted_observed_mouth_frames": int((accepted[:, slot] & observed[:, slot, 0]).sum()),
            "accepted_observed_mouth_fraction": float((accepted[:, slot] & observed[:, slot, 0]).mean()),
            "observed_node_fraction": float(observed[:, slot].mean()),
            "all_nodes_observed_fraction": float(observed[:, slot].all(axis=-1).mean()),
            "observed_nodes": {name: {"frames": int(observed[:, slot, node].sum()),
                                      "fraction": float(observed[:, slot, node].mean())}
                               for node, name in enumerate(("mouth", "pectoral", "tail"))},
        })
    return {
        "native_frames": n,
        "total_fish_frames": n * 3,
        "total_node_frames": n * 9,
        "accepted_identity_frames": int(accepted.sum()),
        "accepted_identity_fraction": float(accepted.mean()),
        "accepted_observed_mouth_fraction": float((accepted & observed[:, :, 0]).mean()),
        "all_three_identities_accepted_fraction": float(accepted.all(axis=1).mean()),
        "observed_node_frames": int(observed.sum()),
        "observed_node_fraction": float(observed.mean()),
        "all_nodes_observed_fish_frame_fraction": float(observed.all(axis=-1).mean()),
        "observed_node_fraction_by_node": {name: float(observed[:, :, node].mean())
                                            for node, name in enumerate(("mouth", "pectoral", "tail"))},
        "slots": slots,
        "meets_identity_coverage_thresholds": bool(accepted.mean() >= .90 and accepted.mean(axis=0).min() >= .80),
        "scope": "Algorithm flag acceptance, not human-confirmed identity. Slots are method-local; male truth is unavailable.",
    }


def reprojection_rows(data, anchors, score, rig):
    result = {}
    for anchor, assigned in zip(anchors, score["anchors"], strict=True):
        require(anchor.frame == assigned["frame"], "scoring and human support differ")
        if assigned["identity"] is None:
            continue
        mouth = data["pose_xyz"][anchor.frame - START, assigned["identity"], 0]
        for view, pixel in anchor.pixels.items():
            prediction = rig.camera(view).project(mouth)
            if np.isfinite(prediction).all():
                result[(anchor.frame, view)] = float(np.linalg.norm(prediction - pixel))
    return result


def describe_errors(errors):
    values = np.asarray(list(errors), dtype=float)
    return {
        "view_observations": int(values.size),
        **{key: (float(function(values)) if values.size else None) for key, function in {
            "mean_px": np.mean, "median_px": np.median, "rmse_px": lambda x: np.sqrt(np.mean(x ** 2)),
            "p95_px": lambda x: np.percentile(x, 95), "max_px": np.max,
        }.items()},
    }


def csv_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output, beam_root):
    require(not output.exists(), f"fresh output directory required: {output}")
    require(output.resolve().is_relative_to(WORK), "evaluation output must remain inside workspace")
    input_hashes = {}
    comparison_path = COMPARISON / "comparison.provenance.json"
    comparison = read_json(comparison_path)
    require(comparison["status"] == "ok" and comparison["stage"] == COMPARISON.name, "successful frozen comparison required")
    input_hashes[str(comparison_path)] = sha256(comparison_path)
    for path in (RAW, SPARSE):
        input_hashes[str(path)] = sealed_input(comparison, path)

    # The labels and original detector-row matches remain exactly those evaluated before.
    anchor_seal_path = ANCHORS / "evaluation.provenance.json"
    anchor_seal = read_json(anchor_seal_path)
    require(anchor_seal["status"] == "ok", "successful fixed anchor evaluation required")
    input_hashes[str(anchor_seal_path)] = sha256(anchor_seal_path)
    for name, digest in FIXED_HASHES.items():
        path = sealed_output(anchor_seal, ANCHORS, name)
        require(sha256(path) == digest, f"fixed 57 anchor input changed: {name}")
        input_hashes[str(path)] = digest
    fixed = read_json(ANCHORS / "fixed_female_anchor_set.json")
    matches = read_json(ANCHORS / "anchor_observations.json")["matches"]
    require(len(matches) == 57 and [row["frame"] for row in matches] == fixed["frames"], "same ordered 57 anchors required")
    require(all(START <= row["frame"] < STOP for row in matches), "anchor outside scientific interval")
    anchors = [MouthAnchor(int(row["frame"]), row["detection_rows"], row["pixels"], row["match_errors_px"]) for row in matches]

    # Load the calibration bound to the frozen streams and verify its four files.
    indexes = []
    for root in (RAW.parent, SPARSE.parent):
        path = root / "input_index.json"
        input_hashes[str(path)] = sealed_input(comparison, path)
        indexes.append(read_json(path)["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"])
    require(indexes[0] == indexes[1], "methods must share the exact calibration and detector inputs")
    manifests = [Path(path) for path in indexes[0] if Path(path).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one calibration manifest required")
    manifest = manifests[0]
    calibration_paths = [manifest] + [manifest.parent / item["path"] for item in read_json(manifest)["model_files"]]
    require(len(calibration_paths) == 4, "three camera calibration files required")
    for path in calibration_paths:
        digest = indexes[0][str(path)]
        require(sha256(path) == digest, f"calibration changed: {path}")
        input_hashes[str(path)] = digest
    rig = load_rig(manifest.parent)

    beam_seal_path = beam_root / "identity.provenance.json"
    beam_seal = read_json(beam_seal_path)
    require(beam_seal["status"] == "ok" and beam_seal["stage"] == beam_root.name, "successful resumed beam output required")
    require(sealed_input(beam_seal, RAW) == input_hashes[str(RAW)], "beam consumed different raw candidate")
    input_hashes[str(beam_seal_path)] = sha256(beam_seal_path)
    beam_path = sealed_output(beam_seal, beam_root, "beam_remapped_stream.npz")
    mapping_path = sealed_output(beam_seal, beam_root, "identity_mapping_and_ambiguity.npz")
    for path in (beam_path, mapping_path):
        input_hashes[str(path)] = sha256(path)
    with np.load(mapping_path, allow_pickle=False) as payload:
        permutation = payload["permutation_by_id"]
        require(np.array_equal(payload["frame_idx"], np.arange(START, STOP)), "mapping support differs")
    require(permutation.shape == (STOP - START, 3), "mapping shape differs")
    require(np.array_equal(np.sort(permutation, axis=1), np.tile(np.arange(3), (STOP - START, 1))), "not a permutation")
    require(np.array_equal(permutation[0], np.arange(3)), "initial numeric mapping is not P012")

    streams = {"sparse_frozen": load_stream(SPARSE), "nodewise_raw": load_stream(RAW), "nodewise_beam": load_stream(beam_path)}
    raw, remapped = streams["nodewise_raw"], streams["nodewise_beam"]
    for key in FIELDS[1:]:
        expected = raw[key][np.arange(STOP - START)[:, None], permutation]
        require(np.array_equal(expected, remapped[key], equal_nan=True), f"beam changed {key} beyond slot permutation")

    results, errors = {}, {}
    for name, data in streams.items():
        scores = {"raw_slots": score_stream(data, anchors), "review_gated": score_stream(data, anchors, data["identity_review"])}
        errors[name] = {mode: reprojection_rows(data, anchors, score, rig) for mode, score in scores.items()}
        results[name] = {
            **scores, "coverage": coverage(data),
            "human_mouth_reprojection_own_support": {mode: describe_errors(values.values()) for mode, values in errors[name].items()},
        }

    # Regression evidence against the already-completed raw comparison.
    previous_path = PRIOR / "summary.json"
    previous = read_json(previous_path)
    input_hashes[str(previous_path)] = sha256(previous_path)
    for current_name, previous_name in (("sparse_frozen", "sparse_old"), ("nodewise_raw", "nodewise_new")):
        for mode in MODES:
            for key in ("anchors", "intervals", "interval_counts", "available_anchors"):
                require(results[current_name][mode][key] == previous["methods"][previous_name][mode][key], "fixed baseline endpoint result changed")

    # Unreviewed raw/beam disagreements remain unaccepted in this additional view.
    disagreement = permutation != np.arange(3)
    conservative_review = remapped["identity_review"] | disagreement
    conservative = {
        "review_gated": score_stream(remapped, anchors, conservative_review),
        "coverage": coverage(remapped, conservative_review),
        "disagreed_fish_frames": int(disagreement.sum()),
        "disagreed_frames": int(disagreement.any(axis=1).sum()),
        "rule": "original permuted review OR beam-ID raw-slot differs from frozen initial P012; no review flag cleared",
    }

    paired, reprojection_table = {}, []
    for mode in MODES:
        common = sorted(set.intersection(*(set(errors[name][mode]) for name in streams)))
        paired[mode] = {
            "common_frame_view_support": [[frame, view] for frame, view in common],
            "common_anchor_frames": len({frame for frame, _ in common}),
            "common_view_observations": len(common),
            "requested_view_observations": sum(len(anchor.pixels) for anchor in anchors),
            "methods": {name: describe_errors(errors[name][mode][key] for key in common) for name in streams},
        }
        for frame, view in common:
            reprojection_table.append({"mode": mode, "frame": frame, "view": view,
                                       **{name + "_error_px": errors[name][mode][(frame, view)] for name in streams}})

    for mode in MODES:
        require(errors["nodewise_raw"][mode].keys() == errors["nodewise_beam"][mode].keys(), "remapping changed female geometric support")
        require(all(errors["nodewise_raw"][mode][key] == errors["nodewise_beam"][mode][key]
                    for key in errors["nodewise_raw"][mode]), "remapping changed female geometry")

    anchor_rows, interval_rows = [], []
    for index, anchor in enumerate(anchors):
        row = {"frame": anchor.frame, "recording_seconds": anchor.frame / FPS, "video_seconds": (anchor.frame - START) / FPS}
        for name in streams:
            for mode in MODES:
                item = results[name][mode]["anchors"][index]
                row.update({f"{name}_{mode}_identity": item["identity"], f"{name}_{mode}_reason": item["reason"]})
        anchor_rows.append(row)
    for index in range(len(anchors) - 1):
        row = {"start": anchors[index].frame, "stop": anchors[index + 1].frame}
        for name in streams:
            for mode in MODES:
                row[f"{name}_{mode}"] = results[name][mode]["intervals"][index]["outcome"]
        interval_rows.append(row)

    summary = {
        "stage": STAGE, "status": "ok", "created_at": datetime.now(timezone.utc).isoformat(),
        "interval": {"start_frame": START, "stop_frame_exclusive": STOP, "native_frames": STOP - START, "fps": FPS},
        "ground_truth": {"fixed_confirmed_female_anchors": 57, "fixed_adjacent_intervals": 56,
                         "labels_created_or_modified": False, "used_for_training_or_tuning": False,
                         "annotation_scope": "same frozen female mouth detector-row matches in two side views; no male labels"},
        "methods": results,
        "beam_original_review_plus_unreviewed_disagreement": conservative,
        "human_mouth_reprojection_paired_support": paired,
        "verification": {"previous_raw_and_review_female_scores_reproduced": True,
                         "beam_is_exact_slot_permutation_of_all_five_evaluation_fields": True,
                         "nodewise_female_geometry_and_support_unchanged_by_beam": True,
                         "same_registered_calibration_and_sleap_inputs": True},
        "inputs": input_hashes,
        "interpretation_limits": [
            "Preserved female endpoints do not exclude swap-and-recovery inside an interval.",
            "No male identity truth is supplied. Algorithm acceptance and smooth geometry cannot establish male identity.",
            "Review-gated unscored anchors remain in the requested denominator; unavailable endpoints are never bridged.",
            "Per-slot observation coverage is method-local; aggregate fish/node denominators share exactly 49,800 native frames.",
            "Human mouth errors measure geometry at source-matched female anchors, not independent identity correctness.",
            "The extra disagreement gate is conservative report-only abstention; saved streams and original flags are unchanged.",
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "summary.json", summary)
    csv_rows(output / "anchor_table.csv", anchor_rows)
    csv_rows(output / "interval_table.csv", interval_rows)
    csv_rows(output / "human_mouth_reprojection.csv", reprojection_table)
    lines = ["# Fixed57 female comparison after full five-minute ID beam", "",
             "Frames `66400:116200`, 49,800 native frames (06:40–11:40). The original 57 confirmed female labels and detector-row matches were reused unchanged.", "",
             "| Stream / scoring | Available anchors | Preserved | Swap intervals | Unscored |", "|---|---:|---:|---:|---:|"]
    for name in streams:
        for mode in MODES:
            score = results[name][mode]
            counts = score["interval_counts"]
            lines.append(f"| {name} / {mode} | {score['available_anchors']}/57 | {counts['preserved']} | {counts['swap']} | {counts['unscored']} |")
    lines += ["", "| Stream | Accepted ID coverage | Per-slot accepted ID coverage (0 / 1 / 2) | Observed nodes |", "|---|---:|---|---:|"]
    for name in streams:
        cov = results[name]["coverage"]
        slots = " / ".join(f"{slot['accepted_identity_fraction']:.2%}" for slot in cov["slots"])
        lines.append(f"| {name} | {cov['accepted_identity_fraction']:.2%} | {slots} | {cov['observed_node_fraction']:.2%} |")
    lines += ["", "Accepted ID coverage means the original algorithm did not request review. It is not human-confirmed accuracy. Coverage promotion requires ≥90% overall and ≥80% for every fish.", "",
              f"The additional raw/beam-disagreement gate leaves {conservative['coverage']['accepted_identity_fraction']:.2%} ID coverage; {conservative['disagreed_frames']} native frames disagree in one or more slots.", "",
              "| Paired support / stream | Same frame-view observations | Median mouth error (px) | Mean (px) | P95 (px) | Max (px) |", "|---|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for name, metrics in paired[mode]["methods"].items():
            number = lambda key: "unscored" if metrics[key] is None else f"{metrics[key]:.3f}"
            lines.append(f"| {mode} / {name} | {metrics['view_observations']} | {number('median_px')} | {number('mean_px')} | {number('p95_px')} | {number('max_px')} |")
    lines += ["", "All paired errors use identical female frame/view observations and the same frozen refractive calibration. Beam remapping preserves every evaluated pose, observation, source and review flag exactly; its female reprojection values equal raw Nodewise.", "",
              "This endpoint evaluation does not establish male identity or exclude a swap followed by recovery between female anchors. Three-fish blinded review remains separate; ambiguous or unreviewed scenes cannot be counted as confirmed passes.", ""]
    (output / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    source_paths = [Path(__file__), REPO / "src/medaka3d/eval/generative.py", REPO / "src/medaka3d/calib.py", REPO / "src/medaka3d/types.py"]
    for path in source_paths:
        input_hashes[str(path)] = sha256(path)
    write_json(output / "evaluation.provenance.json", {
        "stage": STAGE, "status": "ok", "inputs": input_hashes,
        "outputs": {path.name: {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
                    for path in sorted(output.iterdir()) if path.is_file()},
    })
    print(json.dumps({name: {"raw": value["raw_slots"]["interval_counts"],
                             "review": value["review_gated"]["interval_counts"],
                             "coverage": value["coverage"]["accepted_identity_fraction"],
                             "observed": value["coverage"]["observed_node_fraction"]}
                      for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-root", type=Path, default=DEFAULT_BEAM)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.output.resolve(), args.beam_root.resolve())
