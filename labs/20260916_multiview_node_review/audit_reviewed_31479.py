"""Reproduce the reviewed 31479 identity excursion without changing source tracks."""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
from geometry import geometry_metrics

from medaka3d.aquarium import sha256
from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

FIRST, STOP = 31479, 31497
SWAP = [1, 0, 2]


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def location(path, needle, after=None):
    """Retain a short source excerpt and current one-based line citation."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if after in line) if after else 0
    index = next(i for i in range(start, len(lines)) if needle in lines[i])
    return {"path": str(path), "line": index + 1, "excerpt": lines[index].strip()}


def verified_input(record, name, entry):
    path = Path(entry["path"])
    digest = sha256(path)
    if digest != entry["sha256"]:
        raise ValueError(f"Recorded source hash changed: {name}")
    record.input(name, path, sha256=digest)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bidirectional", type=Path, default=Path("work/bidirectional_reconnect_v1/analysis_v2")
    )
    parser.add_argument(
        "--evaluation", type=Path, default=Path("work/multiview_node_review_v1/analysis")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("work/multiview_node_review_v1/human_review_followup"),
    )
    args = parser.parse_args()
    stem = "f31479_diagnosis"
    output = args.output / f"{stem}.json"
    if output.exists() or (args.output / f"{stem}.provenance.json").exists():
        raise FileExistsError("Preserve the existing named diagnosis")
    prior_path = args.bidirectional / "bidirectional_development.provenance.json"
    review_path = args.evaluation / "multiview_node_reassessment.provenance.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))

    with provenance("reviewed_identity_excursion_diagnosis", args.output, name=stem) as record:
        record.params_from(args)
        record.param("hypothetical_interval_half_open", [FIRST, STOP])
        record.param("permutation", SWAP)
        record.param("accepted_identity_corrections", 0)
        for name, path in (
            ("audit_code", Path(__file__)),
            ("bidirectional_provenance", prior_path),
            ("multiview_provenance", review_path),
            ("c30_code", Path("tools/c30_slot_stitch.py")),
            ("c33_code", Path("tools/c33_ray_rejection.py")),
            ("c34_code", Path("tools/c34_microflip.py")),
            ("historical_decision", Path("docs/decisions/2026-09-01-slot-stitching.md")),
        ):
            record.input(name, path, sha256=sha256(path))
        c34_path = verified_input(record, "C34", prior["inputs"]["C34"])
        analysis_path = verified_input(
            record, "analysis", prior["outputs"]["known_31479/analysis.json"]
        )
        evaluation_path = verified_input(
            record, "saved_uv", review["outputs"]["known_31479/evaluation.npz"]
        )
        for key in ("geometry.py", "calibration", "cam0", "cam1", "cam2"):
            verified_input(record, key, review["inputs"][key])
        rig = load_rig(Path(review["inputs"]["calibration"]["path"]).parent)
        with np.load(c34_path, allow_pickle=False) as data:
            xyz = data["xyz"].astype(np.float64)
            source = data["source"]
            residual = data["residual_mm"]
        with np.load(evaluation_path, allow_pickle=False) as data:
            frame_row = int(np.flatnonzero(data["frames"] == FIRST)[0])
            uv = data["original_uv"][frame_row]
            np.testing.assert_array_equal(data["baseline_pose"][frame_row, :, 1], xyz[FIRST])
            np.testing.assert_array_equal(
                data["original_source"][frame_row, :, :, 0], source[FIRST]
            )

        # This is a temporary in-memory comparison, never a saved corrected track.
        frames = np.arange(FIRST - 1, STOP + 2)
        original = xyz[frames]
        hypothetical = original.copy()
        inside = (frames >= FIRST) & (frames < STOP)
        hypothetical[inside] = hypothetical[inside][:, SWAP]
        if not np.isfinite(original).all():
            raise ValueError("The reviewed interval must have all three resolved slots")
        steps = {
            "original": np.linalg.norm(np.diff(original, axis=0), axis=-1),
            "hypothetical": np.linalg.norm(np.diff(hypothetical, axis=0), axis=-1),
        }
        boundaries = []
        for frame in (FIRST, STOP):
            index = int(np.flatnonzero(frames[1:] == frame)[0])
            boundaries.append(
                {
                    "edge_from_frame": frame - 1,
                    "edge_to_frame": frame,
                    "original_step_mm_by_slot": steps["original"][index],
                    "hypothetical_step_mm_by_slot": steps["hypothetical"][index],
                    "original_source_before": source[frame - 1],
                    "original_source_after": source[frame],
                }
            )
        assignments = []
        choices = list(permutations(range(3)))
        for frame in (FIRST, FIRST + 1, FIRST + 2, FIRST + 3, STOP):
            costs = np.array(
                [
                    np.linalg.norm(xyz[frame - 1] - xyz[frame, list(q)], axis=-1).sum()
                    for q in choices
                ]
            )
            best = int(costs.argmin())
            assignments.append(
                {
                    "edge_to_frame": frame,
                    "best_permutation": choices[best],
                    "identity_sum_step_mm": costs[0],
                    "best_sum_step_mm": costs[best],
                    "advantage_over_identity_mm": costs[0] - costs[best],
                }
            )

        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        episode_index, episode = next(
            (i, item)
            for i, item in enumerate(analysis["episodes"])
            if item["start_frame"] <= FIRST < item["stop_frame_exclusive"]
        )
        audit = episode["persistent_audit"]
        marker = f'"start_frame": {episode["start_frame"]},'
        rejection = {
            "evidence_kind": "verbatim fields from completed bidirectional analysis",
            "json_pointer": f"/episodes/{episode_index}/persistent_audit",
            "location": location(analysis_path, '"persistent_audit":', after=marker),
            "status": audit["status"],
            "reason": audit["reason"],
            "config": audit["config"],
        }
        for side in ("left_reference", "right_reference"):
            reference = audit[side]
            interval = xyz[reference["frame_start"] : reference["frame_end"] + 1]
            measured = np.linalg.norm(np.diff(interval, axis=0), axis=-1).max(axis=0)
            np.testing.assert_allclose(measured, reference["max_native_step_mm"], atol=1e-9)
            rejection[side] = {
                key: reference[key]
                for key in (
                    "valid",
                    "frame_start",
                    "frame_end",
                    "supported_counts",
                    "median_fit_residual_mm",
                    "max_native_step_mm",
                    "reason",
                )
            }
            rejection[side]["location"] = location(analysis_path, f'"{side}":', after=marker)

        # Replay only the present S0 source point set, not the unavailable historic C33 run.
        drop_scores = []
        for drop, view in enumerate(VIEWS):
            dropped_uv = uv.copy()
            dropped_uv[:, drop] = np.nan
            geometry = geometry_metrics(rig, dropped_uv)
            node_rms = geometry["ray_rms_mm"][0, :2]
            if not np.isfinite(node_rms).all():
                raise ValueError("C33 replay requires two valid nodes for every two-view fit")
            drop_scores.append(
                {
                    "dropped_view": view,
                    "mouth_pectoral_ray_rms_mm": node_rms,
                    "mean_ray_rms_mm": float(node_rms.mean()),
                }
            )
        ranked = sorted(drop_scores, key=lambda item: item["mean_ray_rms_mm"])
        margin = ranked[1]["mean_ray_rms_mm"] - ranked[0]["mean_ray_rms_mm"]
        if not (np.all(source[FIRST, 0] >= 0) and residual[FIRST, 0] > 3.0):
            replayed_gate = "not_flagged"
        elif ranked[0]["mean_ray_rms_mm"] > 1.5:
            replayed_gate = "not_clean"
        else:
            replayed_gate = "not_sharp" if margin < 1.0 else "would_adopt"
        historical_path = Path("docs/decisions/2026-09-01-slot-stitching.md")
        report = {
            "schema_version": 1,
            "event_frame": FIRST,
            "review_context": "User confirmed the current S0/S1 large-jump swap at 31479",
            "accepted_identity_corrections": 0,
            "source_tracks_modified": False,
            "interval": {
                "start_inclusive": FIRST,
                "stop_exclusive": STOP,
                "last_inclusive": STOP - 1,
                "native_frames": STOP - FIRST,
                "duration_s_at_166_fps": (STOP - FIRST) / 166,
                "all_three_slots_resolved": True,
                "hypothetical_permutation": SWAP,
                "semantics": "Only inside interval: output[i] = original[permutation[i]]",
            },
            "boundaries": boundaries,
            "local_max_native_step_mm": {
                "frame_interval_half_open": [int(frames[0]), int(frames[-1] + 1)],
                **{key: value.max(axis=0) for key, value in steps.items()},
            },
            "adjacent_assignment_recalculation": assignments,
            "saved_bidirectional_refusal": rejection,
            "c33_drop_view_recalculation": {
                "evidence_kind": "Saved S0 point-set replay, not the original C33 run log",
                "frame": FIRST,
                "slot": 0,
                "saved_residual_mm": float(residual[FIRST, 0]),
                "source_rows_by_view": source[FIRST, 0],
                "view_order": list(VIEWS),
                "node_order": ["mouth", "pectoral"],
                "drop_scores": drop_scores,
                "best_drop": ranked[0]["dropped_view"],
                "runnerup_gap_mm": margin,
                "thresholds_mm": {
                    "flag_3_ray_above": 3.0,
                    "adopt_at_most": 1.5,
                    "runnerup_gap_at_least": 1.0,
                },
                "replayed_gate": replayed_gate,
                "scope": "View rejection changes geometry, not temporal animal identity",
                "gate_location": location(
                    Path("tools/c33_ray_rejection.py"), "runner[0] - best[0] < SHARP_MM"
                ),
            },
            "historical_scope": {
                "exit_only_record": location(historical_path, "identity restored from 31497"),
                "c30_suffix_update": location(Path("tools/c30_slot_stitch.py"), "perm[t1:] ="),
                "c34_resolved_frame_cap": location(
                    Path("tools/c34_microflip.py"), "MAX_WINDOW = 3"
                ),
            },
            "interpretation": [
                "C30 restores identity from 31497; the zone audit leaves the interior unchanged.",
                "The 18 resolved-frame excursion exceeds C34's three-frame out-and-back scope.",
                "Bidirectional flanks are unstable; the right flank includes the swap-back.",
                "Local relabeling removes the large exchange; a 12.08 mm entry step remains.",
                "Original error localization needs upstream C27 tracklets and C28 associations.",
                "User-reviewed 323819 has no swap; geometric residual is not identity truth.",
            ],
        }
        output.write_text(json.dumps(clean(report), indent=2, allow_nan=False), encoding="utf-8")
        reread = json.loads(output.read_text(encoding="utf-8"))
        assert reread["interval"]["native_frames"] == 18
        assert reread["saved_bidirectional_refusal"]["reason"] == "invalid_independent_flank"
        assert reread["c33_drop_view_recalculation"]["replayed_gate"] == "not_sharp"
        record.output(output.name, output, sha256=sha256(output))
        # Verify the diagnostic left every registered file input unchanged.
        for entry in record.inputs.values():
            if sha256(Path(entry["path"])) != entry["sha256"]:
                raise AssertionError(f"Input changed during diagnosis: {entry['path']}")
        print(json.dumps({"output": str(output), "boundaries": clean(boundaries)}, indent=2))


if __name__ == "__main__":
    main()
