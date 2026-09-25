"""Read-only scientific scope audit and saved-evidence diagnosis; never retunes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "work/variable_count_tracklets_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def reason_counts(events):
    return dict(Counter(event["reason"] for event in events if event.get("reason") is not None))


def female_annotation_coverage(protocol):
    path = (ROOT / "work/generative_tracking_inputs/assets/groundtruth/fish_sex_gt"
            / "20251225_K_1_2_FCMC/female_sex_gt.csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        confirmed = [row for row in csv.DictReader(stream)
                     if row["gt_status"] == "confirmed" and row["sex_label"] == "F"]
    frames = sorted({int(row["frame"]) for row in confirmed})
    pairs = [(first, second) for first, second in pairwise(frames)
             if second - first == 1]
    return {
        "source": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
        "confirmed_female_annotation_rows": len(confirmed),
        "adjacent_frame_confirmed_anchor_pairs_entire_ledger": pairs,
        "case_coverage": [{
            "case": case["name"],
            "confirmed_frames": [frame for frame in frames
                                 if case["start"] <= frame < case["stop"]],
        } for case in protocol["cases"]],
        "current_c01_detection_mapping_created": False,
        "limitation": (
            "Sparse confirmed female anchors do not directly label adjacent 2D edges, male "
            "identities, or reflection rows. A separate validated mapping from historical "
            "mouth coordinates to current C01 rows would be needed for endpoint evaluation."
        ),
    }


def diagnose(dest, method, view, case, data, events):
    """Reference-label agreement describes a mechanism, never biological identity."""
    p0_path = dest / f"tracking_P0_{view}.npz"
    if not p0_path.exists():
        return None
    p0 = arrays(p0_path)
    frames, rows, points = (data[key] for key in ("native_frame", "global_row", "points"))
    index = {int(row): i for i, row in enumerate(rows)}
    refused = [event for event in events if event.get("kind") == "directional_link_refused"
               and case["start"] <= event["frame"] < case["stop"]]
    ghost_ages = Counter()
    open_refusals, only_same_p0, separated, examples = 0, 0, 0, []
    for event in refused:
        ghosts = [candidate for candidate in event["candidates"]
                  if candidate["age_frames"] > 1 and candidate["score_px"] is not None]
        ghost_ages.update(candidate["age_frames"] for candidate in ghosts)
        if event["reason"] != "open_endpoint_competition":
            continue
        open_refusals += 1
        current = index[event["global_row"]]
        candidates = [candidate for candidate in event["candidates"]
                      if candidate["score_px"] is not None]
        same = bool(candidates) and all(
            p0["labels"][index[candidate["from_global_row"]]] == p0["labels"][current]
            for candidate in candidates
        )
        only_same_p0 += same
        peer_indices = np.flatnonzero((frames == frames[current]) & (rows != rows[current]))
        distances = np.linalg.norm(points[peer_indices, 1] - points[current, 1], axis=1)
        distances = distances[np.isfinite(distances)]
        nearest = float(distances.min()) if len(distances) else None
        far = nearest is not None and nearest > (120 if view == "top_cam" else 100)
        separated += far
        if same and far and len(examples) < 6:
            shown = sorted(candidates, key=lambda item: (
                item["age_frames"] != 1, item["score_px"], item["from_global_row"]
            ))[:8]
            examples.append({
                "frame": int(frames[current]), "global_row": int(rows[current]),
                "direction": event["direction"], "p0_local_id": int(p0["labels"][current]),
                "nearest_other_current_p_px": nearest,
                "candidate_count": len(candidates),
                "candidate_display_policy": "adjacent candidate first, then lowest scores; up to 8",
                "candidates": [{key: candidate.get(key) for key in (
                    "from_global_row", "age_frames", "prediction_error_px", "score_px",
                    "origin_rank", "target_rank", "origin_margin_px", "target_margin_px",
                    "required_margin_px", "uncertainty_px", "reason", "auxiliary_support",
                )} for candidate in shown],
                "interpretation": (
                    "All admissible endpoint hypotheses share the frozen P0 local label; "
                    "this identifies repeated hypotheses, not human identity truth."
                ),
            })
    core = data["core_mask"] & (frames > case["start"])
    p0_link_p1_cut = core & p0["clean_edge"] & ~data["clean_edge"]
    windows = []
    for center in case.get("centers", []):
        start, stop = max(case["start"], center - 83), min(case["stop"], center + 83)
        selected = (frames > start) & (frames < stop)
        old_links, new_links = selected & p0["clean_edge"], selected & data["clean_edge"]
        different = old_links & new_links & (p0["previous_row"] != data["previous_row"])
        window_events = [event for event in refused if start <= event["frame"] < stop]
        windows.append({
            "review_anchor": center, "native_start": start, "native_stop": stop,
            "eligible_observations": int(selected.sum()),
            "p0_adjacent_connections": int(old_links.sum()),
            "p1_adjacent_connections": int(new_links.sum()),
            "p0_connections_held_by_p1": int((old_links & ~new_links).sum()),
            "p1_connections_added_where_p0_held": int((new_links & ~old_links).sum()),
            "both_linked_but_different_predecessor": int(different.sum()),
            "directional_refusal_reasons": reason_counts(window_events),
            "identity_verdict": "unknown; scene review does not label every 2D predecessor",
        })
    return {
        "case": case["name"], "view": view, "method": method,
        "core_directional_refusal_events": len(refused),
        "core_directional_refusal_reasons": reason_counts(refused),
        "admissible_unobserved_competitor_ages": dict(sorted(ghost_ages.items())),
        "core_open_endpoint_refusal_events": open_refusals,
        "open_refusals_all_admissible_endpoints_share_target_p0_label": only_same_p0,
        "open_refusals_nearest_current_p_exceeds_body_scale": separated,
        "p0_adjacent_connections_without_p1_adjacent_connection": int(p0_link_p1_cut.sum()),
        "examples": examples,
        "review_anchor_windows": windows,
        "evidence_scope": "Both directional passes, core target; repeated endpoint events allowed",
        "identity_truth_status": "P0 labels are a frozen comparison, not ground truth",
        "mechanism": (
            "An ambiguous adjacent refusal leaves an old forecast and a fresh observed endpoint. "
            "Multiple forecasts can represent competing hypotheses of the same physical path; "
            "uncertainty-adjusted older forecasts can prevent unique adjacent continuation until "
            "retirement or the 17-frame expiry. Forward and reverse histories then disagree. "
            "The saved evidence quantifies this over-conservative feedback; no parameters changed."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    protocol_path = args.output / "protocol.json"
    protocol = read(protocol_path)
    scope, quality, diagnoses, missing = [], [], [], []
    for case in protocol["cases"]:
        dest = args.output / case["name"]
        for view in VIEWS:
            detections = dest / f"detections_{view}.npz"
            if detections.exists():
                original = arrays(detections)
                mask = np.isfinite(original["points"]).all(axis=-1)
                core = original["core_mask"]
                quality.append({
                    "case": case["name"], "view": view,
                    "core_rows": int(core.sum()),
                    "core_missing_p_rows": int((~mask[core, 1]).sum()),
                    "core_any_missing_node_rows": int((~mask[core].all(axis=1)).sum()),
                    "context_missing_p_rows": int((~mask[:, 1]).sum()),
                    "separation_caveat": (
                        "P-missing rows have unknown separation; the frozen aggregate puts "
                        "their infinite nearest-distance sentinel in the separated stratum."
                    ),
                })
            for method in protocol["conditions"]:
                path = dest / f"tracking_{method}_{view}.npz"
                metrics_path = dest / f"metrics_{method}_{view}.json"
                events_path = dest / f"events_{method}_{view}.json"
                if not all(p.exists() for p in (path, metrics_path, events_path)):
                    missing.append(f"{case['name']}/{method}/{view}")
                    continue
                data, metrics, events = arrays(path), read(metrics_path), read(events_path)
                frames, previous = data["native_frame"], data["previous_row"]
                core = data["core_mask"]
                eligible = core & (frames > case["start"])
                gap = eligible & (data["link_gap"] > 1)
                prior_core = (previous >= 0) & core[np.maximum(previous, 0)]
                both = gap & prior_core
                assert int(gap.sum()) == metrics["gap_connections"]
                all_reasons = reason_counts(events)
                assert all_reasons == metrics["refusal_reasons"]
                directional = [event for event in events
                               if event.get("kind") == "directional_link_refused"]
                examples = np.flatnonzero(gap & ~prior_core)[:6]
                scope.append({
                    "case": case["name"], "view": view, "method": method,
                    "reported_gap_connections_target_core_after_first_frame": int(gap.sum()),
                    "strict_both_endpoints_core_gap_connections": int(both.sum()),
                    "reported_gaps_with_context_predecessor": int((gap & ~prior_core).sum()),
                    "gap_scope_examples": [{
                        "from_frame": int(frames[previous[i]]), "to_frame": int(frames[i]),
                        "from_global_row": int(data["global_row"][previous[i]]),
                        "to_global_row": int(data["global_row"][i]),
                        "elapsed_frames": int(data["link_gap"][i]),
                    } for i in examples],
                    "all_event_kinds_full_context": dict(Counter(e["kind"] for e in events)),
                    "all_event_reasons_full_context": all_reasons,
                    "actual_directional_refusal_events_full_context": len(directional),
                    "actual_directional_refusal_reasons_full_context": reason_counts(directional),
                    "tracking_npz_sha256": sha256(path),
                    "events_json_sha256": sha256(events_path),
                })
                if method in ("P1", "P1aux"):
                    diagnosis = diagnose(dest, method, view, case, data, events)
                    if diagnosis is not None:
                        diagnoses.append(diagnosis)
    if missing and not args.partial:
        raise RuntimeError(f"Numerical comparison incomplete: {len(missing)} missing conditions")
    result = {
        "complete": not missing, "missing_conditions": missing,
        "numerical_inputs_modified": False, "parameters_retuned": False,
        "scope_records": scope, "observation_quality": quality,
        "endpoint_competition_diagnosis": diagnoses,
        "female_annotation_coverage": female_annotation_coverage(protocol),
        "reporting_rules": [
            "Adjacent metrics count both endpoints inside core and use eligible observations.",
            "Frozen gap metrics allow warmup predecessors; strict supplementary counts here.",
            "Frozen refusal_reasons includes all event kinds; label as full-context event reasons.",
            "Directional refusal events are not unique rejected links or identity errors.",
            "Reference P0 continuity and smoothness do not establish identity correctness.",
        ],
        "totals": {
            "method_case_views": len(scope),
            "reported_gaps_with_context_predecessor": sum(
                record["reported_gaps_with_context_predecessor"] for record in scope
            ),
            "core_missing_p_rows_unique_case_view": sum(
                record["core_missing_p_rows"] for record in quality
            ),
        },
    }
    suffix = "_partial" if args.partial else ""
    path = args.output / f"scientific_scope_audit{suffix}.json"
    with provenance("variable_count_scientific_scope_audit", args.output,
                    name="scientific_scope_audit" + suffix) as rec:
        rec.input("audit_code", Path(__file__), sha256=sha256(Path(__file__)))
        rec.input("protocol", protocol_path, sha256=sha256(protocol_path))
        female = result["female_annotation_coverage"]
        rec.input("female_annotation_ledger", ROOT / female["source"], sha256=female["sha256"])
        rec.param("read_only_numerical_inputs", True)
        rec.param("partial", args.partial)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        rec.output("scope_audit", path, sha256=sha256(path))
    print(json.dumps({"output": str(path), "complete": not missing, **result["totals"]}))


if __name__ == "__main__":
    main()
