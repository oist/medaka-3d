"""Classify saved M/C/T mask distances at the existing fixed 6px tolerance.

This reads only saved measurement JSON and the small configuration source.
It does not load masks, SLEAP, trajectories or videos, or introduce a gate.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK / "nodewise_body_mask_containment_v1"
SEAL_SHA = "bc6c28ddf6f8c4cd89f624b085415ebccd5e1be34b857ccc434620796a185138"
CONFIG_SHA = "fb99820558d317a70b4d170f67c8a94c5cf3ead9e319f068568afa651d6054b6"
STAGE = "body_mask_endpoint_classification_v1"
FRAMES, NODES = (82720, 90258), ("M", "C", "T")
VIEWS = ("top_cam", "side1_cam", "side2_cam")
FOCUS = {82720: (18, 19), 90258: (9,)}


def capture(path, expected=None):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("saved input SHA differs: " + str(path))
    return data, dict(path=str(path), sha256=digest, size_bytes=len(data))


def classify_endpoint_distances(distances, mask_indices, available, tolerance):
    if not available:
        return dict(
            classification="unavailable",
            reason="mask evidence unavailable",
            endpoints=[
                dict(node=n, mask_distances_px=None, nearby_mask_indices=None) for n in NODES
            ],
            uncovered_nodes=[],
            ambiguous_nodes=[],
            unique_endpoint_masks=None,
        )
    if (
        len(distances) != 3
        or not mask_indices
        or len(set(mask_indices)) != len(mask_indices)
        or any(type(i) is not int or i < 0 for i in mask_indices)
    ):
        raise ValueError("three endpoint rows and distinct original mask labels required")
    endpoints, choices = [], []
    for node, values in zip(NODES, distances, strict=True):
        if len(values) != len(mask_indices) or any(not math.isfinite(d) or d < 0 for d in values):
            raise ValueError("finite nonnegative distances for every mask required")
        nearby = [index for index, d in zip(mask_indices, values, strict=True) if d <= tolerance]
        choices.append(nearby)
        endpoints.append(
            dict(
                node=node,
                mask_distances_px=values,
                nearby_mask_indices=nearby,
                unique_mask=nearby[0] if len(nearby) == 1 else None,
            )
        )
    uncovered = [n for n, ids in zip(NODES, choices, strict=True) if not ids]
    ambiguous = [n for n, ids in zip(NODES, choices, strict=True) if len(ids) > 1]
    unique = [ids[0] for ids in choices] if not uncovered and not ambiguous else None
    if uncovered:
        classification, reason = "uncovered_endpoint", "at least one endpoint has no mask within6px"
    elif ambiguous:
        classification, reason = (
            "ambiguous_endpoint",
            "all covered but an endpoint fits multiple masks",
        )
    elif len(set(unique)) == 1:
        classification, reason = (
            "same_instance",
            "all endpoints uniquely fit the same original mask",
        )
    else:
        classification, reason = (
            "split_instances",
            "unique endpoints fit two or more original masks",
        )
    return dict(
        classification=classification,
        reason=reason,
        endpoints=endpoints,
        uncovered_nodes=uncovered,
        ambiguous_nodes=ambiguous,
        unique_endpoint_masks=unique,
    )


def run():
    output = WORK / "BODY_MASK_ENDPOINT_CLASSIFICATION_V1.json"
    note = WORK / "BODY_MASK_ENDPOINT_CLASSIFICATION_V1.md"
    if output.exists() or note.exists():
        raise FileExistsError("fresh result files required")
    config_bytes, config_ref = capture(WORK / "partial_body_bundle_score.py", CONFIG_SHA)
    cls = next(
        n
        for n in ast.parse(config_bytes).body
        if isinstance(n, ast.ClassDef) and n.name == "BundleConfig"
    )
    tolerance = ast.literal_eval(
        next(
            n.value
            for n in cls.body
            if isinstance(n, ast.AnnAssign) and n.target.id == "endpoint_tolerance_px"
        )
    )
    if tolerance != 6.0:
        raise ValueError("existing endpoint tolerance differs")
    source_ref = capture(Path(__file__).resolve())[1]
    seal_bytes, seal_ref = capture(ROOT / "measurement.provenance.json", SEAL_SHA)
    seal = json.loads(seal_bytes)
    if seal["stage"] != ROOT.name or seal["status"] != "ok":
        raise ValueError("successful measurement required")
    records, case_refs, counts = [], [], Counter()
    for frame in FRAMES:
        name = f"cases/f{frame}.json"
        descriptor = seal["outputs"][name]
        payload, ref = capture(ROOT / name, descriptor["sha256"])
        if ref != descriptor:
            raise ValueError("case output path or size differs")
        case = json.loads(payload)
        if case["frame_idx"] != frame or case["arm"] != "sparse_instance_rescore":
            raise ValueError("case scope differs")
        indices = case["pool_indices"]
        if len(set(indices)) != len(indices) or len(indices) != len(case["bodies"]):
            raise ValueError("complete-body membership differs")
        case_refs.append(ref)
        for view in VIEWS:
            measured = case["views"][view]["containment"]
            if measured["sample_labels"][:3] != list(NODES):
                raise ValueError("saved sample order differs")
            all_distances = measured.get("distances_by_body_sample_mask")
            if measured["available"] and len(all_distances) != len(indices):
                raise ValueError("per-body sample coverage differs")
            for position, index in enumerate(indices):
                result = classify_endpoint_distances(
                    all_distances[position][:3] if measured["available"] else None,
                    measured.get("mask_indices"),
                    measured["available"],
                    tolerance,
                )
                records.append(
                    dict(
                        frame_idx=frame,
                        pool_index=index,
                        view=view,
                        legacy_body=index < case["legacy_candidate_count"],
                        highlighted=index in FOCUS[frame],
                        **result,
                    )
                )
                counts[result["classification"]] += 1
    if len(records) != 93 or sum(counts.values()) != len(records):
        raise ValueError("all31 bodies by3 views required")
    report = dict(
        schema=STAGE,
        status="ok",
        source=source_ref,
        measurement_seal=seal_ref,
        case_inputs=case_refs,
        configuration_source=config_ref,
        endpoint_tolerance_px=tolerance,
        comparison="distance <=6px, with original detector mask labels",
        classification_precedence=[
            "unavailable",
            "uncovered_endpoint",
            "ambiguous_endpoint",
            "same_instance or split_instances when all three unique",
        ],
        all_complete_bodies=31,
        classified_body_views=len(records),
        counts=dict(counts),
        records=records,
        SLEAP_used=False,
        mask_or_video_or_trajectory_payloads_read=0,
        automatic_rejection=False,
        cost_or_tracking_changed=False,
        physical_identity_truth=False,
        limitations=[
            "D2 can split one fish; split_instances does not confirm cross-fish mixing.",
            "same_instance allows interior leakage or a merged detection.",
            "Mask evidence and complete-body membership inherit the bound measurement.",
            "Uncovered takes precedence over ambiguous; both endpoint lists remain.",
        ],
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    lines = [
        "# M/C/Tの固定6pxマスク対応分類",
        "",
        "保存距離だけから全31完全身体・3視点の93件を分類した。端点の近傍mask集合は距離≤6px。",
        "未被覆を先に、全点被覆時の複数候補を曖昧とし、全3点が一意の場合だけ同一/分離に分類する。",
        "自動reject・費用変更・物理ID認定は行っていない。",
        "",
        "|分類|件数|",
        "|---|---:|",
    ]
    for category in (
        "unavailable",
        "uncovered_endpoint",
        "ambiguous_endpoint",
        "same_instance",
        "split_instances",
    ):
        lines.append(f"|{category}|{counts[category]}|")
    lines += ["", "|対象|視点|分類|M/C/Tの近傍mask集合|", "|---|---|---|---|"]
    for record in records:
        if record["highlighted"]:
            matches = "/".join(str(e["nearby_mask_indices"]) for e in record["endpoints"])
            lines.append(
                f"|{record['frame_idx']} P{record['pool_index']}|{record['view']}|"
                f"{record['classification']}|{matches}|"
            )
    lines += [
        "",
        "各端点の全mask距離・元mask番号・欠落/曖昧理由は"
        "[JSON](BODY_MASK_ENDPOINT_CLASSIFICATION_V1.json)に保存した。",
        "D2が1匹を複数maskへ分割した場合もsplit_instancesとなるため、確定誤結合とは呼ばない。"
        "same_instanceでも線分内点の漏れやmask自身の魚の結合は除外できない。",
        "元mask・動画・SL・trackNPZの読込はなく、保存された先頭3sampleだけを分類した。",
    ]
    with note.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(dict(counts=counts, output=str(output), note=str(note)), flush=True)


if __name__ == "__main__":
    run()
