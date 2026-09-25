"""Score the two five-minute video candidates with fixed female mouth anchors.

This is an endpoint-only evaluation.  It reuses the previously frozen 57-anchor
set and its SLEAP detection-row matches; it does not create labels or tune a
tracker.  A changed raw slot between adjacent anchors is reported as one swap
*interval*, because the annotations do not localize the event within it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from medaka3d.eval.generative import MouthAnchor, score_female_identity  # noqa: E402


START = 66_400
STOP = 116_200
FPS = 166.0
ANCHOR_DIR = REPO / "work/three_body/female_57_five_minute_endpoint_evaluation_v1"
COMPARISON_DIR = REPO / "work/three_body/resumed_nodewise_five_minute_comparison_v1"
OUTPUT_DIR = REPO / "work/three_body/resumed_nodewise_female_57_comparison_v1"
CANDIDATES = {
    "sparse_old": REPO / "work/three_body/sparse_instance_five_minute_continuation_v1/candidate.npz",
    "nodewise_new": REPO / "work/three_body/nodewise_body_row_five_minute_resume_099600_v1/candidate.npz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_sealed_hash(seal: dict, path: Path) -> str:
    target = str(path.resolve()).lower()
    matches = [value for key, value in seal["inputs"].items() if str(Path(key).resolve()).lower() == target]
    if len(matches) != 1:
        raise RuntimeError(f"candidate is not uniquely sealed as an input: {path}")
    return matches[0]


def runs(rows: list[dict]) -> list[dict]:
    available = [row for row in rows if row["identity"] is not None]
    if not available:
        return []
    result = []
    first = previous = available[0]
    count = 1
    for row in available[1:]:
        if row["identity"] == previous["identity"]:
            previous = row
            count += 1
            continue
        result.append({
            "start_frame": first["frame"],
            "stop_frame": previous["frame"],
            "slot": first["identity"],
            "anchor_count": count,
        })
        first = previous = row
        count = 1
    result.append({
        "start_frame": first["frame"],
        "stop_frame": previous["frame"],
        "slot": first["identity"],
        "anchor_count": count,
    })
    return result


def seconds(frame: int) -> float:
    return frame / FPS


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seal_path = COMPARISON_DIR / "comparison.provenance.json"
    seal = load_json(seal_path)
    if seal.get("status") != "ok" or seal.get("stage") != "resumed_nodewise_five_minute_comparison_v1":
        raise RuntimeError("the current five-minute comparison is not sealed status=ok")

    candidate_hashes = {}
    for name, path in CANDIDATES.items():
        actual = sha256(path)
        expected = find_sealed_hash(seal, path)
        if actual != expected:
            raise RuntimeError(f"sealed candidate hash mismatch: {name}")
        candidate_hashes[name] = actual

    anchor_set_path = ANCHOR_DIR / "fixed_female_anchor_set.json"
    observations_path = ANCHOR_DIR / "anchor_observations.json"
    anchor_set = load_json(anchor_set_path)
    observations = load_json(observations_path)
    frozen_frames = list(map(int, anchor_set["frames"]))
    matches = observations["matches"]
    if len(frozen_frames) != 57 or [int(row["frame"]) for row in matches] != frozen_frames:
        raise RuntimeError("the saved observations do not exactly match the frozen 57-anchor list")
    if not all(START <= frame < STOP for frame in frozen_frames):
        raise RuntimeError("an anchor falls outside the comparison interval")
    anchors = [
        MouthAnchor(
            frame=int(row["frame"]),
            detection_rows={key: int(value) for key, value in row["detection_rows"].items()},
            pixels={key: tuple(map(float, value)) for key, value in row["pixels"].items()},
            match_errors_px={key: float(value) for key, value in row["match_errors_px"].items()},
        )
        for row in matches
    ]

    results = {}
    for name, path in CANDIDATES.items():
        with np.load(path, allow_pickle=False) as payload:
            frame_idx = payload["frame_idx"]
            keep = (frame_idx >= START) & (frame_idx < STOP)
            frames = frame_idx[keep]
            if not np.array_equal(frames, np.arange(START, STOP, dtype=frames.dtype)):
                raise RuntimeError(f"{name} does not contain every requested video frame")
            sources = payload["sources"][keep]
            observed = payload["observed"][keep]
            review = payload["identity_review"][keep]
        raw = score_female_identity(frames, sources[:, :, :, 0], observed[:, :, 0], anchors)
        gated = score_female_identity(
            frames, sources[:, :, :, 0], observed[:, :, 0], anchors, review=review
        )
        raw["identity_runs"] = runs(raw["anchors"])
        gated["identity_runs"] = runs(gated["anchors"])
        results[name] = {"raw_slots": raw, "review_gated": gated}

    rows = []
    for index, frame in enumerate(frozen_frames):
        sparse = results["sparse_old"]["raw_slots"]["anchors"][index]
        nodewise = results["nodewise_new"]["raw_slots"]["anchors"][index]
        rows.append({
            "frame": frame,
            "recording_seconds": seconds(frame),
            "video_seconds": seconds(frame - START),
            "sparse_female_slot": sparse["identity"],
            "sparse_reason": sparse["reason"],
            "nodewise_female_slot": nodewise["identity"],
            "nodewise_reason": nodewise["reason"],
        })

    # Compare interval outcomes rather than numeric slots, since each tracker may
    # apply a different single global permutation to the three anonymous slots.
    interval_comparison = []
    for sparse, nodewise in zip(
        results["sparse_old"]["raw_slots"]["intervals"],
        results["nodewise_new"]["raw_slots"]["intervals"],
        strict=True,
    ):
        if (sparse["start"], sparse["stop"]) != (nodewise["start"], nodewise["stop"]):
            raise RuntimeError("the methods were not scored on identical intervals")
        interval_comparison.append({
            "start": sparse["start"],
            "stop": sparse["stop"],
            "start_recording_seconds": seconds(sparse["start"]),
            "stop_recording_seconds": seconds(sparse["stop"]),
            "start_video_seconds": seconds(sparse["start"] - START),
            "stop_video_seconds": seconds(sparse["stop"] - START),
            "sparse_outcome": sparse["outcome"],
            "nodewise_outcome": nodewise["outcome"],
        })

    summary = {
        "stage": "resumed_nodewise_female_57_comparison_v1",
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interval": {
            "start_frame": START,
            "stop_frame_exclusive": STOP,
            "fps": FPS,
            "recording_start_seconds": seconds(START),
            "recording_stop_seconds": seconds(STOP),
            "duration_seconds": seconds(STOP - START),
        },
        "ground_truth": {
            "sex": "female",
            "confirmed_total_in_source_csv": 300,
            "fixed_anchors_in_interval": len(anchors),
            "fixed_adjacent_intervals": len(anchors) - 1,
            "anchor_set_sha256": sha256(anchor_set_path),
            "anchor_observations_sha256": sha256(observations_path),
            "labels_created_or_modified": False,
            "used_for_training_or_tuning": False,
        },
        "candidate_sha256": candidate_hashes,
        "comparison_seal_sha256": sha256(seal_path),
        "evaluator_sha256": sha256(REPO / "src/medaka3d/eval/generative.py"),
        "methods": results,
        "raw_interval_outcome_pairs": dict(Counter(
            f"sparse={row['sparse_outcome']},nodewise={row['nodewise_outcome']}"
            for row in interval_comparison
        )),
        "interval_comparison": interval_comparison,
        "interpretation_limits": [
            "A swap interval means the female maps to different raw slots at two adjacent fixed anchors; it is not an exact event count.",
            "A preserved interval does not rule out an even number of swaps and recoveries between its endpoints.",
            "This evaluates only the female identity. The two males remain unlabeled by this ground truth.",
            "Review-gated results abstain where the tracker itself requests review.",
        ],
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_DIR / "anchor_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "interval_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(interval_comparison[0]))
        writer.writeheader()
        writer.writerows(interval_comparison)

    sparse_counts = results["sparse_old"]["raw_slots"]["interval_counts"]
    nodewise_counts = results["nodewise_new"]["raw_slots"]["interval_counts"]
    differing = [row for row in interval_comparison if row["sparse_outcome"] != row["nodewise_outcome"]]
    report_lines = [
        "# Fixed-female-anchor comparison",
        "",
        f"Interval: frames `{START}:{STOP}` (recording 06:40–11:40), with 57 fixed confirmed-female anchors.",
        "",
        "| Candidate | Available anchors | Preserved intervals | Swap intervals | Unscored |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Sparse (video left) | {results['sparse_old']['raw_slots']['available_anchors']} | "
            f"{sparse_counts['preserved']} | {sparse_counts['swap']} | {sparse_counts['unscored']} |"
        ),
        (
            f"| Nodewise (video right) | {results['nodewise_new']['raw_slots']['available_anchors']} | "
            f"{nodewise_counts['preserved']} | {nodewise_counts['swap']} | {nodewise_counts['unscored']} |"
        ),
        "",
    ]
    if differing:
        report_lines.extend([
            "Sparse changes the female from slot 1 to slot 2 in the fixed-anchor bracket "
            f"`{differing[0]['start']}–{differing[0]['stop']}`. This is video time "
            f"`{differing[0]['start_video_seconds']:.3f}–{differing[0]['stop_video_seconds']:.3f} s` "
            f"and recording time `{differing[0]['start_recording_seconds']:.3f}–"
            f"{differing[0]['stop_recording_seconds']:.3f} s`. Nodewise preserves the female slot across it.",
            "",
        ])
    report_lines.extend([
        "The metric detects changes between annotated endpoints. It does not prove that no swap-and-recovery occurred "
        "inside a preserved interval, and it does not evaluate which male is which.",
        "",
    ])
    report_path = OUTPUT_DIR / "RESULT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    digest = {
        "summary_sha256": sha256(summary_path),
        "anchor_table_sha256": sha256(OUTPUT_DIR / "anchor_table.csv"),
        "interval_table_sha256": sha256(OUTPUT_DIR / "interval_table.csv"),
        "result_md_sha256": sha256(report_path),
    }
    (OUTPUT_DIR / "hashes.json").write_text(
        json.dumps(digest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    compact = {
        name: {
            "raw_counts": result["raw_slots"]["interval_counts"],
            "raw_runs": result["raw_slots"]["identity_runs"],
            "review_counts": result["review_gated"]["interval_counts"],
            "available_raw": result["raw_slots"]["available_anchors"],
            "available_review_gated": result["review_gated"]["available_anchors"],
        }
        for name, result in results.items()
    }
    print(json.dumps({"status": "ok", "methods": compact, "outcome_pairs": summary["raw_interval_outcome_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
