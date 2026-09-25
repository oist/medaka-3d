"""Count repeated uncertainty episodes after inference, including during sticky review.

This is an engineering inspection inventory, not an identity scorer. Scene
counts do not establish that a person can resolve or repair those scenes.
No labels, baseline trajectories, images or tracker are consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from medaka3d.provenance import provenance

FPS = 166
GROUP_GAP_FRAMES = 166
CONTEXT_FRAMES = 83
MARGIN_THRESHOLD = 1.0


def _segments(frames):
    boundaries = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1, len(frames)]
    return list(pairwise(boundaries.tolist()))


def _runs(mask, lo, hi):
    edges = np.diff(np.r_[False, mask[lo:hi], False].astype(np.int8))
    return [(int(a + lo), int(b + lo)) for a, b in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
    )]


def _step_summary(pose, frames, lo, hi, segment_lo, segment_hi, fish):
    # Include the transition out of last support and into first returned support.
    start, stop = max(segment_lo, lo - 1), min(segment_hi, hi + 1)
    steps = np.linalg.norm(np.diff(pose[start:stop, fish], axis=0), axis=-1)
    result = {}
    for node, name in enumerate(("mouth", "pectoral", "tail")):
        finite = np.isfinite(steps[:, node])
        index = int(np.argmax(np.where(finite, steps[:, node], -np.inf))) if finite.any() else None
        result[name] = {
            "maximum_mm": float(steps[index, node]) if index is not None else None,
            "to_frame": int(frames[start + index + 1]) if index is not None else None,
            "finite_steps": int(finite.sum()),
        }
    return result


def extract_review_events(result: dict, *, review_gap_frames: int = 4) -> dict:
    """Extract every long body loss and finite low-margin episode independently.

    Frame gaps split both episodes and scenes. Missing counters are checked
    inside each continuous block; their pre-block history remains unknown.
    ``score_margin`` is scalar per frame, so affected identities for that risk
    cannot be recovered from these arrays and are explicitly unspecified.
    """
    if (isinstance(review_gap_frames, bool) or not isinstance(review_gap_frames, int)
            or review_gap_frames < 1):
        raise ValueError("review_gap_frames must be a positive integer")
    frames = np.asarray(result["frame_idx"])
    if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu"
            or np.any(frames[1:] <= frames[:-1])):
        raise ValueError("frames must be strictly increasing native integers")
    frames = frames.astype(np.int64)
    count = len(frames)
    body, observed = np.asarray(result["body_supported"]), np.asarray(result["observed"])
    review, missing = np.asarray(result["identity_review"]), np.asarray(result["missing"])
    pose, margin = np.asarray(result["pose_xyz"], float), np.asarray(result["score_margin"], float)
    masks = np.asarray(result["mask_supported"])
    for name, value, shape in (
        ("body_supported", body, (count, 3)), ("observed", observed, (count, 3, 3)),
        ("identity_review", review, (count, 3)), ("mask_supported", masks, (count, 3)),
    ):
        if value.shape != shape or value.dtype.kind != "b":
            raise ValueError(f"{name} must be a boolean array of shape {shape}")
    if missing.shape != (count, 3) or missing.dtype.kind not in "iu" or np.any(missing < 0):
        raise ValueError("missing must be nonnegative integer counters (frames, fish)")
    if pose.shape != (count, 3, 3, 3) or np.isinf(pose).any() or margin.shape != (count,):
        raise ValueError("pose_xyz or scalar score_margin has an inconsistent shape/value")
    if np.any(body & (missing != 0)) or np.any(~body & (missing == 0)):
        raise ValueError("missing counters disagree with declared body support")
    if np.any(observed[:, :, 1] & ~body):
        raise ValueError("direct pectoral support must also support the body")
    if np.any(body & ~observed[:, :, 1] & ~masks):
        raise ValueError("body bridge without direct pectoral support requires mask support")
    if np.any(observed & ~np.isfinite(pose).all(axis=-1)):
        raise ValueError("observed nodes must have finite saved poses")
    segments = _segments(frames)
    for lo, hi in segments:
        expected = np.where(body[lo + 1:hi], 0, missing[lo:hi - 1] + 1)
        if not np.array_equal(expected, missing[lo + 1:hi]):
            raise ValueError("missing counters do not increment within a continuous frame block")

    events, short_losses, direct_losses = [], [], []
    union = np.zeros(count, bool)
    margin_risk = np.isfinite(margin) & (margin <= MARGIN_THRESHOLD)
    for block, (block_lo, block_hi) in enumerate(segments):
        for fish in range(3):
            for lo, hi in _runs(~observed[:, fish, 1], block_lo, block_hi):
                direct_losses.append({
                    "fish": fish, "start_frame": int(frames[lo]),
                    "stop_frame_exclusive": int(frames[hi - 1] + 1), "frames": hi - lo,
                    "mask_bridge_frames": int(body[lo:hi, fish].sum()),
                    "body_unsupported_frames": int((~body[lo:hi, fish]).sum()),
                    "left_censored": lo == block_lo, "right_censored": hi == block_hi,
                })
            for lo, hi in _runs(~body[:, fish], block_lo, block_hi):
                item = {
                    "kind": "body_unsupported", "block": block, "affected_fish": [fish],
                    "start_frame": int(frames[lo]),
                    "stop_frame_exclusive": int(frames[hi - 1] + 1),
                    "available_frames": hi - lo, "duration_seconds": (hi - lo) / FPS,
                    "left_censored": lo == block_lo, "right_censored": hi == block_hi,
                    "missing_counter_at_available_start": int(missing[lo, fish]),
                    "maximum_saved_missing_counter": int(missing[lo:hi, fish].max()),
                }
                if missing[lo:hi, fish].max() < review_gap_frames:
                    short_losses.append(item)
                    continue
                crossed = np.flatnonzero(missing[lo:hi, fish] >= review_gap_frames)
                threshold = int(frames[lo + crossed[0]])
                earlier_direct = np.flatnonzero(observed[block_lo:lo, fish, 1])
                returned = hi < block_hi
                item.update(
                    onset_frame=int(frames[lo]) if missing[lo, fish] == 1 else None,
                    last_body_support_frame=int(frames[lo - 1]) if lo > block_lo else None,
                    last_direct_pectoral_frame=(
                        int(frames[block_lo + earlier_direct[-1]]) if len(earlier_direct) else None
                    ),
                    first_return_frame=int(frames[hi]) if returned else None,
                    first_return_support=(
                        "direct_pectoral" if returned and observed[hi, fish, 1]
                        else "mask_bridge" if returned else None
                    ),
                    first_threshold_frame_in_available_input=threshold,
                    actual_threshold_crossing_frame=(
                        threshold if missing[lo, fish] <= review_gap_frames else None
                    ),
                    review_already_true_before_episode=(
                        bool(review[lo - 1, fish]) if lo > block_lo else None
                    ),
                    review_at_available_start=bool(review[lo, fish]),
                    review_frames_within_episode=int(review[lo:hi, fish].sum()),
                    direct_observed_node_frames=dict(zip(
                        ("mouth", "pectoral", "tail"), observed[lo:hi, fish].sum(axis=0).tolist(),
                        strict=True,
                    )),
                    raw_mask_support_while_body_unsupported_frames=int(masks[lo:hi, fish].sum()),
                    saved_latent_pose_step_including_return=_step_summary(
                        pose, frames, lo, hi, block_lo, block_hi, fish
                    ),
                )
                events.append(item)
                union[lo:hi] = True
        for lo, hi in _runs(margin_risk, block_lo, block_hi):
            events.append({
                "kind": "candidate_permutation_risk", "block": block,
                "affected_fish": None, "affected_fish_reason": "scalar margin omits competing IDs",
                "start_frame": int(frames[lo]),
                "stop_frame_exclusive": int(frames[hi - 1] + 1),
                "available_frames": hi - lo, "duration_seconds": (hi - lo) / FPS,
                "left_censored": lo == block_lo, "right_censored": hi == block_hi,
                "minimum_finite_margin": float(margin[lo:hi].min()),
                "maximum_finite_margin": float(margin[lo:hi].max()),
                "reviewed_fish_frames_within_episode": int(review[lo:hi].sum()),
            })
            union[lo:hi] = True
    events.sort(key=lambda item: (item["block"], item["start_frame"], item["kind"],
                                  item["affected_fish"] or []))
    scenes = []
    for index, event in enumerate(events):
        event["event_id"] = index
        if (not scenes or event["block"] != scenes[-1]["block"] or
                event["start_frame"] - scenes[-1]["risk_stop_frame_exclusive"] > GROUP_GAP_FRAMES):
            scenes.append({
                "scene_id": len(scenes), "block": event["block"], "event_ids": [],
                "risk_start_frame": event["start_frame"],
                "risk_stop_frame_exclusive": event["stop_frame_exclusive"],
            })
        scene = scenes[-1]
        scene["event_ids"].append(index)
        scene["risk_stop_frame_exclusive"] = max(
            scene["risk_stop_frame_exclusive"], event["stop_frame_exclusive"]
        )
    for scene in scenes:
        block_lo, block_hi = segments[scene["block"]]
        start = max(int(frames[block_lo]), scene["risk_start_frame"] - CONTEXT_FRAMES)
        stop = min(int(frames[block_hi - 1] + 1),
                   scene["risk_stop_frame_exclusive"] + CONTEXT_FRAMES)
        members = [events[index] for index in scene["event_ids"]]
        selected = (frames >= start) & (frames < stop)
        scene.update(
            context_start_frame=start, context_stop_frame_exclusive=stop,
            context_frames=int(selected.sum()), context_seconds=float(selected.sum() / FPS),
            raw_episode_count=len(members),
            known_affected_fish=sorted({fish for event in members
                                       for fish in event["affected_fish"] or []}),
            contains_unknown_affected_fish=any(event["affected_fish"] is None for event in members),
            risk_union_frames=int(union[selected].sum()),
            context_clipped_before=start > scene["risk_start_frame"] - CONTEXT_FRAMES,
            context_clipped_after=stop < scene["risk_stop_frame_exclusive"] + CONTEXT_FRAMES,
        )
    review_entries = [0, 0, 0]
    for lo, hi in segments:
        for fish in range(3):
            review_entries[fish] += len(_runs(review[:, fish], lo, hi))
    body_events = [event for event in events if event["kind"] == "body_unsupported"]
    return {
        "kind": "post_inference_uncertainty_inventory_not_identity_ground_truth",
        "frames": count, "fps": FPS,
        "available_blocks": [[int(frames[a]), int(frames[b - 1] + 1)] for a, b in segments],
        "settings": {
            "body_loss_threshold_frames": review_gap_frames,
            "finite_margin_threshold": MARGIN_THRESHOLD,
            "scene_max_gap_frames": GROUP_GAP_FRAMES, "context_each_side_frames": CONTEXT_FRAMES,
        },
        "counts": {
            "body_unsupported_episodes": len(body_events),
            "body_unsupported_episodes_per_fish": [
                sum(event["affected_fish"] == [fish] for event in body_events) for fish in range(3)
            ],
            "body_episodes_with_review_already_true": sum(
                event["review_already_true_before_episode"] is True for event in body_events
            ),
            "candidate_permutation_risk_episodes": len(events) - len(body_events),
            "raw_risk_episodes": len(events), "grouped_scenes": len(scenes),
            "raw_episode_frame_sum_includes_overlap": sum(
                event["available_frames"] for event in events
            ),
            "risk_union_frames": int(union.sum()), "risk_union_seconds": float(union.sum() / FPS),
            "scene_context_frames": sum(scene["context_frames"] for scene in scenes),
            "scene_context_seconds": sum(scene["context_seconds"] for scene in scenes),
            "review_true_runs_per_fish_not_risk_event_count": review_entries,
            "direct_pectoral_missing_episodes": len(direct_losses),
            "subthreshold_body_loss_episodes": len(short_losses),
        },
        "support": {
            "direct_pectoral_frames_per_fish": observed[:, :, 1].sum(axis=0).tolist(),
            "body_supported_frames_per_fish": body.sum(axis=0).tolist(),
            "mask_bridge_frames_per_fish": (body & ~observed[:, :, 1]).sum(axis=0).tolist(),
            "mask_positive_but_body_unsupported_frames_per_fish": (
                masks & ~body
            ).sum(axis=0).tolist(),
        },
        "margin": {
            "finite_risk_frames": int(margin_risk.sum()),
            "positive_infinity_no_compared_alternative_frames": int(np.isposinf(margin).sum()),
            "nan_frames": int(np.isnan(margin).sum()),
            "negative_infinity_frames": int(np.isneginf(margin).sum()),
        },
        "raw_events": events, "scenes": scenes,
        "direct_pectoral_missing_episodes": direct_losses,
        "subthreshold_body_loss_episodes": short_losses,
        "limitations": [
            "Repeated body losses are counted even if that fish was already under sticky review.",
            "Latent pose steps mix fitted and predicted positions; no separate forecast is stored.",
            "Scalar margins omit affected pairs and IDs; infinity is not confidence.",
            "Boundaries and frame gaps censor episodes and context; no unseen support is invented.",
            "Scene counts measure flagged material, not human time or correctable identities.",
            "Independent identity truth, event correctness and manual repair are not measured.",
        ],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if len({run.name for run in args.run}) != len(args.run):
        raise ValueError("run directory names must be distinct for separate output reports")
    reports, input_hashes = {}, {}
    for run in args.run:
        candidate_path, summary_path = run / "candidate.npz", run / "summary.json"
        record_path = run / "experiment.provenance.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["status"] != "ok" or summary["technical_completion"] is not True:
            raise ValueError(f"run is not technically complete: {run}")
        for name, path in (("candidate", candidate_path), ("summary", summary_path)):
            if record["outputs"][name]["sha256"] != _sha256(path):
                raise ValueError(f"completed {name} checksum mismatch: {run}")
        if summary["native_fps"] != FPS or summary["temporal_subsampling"] is not False:
            raise ValueError("review events require native 166 fps results")
        with np.load(candidate_path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in (
                "frame_idx", "pose_xyz", "observed", "body_supported", "mask_supported",
                "identity_review", "missing", "score_margin",
            )}
        report = extract_review_events(arrays, review_gap_frames=summary["tracker_config"][
            "review_gap_frames"
        ])
        report["source"] = {
            "run": str(run), "candidate_sha256": _sha256(candidate_path),
            "summary_sha256": _sha256(summary_path), "inference_config": summary["tracker_config"],
            "diagnostic_code_sha256": _sha256(__file__), "human_truth_loaded": False,
        }
        reports[run.name] = report
        input_hashes.update({str(path): _sha256(path) for path in
                             (candidate_path, summary_path, record_path)})
    comparison = {name: report["counts"] for name, report in reports.items()}
    if any(_sha256(path) != digest for path, digest in input_hashes.items()):
        raise ValueError("run output changed during diagnostics")
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("three_body_review_events", args.output, name="review_events") as record:
        for path, digest in input_hashes.items():
            record.input(path, path, sha256=digest)
        record.param("diagnostic_code_sha256", _sha256(__file__))
        record.param("human_truth_loaded", False)
        for name, report in {**reports, "comparison": comparison}.items():
            path = args.output / f"{name}.json"
            path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            record.output(name, path, sha256=_sha256(path))
    print(json.dumps({"output": str(args.output), "runs": comparison}))
    return comparison


if __name__ == "__main__":
    main()
