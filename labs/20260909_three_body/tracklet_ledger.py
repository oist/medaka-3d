"""Risk-bounded native tracklets for future review; no repair or identity truth."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path

import numpy as np
from review_events import MARGIN_THRESHOLD, extract_review_events

FIXED_REVIEW_GAP_FRAMES = 4


def build_tracklet_ledger(result: dict, *, tracker_config: Mapping) -> dict:
    """Preserve repeated risks, unknown review origins, gaps, and raw review flags.

    Tracklets are maximal available native intervals outside risk intervals and
    unknown boundaries. They can contain short missing positions or mask-only
    support: tracklet membership creates no coordinates or observation support.
    Every preceding relevant risk remains unresolved; this module cannot clear it.
    """
    if not isinstance(tracker_config, Mapping):
        raise ValueError("tracker_config must bind the completed run's configuration")
    threshold = tracker_config.get("review_gap_frames")
    if type(threshold) is not int or threshold != FIXED_REVIEW_GAP_FRAMES:
        raise ValueError("this ledger requires explicit review_gap_frames=4")
    margin = tracker_config.get("ambiguity_margin")
    if type(margin) not in (int, float) or not np.isfinite(margin) or margin != MARGIN_THRESHOLD:
        raise ValueError("this ledger requires explicit ambiguity_margin=1.0")
    frames = np.asarray(result["frame_idx"])
    if frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu":
        raise ValueError("nonempty native integer frames required")
    inventory = extract_review_events(result, review_gap_frames=threshold)
    review = np.asarray(result["identity_review"])
    observed = np.asarray(result["observed"])
    body = np.asarray(result["body_supported"])
    mask = np.asarray(result["mask_supported"])
    count = len(frames)
    edges = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1, count]
    blocks = [(int(a), int(b)) for a, b in pairwise(edges)]
    events = []
    excluded = np.zeros((count, 3), bool)
    boundaries = np.zeros((count, 3), bool)

    for source in inventory["raw_events"]:
        affected = source["affected_fish"]
        conservative = list(range(3)) if affected is None else list(affected)
        start, stop = source["start_frame"], source["stop_frame_exclusive"]
        active = (frames >= start) & (frames < stop)
        excluded[np.ix_(active, conservative)] = True
        event = {
            "kind": source["kind"],
            "representation": "available_interval",
            "start_frame": int(start),
            "stop_frame_exclusive": int(stop),
            "affected_slots": conservative,
            "original_affected_slots": affected,
            "affected_scope": "all_three_conservative_unknown_pair"
            if affected is None
            else "recorded_affected_slots_not_biological_truth",
            "block": source["block"],
            "left_censored": source["left_censored"],
            "right_censored": source["right_censored"],
            "recorded_episode_onset_available": not source["left_censored"],
            "recorded_episode_end_available": not source["right_censored"],
            "known_risk_start_frame": None if source["left_censored"] else int(start),
            "known_risk_stop_frame_exclusive": None if source["right_censored"] else int(stop),
            "identity_switch_time_known": False,
            "resolved": False,
            "source_event": copy.deepcopy(source),
        }
        events.append(event)

    def boundary(index, affected, kind, block, **metadata):
        boundaries[index, affected] = True
        events.append(
            {
                "kind": kind,
                "representation": "unknown_extent_boundary",
                "boundary_frame": int(frames[index]),
                "start_frame": int(frames[index]),
                "stop_frame_exclusive": None,
                "affected_slots": list(affected),
                "block": block,
                "cause_onset_frame": None,
                "cause_end_frame": None,
                "identity_switch_time_known": False,
                "resolved": False,
                "meaning": (
                    "uncertainty recorded here; cause time/range unknown, not a one-frame risk"
                ),
                **metadata,
            }
        )

    clearings = []
    for block, (lo, hi) in enumerate(blocks):
        if block:
            boundary(
                lo,
                [0, 1, 2],
                "source_frame_gap",
                block,
                previous_available_frame=int(frames[lo - 1]),
                unavailable_source_interval=[int(frames[lo - 1] + 1), int(frames[lo])],
                raw_review_at_right=review[lo].tolist(),
                history_across_gap="unknown",
            )
        elif review[lo].any():
            boundary(
                lo,
                np.flatnonzero(review[lo]).tolist(),
                "initial_review_history_unknown",
                block,
                raw_review_at_entry=review[lo].tolist(),
            )
        for index in range(lo + 1, hi):
            risen = review[index] & ~review[index - 1]
            unexplained = risen & ~excluded[index]
            if unexplained.any():
                boundary(
                    index,
                    np.flatnonzero(unexplained).tolist(),
                    "review_onset_without_recorded_risk",
                    block,
                    raw_review_before=review[index - 1].tolist(),
                    raw_review_after=review[index].tolist(),
                )
            cleared = np.flatnonzero(~review[index] & review[index - 1])
            if len(cleared):
                clearings.append(
                    {
                        "frame": int(frames[index]),
                        "slots": cleared.tolist(),
                        "action": "raw change preserved; ledger uncertainty not cleared",
                    }
                )

    events.sort(key=lambda e: (e["start_frame"], e["kind"], e["affected_slots"]))
    for index, event in enumerate(events):
        event["risk_id"] = f"risk_{index:04d}"
    tracklets, coverage = [], np.zeros((count, 3), int)
    for block, (block_lo, block_hi) in enumerate(blocks):
        for slot in range(3):
            index = block_lo
            while index < block_hi:
                if excluded[index, slot]:
                    index += 1
                    continue
                lo = index
                index += 1
                while (
                    index < block_hi and not excluded[index, slot] and not boundaries[index, slot]
                ):
                    index += 1
                hi = index
                prior = [
                    event["risk_id"]
                    for event in events
                    if slot in event["affected_slots"] and event["start_frame"] <= frames[lo]
                ]
                state = (
                    "unresolved_after_recorded_risk_or_unknown_boundary"
                    if prior
                    else ("automatic_before_recorded_risk_not_verified_identity")
                )
                item = {
                    "tracklet_id": f"block{block}_slot{slot}_{int(frames[lo])}",
                    "slot": slot,
                    "block": block,
                    "start_frame": int(frames[lo]),
                    "stop_frame_exclusive": int(frames[hi - 1] + 1),
                    "native_frames": hi - lo,
                    "identity_state": state,
                    "unresolved_risk_ids": prior,
                    "raw_review_at_start": bool(review[lo, slot]),
                    "raw_review_true_frames": int(review[lo:hi, slot].sum()),
                    "direct_node_frames": dict(
                        zip(
                            ("mouth", "pectoral", "tail"),
                            observed[lo:hi, slot].sum(axis=0).tolist(),
                            strict=True,
                        )
                    ),
                    "body_supported_frames": int(body[lo:hi, slot].sum()),
                    "mask_only_body_frames": int(
                        (body[lo:hi, slot] & ~observed[lo:hi, slot, 1]).sum()
                    ),
                    "mask_positive_body_unsupported_frames": int(
                        (mask[lo:hi, slot] & ~body[lo:hi, slot]).sum()
                    ),
                    "missing_direct_pectoral_frames": int((~observed[lo:hi, slot, 1]).sum()),
                    "left_at_available_block_boundary": lo == block_lo,
                    "right_at_available_block_boundary": hi == block_hi,
                    "geometry_or_observation_acceptance_changed": False,
                    "human_confirmed": False,
                    "global_identity_link_verified": False,
                }
                tracklets.append(item)
                coverage[lo:hi, slot] += 1
    if not np.array_equal(coverage, (~excluded).astype(int)):
        raise AssertionError("tracklets must cover every non-risk native slot-frame exactly once")
    unresolved_frames = [
        sum(t["native_frames"] for t in tracklets if t["slot"] == s and t["unresolved_risk_ids"])
        for s in range(3)
    ]
    return {
        "schema": "three_body_risk_tracklet_ledger_v1",
        "review_gap_frames": threshold,
        "ambiguity_margin": margin,
        "scope": "post-inference risk bookkeeping; no human correction or identity truth",
        "frame_idx": frames.tolist(),
        "raw_identity_review": review.tolist(),
        "raw_review_clearing_observations": clearings,
        "available_blocks": inventory["available_blocks"],
        "frozen_inventory_counts": inventory["counts"],
        "risk_events": events,
        "tracklets": tracklets,
        "counts": {
            "native_frames": count,
            "available_slot_frames": count * 3,
            "interval_risks": len(inventory["raw_events"]),
            "unknown_extent_boundaries": sum(
                e["representation"] == "unknown_extent_boundary" for e in events
            ),
            "excluded_risk_slot_frames_per_slot": excluded.sum(axis=0).tolist(),
            "tracklets_per_slot": [sum(t["slot"] == s for t in tracklets) for s in range(3)],
            "unresolved_tracklet_frames_per_slot": unresolved_frames,
            "raw_review_true_frames_per_slot": review.sum(axis=0).tolist(),
        },
        "raw_review_preserved": True,
        "coordinates_emitted": False,
        "human_labels_consumed": False,
        "identities_repaired": False,
        "limitations": [
            "Tracklet membership is not continuous observed position or certified identity.",
            "Pre-risk automatic status means no recorded risk, not a correct ID guarantee.",
            "All previous relevant risks stay unresolved; this module has no clearing operation.",
            "Unknown boundaries report flag/gap times, not the true onset or duration of a swap.",
            "Mask-only support and short position gaps remain distinct from direct nodes.",
            "Scene/probe selection and global biological naming are outside this module.",
            "The frozen inventory may omit hidden switches despite good support or large margins.",
        ],
    }


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if len({p.name for p in args.run}) != len(args.run):
        raise ValueError("run folder names must be unique")
    from medaka3d.provenance import provenance

    lab = Path(__file__).resolve().parent
    repo = lab.parents[1]
    code = [
        Path(__file__),
        lab / "review_events.py",
        lab / "HUMAN_CORRECTION_PROTOCOL.md",
        repo / "src/medaka3d/provenance.py",
    ]
    hashes = {str(p): sha(p) for p in code}
    ledgers = {}
    for root in args.run:
        files = {
            name: root / name
            for name in ("candidate.npz", "summary.json", "experiment.provenance.json")
        }
        hashes.update({str(p): sha(p) for p in files.values()})
        summary = json.loads(files["summary.json"].read_text())
        record = json.loads(files["experiment.provenance.json"].read_text())
        if record.get("status") != "ok" or summary.get("technical_completion") is not True:
            raise ValueError("ledger requires a completed run")
        for name in ("candidate", "summary"):
            path = files[f"{name}.npz" if name == "candidate" else "summary.json"]
            if sha(path) != record["outputs"][name]["sha256"]:
                raise ValueError("completed output digest mismatch")
        if summary["native_fps"] != 166 or summary["temporal_subsampling"] is not False:
            raise ValueError("ledger requires native 166-fps output")
        with np.load(files["candidate.npz"], allow_pickle=False) as saved:
            arrays = {
                key: saved[key]
                for key in (
                    "frame_idx",
                    "pose_xyz",
                    "observed",
                    "body_supported",
                    "mask_supported",
                    "identity_review",
                    "missing",
                    "score_margin",
                )
            }
        if not len(arrays["frame_idx"]):
            raise ValueError("completed output must contain native frames")
        if len(arrays["frame_idx"]) != summary["returned_frames"] or summary["frames"] != [
            int(arrays["frame_idx"][0]),
            int(arrays["frame_idx"][-1] + 1),
        ]:
            raise ValueError("saved native frames differ from completed summary")
        ledger = build_tracklet_ledger(arrays, tracker_config=summary["tracker_config"])
        ledger["source"] = {
            "run": str(root),
            "candidate_sha256": sha(files["candidate.npz"]),
            "summary_sha256": sha(files["summary.json"]),
        }
        ledgers[root.name] = ledger
    if any(sha(p) != digest for p, digest in hashes.items()):
        raise ValueError("ledger input/source changed while reading")
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("risk_tracklet_ledger", args.output, name="ledger") as record:
        for p, digest in hashes.items():
            record.input(p, p, sha256=digest)
        for p in code:
            target = args.output / "source_snapshot" / p.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        for name, ledger in ledgers.items():
            path = args.output / f"{name}.json"
            path.write_text(json.dumps(ledger, indent=2, allow_nan=False), encoding="utf-8")
            record.output(name, path, sha256=sha(path))
        if any(sha(p) != digest for p, digest in hashes.items()):
            raise ValueError("ledger input/source changed while saving")
    print(json.dumps({name: ledger["counts"] for name, ledger in ledgers.items()}))
    return ledgers


if __name__ == "__main__":
    main()
