"""Compare original/directional backtraces at the same completed 66706 horizon."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import summarize_chest_candidate_replay as checks
from continuous_tracker import RECORD_FIELDS
from tracker_checkpoint import read_checkpoint

from medaka3d.provenance import provenance

REPO = Path(__file__).resolve().parents[2]
START, STOP = 66400, 66706


def backtrace(arrays):
    offsets = arrays["history_offsets"]
    state, rows = 0, []
    for t in range(len(offsets) - 2, -1, -1):
        if not 0 <= state < offsets[t + 1] - offsets[t]:
            raise ValueError("invalid original parent ancestry")
        row = int(offsets[t]) + state
        rows.append(row)
        state = int(arrays["history_parent"][row])
    rows = np.array(rows[::-1], np.int64)
    result = {k: arrays["history_" + k][rows].copy() for k in RECORD_FIELDS if k != "parent"}
    result["xyz"] = np.where(result["observed"][:, :, 1, None], result["pose_xyz"][:, :, 1], np.nan)
    result["accepted_xyz"] = np.where(result["identity_review"][:, :, None], np.nan, result["xyz"])
    result["body_xyz"] = np.where(
        result["body_supported"][:, :, None], result["pose_xyz"][:, :, 1], np.nan
    )
    return result


def stats(candidate, start, stop):
    frames = candidate["frame_idx"]
    keep = (frames >= start) & (frames < stop)
    observed, review = candidate["observed"][keep], candidate["identity_review"][keep]
    step = np.r_[
        np.full((1, 3), np.nan),
        np.linalg.norm(np.diff(candidate["pose_xyz"][:, :, 1], axis=0), axis=-1),
    ][keep]
    return dict(
        frames=int(keep.sum()),
        direct_node_support=observed.sum((0, 1)).tolist(),
        all_three_chest_frames=int(observed[:, :, 1].all(1).sum()),
        identity_review_fish_frames=int(review.sum()),
        accepted_chest_fish_frames=int((observed[:, :, 1] & ~review).sum()),
        chest_steps_gt10mm=int((step > 10).sum()),
        max_chest_step_mm=float(np.nanmax(step)),
        mean_fitting_mask_error=float(candidate["loss_parts"][keep, 4].mean()),
        selected_chest_recovery_frames=int(
            (candidate["proposal_kind"][keep] == "chest_recovery").sum()
        ),
    )


def main(original, branch, output):
    hashes = {str(Path(__file__).resolve()): checks.sha(__file__)}
    checks.verify(
        original, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
    )
    checks.verify(branch, "replay.provenance.json", "directional_chest_continuous_replay", hashes)
    traces = []
    for root in (original, branch):
        for filename in ("manifest.json", "metadata.json", "arrays.npz"):
            checks.require(
                str(root / "checkpoint" / filename) in hashes, "unsealed checkpoint component"
            )
        manifest = checks.read(root / "checkpoint/manifest.json")
        arrays, meta, _ = read_checkpoint(
            root / "checkpoint", expected_bindings=manifest["bindings"]
        )
        checks.require(
            meta["previous_frame"] == STOP - 1 and meta["initialize_count"] == 1,
            "different end horizon or lifecycle",
        )
        traces.append(backtrace(arrays))
    with np.load(branch / "candidate.npz", allow_pickle=False) as z:
        checks.require(
            set(z.files) == set(traces[1]) and len(z.files) == 21, "branch candidate schema differs"
        )
        for key in z.files:
            a, b = z[key], traces[1][key]
            exact = (
                np.array_equal(a, b)
                if a.dtype.kind == b.dtype.kind == "U"
                else checks.equal_bytes(a, b)
            )
            checks.require(exact, "independent backtrace disagrees with saved branch: " + key)
    old, new = traces
    checks.require(
        checks.equal_bytes(old["frame_idx"], new["frame_idx"]), "different native intervals"
    )
    result = dict(
        schema="directional_continuous_same_horizon_comparison_v1",
        status="ok",
        horizon_stop=STOP,
        native_interval=[START, STOP],
        original_306=stats(old, START, STOP),
        directional_306=stats(new, START, STOP),
        original_target50=stats(old, 66656, STOP),
        directional_target50=stats(new, 66656, STOP),
        new_backtrace_matches_all21_saved_arrays=True,
        independent_ID_truth=False,
        adopted=False,
        one_minute_directional_run_completed=False,
    )
    rows = []
    for i in np.flatnonzero(new["frame_idx"] >= START):
        row = dict(frame_idx=int(new["frame_idx"][i]))
        for label, candidate in (("original", old), ("directional", new)):
            counts = candidate["observed"][i].sum(0)
            for node, count in zip(("mouth", "chest", "tail"), counts, strict=True):
                row[label + "_" + node + "_supported"] = int(count)
            row[label + "_review_fish"] = int(candidate["identity_review"][i].sum())
            row[label + "_proposal"] = str(candidate["proposal_kind"][i])
            row[label + "_mask_fit_error"] = float(candidate["loss_parts"][i, 4])
            row[label + "_max_chest_step_mm"] = float(
                np.linalg.norm(
                    candidate["pose_xyz"][i, :, 1] - candidate["pose_xyz"][i - 1, :, 1], axis=-1
                ).max()
            )
        rows.append(row)
    checks.require(len(rows) == 306, "incomplete suffix")
    output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "directional_continuous_same_horizon_comparison", output, name="comparison"
    ) as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        np.savez_compressed(output / "original_same_horizon_candidate.npz", **old)
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (output / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=checks.sha(path))
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "bound input changed")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--branch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    target = args.output.resolve()
    checks.require(
        target.is_relative_to(REPO / "work") and not target.exists(),
        "new workspace output required",
    )
    main(args.original.resolve(), args.branch.resolve(), target)
