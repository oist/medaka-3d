"""CPU counterfactual: expose existing body recovery to persistent chest loss.

No tracking, fitting, candidate replacement, or identity decision is performed.
The gate is fixed before running: original first parent body missing all zero,
chest missing >=4, and current predicted chest not directly supported. Use the
unchanged node pool/recovery helper with keep=3, original shape and bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import summarize_chest_candidate_replay as summary_tools

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]
from chest_candidate_diagnostics import support_snapshot  # noqa: E402
from node_geometry import (  # noqa: E402
    RayObservations,
    assignment_plans,
    node_pools,
    recovery_poses,
)
from run_experiment import CALIBRATION_DIR  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

require, sha, read = summary_tools.require, summary_tools.sha, summary_tools.read


def main(root, output):
    hashes = {str(Path(__file__).resolve()): sha(__file__)}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(REPO):
                hashes[str(path)] = sha(path)
    run = summary_tools.verify(
        root, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
    )
    status = read(root / "summary.json")
    require(
        status["replay_complete"] and status["exact_frames"] == 306,
        "completed exact original replay required",
    )
    plan = read(root / "registration/plan.json")
    binding = plan["original_binding"]

    def bound(path):
        path = Path(path).resolve()
        item = run["inputs"][str(path)]
        require(sha(path) == item["sha256"], "replay input changed")
        hashes[str(path)] = item["sha256"]
        return path

    for filename in binding["code_sha256"]:
        bound(filename)
    calibration = Path(binding["inputs"]["seed"]["input_root"]) / CALIBRATION_DIR
    for path in calibration.iterdir():
        if path.is_file():
            bound(path)
    seed_root = Path(binding["inputs"]["seed"]["registration_path"]).parent
    initial = np.load(bound(seed_root / "initial_pose_xyz.npy"), allow_pickle=False)
    lengths = np.linalg.norm(np.diff(initial, axis=1), axis=-1)
    rig = load_rig(calibration)
    lower, upper = np.asarray([0.0, 0.0, -6.0]), np.asarray([192.0, 192.0, 186.0])
    cfg = plan["original_context"]["tracker_config"]
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    with provenance("chest_recovery_counterfactual_proposals", output, name="probe") as record:
        record.param(
            "fixed_rule",
            dict(
                first_parent_only=True,
                original_body_missing_all_zero=True,
                minimum_chest_missing_frames=4,
                current_chest_not_directly_supported=True,
                original_helpers_unchanged=True,
                keep=3,
                additional_fit=False,
            ),
        )
        for native in range(66656, 66706):
            folder = root / "diagnostics" / f"frame_{native:06d}"
            require(
                str(folder / "arrays.npz") in hashes and str(folder / "metadata.json") in hashes,
                "unsealed source diagnostic",
            )
            with np.load(folder / "arrays.npz", allow_pickle=False) as z:
                a = {key: z[key] for key in z.files}
            meta = read(folder / "metadata.json")
            require(
                meta["proposal_kind"][0] == "predict"
                and a["proposal_parent"][0] == 0
                and np.array_equal(a["proposal_permutation"][0], [0, 1, 2]),
                "original first-parent prediction differs",
            )
            age = a["parent_node_missing"][0, :, 1]
            eligible = (age >= 4) & ~a["generated_strict_observed"][0, :, 1]
            original_gate_closed = not a["parent_missing"][0].any()
            enabled = bool(original_gate_closed and eligible.any())
            row = dict(
                frame_idx=native,
                original_body_gate_closed=original_gate_closed,
                eligible_slots=np.flatnonzero(eligible).tolist(),
                enabled=enabled,
                returned_candidates=0,
                all_three_chests_supported_candidates=0,
            )
            if enabled:
                rays = RayObservations.from_points(rig, {v: a[v + "_points"] for v in VIEWS})
                for v, view in enumerate(VIEWS):
                    require(
                        summary_tools.equal_bytes(rays.origins[v], a[view + "_ray_origins"])
                        and summary_tools.equal_bytes(
                            rays.directions[v], a[view + "_ray_directions"]
                        ),
                        "fresh calibrated rays differ from exact replay",
                    )
                pools = node_pools(rig, rays, lower, upper)
                missing = np.where(eligible, age, 0)
                candidates = recovery_poses(
                    a["proposal_pose"][0].copy(),
                    missing,
                    lengths,
                    pools,
                    a["proposal_plan"][0].copy(),
                    lower,
                    upper,
                    keep=3,
                )
                row["node_pool_counts"] = [len(pool) for pool in pools]
                row["returned_candidates"] = len(candidates)
                if candidates:
                    poses = np.stack([c[1] for c in candidates])
                    plans = np.stack(
                        [
                            assignment_plans(
                                pose, rays, cfg["ray_sigma_mm"], cfg["missing_cost"], keep=1
                            )[0][1]
                            for pose in poses
                        ]
                    )
                    support = support_snapshot(poses, plans, rays, rig)
                    require(not support["plan_source_collision"].any(), "source reuse in proposal")
                    all_chests = support["strict_observed"][:, :, 1].all(1)
                    row["all_three_chests_supported_candidates"] = int(all_chests.sum())
                    target = np.asarray([c[2] for c in candidates], np.int64)
                    row["target_chest_supported_candidates"] = int(
                        support["strict_observed"][np.arange(len(target)), target, 1].sum()
                    )
                    path = output / f"frame_{native:06d}.npz"
                    np.savez_compressed(
                        path,
                        pose_xyz=poses,
                        plan=plans,
                        target_slot=target,
                        helper_cost=np.asarray([c[0] for c in candidates]),
                        **support,
                    )
                    record.output(path.name, path, sha256=sha(path))
            rows.append(row)
        result = dict(
            schema="chest_recovery_counterfactual_proposals_v1",
            frames=len(rows),
            enabled_frames=sum(r["enabled"] for r in rows),
            frames_with_candidates=sum(r["returned_candidates"] > 0 for r in rows),
            frames_with_all_three_supported=sum(
                r["all_three_chests_supported_candidates"] > 0 for r in rows
            ),
            candidates=sum(r["returned_candidates"] for r in rows),
            all_three_supported_candidates=sum(
                r["all_three_chests_supported_candidates"] for r in rows
            ),
            tracking_performed=False,
            fitting_performed=False,
            masks_scored=False,
            biological_ID_truth=False,
            adopted=False,
            limitation=("Geometric availability only; image fit, motion cost "
                        "and identity are untested."),
        )
        for name, data in (("summary.json", result), ("frames.json", rows)):
            path = output / name
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            record.output(name, path, sha256=sha(path))
        for module in tuple(sys.modules.values()):
            name = getattr(module, "__file__", None)
            if name:
                path = Path(name).resolve()
                if path.suffix == ".py" and path.is_relative_to(REPO):
                    require(str(path) in hashes, "late unregistered local source import")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
            require(sha(path) == digest, "evidence changed during proposal probe")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    require(target.is_relative_to(REPO / "work") and not target.exists(), "new output required")
    main(args.replay.resolve(), target)
