"""Paired arithmetic on fixed isotropic/directional candidates, without fitting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import summarize_chest_candidate_replay as evidence

from medaka3d.provenance import provenance

PARTS = ("node", "shape", "chest_motion", "orientation_motion", "repulsion", "silhouette")
require, sha, read = evidence.require, evidence.sha, evidence.read


def main(old, new, output):
    hashes = {str(Path(__file__).resolve()): sha(__file__)}
    evidence.verify(
        old, "fit.provenance.json", "chest_recovery_same_objective_local_CPU_fit", hashes
    )
    seal = evidence.verify(new, "fit.provenance.json", "directional_chest_local_CPU_fit", hashes)
    require(
        seal["inputs"][str(old / "fit.provenance.json")]["sha256"]
        == hashes[str(old / "fit.provenance.json")],
        "different original fit",
    )
    frame_rows, candidate_rows = [], []
    for info in read(new / "frames.json"):
        native = info["frame_idx"]
        arrays = []
        for root in (old, new):
            path = root / f"frame_{native:06d}.npz"
            require(str(path) in hashes and sha(path) == hashes[str(path)], "unbound frame")
            with np.load(path, allow_pickle=False) as z:
                arrays.append({k: z[k].copy() for k in z.files})
        a, b = arrays
        for key in (
            "initial_pose_xyz",
            "predicted_pose_xyz",
            "proposal_plan",
            "proposal_parent",
            "proposal_permutation",
            "new_candidate",
            "parent_identity_review",
        ):
            require(evidence.equal_bytes(a[key], b[key]), "paired inputs differ: " + key)
        ia, ib = int(a["selected_indices"][0]), int(b["selected_indices"][0])
        old_best_in_new = int(b["original_selected_indices"][0])
        all_support = b["after_strict_observed"][:, :, 1].all(1)
        require(
            np.allclose(b["final_weighted_six_parts"].sum(1), b["local_cost"], atol=1e-10, rtol=0),
            "part sum differs",
        )
        row = dict(
            frame_idx=native,
            old_best=ia,
            new_best=ib,
            old_best_permutation=a["proposal_permutation"][ia].tolist(),
            new_best_permutation=b["proposal_permutation"][ib].tolist(),
            best_proposal_changed=ia != ib,
            old_best_observed_nodes=a["after_strict_observed"][ia].sum(0).tolist(),
            new_best_observed_nodes=b["after_strict_observed"][ib].sum(0).tolist(),
            old_best_mask_support=a["mask_supported"][ia].tolist(),
            new_best_mask_support=b["mask_supported"][ib].tolist(),
            selected_new=int(b["new_candidate"][b["selected_indices"]].sum()),
            selected_all_three=int(all_support[b["selected_indices"]].sum()),
        )
        frame_rows.append(row)
        for idx in np.flatnonzero(b["new_candidate"]):
            part_gap = (
                b["final_weighted_six_parts"][idx] - b["final_weighted_six_parts"][old_best_in_new]
            )
            part_during = b["final_weighted_six_parts"][idx] - b["initial_weighted_six_parts"][idx]
            c = dict(
                frame_idx=native,
                candidate=int(idx),
                original_best_in_directional_objective=old_best_in_new,
                parent=int(b["proposal_parent"][idx]),
                selected=bool(np.isin(idx, b["selected_indices"])),
                stable_rank=int(np.flatnonzero(b["stable_path_order"] == idx)[0]) + 1,
                all_three_chests=bool(all_support[idx]),
                mouth_support_before=int(b["before_strict_observed"][idx, :, 0].sum()),
                mouth_support_after=int(b["after_strict_observed"][idx, :, 0].sum()),
                tail_support_before=int(b["before_strict_observed"][idx, :, 2].sum()),
                tail_support_after=int(b["after_strict_observed"][idx, :, 2].sum()),
                mask_supported_fish=int(b["mask_supported"][idx].sum()),
                local_cost_gap=float(b["local_cost"][idx] - b["local_cost"][old_best_in_new]),
                path_cost_gap=float(b["path_cost"][idx] - b["path_cost"][old_best_in_new]),
            )
            c.update({"gap_" + k: float(v) for k, v in zip(PARTS, part_gap, strict=True)})
            c.update({"during_" + k: float(v) for k, v in zip(PARTS, part_during, strict=True)})
            candidate_rows.append(c)
    require(len(frame_rows) == 20 and len(candidate_rows) == 31, "fixed scope incomplete")
    result = dict(
        schema="directional_chest_fit_analysis_v1",
        status="ok",
        frames=20,
        new_candidates=31,
        supported_new_candidates=sum(r["all_three_chests"] for r in candidate_rows),
        selected_new_candidates=sum(r["selected"] for r in candidate_rows),
        frames_best_proposal_changed=sum(r["best_proposal_changed"] for r in frame_rows),
        old_best_node_support=np.sum(
            [r["old_best_observed_nodes"] for r in frame_rows], axis=0
        ).tolist(),
        new_best_node_support=np.sum(
            [r["new_best_observed_nodes"] for r in frame_rows], axis=0
        ).tolist(),
        frames_explicit_permutation_old=sum(
            r["old_best_permutation"] != [0, 1, 2] for r in frame_rows
        ),
        frames_explicit_permutation_new=sum(
            r["new_best_permutation"] != [0, 1, 2] for r in frame_rows
        ),
        new_candidates_mouth_support_lost=sum(
            r["mouth_support_after"] < r["mouth_support_before"] for r in candidate_rows
        ),
        new_candidates_tail_support_lost=sum(
            r["tail_support_after"] < r["tail_support_before"] for r in candidate_rows
        ),
        mean_gap_parts={k: float(np.mean([r["gap_" + k] for r in candidate_rows])) for k in PARTS},
        mean_during_parts={
            k: float(np.mean([r["during_" + k] for r in candidate_rows])) for k in PARTS
        },
        new_candidate_rank_min=min(r["stable_rank"] for r in candidate_rows),
        new_candidate_rank_max=max(r["stable_rank"] for r in candidate_rows),
        independent_ID_truth=False,
        adopted=False,
        tracking_performed=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    with provenance("directional_chest_fit_analysis", output, name="analysis") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for name, value in (("summary.json", result), ("frames.json", frame_rows)):
            (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
        with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
            writer.writeheader()
            writer.writerows(candidate_rows)
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=sha(path))
        for path, digest in hashes.items():
            require(sha(path) == digest, "input changed")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    target = args.output.resolve()
    require(
        target.is_relative_to(Path(__file__).resolve().parents[2] / "work") and not target.exists(),
        "new workspace output required",
    )
    main(args.old.resolve(), args.new.resolve(), target)
