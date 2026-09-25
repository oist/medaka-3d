"""Measure whether a recovery displacement follows an unchanged single-view ray."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import summarize_chest_candidate_replay as checks

from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

REPO = Path(__file__).resolve().parents[2]


def main(fit, replay, recovery, output):
    hashes = {str(Path(__file__).resolve()): checks.sha(__file__)}
    fit_seal = checks.verify(
        fit, "fit.provenance.json", "chest_recovery_same_objective_local_CPU_fit", hashes
    )
    checks.verify(replay, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes)
    checks.verify(
        recovery, "probe.provenance.json", "chest_recovery_counterfactual_proposals", hashes
    )
    for root, filename in ((replay, "replay.provenance.json"), (recovery, "probe.provenance.json")):
        path = str(root / filename)
        checks.require(
            fit_seal["inputs"][path]["sha256"] == hashes[path], "different input artifact"
        )
    frames = checks.read(fit / "frames.json")
    rows = []
    for frame in frames:
        native = frame["frame_idx"]
        paths = [
            fit / f"frame_{native:06d}.npz",
            recovery / f"frame_{native:06d}.npz",
            replay / "diagnostics" / f"frame_{native:06d}" / "arrays.npz",
        ]
        data = []
        for path in paths:
            checks.require(str(path) in hashes, "unsealed numerical input")
            with np.load(path, allow_pickle=False) as z:
                data.append({key: z[key].copy() for key in z.files})
        fitted, counter, original = data
        for candidate, fish in zip(
            np.flatnonzero(fitted["new_candidate"]), counter["target_slot"], strict=True
        ):
            fish = int(fish)
            old_source = fitted["before_strict_sources"][0, fish, :, 1]
            new_source = fitted["before_strict_sources"][candidate, fish, :, 1]
            held = np.flatnonzero(old_source >= 0)
            shared = np.flatnonzero((old_source >= 0) & (new_source == old_source))
            delta = (
                fitted["initial_pose_xyz"][candidate, fish, 1]
                - fitted["predicted_pose_xyz"][candidate, fish, 1]
            )
            row = dict(
                frame_idx=native,
                candidate=int(candidate),
                target_slot=fish,
                original_predict_supported_views=len(held),
                shared_source_views=len(shared),
                displacement_mm=float(np.linalg.norm(delta)),
                old_sources=old_source.tolist(),
                new_sources=new_source.tolist(),
                shared_view=None,
                parallel_mm=None,
                perpendicular_mm=None,
            )
            if len(held) == 1 and np.array_equal(held, shared):
                view = int(held[0])
                direction = original[VIEWS[view] + "_ray_directions"][old_source[view], 1]
                checks.require(
                    np.isclose(np.linalg.norm(direction), 1.0, rtol=0, atol=1e-10),
                    "ray is not a unit direction",
                )
                signed = float(np.dot(delta, direction))
                perpendicular = float(np.linalg.norm(delta - signed * direction))
                checks.require(
                    np.isclose(
                        signed * signed + perpendicular * perpendicular,
                        np.dot(delta, delta),
                        rtol=0,
                        atol=1e-8,
                    ),
                    "orthogonal decomposition failed",
                )
                row.update(
                    shared_view=VIEWS[view], parallel_mm=abs(signed), perpendicular_mm=perpendicular
                )
            rows.append(row)
    qualified = [r for r in rows if r["parallel_mm"] is not None]

    def stats(key):
        values = np.asarray([r[key] for r in qualified], float)
        return (
            dict(
                minimum=float(values.min()),
                median=float(np.median(values)),
                maximum=float(values.max()),
            )
            if len(values)
            else None
        )

    summary = dict(
        schema="chest_recovery_shared_ray_axis_v1",
        candidates=len(rows),
        same_single_source_candidates=len(qualified),
        source_views_histogram={
            str(n): sum(r["original_predict_supported_views"] == n for r in rows) for n in range(4)
        },
        no_shared_source_candidates=sum(r["shared_source_views"] == 0 for r in rows),
        parallel_mm=stats("parallel_mm"),
        perpendicular_mm=stats("perpendicular_mm"),
        biological_ID_truth=False,
        additional_fit=False,
        limitation=("A shared detection ray is an observation compatibility check, "
                    "not identity proof."),
    )
    output.mkdir(parents=True, exist_ok=False)
    with provenance("chest_recovery_shared_ray_axis", output, name="analysis") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (output / "candidates.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=checks.sha(path))
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "input changed during ray analysis")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    checks.require(
        target.is_relative_to(REPO / "work") and not target.exists(), "new output required"
    )
    main(args.fit.resolve(), args.replay.resolve(), args.recovery.resolve(), target)
