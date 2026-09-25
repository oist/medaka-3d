"""Compare fixed same-horizon tracks on unused pixels of all 50 development frames."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import fit_chest_recovery_candidates as base
import numpy as np
import torch
from prepare_sequence import decode_frame_masks
from renderer import RefractiveFishRenderer, RendererConfig
from score_temporal_disjoint_pixels import COUNTS, SEED_OFFSET, unused_samples

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks = base.evidence_tools
REPO = Path(__file__).resolve().parents[2]


def main(original, branch, comparison, output):
    code = base.local_code()
    hashes = dict(code)
    replay = checks.verify(
        original, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
    )
    checks.verify(branch, "replay.provenance.json", "directional_chest_continuous_replay", hashes)
    seal = checks.verify(
        comparison,
        "comparison.provenance.json",
        "directional_continuous_same_horizon_comparison",
        hashes,
    )
    for root in (original, branch):
        p = str(root / "replay.provenance.json")
        checks.require(
            seal["inputs"][p]["sha256"] == hashes[p], "comparison used different trajectories"
        )
    candidates = []
    for path in (comparison / "original_same_horizon_candidate.npz", branch / "candidate.npz"):
        candidates.append(base.load_arrays(path, hashes))
    checks.require(
        checks.equal_bytes(candidates[0]["frame_idx"], candidates[1]["frame_idx"]),
        "native frames differ",
    )
    registration = checks.read(original / "registration/plan.json")
    calibration = (
        Path(registration["original_binding"]["inputs"]["seed"]["input_root"])
        / base.CALIBRATION_DIR
    )
    for p in calibration.iterdir():
        if p.is_file():
            expected = replay["inputs"][str(p)]["sha256"]
            checks.require(checks.sha(p) == expected, "calibration changed")
            hashes[str(p)] = expected
    rig, packed = load_rig(calibration), {}
    for view in VIEWS:
        paths = [
            Path(item["path"])
            for item in replay["inputs"].values()
            if "f066400-068060" in item["path"] and Path(item["path"]).name == view + ".npz"
        ]
        checks.require(len(paths) == 1, "unique original chunk required")
        p = paths[0]
        hashes[str(p)] = replay["inputs"][str(p)]["sha256"]
        packed[view] = base.load_arrays(p, hashes)
        checks.require(
            np.array_equal(packed[view]["image_shape_hw"], [1200, 1200]), "native dimensions differ"
        )
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    rows, membership = [], []
    with provenance("directional_continuous_unused_pixels", output, name="evaluation") as record:
        record.param(
            "sampling",
            dict(counts=COUNTS, seed_offset=SEED_OFFSET, fitted_coordinates_excluded=True),
        )
        record.param("native_interval", [66656, 66706])
        record.param("raw_video_decoded", False)
        record.param("temporal_holdout", False)
        for native in range(66656, 66706):
            old, new = [
                base.load_arrays(
                    root / "diagnostics" / f"frame_{native:06d}" / "arrays.npz", hashes
                )
                for root in (original, branch)
            ]
            for key in (
                "fit_target",
                "fit_weights",
                "fit_foreground",
                *[v + "_fit_pixels" for v in VIEWS],
            ):
                checks.require(
                    checks.equal_bytes(old[key], new[key]),
                    "original/branch fitting evidence differs: " + key,
                )
            masks = {}
            for view in VIEWS:
                data = packed[view]
                index = int(np.searchsorted(data["frame_idx"], native))
                checks.require(data["frame_idx"][index] == native, "wrong mask frame")
                value = decode_frame_masks(data, index)
                checks.require(
                    hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    == bytes(data["frame_mask_sha256"][index]).decode(),
                    "native decoded mask mismatch",
                )
                masks[view] = value if data["view_available"][index] else None
            frame = SimpleNamespace(
                frame_idx=native,
                masks=masks,
                images_bgr={v: SimpleNamespace(shape=(1200, 1200, 3)) for v in VIEWS},
            )
            pixels, target, weights, overlap, audit = unused_samples(
                frame, np.stack([old[v + "_fit_pixels"] for v in VIEWS])
            )
            checks.require(not overlap.any(), "evaluation reused fit pixels")
            membership.extend(audit)
            index = int(np.searchsorted(candidates[0]["frame_idx"], native))
            checks.require(candidates[0]["frame_idx"][index] == native, "trajectory frame mismatch")
            pose = np.stack([c["pose_xyz"][index] for c in candidates])
            renderer = RefractiveFishRenderer(
                rig,
                {v: pixels[i] for i, v in enumerate(VIEWS)},
                RendererConfig(width_ratio=0.06),
                device="cpu",
                dtype=torch.float64,
            )
            with torch.no_grad():
                alpha = renderer.render(pose).union_alpha.numpy()
            valid = renderer.ray_valid.numpy()
            error = ((alpha - target) ** 2 * weights * valid).sum(-1)
            rows.append(
                dict(
                    frame_idx=native,
                    original_error=error[0].tolist(),
                    directional_error=error[1].tolist(),
                    difference_by_view=(error[1] - error[0]).tolist(),
                    mean_difference=float((error[1] - error[0]).mean()),
                )
            )
            p = output / f"frame_{native:06d}.npz"
            np.savez_compressed(
                p,
                poses=pose,
                pixels=pixels,
                target=target,
                weights=weights,
                ray_valid=valid,
                union_alpha=alpha,
                per_view_error=error,
                fitting_pixels=np.stack([old[v + "_fit_pixels"] for v in VIEWS]),
                reused_fit_pixel=overlap,
            )
            record.output(p.name, p, sha256=checks.sha(p))
        original_error = float(np.mean([r["original_error"] for r in rows]))
        directional_error = float(np.mean([r["directional_error"] for r in rows]))
        result = dict(
            status="ok",
            schema="directional_continuous_unused_pixels_v1",
            frames=len(rows),
            original_mean_error=original_error,
            directional_mean_error=directional_error,
            relative_error_reduction_percent=100
            * (original_error - directional_error)
            / original_error,
            mean_difference_by_view=np.mean(
                [r["difference_by_view"] for r in rows], axis=0
            ).tolist(),
            frames_lower_error=sum(r["mean_difference"] < 0 for r in rows),
            frames_higher_error=sum(r["mean_difference"] > 0 for r in rows),
            native_mask_checks=len(membership),
            fitted_coordinate_overlap=0,
            temporal_holdout=False,
            independent_ID_truth=False,
            candidate_reselected=False,
            raw_video_decoded=False,
            adopted=False,
        )
        for name, value in (
            ("summary.json", result),
            ("frames.json", rows),
            ("membership.json", membership),
        ):
            p = output / name
            p.write_text(json.dumps(value, indent=2), encoding="utf-8")
            record.output(name, p, sha256=checks.sha(p))
        checks.require(code == base.local_code(), "source set changed")
        for p, digest in hashes.items():
            checks.require(checks.sha(p) == digest, "bound input changed")
            record.input(p, p, sha256=digest)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "branch", "comparison", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(),
        "new workspace output required",
    )
    main(args.original.resolve(), args.branch.resolve(), args.comparison.resolve(), output)
