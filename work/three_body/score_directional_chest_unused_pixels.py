"""Score fixed fits on spatially unused mask pixels, not held-out time or ID truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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
PLAN = Path(__file__).with_name("CHEST_SPATIAL_EVALUATION_PLAN.md")
REPO = Path(__file__).resolve().parents[2]


def main(replay, old, new, output):
    code = {**base.local_code(), str(PLAN): checks.sha(PLAN)}
    hashes = dict(code)
    replay_seal = checks.verify(
        replay, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
    )
    checks.verify(old, "fit.provenance.json", "chest_recovery_same_objective_local_CPU_fit", hashes)
    checks.verify(new, "fit.provenance.json", "directional_chest_local_CPU_fit", hashes)
    registration = checks.read(replay / "registration/plan.json")
    calibration = (
        Path(registration["original_binding"]["inputs"]["seed"]["input_root"])
        / base.CALIBRATION_DIR
    )
    for p in calibration.iterdir():
        if p.is_file():
            expected = replay_seal["inputs"][str(p)]["sha256"]
            checks.require(checks.sha(p) == expected, "calibration differs")
            hashes[str(p)] = expected
    rig = load_rig(calibration)
    packed = {}
    for view in VIEWS:
        matches = [
            Path(item["path"])
            for item in replay_seal["inputs"].values()
            if "f066400-068060" in item["path"] and Path(item["path"]).name == view + ".npz"
        ]
        checks.require(len(matches) == 1, "unique registered mask chunk required")
        path = matches[0]
        expected = replay_seal["inputs"][str(path)]["sha256"]
        checks.require(checks.sha(path) == expected, "mask chunk bytes differ")
        hashes[str(path)] = expected
        packed[view] = base.load_arrays(path, hashes)
        checks.require(
            np.array_equal(packed[view]["image_shape_hw"], [1200, 1200]),
            "unexpected native mask shape",
        )
    frames = [r["frame_idx"] for r in checks.read(new / "frames.json")]
    checks.require(
        len(frames) == 20 and frames[0] == 66660 and frames[-1] == 66696, "fixed scope differs"
    )
    width = registration["original_context"]["tracker_config"]["width_ratio"]
    checks.require(width == 0.06, "renderer width changed")
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    rows, membership = [], []
    start = time.perf_counter()
    with provenance(
        "directional_chest_spatial_mask_evaluation", output, name="evaluation"
    ) as record:
        record.param(
            "sampler",
            dict(
                counts=COUNTS, seed_offset=SEED_OFFSET, exclusion="all original fitting coordinates"
            ),
        )
        record.param("native_frames", frames)
        record.param("raw_video_decoded", False)
        record.param(
            "scope", "unused pixels of the same development frames; no temporal or ID holdout"
        )
        for native in frames:
            fits = [
                base.load_arrays(root / f"frame_{native:06d}.npz", hashes) for root in (old, new)
            ]
            for key in (
                "initial_pose_xyz",
                "proposal_plan",
                "proposal_parent",
                "proposal_permutation",
                "new_candidate",
            ):
                checks.require(
                    checks.equal_bytes(fits[0][key], fits[1][key]), "different paired fit inputs"
                )
            diag = base.load_arrays(
                replay / "diagnostics" / f"frame_{native:06d}" / "arrays.npz", hashes
            )
            fitted_pixels = np.stack([diag[v + "_fit_pixels"] for v in VIEWS])
            masks = {}
            for view in VIEWS:
                data = packed[view]
                index = int(np.searchsorted(data["frame_idx"], native))
                checks.require(int(data["frame_idx"][index]) == native, "wrong mask frame")
                value = decode_frame_masks(data, index)
                actual_sha = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                checks.require(
                    actual_sha == bytes(data["frame_mask_sha256"][index]).decode("ascii"),
                    "decoded mask differs",
                )
                masks[view] = value if data["view_available"][index] else None
            # The reused sampler reads only image.shape, never raw pixel values.
            # A shape-only object avoids fabricating an image or claiming raw validation.
            frame = SimpleNamespace(
                frame_idx=native,
                masks=masks,
                images_bgr={v: SimpleNamespace(shape=(1200, 1200, 3)) for v in VIEWS},
            )
            pixels, target, weights, overlap, audit = unused_samples(frame, fitted_pixels)
            checks.require(not overlap.any(), "spatial samples overlap fitting")
            membership.extend(audit)
            renderer = RefractiveFishRenderer(
                rig,
                {v: pixels[i] for i, v in enumerate(VIEWS)},
                RendererConfig(width_ratio=width),
                device="cpu",
                dtype=torch.float64,
            )
            pose = np.concatenate([a["fitted_pose_xyz"] for a in fits])
            # Keep per-candidate calculations independent and bound CPU peak memory.
            with torch.no_grad():
                union = np.concatenate(
                    [
                        renderer.render(pose[i : i + 8]).union_alpha.numpy()
                        for i in range(0, len(pose), 8)
                    ]
                )
            valid = renderer.ray_valid.numpy()
            error = ((union - target) ** 2 * weights * valid).sum(-1)
            count = len(fits[0]["fitted_pose_xyz"])
            checks.require(len(pose) == 2 * count, "paired candidate count differs")
            ia, ib = int(fits[0]["selected_indices"][0]), int(fits[1]["selected_indices"][0])
            old_error, new_error = error[ia], error[count + ib]
            row = dict(
                frame_idx=native,
                old_selected=ia,
                new_selected=ib,
                old_selected_all_three_chests=bool(
                    fits[0]["after_strict_observed"][ia, :, 1].all()
                ),
                new_selected_all_three_chests=bool(
                    fits[1]["after_strict_observed"][ib, :, 1].all()
                ),
                old_selected_view_error=old_error.tolist(),
                new_selected_view_error=new_error.tolist(),
                difference_by_view=(new_error - old_error).tolist(),
                old_selected_error=float(old_error.mean()),
                new_selected_error=float(new_error.mean()),
            )
            rows.append(row)
            path = output / f"frame_{native:06d}.npz"
            np.savez_compressed(
                path,
                poses=pose,
                pixels=pixels,
                target=target,
                weights=weights,
                ray_valid=valid,
                fit_pixels=fitted_pixels,
                reused_fit_pixel=overlap,
                union_alpha=union,
                per_view_error=error,
                mode=np.repeat(np.array(["isotropic", "directional"]), count),
                candidate=np.tile(np.arange(count), 2),
                selected=np.array([ia, count + ib]),
            )
            record.output(path.name, path, sha256=checks.sha(path))
            print(json.dumps(row), flush=True)
        result = dict(
            schema="directional_chest_spatial_mask_evaluation_v1",
            status="ok",
            frames=len(rows),
            pixel_overlap_count=0,
            decoded_mask_checks=len(membership),
            views=list(VIEWS),
            old_selected_mean_error=float(np.mean([r["old_selected_error"] for r in rows])),
            new_selected_mean_error=float(np.mean([r["new_selected_error"] for r in rows])),
            mean_difference_by_view=np.mean(
                [r["difference_by_view"] for r in rows], axis=0
            ).tolist(),
            frames_lower_error=sum(r["new_selected_error"] < r["old_selected_error"] for r in rows),
            frames_higher_error=sum(
                r["new_selected_error"] > r["old_selected_error"] for r in rows
            ),
            frames_new_best_chest_supported=sum(r["new_selected_all_three_chests"] for r in rows),
            seconds=time.perf_counter() - start,
            independent_ID_truth=False,
            held_out_time=False,
            pose_fitting_performed=False,
            raw_video_decoded=False,
            adopted=False,
        )
        for name, value in (
            ("summary.json", result),
            ("frames.json", rows),
            ("membership.json", membership),
        ):
            path = output / name
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
            record.output(path.name, path, sha256=checks.sha(path))
        checks.require(
            code == {**base.local_code(), str(PLAN): checks.sha(PLAN)}, "source set changed"
        )
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "bound input changed")
            record.input(path, path, sha256=digest)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(),
        "new workspace output required",
    )
    main(args.replay.resolve(), args.old.resolve(), args.new.resolve(), output)
