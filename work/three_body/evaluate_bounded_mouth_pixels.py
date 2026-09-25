"""CPU-only unused-pixel comparison of already selected local three-fish fits.

All 33 development frames, all three views, same samples for both fits. No fit,
candidate reselection, temporal inference or biological identity claim is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fit_bounded_mouth_packets as local
import numpy as np
import torch
from renderer import RefractiveFishRenderer, RendererConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

base, checks, REPO = local.base, local.checks, local.REPO
STAGE = "bounded_mouth_unused_pixel_evaluation"
PACKETS = REPO / "work/three_body/bounded_mouth_local_fit_packets_33_v1"
PACKET_SHA = "c1b4fc4b0f369f7f4a376f8bcd4a5b1736ad42efdcc0101118f9a407243da080"


def disjoint_pixels(fitted, unused):
    """Require native pixel centers and no frame/view overlap with fit coordinates."""
    checks.require(fitted.shape == (3, 1024, 2) and unused.shape == (3, 256, 2), "pixel shape")
    for points in (fitted, unused):
        checks.require(
            np.isfinite(points).all()
            and np.array_equal(points, np.rint(points))
            and np.all((points >= 0) & (points < 1200)),
            "invalid native pixel centers",
        )
    for a, b in zip(fitted, unused, strict=True):
        fit_set = set(map(tuple, a.astype(np.int64)))
        checks.require(not any(tuple(x) in fit_set for x in b.astype(np.int64)), "fit pixel reused")


def run(args):
    fit, samples, output = (
        args.fit.resolve(strict=True),
        args.samples.resolve(strict=True),
        args.output.resolve(),
    )
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = base.local_code()
    hashes = dict(code)
    output.mkdir(parents=True)
    rows = []
    with provenance(STAGE, output, name="evaluation") as record:
        try:
            for root, name, digest in (
                (fit, "fit.provenance.json", args.fit_sha),
                (samples, "preparation.provenance.json", args.sample_sha),
                (PACKETS, "preparation.provenance.json", PACKET_SHA),
            ):
                checks.require(
                    checks.sha(root / name) == digest, "required successful seal changed"
                )
            fitted = checks.verify(fit, "fit.provenance.json", local.STAGE, hashes)
            sample_seal = checks.verify(
                samples,
                "preparation.provenance.json",
                "bounded_mouth_unused_pixel_preparation_v1",
                hashes,
            )
            producer = checks.verify(
                PACKETS, "preparation.provenance.json", local.PACKET_STAGE, hashes
            )
            summary = checks.read(local.member(fit, "summary.json", hashes))
            checks.require(
                summary["status"] == "ok"
                and summary["device"] == "cuda"
                and tuple(summary["frames"]) == local.FRAMES
                and summary["control_all19_records_reproduced"]
                and tuple(summary["CPU_CUDA_all_candidate_frames"]) == local.CPU_FRAMES,
                "complete CPU-qualified CUDA comparison required",
            )
            for seal in (fitted, sample_seal):
                local.bind_input(seal, PACKETS / "preparation.provenance.json", hashes)
                for path in set(code) & set(seal["inputs"]):
                    local.bind_input(seal, path, hashes)
            metadata = checks.read(local.member(PACKETS, "metadata.json", hashes))
            cfg = metadata["tracker_config"]
            checks.require(cfg["width_ratio"] == 0.06, "frozen width ratio required")
            sample_metadata = checks.read(local.member(samples, "metadata.json", hashes))
            checks.require(
                tuple(sample_metadata["frames"]) == local.FRAMES
                and sample_metadata["counts"] == [96, 96, 64]
                and sample_metadata["seed_offset"] == 8803010
                and sample_metadata["sampler_config"] == cfg
                and sample_metadata["packet_seal_sha256"] == PACKET_SHA,
                "fixed unused sampler configuration differs",
            )
            calibration = Path(metadata["calibration_directory"])
            for path in calibration.iterdir():
                if path.is_file():
                    for seal in (producer, fitted, sample_seal):
                        local.bind_input(seal, path, hashes)
            rig = load_rig(calibration)
            prepared = base.load_arrays(local.member(samples, "samples.npz", hashes), hashes)
            checks.require(
                np.array_equal(prepared["frame_idx"], local.FRAMES), "fixed samples frame order"
            )
            checks.require(np.array_equal(prepared["views"], VIEWS), "fixed samples view order")
            checks.require(not prepared["reused_fitted_coordinate"].any(), "claimed reused pixels")
            torch.set_num_threads(2)
            for t, native in enumerate(local.FRAMES):
                packet = base.load_arrays(
                    local.member(PACKETS, f"f{native}_packet.npz", hashes), hashes
                )
                result = base.load_arrays(local.member(fit, f"f{native}_fit.npz", hashes), hashes)
                for key in ("fit_pixels", "fit_target", "fit_weights", "fit_foreground"):
                    checks.require(
                        checks.equal_bytes(packet[key], prepared[key][t]), "fit sampler differs"
                    )
                pixels, target, weights = (prepared[k][t] for k in ("pixels", "target", "weights"))
                disjoint_pixels(packet["fit_pixels"], pixels)
                indices = np.array(
                    [result[k][0] for k in ("original_selected_indices", "selected_indices")]
                )
                poses = result["fitted_pose_xyz"][indices].copy()
                checks.require(
                    checks.equal_bytes(poses[0], result["control_pose_xyz"][0])
                    and checks.equal_bytes(poses[1], result["combined_pose_xyz"][0]),
                    "saved selected pose differs",
                )
                renderer = RefractiveFishRenderer(
                    rig,
                    {v: pixels[i] for i, v in enumerate(VIEWS)},
                    RendererConfig(width_ratio=cfg["width_ratio"]),
                    device="cpu",
                    dtype=torch.float64,
                )
                with torch.no_grad():
                    alpha = renderer.render(poses).union_alpha.numpy()
                valid = renderer.ray_valid.numpy()
                error = ((alpha - target) ** 2 * weights * valid).sum(-1)
                checks.require(np.isfinite(error).all(), "nonfinite rendering error")
                rows.append(
                    dict(
                        frame_idx=native,
                        control_error=error[0].tolist(),
                        combined_error=error[1].tolist(),
                    )
                )
                np.savez_compressed(
                    output / f"f{native}_pixels.npz",
                    poses=poses,
                    selected_indices=indices,
                    union_alpha=alpha,
                    ray_valid=valid,
                    per_view_error=error,
                    pixels=pixels,
                    target=target,
                    weights=weights,
                    fit_pixels=packet["fit_pixels"],
                )
            errors = np.array([[r["control_error"], r["combined_error"]] for r in rows])
            means = errors.mean((0, 2))
            differences = (errors[:, 1] - errors[:, 0]).mean(1)
            result = dict(
                status="ok",
                frames=local.FRAMES,
                control_mean_error=float(means[0]),
                combined_mean_error=float(means[1]),
                mean_difference=float(means[1] - means[0]),
                relative_reduction_percent=float(100 * (means[0] - means[1]) / means[0]),
                frames_better=int((differences < 0).sum()),
                frames_worse=int((differences > 0).sum()),
                frames_equal=int((differences == 0).sum()),
                mean_difference_by_view=(errors[:, 1] - errors[:, 0]).mean(0).tolist(),
                fitted_coordinate_overlap=0,
                temporal_holdout=False,
                anatomical_or_ID_truth=False,
                candidate_reselected=False,
                tracking_performed=False,
                GPU_used=False,
            )
            local.write_json(output / "summary.json", result)
        finally:
            local.write_json(output / "frames.json", rows)
            for path, digest in hashes.items():
                checks.require(checks.sha(path) == digest, "bound input changed")
                record.input(path, path, sha256=digest)
            checks.require(code == base.local_code(), "loaded code changed")
            for path in output.rglob("*"):
                if path.is_file() and path.name != "evaluation.provenance.json":
                    record.output(str(path.relative_to(output)), path, sha256=checks.sha(path))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("fit", "samples", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--fit-sha", required=True)
    parser.add_argument("--sample-sha", required=True)
    run(parser.parse_args())
