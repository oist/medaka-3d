"""Compare a completed bounded minute using the frozen uniform 60 pixel samples.

Only the bounded candidate is rendered again. Original/directional renders are
reused from their successful, pinned assay after checking their poses, samples,
ray validity and loss arithmetic. No fitting, reselection or temporal holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import score_directional_minute_pixels as prior
import torch
from evaluate_bounded_mouth_pixels import disjoint_pixels
from renderer import RefractiveFishRenderer, RendererConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

base, checks, REPO = prior.base, prior.checks, prior.REPO
STAGE = "bounded_minute_unused_pixel_evaluation_v1"
COMPARISON_STAGE = "bounded_one_minute_comparison_v1"
PREP_SHA = "a219bbad9980acfc1e4d17928829fad0a39792c97be6bd9e14abc5e8938cfa4e"
PRIOR_SHA = "7422c228301c9dc4c4e51f8c216e789d8e219c52361fe03d8994cfba0af9e0b3"
PRIOR_CODE_SHA = "ef0e2a9ac06d582347b14832c5f344d0b29c637a5aa2294d54a66a1ecf102855"
METHODS = ("original", "directional", "bounded")
RUN_SPECS = (
    ("continuation.provenance.json", "three_body_long_continuation"),
    ("tracking.provenance.json", "directional_one_minute_continuous"),
    ("tracking.provenance.json", "bounded_mouth_one_minute_continuous"),
)


def bound_input(seal, path, hashes):
    path = str(Path(path).resolve())
    checks.require(path in hashes, "input not independently verified: " + path)
    checks.require(
        path in seal["inputs"] and seal["inputs"][path]["sha256"] == hashes[path],
        "sealed input differs: " + path,
    )


def squared_error(alpha, target, weights, valid):
    checks.require(
        alpha.ndim == 3
        and alpha.shape[1:] == (3, 256)
        and target.shape == weights.shape == valid.shape == (3, 256),
        "pixel assay shape differs",
    )
    checks.require(
        all(np.isfinite(a).all() for a in (alpha, target, weights))
        and (weights >= 0).all()
        and valid.dtype == np.bool_,
        "invalid pixel assay values",
    )
    return ((alpha - target) ** 2 * weights * valid).sum(-1)


def validate_saved_frame(saved, sample, expected_poses):
    checks.require(
        expected_poses.shape == (2, 3, 3, 3) and checks.equal_bytes(saved["poses"], expected_poses),
        "previous assay selected pose differs",
    )
    for key, value in sample.items():
        checks.require(
            key in saved and checks.equal_bytes(saved[key], value),
            "previous assay sample differs: " + key,
        )
    checks.require(not sample["reused_fitted_coordinate"].any(), "claimed fitted overlap")
    disjoint_pixels(sample["fit_pixels"], sample["pixels"])
    error = squared_error(
        saved["union_alpha"], sample["target"], sample["weights"], saved["ray_valid"]
    )
    checks.require(
        error.shape == (2, 3) and checks.equal_bytes(error, saved["per_view_error"]),
        "previous assay saved loss differs",
    )
    return error


def summary_errors(errors):
    checks.require(errors.shape == (60, 3, 3) and np.isfinite(errors).all(), "full 60x3x3 assay")
    means = errors.mean((0, 2))
    comparisons = {}
    for reference in (0, 1):
        delta = (errors[:, 2] - errors[:, reference]).mean(1)
        comparisons[METHODS[reference]] = dict(
            mean_difference=float(means[2] - means[reference]),
            relative_reduction_percent=(
                float(100 * (means[reference] - means[2]) / means[reference])
                if means[reference] > 0
                else None
            ),
            mean_difference_by_view=(errors[:, 2] - errors[:, reference]).mean(0).tolist(),
            frames_better=int((delta < 0).sum()),
            frames_worse=int((delta > 0).sum()),
            frames_equal=int((delta == 0).sum()),
        )
    return dict(
        mean_error_by_method=dict(zip(METHODS, map(float, means), strict=True)),
        bounded_compared_with=comparisons,
    )


def run(args):
    original, directional, bounded, comparison, prepared, previous = (
        getattr(args, key).resolve(strict=True)
        for key in (*METHODS, "comparison", "prepared", "previous")
    )
    roots = (original, directional, bounded)
    output = prior.new_output(args.output)
    code = base.local_code()
    hashes = dict(code)
    rows = []
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="evaluation") as record:
        try:
            for root, name, digest in (
                (prepared, "preparation.provenance.json", PREP_SHA),
                (previous, "evaluation.provenance.json", PRIOR_SHA),
                (comparison, "comparison.provenance.json", args.comparison_sha),
            ):
                checks.require(checks.sha(root / name) == digest, "pinned successful seal differs")
            cmp = checks.verify(comparison, "comparison.provenance.json", COMPARISON_STAGE, hashes)
            result = checks.read(prior.require_member(comparison, "summary.json", hashes))
            checks.require(
                result["status"] == "ok"
                and result["native_frames"] == 9960
                and result["restores_during_one_minute"] == 0,
                "successful complete minute comparison required",
            )
            prep_seal = checks.verify(
                prepared, "preparation.provenance.json", prior.PREP_STAGE, hashes
            )
            old_eval = checks.verify(
                previous,
                "evaluation.provenance.json",
                "directional_minute_unused_pixel_evaluation",
                hashes,
            )
            bound_input(old_eval, prepared / "preparation.provenance.json", hashes)
            prep = checks.read(prior.require_member(prepared, "summary.json", hashes))
            old_summary = checks.read(prior.require_member(previous, "summary.json", hashes))
            checks.require(
                old_summary["status"] == "ok"
                and old_summary["scored_native_frames"] == prior.SCORED.tolist()
                and old_summary["scored_frame_count"] == 60
                and not old_summary["candidate_reselected"],
                "previous full fixed assay required",
            )
            checks.require(
                checks.sha(prior.__file__) == PRIOR_CODE_SHA, "frozen assay source differs"
            )
            shared_code = {p: sha for p, sha in code.items() if p in prep_seal["inputs"]}
            # Required renderer and sampler modules must be present, not merely any intersection.
            for module in (
                prior,
                sys.modules["renderer"],
                sys.modules["observations"],
                sys.modules["score_temporal_disjoint_pixels"],
            ):
                path = str(Path(module.__file__).resolve())
                checks.require(path in shared_code, "required shared sampler source absent")
            for path in shared_code:
                bound_input(old_eval, path, hashes)
            seals, candidates = [], []
            for index, (root, (filename, stage)) in enumerate(zip(roots, RUN_SPECS, strict=True)):
                seal = checks.verify(root, filename, stage, hashes)
                seals.append(seal)
                bound_input(cmp, root / filename, hashes)
                path = prior.require_member(root, "candidate.npz", hashes)
                bound_input(cmp, path, hashes)
                if index < 2:
                    bound_input(old_eval, root / filename, hashes)
                    bound_input(old_eval, path, hashes)
                candidate = base.load_arrays(path, hashes)
                checks.require(
                    np.array_equal(candidate["frame_idx"], np.arange(49800, prior.STOP)),
                    "candidate complete native horizon differs",
                )
                candidates.append(candidate)
            metadata = checks.read(
                prior.require_member(original, "checkpoint/metadata.json", hashes)
            )
            config = metadata["context"]["tracker_config"]
            prior.validate_preparation(prep, prep_seal, shared_code, config)
            checks.require(
                prep["original_seal_sha256"] == hashes[str(original / RUN_SPECS[0][0])],
                "prepared reference differs",
            )
            new_meta = checks.read(
                prior.require_member(bounded, "checkpoints/through_076360/metadata.json", hashes)
            )
            checks.require(
                new_meta["context"]["tracker_config"] == config,
                "bounded sampler configuration differs",
            )
            for path, digest in prep["source_mask_sha256"].items():
                checks.require(checks.sha(path) == digest, "prepared segmentation changed")
                hashes[path] = digest
                bound_input(seals[2], path, hashes)
            for native in prior.DIAGNOSTIC:
                sample = base.load_arrays(
                    prior.require_member(prepared, f"frame_{native:06d}.npz", hashes), hashes
                )
                path = prior.require_member(
                    bounded, f"diagnostics/frame_{native:06d}/arrays.npz", hashes
                )
                prior.validate_diagnostic(sample, base.load_arrays(path, hashes))
            calibration = Path(prep["calibration"])
            for path in calibration.iterdir():
                if path.is_file():
                    digest = prep_seal["inputs"][str(path)]["sha256"]
                    checks.require(checks.sha(path) == digest, "prepared calibration differs")
                    hashes[str(path)] = digest
                    bound_input(old_eval, path, hashes)
                    bound_input(seals[2], path, hashes)
            rig = load_rig(calibration)
            torch.set_num_threads(2)
            errors = []
            for native in prior.SCORED:
                sample = base.load_arrays(
                    prior.require_member(prepared, f"frame_{native:06d}.npz", hashes), hashes
                )
                old = base.load_arrays(
                    prior.require_member(previous, f"frame_{native:06d}.npz", hashes), hashes
                )
                poses = np.stack([c["pose_xyz"][int(native) - 49800] for c in candidates])
                old_error = validate_saved_frame(old, sample, poses[:2])
                renderer = RefractiveFishRenderer(
                    rig,
                    {v: sample["pixels"][i] for i, v in enumerate(VIEWS)},
                    RendererConfig(width_ratio=0.06),
                    device="cpu",
                    dtype=torch.float64,
                )
                with torch.no_grad():
                    alpha = renderer.render(poses[2:3]).union_alpha.numpy()
                valid = renderer.ray_valid.numpy()
                checks.require(
                    checks.equal_bytes(valid, old["ray_valid"]), "renderer valid rays differ"
                )
                error = np.concatenate(
                    (old_error, squared_error(alpha, sample["target"], sample["weights"], valid))
                )
                errors.append(error)
                rows.append(dict(frame_idx=int(native), per_view_error=error.tolist()))
                np.savez_compressed(
                    output / f"frame_{native:06d}.npz",
                    poses=poses,
                    union_alpha=np.concatenate((old["union_alpha"], alpha)),
                    ray_valid=valid,
                    per_view_error=error,
                    **sample,
                )
            result = dict(
                status="ok",
                native_interval=[prior.START, prior.STOP],
                scored_native_frames=prior.SCORED.tolist(),
                scored_frame_count=60,
                methods=METHODS,
                **summary_errors(np.array(errors)),
                regenerated_bounded_diagnostic_checks=166,
                diagnostic_check_scope=(
                    "previously prepared 166 frames; additional 155 mouth frames not used"
                ),
                reused_previous_render_frames=60,
                new_render_method="bounded",
                GPU_used=False,
                fitted_coordinate_overlap=0,
                temporal_holdout=False,
                all_three_ID_truth=False,
                candidate_reselected=False,
                adopted=False,
            )
            for index, key in enumerate(("original_mean_error", "directional_mean_error")):
                checks.require(
                    result["mean_error_by_method"][METHODS[index]] == old_summary[key],
                    "reused prior aggregate differs",
                )
            prior.write_json(output / "summary.json", result)
            means = result["mean_error_by_method"]
            text = (
                "# 同じ1分・固定60時刻の未使用輪郭画素\n\n"
                f"平均誤差: 元 {means['original']:.9f}、胸部改良 {means['directional']:.9f}、"
                f"口候補追加 {means['bounded']:.9f}。\n\n"
                "前回と同じ時刻・画素で、最終76360終了時点の選択済み3匹姿勢を評価した。"
                "元・胸部改良の保存renderは姿勢・画素・損失を照合して再利用し、口候補追加のみ再renderした。"
                "全166共通診断時刻の最適化画素・目標・重み・前景は準備時と完全一致した。"
                "同じ開発画像の別画素による診断であり、時間holdoutや個体IDの正解率ではない。\n"
            )
            (output / "RESULT.md").write_text(text, encoding="utf-8")
        finally:
            prior.write_json(output / "frames.json", rows)
            prior.finish(record, output, hashes, code)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (*METHODS, "comparison", "prepared", "previous", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--comparison-sha", required=True)
    run(parser.parse_args())
