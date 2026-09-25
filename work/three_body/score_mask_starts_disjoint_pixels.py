"""Compare six new mask-start poses on the exact previously saved unused pixels.

This reuses data, targets and weights without sampling or fitting. The reference
poses are original v1 and the 400-step v2 H1/H11. Spatially disjoint pixels of the
same images do not provide independent frame, anatomical or identity truth.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from renderer import RendererConfig  # noqa: E402
from temporal_renderer import TemporalUnionRenderer  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("work/three_body/temporal_v3_mask_starts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), LAB / "renderer.py", LAB / "temporal_renderer.py",
                    REPO / "src/medaka3d/calib.py"]
    hashes = {p.relative_to(REPO).as_posix(): digest(p) for p in source_paths}
    snapshot = args.output / "code_snapshot"
    snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshot / path.name)
    original_root = REPO / "work/three_body/temporal_v2_disjoint_pixels_v1"
    reference_root = REPO / "work/three_body/temporal_v2_400_steps_disjoint_pixels"
    sample_name = "evaluation_samples_and_scores.npz"
    original, previous = load(original_root / sample_name), load(reference_root / sample_name)
    original_report = json.loads((original_root / "results.json").read_text(encoding="utf-8"))
    previous_report = json.loads((reference_root / "results.json").read_text(encoding="utf-8"))
    if (digest(original_root / sample_name) != original_report["samples_sha256"]
            or digest(reference_root / sample_name) != previous_report["samples_sha256"]):
        raise ValueError("saved spatial evaluation samples changed")
    for name in ("frame_idx", "views", "pixels", "target", "weights", "fitted_pixels",
                 "reused_fitted_coordinate"):
        np.testing.assert_array_equal(previous[name], original[name])
    if original["reused_fitted_coordinate"].any():
        raise ValueError("saved targets overlap the fitted coordinates")
    provenance_path = args.run / "refinement.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["status"] != "ok":
        raise ValueError("new refinement must have completed successfully")
    for relative, expected in provenance["params"]["code_sha256"].items():
        relative = relative.replace("\\", "/")
        if relative in hashes and hashes[relative] != expected:
            raise ValueError(f"changed renderer dependency: {relative}")
    hypotheses = load(args.run / "hypotheses.npz")
    combined = load(args.run / "combined_hypotheses.npz")
    fitted = load(args.run / "sampled_targets.npz")
    np.testing.assert_array_equal(hypotheses["frame_idx"], original["frame_idx"])
    np.testing.assert_array_equal(fitted["pixels"], original["fitted_pixels"])
    reference_indexes = [list(previous["labels"]).index(label)
                         for label in ("original_v1", "H1", "H11")]
    np.testing.assert_array_equal(combined["pose_xyz"][:12], previous["poses"][1:])
    np.testing.assert_array_equal(combined["pose_xyz"][12:], hypotheses["pose_xyz"])
    poses = np.concatenate((previous["poses"][reference_indexes], hypotheses["pose_xyz"]))
    labels = ["original_v1", "v2_400_H1", "v2_400_H11",
              *[f"v3_H{i}" for i in range(len(hypotheses["pose_xyz"]))]]
    config = provenance["params"]["config"]
    if config["width_ratio"] != float(original["width_ratio"]):
        raise ValueError("new poses must use the same fixed body width")
    base_run = Path(provenance["params"]["base_refinement"])
    rig = load_rig(base_run / "calibration_snapshot")
    torch.set_num_threads(2)
    pixels, target, weights = (original[name] for name in ("pixels", "target", "weights"))
    renderer = TemporalUnionRenderer(
        rig, [{view: p[vi] for vi, view in enumerate(VIEWS)} for p in pixels],
        RendererConfig(width_ratio=config["width_ratio"]), dtype=torch.float64, device="cpu",
        time_chunk_size=5, checkpoint_chunks=False)
    np.testing.assert_array_equal(renderer.ray_valid.numpy(), original["ray_valid"])
    start = time.perf_counter()
    with torch.no_grad():
        union = renderer.render(poses).numpy()
    seconds = time.perf_counter() - start
    view_frame_error = ((union - target) ** 2 * weights * original["ray_valid"]).sum(-1)
    np.testing.assert_array_equal(
        view_frame_error[:3], previous["view_frame_error"][reference_indexes])
    scores = view_frame_error.mean((1, 2))
    per_view = view_frame_error.mean(1)
    saved = args.output / "evaluation_samples_and_scores.npz"
    np.savez_compressed(saved, frame_idx=original["frame_idx"], views=original["views"],
                        labels=np.asarray(labels), pixels=pixels, target=target, weights=weights,
                        fitted_pixels=original["fitted_pixels"],
                        reused_fitted_coordinate=original["reused_fitted_coordinate"],
                        ray_valid=original["ray_valid"], poses=poses,
                        width_ratio=config["width_ratio"], union_alpha=union,
                        view_frame_error=view_frame_error, mean_error=scores)
    input_paths = [original_root / sample_name, original_root / "results.json",
                   reference_root / sample_name, reference_root / "results.json",
                   provenance_path, args.run / "hypotheses.npz",
                   args.run / "combined_hypotheses.npz", args.run / "sampled_targets.npz",
                   *(base_run / "calibration_snapshot").glob("*.json")]
    candidates = [dict(label=label, mean_image_error=float(scores[i]),
                       mean_image_error_by_view=per_view[i].tolist(),
                       mean_image_error_by_frame=view_frame_error[i].mean(1).tolist(),
                       differences_vs_original_H1_H11=(scores[i] - scores[:3]).tolist(),
                       lower_error_than_original_H1_H11=(scores[i] < scores[:3]).tolist())
                  for i, label in enumerate(labels)]
    if {p.relative_to(REPO).as_posix(): digest(p) for p in source_paths} != hashes:
        raise RuntimeError("source changed during scoring")
    report = dict(
        status="ok", device="CPU only", dtype="float64", render_seconds=seconds,
        source_sha256=hashes, input_sha256={str(p): digest(p) for p in input_paths},
        samples_sha256=digest(saved), exact_saved_pixels_targets_weights_reused=True,
        preserved_original_H1_H11_errors_bitwise_exact=True,
        fitted_coordinate_overlap_count=int(original["reused_fitted_coordinate"].sum()),
        image_only_ranking=[labels[i] for i in np.argsort(scores, kind="stable")],
        best_new_image_label=labels[3 + int(np.argmin(scores[3:]))], candidates=candidates,
        frames=[int(original["frame_idx"][0]), int(original["frame_idx"][-1]) + 1],
        membership_audit=original_report["membership_audit"],
        error_definition="mean over T,V of sum_N weight*(union-target)^2; same fixed class weights",
        limits=["No resampling, optimization, model update or identity label enters this scoring.",
                "Same images and masks as fitting, with spatially disjoint sampled coordinates.",
                "This is not held-out frames or statistically independent validation.",
                "Union image fit does not establish anatomical locations or persistent identity.",
                "All 18 combined pose arrays were checked against their preserved source runs."],
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                             encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "render_seconds", "image_only_ranking", "best_new_image_label")}, indent=2))
    print(json.dumps([{k: c[k] for k in ("label", "mean_image_error", "mean_image_error_by_view",
                                        "differences_vs_original_H1_H11")}
                      for c in candidates], indent=2))


if __name__ == "__main__":
    main()
