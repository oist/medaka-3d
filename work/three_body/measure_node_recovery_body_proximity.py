"""CPU body-proximity ablation diagnostic on ten already-fitted saved poses."""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from measure_scene0_body_proximity import (
    PAIRS,
    covariance,
    gaussian_pair_distances,
    polyline_closest,
    require,
)

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from prepare_sequence import sha256  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

SOURCE = REPO / "work/three_body/node_recovery_local_fit_v1"
MINUTE = REPO / "work/three_body/v1_continuous_one_minute_detectron2"
PRIOR = REPO / "work/three_body/minute_scene0_body_proximity"
OUT = REPO / "work/three_body/node_recovery_body_proximity_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def proximity_surrogate(standardized_distances):
    """Fixed requested engineering summary; never added to any fitting objective."""
    d = np.asarray(standardized_distances, dtype=float)
    require(d.shape[-3:] == (3, 8, 8), "Expected three fish pairs and all 8-by-8 Gaussians")
    require(np.isfinite(d).all() and (d >= 0).all(), "Distances must be finite and nonnegative")
    pair_values = np.exp(-0.5 * d**2).max(axis=(-2, -1))
    return pair_values, pair_values.mean(axis=-1)


def main():
    started = time.perf_counter()
    require(not OUT.exists(), "No overwrite of a proximity diagnostic")
    seal_path, summary_path = SOURCE / "fit.provenance.json", SOURCE / "summary.json"
    seal, summary = read(seal_path), read(summary_path)
    require(seal["status"] == "ok" and summary["status"] == "completed", "Local fit must be sealed")
    for item in seal["outputs"].values():
        require(sha256(item["path"]) == item["sha256"], "A completed local-fit output changed")
    needed = [
        Path(__file__),
        Path(__file__).with_name("test_node_recovery_body_proximity.py"),
        Path(__file__).with_name("measure_scene0_body_proximity.py"),
        seal_path,
        summary_path,
        SOURCE / "local_fit_arrays.npz",
        PRIOR / "body_proximity.provenance.json",
        PRIOR / "summary.json",
        LAB / "renderer.py",
        LAB / "prepare_sequence.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
        *sorted((MINUTE / "calibration_snapshot").glob("*.json")),
    ]
    inputs = {p.resolve(): sha256(p) for p in needed}
    source_inputs = {Path(p).resolve(): d for p, d in summary["inputs"].items()}
    for path, digest in inputs.items():
        if path in source_inputs:
            require(digest == source_inputs[path], "Renderer/calibration differs from local fit")
    prior_seal = read(PRIOR / "body_proximity.provenance.json")
    require(prior_seal["status"] == "ok", "Prior geometry diagnostic incomplete")
    helper = Path(__file__).with_name("measure_scene0_body_proximity.py").resolve()
    require(
        inputs[helper] == prior_seal["inputs"][str(helper)]["sha256"], "Geometry helper changed"
    )
    poses, metadata = [], []
    with np.load(SOURCE / "local_fit_arrays.npz", allow_pickle=False) as data:
        for case in summary["cases"]:
            frame = case["frame"]
            fitted = data[f"f{frame}_fitted_pose"]
            scores, parts = data[f"f{frame}_scores"], data[f"f{frame}_loss_parts"]
            require(
                fitted.shape == (len(case["choices"]), 3, 3, 3), "Fit count or body schema differs"
            )
            for j, choice in enumerate(case["choices"]):
                require(choice["index"] == j, "Choice order differs")
                require(
                    np.array_equal(fitted[j], choice["fitted_pose_xyz"]),
                    "Saved JSON/NPZ pose mismatch",
                )
                require(scores[j] == choice["fitted_score"], "Saved JSON/NPZ score mismatch")
                poses.append(fitted[j])
                metadata.append(
                    dict(
                        frame_idx=frame,
                        choice_index=j,
                        kind=choice["kind"],
                        target_fish=choice["target_fish"],
                        fitted_score=float(scores[j]),
                        fitted_loss_parts=parts[j].tolist(),
                        fitted_observed=choice["fitted_direct"],
                        identity_review_unchanged=case["identity_review_unchanged"],
                        is_original_local_objective_best=j == case["best_index"],
                    )
                )
    pose = np.stack(poses)
    require(
        pose.shape == (10, 3, 3, 3) and np.isfinite(pose).all(),
        "Expected all ten finite final poses",
    )
    renderer_config = RendererConfig(**read(PRIOR / "summary.json")["renderer_config"])
    require(
        renderer_config.width_ratio == summary["config"]["width_ratio"], "Different body widths"
    )
    rig = load_rig(MINUTE / "calibration_snapshot")
    torch.set_num_threads(2)
    renderer = RefractiveFishRenderer(
        rig,
        {rig.views[0]: np.array([[600.0, 600.0]])},
        renderer_config,
        device="cpu",
        dtype=torch.float64,
    )
    with torch.inference_mode():
        bodies = renderer.gaussians(pose)
    body_arrays = {
        f.name: getattr(bodies, f.name).cpu().numpy() for f in dataclasses.fields(bodies)
    }
    cov = covariance(
        body_arrays["tangents"], body_arrays["axial_sigma_mm"], body_arrays["radial_sigma_mm"]
    )
    euclidean_all, standardized_all, pair_rows = [], [], []
    for i, info in enumerate(metadata):
        euclidean_frame, standardized_frame = [], []
        for a, b in PAIRS:
            e, d = gaussian_pair_distances(
                body_arrays["centers"][i, a],
                cov[i, a],
                body_arrays["centers"][i, b],
                cov[i, b],
            )
            ei, di = np.unravel_index(e.argmin(), e.shape), np.unravel_index(d.argmin(), d.shape)
            row = dict(
                frame_idx=info["frame_idx"],
                choice_index=info["choice_index"],
                fish_pair=[a, b],
                pectoral_distance_mm=float(np.linalg.norm(pose[i, a, 1] - pose[i, b, 1])),
                polyline_closest=polyline_closest(pose[i, a], pose[i, b]),
                minimum_center_pair=dict(
                    indices=list(map(int, ei)),
                    center_distance_mm=float(e[ei]),
                    standardized_distance=float(d[ei]),
                ),
                minimum_standardized_pair=dict(
                    indices=list(map(int, di)),
                    center_distance_mm=float(e[di]),
                    standardized_distance=float(d[di]),
                ),
            )
            for key in ("minimum_center_pair", "minimum_standardized_pair"):
                selected = row[key]
                selection = list(zip((a, b), selected["indices"], strict=True))
                selected["centers_xyz"] = [
                    body_arrays["centers"][i, f, k].tolist() for f, k in selection
                ]
                selected["axial_sigma_mm"] = [
                    float(body_arrays["axial_sigma_mm"][i, f, k]) for f, k in selection
                ]
                selected["radial_sigma_mm"] = [
                    float(body_arrays["radial_sigma_mm"][i, f, k]) for f, k in selection
                ]
            pair_rows.append(row)
            euclidean_frame.append(e)
            standardized_frame.append(d)
        euclidean_all.append(euclidean_frame)
        standardized_all.append(standardized_frame)
    standardized_all = np.asarray(standardized_all)
    pair_values, surrogate = proximity_surrogate(standardized_all)
    differences = []
    for i, info in enumerate(metadata):
        base = next(
            j
            for j, item in enumerate(metadata)
            if item["frame_idx"] == info["frame_idx"] and item["choice_index"] == 0
        )
        info["proximity_surrogate"] = float(surrogate[i])
        info["proximity_surrogate_by_fish_pair"] = pair_values[i].tolist()
        info["surrogate_delta_vs_fitted_base"] = float(surrogate[i] - surrogate[base])
        info["fitted_score_delta_vs_base"] = info["fitted_score"] - metadata[base]["fitted_score"]
        distances = np.linalg.norm(
            pose[i, np.asarray(PAIRS)[:, 0], 1] - pose[i, np.asarray(PAIRS)[:, 1], 1], axis=-1
        )
        actual_repulsion = float(np.exp(-(distances**2) / 8).mean())
        differences.append(actual_repulsion - info["fitted_loss_parts"][3])
    require(np.max(np.abs(differences)) < 1e-12, "Saved pectoral repulsion differs")
    OUT.mkdir(parents=True)
    with provenance("local_fit_body_proximity_diagnostic", OUT, name="diagnostic") as record:
        for path, digest in inputs.items():
            record.input(str(path), path, sha256=digest)
            if path.suffix == ".py":
                target = OUT / "source_snapshot" / path.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                require(sha256(target) == digest, "Source snapshot mismatch")
                record.output(str(target.relative_to(OUT)), target, sha256=digest)
        path = OUT / "bodies_and_distances.npz"
        np.savez_compressed(
            path,
            pose_xyz=pose,
            frame_idx=np.array([m["frame_idx"] for m in metadata]),
            choice_index=np.array([m["choice_index"] for m in metadata]),
            **body_arrays,
            covariance_xyz=cov,
            gaussian_center_distances_mm=np.asarray(euclidean_all),
            gaussian_standardized_distances=standardized_all,
            proximity_surrogate=surrogate,
            proximity_surrogate_by_pair=pair_values,
        )
        record.output("arrays", path, sha256=sha256(path))
        path = OUT / "summary.json"
        result = dict(
            status="complete_geometry_diagnostic_no_new_loss",
            inputs_sha256_before={str(p): d for p, d in inputs.items()},
            inputs_sha256_after={str(p): d for p, d in inputs.items()},
            poses=metadata,
            pair_rows=pair_rows,
            fish_pair_order=PAIRS,
            surrogate_definition=(
                "mean over 3 fish pairs of max over 64 Gaussian pairs exp(-0.5*d^2)"
            ),
            standardized_distance_definition=(
                "d=sqrt(delta^T(Sigma_f+Sigma_g)^-1delta), same fixed renderer sigmas "
                "as prior 9-frame diagnostic"
            ),
            renderer_config=dataclasses.asdict(renderer_config),
            source_final_pose_count=10,
            diagnostic_geometry_rows=30,
            max_saved_repulsion_error=float(np.max(np.abs(differences))),
            runtime=dict(
                device="cpu",
                dtype="float64",
                torch_threads=2,
                elapsed_s=time.perf_counter() - started,
            ),
            optimizer_steps=0,
            surrogate_added_to_objective=False,
            weight_chosen=False,
            thresholds_chosen=False,
            biological_identity_labels_used=False,
            limitations=[
                "The dimensionless surrogate is an engineering proximity summary, "
                "not overlap volume or an identity likelihood.",
                "Each frame has an independent fitted base; scores and surrogate "
                "differences are only compared within that frame.",
                "No candidate pose, score, review flag, acceptance or tracking output is modified.",
                "Geometric proximity does not prove a false connection or justify "
                "penalizing true physical contact.",
                "These ten already-inspected candidates form a post-hoc mechanistic "
                "diagnostic, not held-out validation.",
            ],
        )
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        path = OUT / "candidate_comparison.csv"
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "frame",
                    "choice",
                    "kind",
                    "target_fish",
                    "original_fit_score",
                    "score_delta_vs_base",
                    "surrogate",
                    "surrogate_delta_vs_base",
                    "pair01",
                    "pair02",
                    "pair12",
                ]
            )
            for item in metadata:
                writer.writerow(
                    [
                        item["frame_idx"],
                        item["choice_index"],
                        item["kind"],
                        item["target_fish"],
                        item["fitted_score"],
                        item["fitted_score_delta_vs_base"],
                        item["proximity_surrogate"],
                        item["surrogate_delta_vs_fitted_base"],
                        *item["proximity_surrogate_by_fish_pair"],
                    ]
                )
        record.output("comparison", path, sha256=sha256(path))
        require(all(sha256(p) == d for p, d in inputs.items()), "A frozen source/input changed")
    print(
        json.dumps(
            dict(
                out=str(OUT),
                max_repulsion_error=result["max_saved_repulsion_error"],
                elapsed_s=result["runtime"]["elapsed_s"],
            )
        )
    )


if __name__ == "__main__":
    main()
