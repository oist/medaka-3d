"""Audit sealed local fits and fixed-pixel losses; never render, fit, or read GT."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path[:0] = [str(LAB), str(REPO / "src")]
from body_local_fit import replay_chosen_parents  # noqa: E402
from node_geometry import RayObservations, assignment_plans, supported_nodes  # noqa: E402
from observations import CachedNodes  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

FRAMES = (79316, 79318, 79322, 79355, 79370, 79372, 79381, 79394, 79396, 79405, 79547, 82719, 82720)
ARMS = ("direct", "unit", "fresnel")
PROTOCOL_SHA = "07c21ec5b75f206959491f381f085b1e45efd32363b6dee28e54bbff66db3cc5"
PRODUCER_SHA = "0fe7abf1083de518b95a0298ba59e4089167ecb6ec35609d7991cf69265cd0e4"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        require(len(set(data.files)) == len(data.files), "duplicate NPZ fields")
        return {key: data[key] for key in data.files}


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    require(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
        "array dtype/shape/bytes differ",
    )


def near(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def codes():
    result = {str(Path(__file__).resolve()): sha(__file__)}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src")
            ):
                result[str(path)] = sha(path)
    return result


def unchanged(bindings):
    for path, digest in bindings.items():
        require(sha(path) == digest, f"changed evidence/source: {path}")


def verified_outputs(root, name, stage, bindings):
    path = root / name
    bindings[str(path)] = sha(path)
    seal = read(path)
    require(seal["status"] == "ok" and seal["stage"] == stage, "successful matching seal required")
    for entry in seal["outputs"].values():
        path = Path(entry["path"]).resolve(strict=True)
        require(path.is_relative_to(root), "foreign output path")
        require(sha(path) == entry["sha256"], f"sealed output SHA differs: {path}")
        bindings[str(path)] = entry["sha256"]
    return seal


def bounded_input(path, seal, bindings):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO), "input outside workspace")
    entry = seal["inputs"].get(str(path))
    require(
        entry is not None and Path(entry["path"]).resolve() == path,
        f"input not bound by producer: {path}",
    )
    require(sha(path) == entry["sha256"], f"input SHA differs: {path}")
    bindings[str(path)] = entry["sha256"]
    return path


def support_check(source, observed, plan):
    require(source.shape == (3, 3, 3) and source.dtype.kind in "iu", "wrong source schema")
    require(
        np.all(source >= -1) and np.all((source == -1) | (source == plan)),
        "source no longer a subset of the fixed assignment",
    )
    exact(observed, np.sum(source >= 0, axis=1) >= 2)
    for view in range(3):
        for node in range(3):
            held = source[:, view, node]
            held = held[held >= 0]
            require(len(held) == len(np.unique(held)), "nonexclusive source")


def pixel_losses(saved, pixels):
    target, weights, strata = (pixels["eval_" + k] for k in ("target", "weights", "strata"))
    valid = saved["eval_calibrated_valid"]
    require(valid.dtype == bool and valid.shape == target.shape, "invalid pixel validity")
    result = []
    for model in ARMS:
        union = saved["eval_" + model + "_union"]
        require(
            union.shape == (3, *target.shape)
            and np.isfinite(union).all()
            and np.all((union >= 0) & (union <= 1)),
            "saved response invalid",
        )
        weighted_error = (union - target) ** 2 * weights * valid
        parts = np.stack(
            [np.sum(weighted_error * (strata == s), axis=-1) for s in range(3)], axis=-1
        )
        per_view = parts.sum(-1)
        means = per_view.mean(-1)
        for key, value in (
            ("loss_by_stratum", parts),
            ("loss_by_view", per_view),
            ("mean_view_loss", means),
            ("foreground_loss_by_view", parts[..., 0]),
            ("background_loss_by_view", parts[..., 1:].sum(-1)),
        ):
            near(saved["eval_" + model + "_" + key], value)
        counts = np.stack([np.sum(valid & (strata == s), axis=1) for s in range(3)], axis=-1)
        exact(saved["eval_" + model + "_valid_counts"], counts)
        result.append(parts)
    return np.stack(result, axis=1)  # fitted arm, scorer, view, stratum


def verify_gates(summary, parts, observed):
    # Independently use the saved pixel sums, without calling producer.screening.
    losses = parts.sum(-1).mean(-1)  # frame, fitted arm, scorer
    matrix = losses.mean(0)
    rows = {}
    for a, arm in enumerate(ARMS):
        pect = observed[:, 0, :, 1] & ~observed[:, a, :, 1]
        mouth = observed[:, 0, :, 0] & ~observed[:, a, :, 0]
        recovery = mouth & np.isin(FRAMES, (82719, 82720))[:, None]

        def records(mask):
            return [dict(frame_idx=FRAMES[t], slot=int(f)) for t, f in np.argwhere(mask)]

        first = bool(matrix[a, a] < matrix[0, 0] - 1e-10)
        second = bool(matrix[a, a] < matrix[0, a] - 1e-10)
        passed = bool(a != 0 and first and second and not pect.any() and not recovery.any())
        expected = summary["screening"][arm]
        require(
            expected["average_chosen_error_lower_than_direct"] == first
            and expected["same_model_error_lower_than_direct_arm_pose"] == second
            and expected["lost_chest_scene_slots"] == records(pect)
            and expected["lost_recovery_mouth_scene_slots"] == records(recovery)
            and expected["gate_pass"] == passed
            and expected["gate_error_tolerance"] == 1e-10,
            "saved development gate differs",
        )
        near(expected["cross_model_mean_errors"], matrix[a])
        near(expected["mean_chosen_mask_error"], matrix[a, a])
        near(expected["mean_common_direct_error"], matrix[a, 0])
        rows[arm] = dict(
            gate_pass=passed,
            image_gate_vs_direct=first,
            image_gate_same_model=second,
            mean_selected_error=float(matrix[a, a]),
            selected_minus_direct=float(matrix[a, a] - matrix[0, 0]),
            same_model_pose_change=float(matrix[a, a] - matrix[0, a]),
            common_direct_pose_change=float(matrix[a, 0] - matrix[0, 0]),
            lost_chest=records(pect),
            lost_recovery_mouth=records(recovery),
            lost_any_mouth=records(mouth),
            supported_nodes_by_semantic_node=observed[:, a].sum((0, 1)).tolist(),
        )
    return matrix, losses, rows


def report(output, parts, losses, matrix, observed, gates, summaries):
    columns = [
        "frame_idx",
        "fitted_arm",
        "scoring_model",
        "view",
        "foreground",
        "near",
        "global",
        "total",
        "same_model_minus_direct_fit",
    ]
    with (output / "all_view_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for t, frame in enumerate(FRAMES):
            for a, arm in enumerate(ARMS):
                for r, model in enumerate(ARMS):
                    for v, view in enumerate(VIEWS):
                        p = parts[t, a, r, v]
                        writer.writerow(
                            dict(
                                zip(
                                    columns,
                                    (
                                        frame,
                                        arm,
                                        model,
                                        view,
                                        *p,
                                        p.sum(),
                                        (p - parts[t, 0, r, v]).sum(),
                                    ),
                                    strict=True,
                                )
                            )
                        )
    lines = [
        "# Independent audit of local reflection fits",
        "",
        "All 39 local results, all 585 stored iterates, fixed samples, final source support "
        "and both registered image/support gates were checked. No rendering or fitting was "
        "repeated; no raw video or GT was read.",
        "",
        "Rows below are fitted poses; columns are scoring models. Every value averages the "
        "same 13 opened development frames and three views. Errors are engineering mask "
        "responses against the same D2 labels, not identity probabilities.",
        "",
        "| Fitted arm | Direct scorer | Unit scorer | Fresnel scorer |",
        "|---|---:|---:|---:|",
    ]
    for a, arm in enumerate(ARMS):
        lines.append(f"| {arm} | " + " | ".join(f"{v:.9f}" for v in matrix[a]) + " |")
    lines += [
        "",
        "| Arm | Model-change gate | Same-model pose gate | Chest lost | "
        "Recovery mouth lost | Pass |",
        "|---|---|---|---:|---:|---|",
    ]
    for arm, row in gates.items():
        lines.append(
            f"| {arm} | {row['image_gate_vs_direct']} | {row['image_gate_same_model']} | "
            f"{len(row['lost_chest'])} | {len(row['lost_recovery_mouth'])} | "
            f"{row['gate_pass']} |"
        )
    lines += [
        "",
        "Positive changes below mean worse error. Each reflected arm is compared with "
        "the refitted direct pose under that same reflected scoring model.",
        "",
        "| Frame | Unit change | Fresnel change | Unit lost chest | Fresnel lost chest |",
        "|---|---:|---:|---:|---:|",
    ]
    for t, frame in enumerate(FRAMES):
        lost = [int((observed[t, 0, :, 1] & ~observed[t, a, :, 1]).sum()) for a in (1, 2)]
        lines.append(
            f"| {frame} | {losses[t, 1, 1] - losses[t, 0, 1]:+.9f} | "
            f"{losses[t, 2, 2] - losses[t, 0, 2]:+.9f} | {lost[0]} | {lost[1]} |"
        )
    lines += [
        "",
        "The CSV retains every view and foreground/near/global component, including "
        "worsening cases. The support table uses the original 2 mm ray / 12 pixel "
        "projection rule. Direct support still depends on imperfect SLEAP detections; "
        "a development gate pass would not establish physical geometry or biological IDs.",
        "",
        "The unbounded direct branch remains an inherited approximation. Nominal "
        "reflection faces are fixed; no floor survey, radiometry, temporal recovery or "
        "tracking validation was added. Large raw videos were not rehashed: their existing "
        "successful producer verification is inherited and explicitly outside this audit.",
    ]
    (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True)
    im = axes[0].imshow(matrix, cmap="viridis")
    axes[0].set_xticks(range(3), ARMS)
    axes[0].set_yticks(range(3), ARMS)
    axes[0].set(
        xlabel="Scoring model", ylabel="Fitted arm", title="Same sampled pixels / mean error"
    )
    for a in range(3):
        for r in range(3):
            axes[0].text(r, a, f"{matrix[a, r]:.5f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes[0], shrink=0.6)
    for ax, arm in zip(axes[1:], (1, 2), strict=True):
        delta = parts[:, arm, arm].sum(-1) - parts[:, 0, arm].sum(-1)
        limit = max(1e-12, float(np.max(np.abs(delta))))
        im = ax.imshow(delta, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(3), ["top", "side1", "side2"])
        ax.set_yticks(range(13), FRAMES)
        ax.set_title(f"{ARMS[arm]}: same-model pose change\nred = worse, blue = lower error")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Local development diagnostic; no tracking or biological ID validation")
    fig.savefig(output / "cross_evaluation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run.resolve(strict=True), args.output.resolve()
    require(root.is_relative_to(REPO) and output.is_relative_to(REPO), "workspace paths required")
    source = codes()
    bindings = {}
    seal = verified_outputs(root, "probe.provenance.json", "reflection_local_probe", bindings)
    require(
        str(root / "summary.json") in bindings and str(root / "plan.json") in bindings,
        "plan and summary must be bound outputs",
    )
    plan, summary = read(root / "plan.json"), read(root / "summary.json")
    require(
        plan["protocol_sha256"] == PROTOCOL_SHA
        and plan["frames"] == list(FRAMES)
        and plan["arms"] == list(ARMS)
        and plan["scoring_models"] == list(ARMS),
        "registered frame/model/protocol mismatch",
    )
    require(
        plan["tracker_config"]["opt_steps"] == 14
        and plan["tracker_config"]["device"] == "cpu"
        and plan["tracker_config"]["learning_rate_mm"] == 0.25
        and plan["threads"] == 2,
        "optimization conditions differ",
    )
    require(
        plan["code_hashes"][str(LAB / "run_reflection_local_probe.py")] == PRODUCER_SHA,
        "unexpected producer source",
    )
    for path, digest in plan["code_hashes"].items():
        actual = bounded_input(path, seal, bindings)
        require(bindings[str(actual)] == digest, "plan/live source mismatch")
        copied = root / "code_snapshot" / actual.relative_to(REPO)
        require(bindings.get(str(copied)) == digest, "missing/changed producer source snapshot")
    registrations = [
        Path(k) for k in seal["inputs"] if Path(k).name == "registration.provenance.json"
    ]
    require(len(registrations) == 1, "exactly one local registration seal required")
    bounded_input(registrations[0], seal, bindings)
    reg = verified_outputs(
        registrations[0].parent, registrations[0].name, "reflection_local_registration", bindings
    )
    require(read(reg["outputs"]["plan"]["path"]) == plan, "run differs from prior registration")
    require(reg["finished_at"] < seal["started_at"], "registration did not precede execution")
    geometry = Path(plan["run"])
    producer_path = bounded_input(geometry / "experiment.provenance.json", seal, bindings)
    producer = read(producer_path)
    require(producer["status"] == "ok", "original geometry producer incomplete")
    params = producer["params"]
    sequence = Path(params["sequence"])
    manifest = read(bounded_input(sequence / "manifest.json", seal, bindings))
    original = arrays(bounded_input(geometry / "candidate.npz", seal, bindings))
    exact(original["frame_idx"], np.arange(76620, 86580, dtype=np.int64))
    np.testing.assert_array_equal(original["frame_idx"], manifest["frames"])
    initial = np.load(
        bounded_input(geometry / "initial_pose_xyz.npy", seal, bindings), allow_pickle=False
    )
    parents = replay_chosen_parents(original, initial)
    calib = geometry / "calibration_snapshot"
    for path in calib.glob("*.json"):
        bounded_input(path, seal, bindings)
    rig = load_rig(calib)
    for view in VIEWS:
        bounded_input(
            Path(params["input_root"])
            / "work/c01_s16_predictions/bottomup"
            / f"{manifest['experiment']}__{view}.npz",
            seal,
            bindings,
        )
    nodes = CachedNodes(
        params["input_root"], manifest["experiment"], np.asarray(manifest["frames"])
    )
    output.mkdir(parents=True, exist_ok=False)
    with provenance("independent_reflection_local_output_audit", output, name="audit") as record:
        for path, digest in source.items():
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha(target) == digest, "audit snapshot mismatch")
            record.output(str(target.relative_to(output)), target, sha256=digest)
        all_parts, all_observed, frame_rows = [], [], []
        for t, frame in enumerate(FRAMES):
            fit_path, pixel_path = root / f"f{frame}_fits.npz", root / f"f{frame}_pixels.npz"
            require(
                str(fit_path) in bindings and str(pixel_path) in bindings, "unsealed frame output"
            )
            fitted, pixels = arrays(fit_path), arrays(pixel_path)
            require(
                int(fitted["frame_idx"]) == int(pixels["frame_idx"]) == frame, "wrong native frame"
            )
            prior = bounded_input(Path(plan["pixels_run"]) / pixel_path.name, seal, bindings)
            original_pixels = arrays(prior)
            require(set(pixels) == set(original_pixels), "pixel schema differs")
            for key in pixels:
                exact(pixels[key], original_pixels[key])
            index = frame - int(original["frame_idx"][0])
            exact(fitted["input_pose_xyz"], original["pose_xyz"][index])
            exact(fitted["original_identity_review"], original["identity_review"][index])
            exact(fitted["original_sources"], original["sources"][index])
            exact(fitted["original_observed"], original["observed"][index])
            for key in ("predicted_pose_xyz", "parent_missing", "parent_velocity"):
                exact(fitted[key], parents[key][index])
            rays = RayObservations.from_points(rig, {v: nodes.at(v, frame) for v in VIEWS})
            cfg = plan["tracker_config"]
            fixed = assignment_plans(
                fitted["input_pose_xyz"], rays, cfg["ray_sigma_mm"], cfg["missing_cost"], keep=1
            )[0][1]
            exact(fitted["fixed_plan"], fixed)
            require(
                fitted["iteration_pose_xyz"].shape == (3, 15, 3, 3, 3), "incomplete iterate poses"
            )
            for key, shape in (("iteration_scores", (3, 15)), ("iteration_loss_parts", (3, 15, 5))):
                require(
                    fitted[key].shape == shape and np.isfinite(fitted[key]).all(),
                    "nonfinite or incomplete objective history",
                )
            for a in range(3):
                history = fitted["iteration_pose_xyz"][a]
                require(
                    np.isfinite(history).all()
                    and np.all(history >= [0, 0, -6])
                    and np.all(history <= [192, 192, 186]),
                    "nonfinite/outside iterate",
                )
                exact(history[0], fitted["input_pose_xyz"])
                best = int(np.argmin(fitted["iteration_scores"][a]))
                require(best == fitted["best_step"][a], "best index differs")
                exact(fitted["pose_xyz"][a], history[best])
                near(fitted["score"][a], fitted["iteration_scores"][a, best])
                near(fitted["loss_parts"][a], fitted["iteration_loss_parts"][a, best])
                for step in range(15):
                    support_check(
                        fitted["iteration_sources"][a, step],
                        fitted["iteration_observed"][a, step],
                        fixed,
                    )
                expected_source, expected_observed, _ = supported_nodes(
                    fitted["pose_xyz"][a], fixed, rays, rig
                )
                exact(fitted["sources"][a], expected_source)
                exact(fitted["observed"][a], expected_observed)
                exact(fitted["sources"][a], fitted["iteration_sources"][a, best])
                exact(fitted["observed"][a], fitted["iteration_observed"][a, best])
            pose = fitted["pose_xyz"]
            exact(fitted["boundary_coordinates"], (pose <= [0, 0, -6]) | (pose >= [192, 192, 186]))
            near(fitted["body_segment_lengths_mm"], np.linalg.norm(np.diff(pose, axis=2), axis=-1))
            near(
                fitted["chest_pair_distances_mm"],
                np.stack(
                    [
                        np.linalg.norm(pose[:, a, 1] - pose[:, b, 1], axis=-1)
                        for a, b in ((0, 1), (0, 2), (1, 2))
                    ],
                    axis=-1,
                ),
            )
            near(
                fitted["predicted_chest_displacement_mm"],
                pose[:, :, 1] - fitted["predicted_pose_xyz"][:, 1],
            )
            parts = pixel_losses(fitted, pixels)
            means = parts.sum(-1).mean(-1)
            require(summary["rows"][t]["frame_idx"] == frame, "summary frame order differs")
            near(means, summary["rows"][t]["mean_errors_fit_arm_by_scoring_model"])
            exact(fitted["observed"], np.asarray(summary["rows"][t]["observed"], bool))
            all_parts.append(parts)
            all_observed.append(fitted["observed"])
            frame_rows.append(
                dict(
                    frame_idx=frame,
                    best_step=fitted["best_step"].tolist(),
                    boundary_counts=fitted["boundary_coordinates"].sum((1, 2, 3)).tolist(),
                    body_lengths_mm=fitted["body_segment_lengths_mm"].tolist(),
                    pair_distances_mm=fitted["chest_pair_distances_mm"].tolist(),
                    pose_displacement_from_direct_mm=np.linalg.norm(
                        pose - pose[0], axis=-1
                    ).tolist(),
                    support_changes={
                        arm: [
                            dict(
                                slot=int(f),
                                node=("mouth", "pectoral", "tail")[n],
                                direct=bool(fitted["observed"][0, f, n]),
                                candidate=bool(fitted["observed"][a, f, n]),
                            )
                            for f, n in np.argwhere(fitted["observed"][a] != fitted["observed"][0])
                        ]
                        for a, arm in enumerate(ARMS)
                    },
                )
            )
        parts, observed = np.stack(all_parts), np.stack(all_observed)
        matrix, losses, gates = verify_gates(summary, parts, observed)
        require(
            summary["scene_fits"] == 39
            and summary["frames"] == list(FRAMES)
            and summary["tracking"] is False
            and summary["identity_updated"] is False
            and summary["GT_read"] is False,
            "summary scope/completeness differs",
        )
        report(output, parts, losses, matrix, observed, gates, frame_rows)
        result = dict(
            status="passed",
            frames=list(FRAMES),
            fits=39,
            stored_iterates=585,
            strict_final_pose_supports_recomputed=39,
            all_outputs_verified=len(seal["outputs"]),
            cross_model_matrix=matrix.tolist(),
            cross_model_view_stratum_means=parts.mean(0).tolist(),
            screening=gates,
            frame_geometry=frame_rows,
            raw_decodes=0,
            raw_video_rehashes=0,
            pixel_renders=0,
            pose_fits=0,
            gt_read=False,
            identity_validation=False,
            sources_unchanged=True,
            limits="Large video/D2 provenance inherited from successful producer; final ray gates "
            "and every saved pixel loss recomputed. Intermediate ray projections not rerun.",
        )
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        np.savez_compressed(
            output / "recomputed_metrics.npz",
            frame_idx=np.asarray(FRAMES),
            per_view_stratum_errors=parts,
            mean_view_errors=losses,
            observed=observed,
        )
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=sha(path))
        for path, digest in {**bindings, **source}.items():
            record.input(path, path, sha256=digest)
        unchanged(bindings)
        unchanged(source)
        unchanged({v["path"]: v["sha256"] for v in record.outputs.values()})
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
