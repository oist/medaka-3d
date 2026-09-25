"""Read-only CPU diagnostics of saved three-dimensional body proximity.

Gaussian proximity uses the existing renderer's centers, axes and sigmas.
It is neither a finite-volume intersection test nor an identity criterion.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from prepare_sequence import sha256  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

ATTRIBUTION = REPO / "work/three_body/minute_scene0_silhouette_attribution"
RUN = REPO / "work/three_body/v1_continuous_one_minute_detectron2"
OUT = REPO / "work/three_body/minute_scene0_body_proximity"
FRAMES = [79316, 79318, 79322, 79355, 79370, 79372, 79381, 79405, 79547]
PAIRS = tuple(itertools.combinations(range(3), 2))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def segment_closest(a0, a1, b0, b1):
    """Global minimum on two closed 3-D segments, including zero length cases.

    The quadratic objective on [0,1]^2 reaches its minimum either at its
    unconstrained stationary point or at an edge's scalar minimum.
    """
    a0, a1, b0, b1 = (np.asarray(x, dtype=np.float64) for x in (a0, a1, b0, b1))
    require(
        all(x.shape == (3,) and np.isfinite(x).all() for x in (a0, a1, b0, b1)),
        "Segments must contain finite 3-D points",
    )
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc = float(u @ u), float(u @ v), float(v @ v)
    dd, ee = float(u @ w), float(v @ w)
    locations = []
    for s in (0.0, 1.0):
        t = float(np.clip((ee + bb * s) / cc, 0, 1)) if cc else 0.0
        locations.append((s, t))
    for t in (0.0, 1.0):
        s = float(np.clip((bb * t - dd) / aa, 0, 1)) if aa else 0.0
        locations.append((s, t))
    denominator = aa * cc - bb * bb
    if denominator > 64 * np.finfo(float).eps * aa * cc:
        s = (bb * ee - cc * dd) / denominator
        t = (aa * ee - bb * dd) / denominator
        if 0 <= s <= 1 and 0 <= t <= 1:
            locations.append((s, t))
    distances = [float(np.linalg.norm(w + s * u - t * v)) for s, t in locations]
    index = int(np.argmin(distances))
    s, t = locations[index]
    return dict(
        distance_mm=distances[index],
        fractions=[s, t],
        points_xyz=[(a0 + s * u).tolist(), (b0 + t * v).tolist()],
    )


def polyline_closest(a, b):
    require(np.shape(a) == np.shape(b) == (3, 3), "Each polyline must contain three 3-D nodes")
    choices = []
    for i, j in itertools.product(range(2), repeat=2):
        result = segment_closest(a[i], a[i + 1], b[j], b[j + 1])
        choices.append(dict(segment_indices=[i, j], **result))
    return min(choices, key=lambda item: item["distance_mm"])


def covariance(tangents, axial_sigma, radial_sigma):
    tangents = np.asarray(tangents, dtype=np.float64)
    axial_sigma, radial_sigma = np.asarray(axial_sigma), np.asarray(radial_sigma)
    require(
        np.allclose(np.linalg.norm(tangents, axis=-1), 1, rtol=0, atol=1e-12),
        "Gaussian tangents must be unit vectors",
    )
    require(
        np.isfinite(axial_sigma).all()
        and np.isfinite(radial_sigma).all()
        and (axial_sigma > 0).all()
        and (radial_sigma > 0).all(),
        "Gaussian sigmas must be finite and positive",
    )
    outer = tangents[..., :, None] * tangents[..., None, :]
    return (
        radial_sigma[..., None, None] ** 2 * np.eye(3)
        + (axial_sigma**2 - radial_sigma**2)[..., None, None] * outer
    )


def gaussian_pair_distances(centers_a, covariance_a, centers_b, covariance_b):
    delta = centers_a[:, None] - centers_b[None, :]
    combined = covariance_a[:, None] + covariance_b[None, :]
    require(
        np.isfinite(combined).all() and (np.linalg.eigvalsh(combined) > 0).all(),
        "Combined covariance must be positive definite",
    )
    inverse_delta = np.linalg.solve(combined, delta[..., None])[..., 0]
    standardized_sq = np.einsum("...i,...i->...", delta, inverse_delta)
    require(
        np.isfinite(standardized_sq).all() and (standardized_sq >= -1e-12).all(),
        "Nonfinite or negative squared standardized distance",
    )
    return np.linalg.norm(delta, axis=-1), np.sqrt(np.maximum(standardized_sq, 0))


def closest_gaussian_record(index, euclidean, standardized, centers_a, centers_b):
    i, j = (int(v) for v in index)
    return dict(
        gaussian_indices=[i, j],
        center_distance_mm=float(euclidean[i, j]),
        standardized_distance=float(standardized[i, j]),
        centers_xyz=[centers_a[i].tolist(), centers_b[j].tolist()],
    )


def plot_rows(rows, output):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    for pair, color in zip(PAIRS, ("tab:red", "tab:blue", "tab:green"), strict=True):
        selected = [row for row in rows if tuple(row["fish_pair"]) == pair]
        x = [row["frame_idx"] for row in selected]
        axes[0].plot(
            x,
            [row["pectoral_distance_mm"] for row in selected],
            "o-",
            color=color,
            label=f"S{pair[0]} / S{pair[1]}",
        )
        axes[1].plot(
            x, [row["polyline_closest"]["distance_mm"] for row in selected], "o-", color=color
        )
        axes[2].plot(
            x,
            [row["minimum_standardized_pair"]["standardized_distance"] for row in selected],
            "o-",
            color=color,
        )
    for ax, label in zip(
        axes,
        (
            "Pectoral distance (mm)",
            "Polyline distance (mm)",
            "Minimum combined-covariance distance",
        ),
        strict=True,
    ):
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[0].legend()
    axes[-1].set_xlabel(
        "Native frame index (nine selected measurements; connecting lines are visual guides)"
    )
    fig.suptitle("Saved 3-D body proximity: no refit, collision threshold or identity judgment")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    started = time.perf_counter()
    require(not OUT.exists(), "Body-proximity output already exists; no overwrite")
    attribution = read(ATTRIBUTION / "summary.json")
    seal = read(ATTRIBUTION / "attribution.provenance.json")
    require(
        seal["status"] == "ok" and attribution["frames"] == FRAMES,
        "Completed nine-frame attribution is required",
    )
    for item in seal["outputs"].values():
        require(sha256(item["path"]) == item["sha256"], "A sealed attribution output changed")
    previous_hashes = {
        Path(item["path"]).resolve(): item["sha256"] for item in seal["inputs"].values()
    }
    code = [
        Path(__file__),
        Path(__file__).with_name("test_scene0_body_proximity.py"),
        LAB / "renderer.py",
        LAB / "prepare_sequence.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
    ]
    needed = [
        RUN / "candidate.npz",
        RUN / "summary.json",
        RUN / "experiment.provenance.json",
        ATTRIBUTION / "summary.json",
        ATTRIBUTION / "attribution.provenance.json",
        *sorted((RUN / "calibration_snapshot").glob("*.json")),
        *code,
    ]
    inputs = {path.resolve(): sha256(path) for path in needed}
    for path, digest in inputs.items():
        if path in previous_hashes:
            require(
                digest == previous_hashes[path], f"Changed candidate/source/calibration: {path}"
            )
    require(
        inputs[(RUN / "candidate.npz").resolve()] == attribution["candidate_sha256"],
        "Candidate binding differs",
    )
    run = read(RUN / "summary.json")
    with np.load(RUN / "candidate.npz", allow_pickle=False) as candidate:
        require(len(candidate.files) == 19, "Expected frozen 19-array minute candidate")
        indices = np.searchsorted(candidate["frame_idx"], FRAMES)
        require(
            np.array_equal(candidate["frame_idx"][indices], FRAMES), "Exact source frames absent"
        )
        pose = candidate["pose_xyz"][indices]
        observed = candidate["observed"][indices]
        review = candidate["identity_review"][indices]
        saved_repulsion = candidate["loss_parts"][indices, 3]
    require(np.isfinite(pose).all(), "Latent bodies must be finite")
    cfg = RendererConfig(**attribution["renderer_config"])
    require(
        cfg.gaussians_per_fish == 8 and cfg.width_ratio == run["tracker_config"]["width_ratio"],
        "Renderer geometry differs from saved run",
    )
    rig = load_rig(RUN / "calibration_snapshot")
    torch.set_num_threads(2)
    # One valid native ray only initializes the existing API; gaussians() uses no image rays.
    renderer = RefractiveFishRenderer(
        rig, {rig.views[0]: np.array([[600.0, 600.0]])}, cfg, device="cpu", dtype=torch.float64
    )
    with torch.inference_mode():
        bodies = renderer.gaussians(pose)
    arrays = {
        field.name: getattr(bodies, field.name).cpu().numpy()
        for field in dataclasses.fields(bodies)
    }
    cov = covariance(arrays["tangents"], arrays["axial_sigma_mm"], arrays["radial_sigma_mm"])
    rows, euclidean_all, standardized_all, pectoral_terms = [], [], [], []
    for t, frame in enumerate(FRAMES):
        frame_euclidean, frame_standardized, frame_terms = [], [], []
        for a, b in PAIRS:
            distance = float(np.linalg.norm(pose[t, a, 1] - pose[t, b, 1]))
            e, d = gaussian_pair_distances(
                arrays["centers"][t, a], cov[t, a], arrays["centers"][t, b], cov[t, b]
            )
            ei, di = (
                np.unravel_index(np.argmin(e), e.shape),
                np.unravel_index(np.argmin(d), d.shape),
            )
            repulsion = float(np.exp(-(distance**2) / 8))
            rows.append(
                dict(
                    frame_idx=frame,
                    fish_pair=[a, b],
                    pectoral_distance_mm=distance,
                    existing_pectoral_repulsion=repulsion,
                    polyline_closest=polyline_closest(pose[t, a], pose[t, b]),
                    minimum_center_pair=closest_gaussian_record(
                        ei, e, d, arrays["centers"][t, a], arrays["centers"][t, b]
                    ),
                    minimum_standardized_pair=closest_gaussian_record(
                        di, e, d, arrays["centers"][t, a], arrays["centers"][t, b]
                    ),
                    observed=[observed[t, a].tolist(), observed[t, b].tolist()],
                    identity_review=[bool(review[t, a]), bool(review[t, b])],
                )
            )
            for key in ("minimum_center_pair", "minimum_standardized_pair"):
                pair_record = rows[-1][key]
                locations = list(zip((a, b), pair_record["gaussian_indices"], strict=True))
                for field in ("axial_sigma_mm", "radial_sigma_mm"):
                    pair_record[field] = [float(arrays[field][t, f, k]) for f, k in locations]
                pair_record["segment_indices"] = [0 if k < 2 else 1 for _, k in locations]
            frame_euclidean.append(e)
            frame_standardized.append(d)
            frame_terms.append(repulsion)
        euclidean_all.append(frame_euclidean)
        standardized_all.append(frame_standardized)
        pectoral_terms.append(frame_terms)
    terms = np.asarray(pectoral_terms)
    difference = terms.mean(-1) - saved_repulsion
    require(
        np.max(np.abs(difference)) < 1e-12,
        "Existing pectoral repulsion does not reproduce saved objective",
    )
    OUT.mkdir(parents=True)
    with provenance("fixed_pose_scene0_body_proximity", OUT, name="body_proximity") as record:
        for path, digest in inputs.items():
            record.input(str(path), path, sha256=digest)
        for path in code:
            target = OUT / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha256(target) == inputs[path.resolve()], "Source snapshot differs")
            record.output(str(target.relative_to(OUT)), target, sha256=sha256(target))
        path = OUT / "bodies_and_distances.npz"
        np.savez_compressed(
            path,
            frame_idx=np.asarray(FRAMES, np.int64),
            fish_pairs=np.asarray(PAIRS, np.int64),
            pose_xyz=pose,
            observed=observed,
            identity_review=review,
            **arrays,
            covariance_xyz=cov,
            gaussian_center_distances_mm=np.asarray(euclidean_all),
            gaussian_standardized_distances=np.asarray(standardized_all),
            existing_pectoral_repulsion_terms=terms,
            saved_repulsion=saved_repulsion,
            repulsion_difference=difference,
        )
        record.output("arrays", path, sha256=sha256(path))
        path = OUT / "pairs.csv"
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "frame_idx",
                    "fish_a",
                    "fish_b",
                    "pectoral_distance_mm",
                    "polyline_distance_mm",
                    "closest_center_mm",
                    "standardized_at_closest_center",
                    "minimum_standardized",
                    "center_mm_at_minimum_standardized",
                    "existing_pectoral_repulsion",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        row["frame_idx"],
                        *row["fish_pair"],
                        row["pectoral_distance_mm"],
                        row["polyline_closest"]["distance_mm"],
                        row["minimum_center_pair"]["center_distance_mm"],
                        row["minimum_center_pair"]["standardized_distance"],
                        row["minimum_standardized_pair"]["standardized_distance"],
                        row["minimum_standardized_pair"]["center_distance_mm"],
                        row["existing_pectoral_repulsion"],
                    ]
                )
        record.output("pairs", path, sha256=sha256(path))
        path = OUT / "proximity.png"
        plot_rows(rows, path)
        record.output("plot", path, sha256=sha256(path))
        require(
            all(sha256(path) == digest for path, digest in inputs.items()),
            "An input or frozen source changed",
        )
        summary = dict(
            status="complete_fixed_pose_diagnostic_not_collision_or_identity_truth",
            frame_idx=FRAMES,
            fish_pairs=PAIRS,
            rows=rows,
            renderer_config=dataclasses.asdict(cfg),
            candidate_sha256=attribution["candidate_sha256"],
            definitions=dict(
                pectoral="Euclidean distance between saved node1 centers",
                polyline=(
                    "Minimum Euclidean distance over all four pairs of closed mouth-pectoral / "
                    "pectoral-tail centerline segments; endpoints and degenerate segments "
                    "included; body thickness excluded"
                ),
                covariance=(
                    "Sigma = radial_sigma^2 I + (axial_sigma^2-radial_sigma^2) tangent tangent^T "
                    "from unmodified renderer.gaussians()"
                ),
                standardized=(
                    "sqrt(delta^T (Sigma_a + Sigma_b)^-1 delta), symmetric combined-covariance "
                    "scale; dimensionless, not overlap mass"
                ),
                gaussian_pair_selection=(
                    "Minimum Euclidean pair and minimum standardized pair are reported "
                    "separately; all 8x8 matrices saved"
                ),
                repulsion=(
                    "Existing per-pair exp(-pectoral_distance_mm^2/8); saved loss_part is "
                    "mean over the three pairs"
                ),
                interpretation=(
                    "Continuous model-proximity diagnostics only: no finite Gaussian support, "
                    "collision threshold, probability or true body-volume intersection is implied"
                ),
            ),
            original_repulsion_weight=run["tracker_config"]["repulsion_weight"],
            max_abs_saved_repulsion_difference=float(np.abs(difference).max()),
            repulsion_difference=difference.tolist(),
            all_bodies_nondegenerate=bool(arrays["nondegenerate"].all()),
            inputs_sha256_before={str(p): d for p, d in inputs.items()},
            inputs_sha256_after={str(p): d for p, d in inputs.items()},
            input_pose_changed=False,
            optimizer_steps=0,
            new_penalty=False,
            identity_assertion=False,
            runtime=dict(
                device="cpu",
                dtype="float64",
                torch_threads=2,
                python=sys.version,
                torch=str(torch.__version__),
                numpy=np.__version__,
                elapsed_s=time.perf_counter() - started,
            ),
            limitations=[
                "These are fitted latent bodies, including unsupported nodes, "
                "not measured physical animal volumes.",
                "Even exact contact between modeled body segments need not imply "
                "an identity violation.",
                "Two silhouettes may overlap in projection while separated in 3-D; "
                "no 2-D-to-3-D collision inference is applied.",
                "Eight tapered Gaussian ellipsoids have infinite tails and use engineering "
                "widths, not individually measured fish thickness.",
                "Only nine selected instants are measured; no temporal collision history "
                "or causal improvement of a new penalty is tested.",
            ],
        )
        path = OUT / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
    print(
        json.dumps(
            dict(
                out=str(OUT),
                elapsed_s=summary["runtime"]["elapsed_s"],
                max_repulsion_error=summary["max_abs_saved_repulsion_difference"],
            )
        )
    )


if __name__ == "__main__":
    main()
