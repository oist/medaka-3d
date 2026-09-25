"""CPU-only diagnosis of saved mouth support; no pose optimization or ID correction."""

import hashlib
import json
import sys
from itertools import combinations, pairwise, product
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from observations import CachedNodes  # noqa: E402
from prepare_sequence import decode_frame_masks  # noqa: E402

from medaka3d.calib import load_rig, triangulate  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

RUN = ROOT / "work/three_body/v1_continuous_one_minute_detectron2"
SEQUENCE = ROOT / "work/three_body/inputs/continuous_one_minute_76620-86580"
NODE_ROOT = ROOT / "work/generative_tracking_inputs"
OUT = ROOT / "work/three_body/minute_scene1_mouth_diagnosis_v2"
START, STOP = 80261, 80432
ANALYSIS_START, ANALYSIS_STOP = 79964, 80464
LOWER, UPPER = np.array([0.0, 0.0, -6.0]), np.array([192.0, 192.0, 186.0])
SELECTED = (80261, 80320, 80339, 80344, 80370, 80431, 80463)
COLORS = ((255, 180, 40), (0, 170, 255), (220, 80, 220))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def runs(flag, frames):
    edge = np.diff(np.r_[False, flag, False].astype(int))
    return [
        [int(frames[a]), int(frames[b - 1]) + 1]
        for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1), strict=True)
    ]


def nearest_geometry(saved, rays, free_rows):
    cases = []
    for n_views in (3, 2):
        for views in combinations(range(3), n_views):
            for rows in product(*(free_rows[v] for v in views)):
                origin = np.array([rays[v][0][r, 0] for v, r in zip(views, rows, strict=True)])
                direction = np.array([rays[v][1][r, 0] for v, r in zip(views, rows, strict=True)])
                xyz, rms = triangulate(origin, direction)
                if not np.isfinite(xyz).all() or rms > 1.5:
                    continue
                if np.any(((xyz - origin) * direction).sum(-1) < 0):
                    continue
                cases.append(
                    dict(
                        xyz=xyz,
                        ray_rms_mm=float(rms),
                        views=[VIEWS[v] for v in views],
                        source_rows=list(rows),
                        distance_from_saved_mm=float(np.linalg.norm(xyz - saved)),
                        within_engineering_box=bool(np.all(xyz >= LOWER) and np.all(xyz <= UPPER)),
                    )
                )
    cases.sort(key=lambda row: (row["distance_from_saved_mm"], row["ray_rms_mm"]))
    return cases[:3]


def overlay(image, pose_uv, observed, points, masks, frame, view):
    canvas = image.copy()
    union = np.any(masks, axis=0)
    contours, _ = cv2.findContours(
        union.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, (60, 180, 60), 1)
    for row, instance in enumerate(points):
        mouth = instance[0]
        if np.isfinite(mouth).all():
            xy = tuple(np.rint(mouth).astype(int))
            cv2.circle(canvas, xy, 4, (255, 255, 255), 1)
            cv2.putText(
                canvas,
                f"r{row}",
                (xy[0] + 5, xy[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    for fish, uv in enumerate(pose_uv):
        for first, second in pairwise(uv):
            if np.isfinite(first).all() and np.isfinite(second).all():
                cv2.line(
                    canvas,
                    tuple(np.rint(first).astype(int)),
                    tuple(np.rint(second).astype(int)),
                    COLORS[fish],
                    2,
                    cv2.LINE_AA,
                )
        for node, pixel in enumerate(uv):
            if not np.isfinite(pixel).all():
                continue
            xy = tuple(np.rint(pixel).astype(int))
            if observed[fish, node]:
                cv2.circle(canvas, xy, 5, COLORS[fish], 2, cv2.LINE_AA)
            else:
                cv2.drawMarker(canvas, xy, COLORS[fish], cv2.MARKER_TILTED_CROSS, 11, 2)
        pixel = uv[1]
        if np.isfinite(pixel).all():
            cv2.putText(
                canvas,
                f"S{fish}",
                tuple(np.rint(np.add(pixel, [8, 13])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS[fish],
                2,
                cv2.LINE_AA,
            )
    pool = np.concatenate((pose_uv.reshape(-1, 2), points.reshape(-1, 2)))
    pool = pool[np.isfinite(pool).all(-1)]
    low = np.maximum(0, np.floor(pool.min(0) - 45).astype(int))
    high = np.minimum([1200, 1200], np.ceil(pool.max(0) + 45).astype(int))
    crop = canvas[low[1] : high[1], low[0] : high[0]]
    panel = np.full((650, 560, 3), 24, np.uint8)
    scale = min(550 / crop.shape[1], 565 / crop.shape[0])
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    panel[45 : 45 + len(resized), 5 : 5 + resized.shape[1]] = resized
    cv2.putText(
        panel,
        f"{frame}  {view}  ROI {low.tolist()}:{high.tolist()}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "O: 2+view direct   X: latent only",
        (8, 628),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "white r#: SLEAP mouth   green: D2 mask",
        (8, 646),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return panel, dict(native_crop_xyxy=[*low.tolist(), *high.tolist()], display_scale=scale)


def main():
    OUT.mkdir(exist_ok=False)
    seal = json.loads((RUN / "experiment.provenance.json").read_text())
    assert seal["status"] == "ok"
    candidate = RUN / "candidate.npz"
    assert sha256(candidate) == seal["outputs"]["candidate"]["sha256"]
    with np.load(candidate, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    frames = result["frame_idx"]
    manifest_path = SEQUENCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = json.loads((RUN / "sequence_manifest.json").read_text())
    assert manifest == expected_manifest
    calibration = RUN / "calibration_snapshot"
    rig = load_rig(calibration)
    nodes = CachedNodes(NODE_ROOT, manifest["experiment"], frames)
    paths = [
        Path(__file__),
        candidate,
        RUN / "experiment.provenance.json",
        manifest_path,
        RUN / "sequence_manifest.json",
        *(calibration.glob("*.json")),
        *nodes.paths.values(),
        LAB / "node_geometry.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        ROOT / "src/medaka3d/calib.py",
    ]
    hashes = {str(path): sha256(path) for path in paths}
    with provenance("diagnose_saved_minute_scene1_mouths", OUT, name="diagnosis") as record:
        record.param("tracking_or_optimization_executed", False)
        record.param("human_gt_or_baseline_trajectory_read", False)
        record.param("context_unchanged", [START, STOP])
        record.param("read_only_diagnostic_extent", [ANALYSIS_START, ANALYSIS_STOP])
        record.param("raw_full_file_hash_not_recomputed", True)
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        rows = []
        for frame in range(ANALYSIS_START, ANALYSIS_STOP):
            t = int(frame - frames[0])
            pose = result["pose_xyz"][t]
            points = {view: nodes.at(view, frame) for view in VIEWS}
            rays = [rig.camera(view).rays(points[view]) for view in VIEWS]
            projected = [rig.camera(view).project(pose) for view in VIEWS]
            for fish in range(3):
                per_view, free_rows = [], []
                for v, view in enumerate(VIEWS):
                    raw = points[view][:, 0]
                    # Match the original support routine's float32 assigned rays.
                    origin, direction = (x[:, 0].astype(np.float32) for x in rays[v])
                    delta = pose[fish, 0] - origin
                    residual = delta - (delta * direction).sum(-1)[:, None] * direction
                    distance = np.linalg.norm(residual, axis=-1)
                    pixel_error = np.linalg.norm(raw - projected[v][fish, 0], axis=-1)
                    other = set(result["sources"][t, np.arange(3) != fish, v, 0].tolist())
                    available = np.isfinite(raw).all(-1)
                    unused = available & ~np.isin(np.arange(len(raw)), list(other))
                    free_rows.append(np.flatnonzero(unused).tolist())
                    possible = available & (distance <= 2.0) & (pixel_error <= 12.0)
                    metric = (distance / 2.0) ** 2 + (pixel_error / 12.0) ** 2
                    metric[~unused] = np.inf
                    best = int(np.argmin(metric)) if np.isfinite(metric).any() else None
                    per_view.append(
                        dict(
                            view=view,
                            projected_mouth_uv=projected[v][fish, 0],
                            saved_mouth_source=int(result["sources"][t, fish, v, 0]),
                            raw_rows=len(raw),
                            finite_mouth_rows=int(available.sum()),
                            candidate_rows_passing_saved_pose_support=np.flatnonzero(
                                possible
                            ).tolist(),
                            unclaimed_rows_passing_saved_pose_support=np.flatnonzero(
                                possible & unused
                            ).tolist(),
                            nearest_unclaimed_row=best,
                            nearest_unclaimed_pixel_error_px=None
                            if best is None
                            else pixel_error[best],
                            nearest_unclaimed_ray_error_mm=None if best is None else distance[best],
                            raw_mouth_pixels=raw,
                        )
                    )
                rows.append(
                    dict(
                        frame=frame,
                        fish=fish,
                        mouth_xyz=pose[fish, 0],
                        mouth_direct=bool(result["observed"][t, fish, 0]),
                        all_nodes_direct=result["observed"][t, fish],
                        mask_supported=bool(result["mask_supported"][t, fish]),
                        body_supported=bool(result["body_supported"][t, fish]),
                        mouth_on_lower=np.isclose(pose[fish, 0], LOWER, rtol=0, atol=1e-8),
                        mouth_on_upper=np.isclose(pose[fish, 0], UPPER, rtol=0, atol=1e-8),
                        views=per_view,
                        nearest_unclaimed_ray_intersections=nearest_geometry(
                            pose[fish, 0], rays, free_rows
                        ),
                    )
                )
        diagnostic_path = OUT / "native_geometry.json"
        diagnostic_path.write_text(json.dumps(clean(rows), indent=2, allow_nan=False))
        record.output("native_geometry", diagnostic_path, sha256=sha256(diagnostic_path))
        mouth = result["observed"][:, :, 0]
        context = (frames >= START) & (frames < STOP)
        summary = dict(
            context=[START, STOP],
            context_frames=int(context.sum()),
            mouth_supported_per_slot=mouth[context].sum(0).tolist(),
            all_three_mouth_supported_context_frames=frames[context & mouth.all(1)].tolist(),
            nearest_all_three_supported_before_context=frames[(frames < START) & mouth.all(1)][
                -1:
            ].tolist(),
            nearest_all_three_supported_after_context=frames[(frames >= STOP) & mouth.all(1)][
                :1
            ].tolist(),
            fish=[],
            selected_images=[],
            input_sha256=hashes,
            limitations=[
                "Saved slot labels are not independently verified identities.",
                "Pre-fit plans were not saved; exact rejection stage is unavailable.",
                "Ray intersections are diagnostic; poses, bounds, IDs and probes stay fixed.",
                "D2 silhouette support does not directly observe an anatomical mouth.",
                "Selected pixels are SHA-verified; full video hashes are inherited.",
            ],
        )
        for fish in range(3):
            all_runs = runs(~mouth[:, fish], frames)
            selected_rows = [r for r in rows if r["fish"] == fish and START <= r["frame"] < STOP]
            summary["fish"].append(
                dict(
                    fish=fish,
                    missing_runs_touching_context=[
                        r for r in all_runs if r[0] < STOP and r[1] > START
                    ],
                    nearest_direct_before_context=frames[(frames < START) & mouth[:, fish]][
                        -1:
                    ].tolist(),
                    nearest_direct_after_context=frames[(frames >= STOP) & mouth[:, fish]][
                        :1
                    ].tolist(),
                    boundary_counts_lower=np.sum(
                        [r["mouth_on_lower"] for r in selected_rows], axis=0
                    ),
                    boundary_counts_upper=np.sum(
                        [r["mouth_on_upper"] for r in selected_rows], axis=0
                    ),
                    saved_sources_per_view=np.sum(
                        result["sources"][context, fish, :, 0] >= 0, axis=0
                    ),
                    at_least_two_unclaimed_rows_pass_final_pose=sum(
                        sum(
                            bool(v["unclaimed_rows_passing_saved_pose_support"]) for v in r["views"]
                        )
                        >= 2
                        for r in selected_rows
                        if not r["mouth_direct"]
                    ),
                    nearest_unclaimed_intersection_outside_box=sum(
                        bool(r["nearest_unclaimed_ray_intersections"])
                        and not r["nearest_unclaimed_ray_intersections"][0][
                            "within_engineering_box"
                        ]
                        for r in selected_rows
                        if not r["mouth_direct"]
                    ),
                )
            )
        panels_by_frame = {frame: [] for frame in SELECTED}
        for view in VIEWS:
            mask_path = SEQUENCE / manifest["masks"][view]["path"]
            assert sha256(mask_path) == manifest["masks"][view]["sha256"]
            record.input("masks:" + view, mask_path, sha256=sha256(mask_path))
            with np.load(mask_path, allow_pickle=False) as archive:
                masks = {key: archive[key] for key in archive.files}
            video_path = SEQUENCE / manifest["raw"][view]["path"]
            stat_before = (video_path.stat().st_size, video_path.stat().st_mtime_ns)
            capture = cv2.VideoCapture(str(video_path))
            assert capture.isOpened()
            try:
                for frame in SELECTED:
                    t = int(frame - frames[0])
                    assert capture.set(cv2.CAP_PROP_POS_FRAMES, t)
                    ok, image = capture.read()
                    assert ok and capture.get(cv2.CAP_PROP_POS_FRAMES) == t + 1
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    assert digest == manifest["raw"][view]["frame_bgr_sha256"][t]
                    raw_path = OUT / f"raw_f{frame}_{view}.png"
                    assert cv2.imwrite(str(raw_path), image)
                    record.output(raw_path.name, raw_path, sha256=sha256(raw_path))
                    decoded = decode_frame_masks(masks, t)
                    uv = rig.camera(view).project(result["pose_xyz"][t])
                    panel, crop = overlay(
                        image,
                        uv,
                        result["observed"][t],
                        nodes.at(view, frame),
                        decoded,
                        frame,
                        view,
                    )
                    panels_by_frame[frame].append(panel)
                    summary["selected_images"].append(
                        dict(
                            frame=frame,
                            view=view,
                            raw_bgr_sha256=digest,
                            **crop,
                            predicted_nodes_uv=uv,
                            saved_source_rows=result["sources"][t, :, VIEWS.index(view)],
                            untracked_sleap_nodes=nodes.at(view, frame),
                            d2_instance_count=len(decoded),
                            d2_empty_masks=int((~decoded.any((1, 2))).sum()),
                        )
                    )
            finally:
                capture.release()
            assert stat_before == (video_path.stat().st_size, video_path.stat().st_mtime_ns)
            record.param(
                "raw:" + view,
                dict(
                    path=str(video_path),
                    bytes=stat_before[0],
                    declared_sha256=manifest["raw"][view]["sha256"],
                ),
            )
        for frame, panels in panels_by_frame.items():
            path = OUT / f"overlay_f{frame}.png"
            assert cv2.imwrite(str(path), np.hstack(panels))
            record.output(path.name, path, sha256=sha256(path))
        assert {path: sha256(path) for path in hashes} == hashes
        path = OUT / "summary.json"
        path.write_text(json.dumps(clean(summary), indent=2, allow_nan=False))
        record.output("summary", path, sha256=sha256(path))
        (OUT / "audit.py").write_bytes(Path(__file__).read_bytes())
        record.output("audit.py", OUT / "audit.py", sha256=sha256(OUT / "audit.py"))
        print(
            json.dumps(
                clean(
                    {
                        key: summary[key]
                        for key in (
                            "mouth_supported_per_slot",
                            "nearest_all_three_supported_before_context",
                            "nearest_all_three_supported_after_context",
                            "fish",
                        )
                    }
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
