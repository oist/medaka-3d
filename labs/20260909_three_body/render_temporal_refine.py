"""CPU-only raw-image review of forward tracking and hypothetical temporal fits."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(REPO / "src"))
from observations import SequenceReader  # noqa: E402
from run_temporal_refine import (  # noqa: E402
    CALIBRATION_DIR,
    load_sealed_forward,
    sha256,
    verify_file,
)

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

COLORS = ((40, 90, 255), (30, 230, 50), (255, 170, 20))
STILL_FRAMES = (333536, 333544, 333550)


def fixed_crops(projections):
    """One square native crop per camera, shared by every row/frame/hypothesis."""
    boxes = {}
    for view in VIEWS:
        points = np.concatenate([value[view].reshape(-1, 2) for value in projections])
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            boxes[view] = (0, 0, 1200, 1200)
            continue
        lo, hi = points.min(axis=0) - 90, points.max(axis=0) + 90
        extent = min(1200, max(360, int(np.ceil((hi - lo).max()))))
        corner = np.clip(np.floor((lo + hi - extent) / 2).astype(int), 0, 1200 - extent)
        boxes[view] = tuple(int(v) for v in [*corner, *(corner + extent)])
    return boxes


def draw_panel(frame, pixels, observed, inherited_review, view, crop, title, width=600):
    x0, y0, x1, y1 = crop
    native = frame.images_bgr[view].copy()
    masks = frame.masks[view]
    if masks is not None and len(masks):
        union = np.any(masks, axis=0).astype(np.uint8)
        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(native, contours, -1, (255, 255, 0), 1)
    for fish in range(3):
        uv = pixels[fish]
        if not np.isfinite(uv).all():
            continue
        xy = np.rint(uv).astype(np.int32)
        color = COLORS[fish]
        cv2.polylines(native, [xy], False, color, 2, cv2.LINE_AA)
        for node, point in enumerate(xy):
            cv2.circle(
                native,
                tuple(point),
                5 if node == 1 else 3,
                color,
                -1 if observed[fish, node] else 1,
                cv2.LINE_AA,
            )
        text = f"S{fish}" + (" ?" if inherited_review[fish] else "")
        cv2.putText(
            native,
            text,
            tuple(xy[1] + [8, -8]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    resized = cv2.resize(native[y0:y1, x0:x1], (width, width), interpolation=cv2.INTER_AREA)
    canvas = np.full((width + 82, width, 3), 24, np.uint8)
    canvas[82:] = resized
    lines = (
        title,
        f"{view} | source frame {frame.frame_idx}",
        "cyan: D2 | filled: observed | hollow/lines: latent",
        "?: original forward ID review (retained)",
    )
    for row, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (9, 18 + row * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def montage(
    frame,
    t,
    forward_uv,
    candidate_uv,
    forward_observed,
    candidate_observed,
    review,
    crops,
    hypothesis,
    permutation,
    slow_factor,
):
    rows = []
    for title, uv, held in (
        ("V1 forward estimate", forward_uv, forward_observed),
        (
            f"V2 H{hypothesis} hypothetical | right slots {[int(v) for v in permutation]}",
            candidate_uv,
            candidate_observed,
        ),
    ):
        rows.append(
            np.concatenate(
                [
                    draw_panel(frame, uv[view][t], held[t], review[t], view, crops[view], title)
                    for view in VIEWS
                ],
                axis=1,
            )
        )
    body = np.concatenate(rows, axis=0)
    canvas = np.full((body.shape[0] + 72, body.shape[1], 3), 18, np.uint8)
    canvas[72:] = body
    for y, text in (
        (
            24,
            f"Offline comparison | source frame {frame.frame_idx} | {slow_factor:g}x slower",
        ),
        (
            51,
            "V2: hypothetical, not accepted identity. Same raw pixels and crops in both rows.",
        ),
    ):
        cv2.putText(
            canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 1, cv2.LINE_AA
        )
    return canvas


def decode_qa(video_path, out, frames, fps, size):
    cap = cv2.VideoCapture(str(video_path))
    count, pictures = 0, []
    selected = {0, len(frames) // 2, len(frames) - 1}
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - fps) > 1e-5:
            raise ValueError("comparison video could not be decoded at its declared fps")
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if image.shape[:2] != (size[1], size[0]):
                raise ValueError("comparison video decoded with unexpected dimensions")
            if count in selected:
                path = out / f"decoded_{count:03d}_f{frames[count]}.png"
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError("could not save decoded QA frame")
                pictures.append(path.name)
            count += 1
    finally:
        cap.release()
    if count != len(frames):
        raise ValueError("comparison video lost or added source frames")
    return {
        "all_frames_decoded": count,
        "fps": fps,
        "dimensions_wh": list(size),
        "qa_images": pictures,
        "duration_seconds": count / fps,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("work/generative_tracking_inputs"))
    parser.add_argument("--slow-factor", type=float, default=16.0)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite temporal review: {args.output}")
    if not np.isfinite(args.slow_factor) or args.slow_factor < 1:
        raise ValueError("slow factor must be finite and at least one")
    provenance_path = args.refinement / "refinement.provenance.json"
    ref_record = json.loads(provenance_path.read_text(encoding="utf-8"))
    if ref_record["status"] != "ok":
        raise ValueError("temporal experiment must finish successfully before review")
    summary_path, hypotheses_path = (
        args.refinement / "summary.json",
        args.refinement / "hypotheses.npz",
    )
    for path, name in ((summary_path, "summary"), (hypotheses_path, "hypotheses")):
        verify_file(
            path, ref_record["outputs"][name]["sha256"], ref_record["outputs"][name]["size"]
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    start, stop = summary["frames"]
    forward, frames, pose, observed, _, input_hashes, _ = load_sealed_forward(
        args.run, args.sequence, args.input_root, start, stop
    )
    if sha256(args.run / "candidate.npz") != summary["forward_candidate_sha256"]:
        raise ValueError("refinement did not use this sealed forward candidate")
    if not set(STILL_FRAMES).issubset(frames):
        raise ValueError("this bounded review requires its three requested development frames")
    with np.load(hypotheses_path, allow_pickle=False) as data:
        hypotheses = {k: data[k] for k in data.files}
    np.testing.assert_array_equal(hypotheses["frame_idx"], frames)
    rankings = sorted(summary["permutation_rankings"], key=lambda x: x["rank"])
    best, second = (int(row["best_hypothesis"]) for row in rankings[:2])
    if best != int(np.argmin(hypotheses["costs"])):
        raise ValueError("summary best hypothesis disagrees with saved costs")
    with np.load(args.run / "candidate.npz", allow_pickle=False) as data:
        indexes = np.searchsorted(data["frame_idx"], frames)
        inherited_review = data["identity_review"][indexes]
    if inherited_review.dtype != bool or inherited_review.shape != (len(frames), 3):
        raise ValueError("forward ID-review flags have an invalid shape")
    rig = load_rig(args.input_root / CALIBRATION_DIR)
    poses = (pose, hypotheses["pose_xyz"][best], hypotheses["pose_xyz"][second])
    projections = [{view: rig.camera(view).project(value) for view in VIEWS} for value in poses]
    crops = fixed_crops(projections)
    reader = SequenceReader(args.sequence, args.input_root)
    sources = [
        Path(__file__),
        LAB / "run_temporal_refine.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        LAB / "run_experiment.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
    ]
    code_hashes = {p.relative_to(REPO).as_posix(): sha256(p) for p in sources}
    for path in (provenance_path, summary_path, hypotheses_path):
        input_hashes[str(path)] = sha256(path)
    args.output.mkdir(parents=True, exist_ok=False)
    with provenance("hypothetical_temporal_raw_review", args.output, name="review") as record:
        record.params_from(args)
        record.param("code_sha256", code_hashes)
        record.param("gpu_used", False)
        record.param("selected_hypothesis", best)
        record.param("second_permutation_hypothesis", second)
        for path, digest in input_hashes.items():
            record.input(path, path, sha256=digest)
        for source in sources:
            target = args.output / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        derived_path = args.output / "display_arrays.npz"
        np.savez_compressed(
            derived_path,
            frame_idx=frames,
            forward_pose_xyz=pose,
            forward_observed=observed,
            inherited_forward_identity_review=inherited_review,
            hypothesis_indices=np.asarray([best, second]),
            hypothesis_pose_xyz=hypotheses["pose_xyz"][[best, second]],
            hypothesis_observed=hypotheses["observed"][[best, second]],
            hypothesis_requires_identity_review=np.ones((2, len(frames), 3), bool),
            hypothesis_permutations=hypotheses["permutations"][[best, second]],
        )
        record.output("display_arrays", derived_path, sha256=sha256(derived_path))
        fps = 166 / args.slow_factor
        size, written, stills = (1800, 1436), [], []
        video_path = args.output / "forward_vs_hypothesis.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise RuntimeError(
                "OpenCV MP4 writer unavailable; no dependency installation attempted"
            )
        iterator = iter(reader)
        try:
            for frame in iterator:
                if frame.frame_idx < start:
                    continue
                if frame.frame_idx >= stop:
                    break
                t = frame.frame_idx - start
                image = montage(
                    frame,
                    t,
                    projections[0],
                    projections[1],
                    observed,
                    hypotheses["observed"][best],
                    inherited_review,
                    crops,
                    best,
                    hypotheses["permutations"][best],
                    args.slow_factor,
                )
                if image.shape[:2] != (size[1], size[0]):
                    raise ValueError("montage dimensions disagree with video writer")
                writer.write(image)
                written.append(frame.frame_idx)
                if frame.frame_idx in STILL_FRAMES:
                    path = args.output / f"f{frame.frame_idx}_forward_vs_H{best}.png"
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError("could not write raw comparison montage")
                    stills.append(path.name)
                if frame.frame_idx == 333544:
                    image = montage(
                        frame,
                        t,
                        projections[0],
                        projections[2],
                        observed,
                        hypotheses["observed"][second],
                        inherited_review,
                        crops,
                        second,
                        hypotheses["permutations"][second],
                        args.slow_factor,
                    )
                    path = args.output / f"f{frame.frame_idx}_forward_vs_H{second}_competitor.png"
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError("could not write competing hypothesis montage")
                    stills.append(path.name)
        finally:
            iterator.close()
            writer.release()
        np.testing.assert_array_equal(written, frames)
        qa = decode_qa(video_path, args.output, frames, fps, size)
        for path, digest in input_hashes.items():
            verify_file(Path(path), digest)
        if any(sha256(REPO / path) != digest for path, digest in code_hashes.items()):
            raise RuntimeError("review source changed during rendering")
        result = {
            "status": "ok",
            "source_frames": [start, stop],
            "native_fps": 166,
            "display_sampling_stride": 1,
            "display_slow_factor": args.slow_factor,
            "fps": fps,
            "selected_hypothesis": best,
            "selected_permutation": hypotheses["permutations"][best].tolist(),
            "selected_cost": float(hypotheses["costs"][best]),
            "second_permutation_hypothesis": second,
            "second_permutation": hypotheses["permutations"][second].tolist(),
            "second_cost": float(hypotheses["costs"][second]),
            "cost_gap": float(hypotheses["costs"][second] - hypotheses["costs"][best]),
            "crop_xyxy_by_view": crops,
            "stills": stills,
            "video": video_path.name,
            "video_decode_qa": qa,
            "code_sha256": code_hashes,
            "hypotheses_npz_sha256": input_hashes[str(hypotheses_path)],
            "forward_candidate_sha256": summary["forward_candidate_sha256"],
            "forward_mode": forward["mode"],
            "opencv": cv2.__version__,
            "limitations": [
                "All 12 source hypotheses are preserved; selecting a display is not ID acceptance.",
                "Forward review flags are retained; every V2 candidate remains hypothetical.",
                "The second display represents the second-ranked endpoint permutation.",
                "Filled nodes show direct support, not independent biological identity truth.",
            ],
        }
        metadata_path = args.output / "review_manifest.json"
        metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        for filename in [video_path.name, *stills, *qa["qa_images"], metadata_path.name]:
            path = args.output / filename
            record.output(filename, path, sha256=sha256(path))
        print(
            json.dumps(
                {
                    "status": "ok",
                    "video_frames": len(written),
                    "stills": stills,
                    "hypothesis": best,
                    "second": second,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
