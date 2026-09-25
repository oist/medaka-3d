"""Render the sealed five-minute Sparse/Nodewise trajectories in a 3D tank view."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work" / "three_body"
DEFAULT_COMPARISON = WORK / "resumed_nodewise_five_minute_comparison_v1"
DEFAULT_SPARSE = WORK / "sparse_instance_five_minute_continuation_v1" / "candidate.npz"
DEFAULT_NODEWISE = (
    WORK / "nodewise_body_row_five_minute_resume_099600_v1" / "candidate.npz"
)
DEFAULT_OUTPUT = WORK / "nodewise_five_minute_3d_comparison_video_v1"

START = 66_400
STOP = 116_200
SOURCE_FPS = 166.0
WIDTH = 1600
HEIGHT = 900
PANEL_RECTS = ((25, 115, 785, 810), (815, 115, 1575, 810))
TANK_LOWER = np.array([0.0, 0.0, -6.0])
TANK_UPPER = np.array([192.0, 192.0, 186.0])
COLORS = ((54, 166, 255), (232, 207, 61), (220, 87, 218))  # BGR
NODE_NAMES = ("M", "C", "T")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"frame_idx", "pose_xyz", "observed", "identity_review"}
        if not required.issubset(archive.files):
            raise ValueError(f"missing candidate arrays in {path}")
        result = {name: archive[name] for name in required}
    frames = result["frame_idx"]
    if (
        frames.ndim != 1
        or not np.array_equal(frames, np.arange(int(frames[0]), int(frames[-1]) + 1))
        or result["pose_xyz"].shape != (len(frames), 3, 3, 3)
        or result["observed"].shape != (len(frames), 3, 3)
        or result["identity_review"].shape != (len(frames), 3)
        or not np.isfinite(result["pose_xyz"]).all()
    ):
        raise ValueError(f"invalid contiguous three-body candidate: {path}")
    keep = (frames >= START) & (frames < STOP)
    if not np.array_equal(frames[keep], np.arange(START, STOP)):
        raise ValueError(f"candidate does not cover the sealed five-minute interval: {path}")
    return {name: value[keep] for name, value in result.items()}


def view_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(42.0)
    elevation = np.deg2rad(24.0)
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    vertical = np.array(
        [
            -np.sin(elevation) * np.cos(azimuth),
            -np.sin(elevation) * np.sin(azimuth),
            np.cos(elevation),
        ]
    )
    depth = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    return horizontal, vertical, depth


HORIZONTAL, VERTICAL, DEPTH = view_basis()
BOX_CORNERS = np.array(
    [
        [x, y, z]
        for x in (TANK_LOWER[0], TANK_UPPER[0])
        for y in (TANK_LOWER[1], TANK_UPPER[1])
        for z in (TANK_LOWER[2], TANK_UPPER[2])
    ],
    dtype=float,
)
BOX_EDGES = tuple(
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if np.count_nonzero(BOX_CORNERS[i] != BOX_CORNERS[j]) == 1
)


def make_projector(rect: tuple[int, int, int, int]):
    x0, y0, x1, y1 = rect
    center = (TANK_LOWER + TANK_UPPER) / 2
    raw = np.stack(
        ((BOX_CORNERS - center) @ HORIZONTAL, (BOX_CORNERS - center) @ VERTICAL), axis=-1
    )
    lo, hi = raw.min(axis=0), raw.max(axis=0)
    scale = min((x1 - x0 - 70) / (hi[0] - lo[0]), (y1 - y0 - 70) / (hi[1] - lo[1]))
    midpoint = (lo + hi) / 2
    screen_center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        relative = points - center
        plane = np.stack((relative @ HORIZONTAL, relative @ VERTICAL), axis=-1)
        pixels = (plane - midpoint) * np.array([scale, -scale]) + screen_center
        return pixels, relative @ DEPTH

    return project


PROJECTORS = tuple(make_projector(rect) for rect in PANEL_RECTS)


def text(
    image: np.ndarray,
    value: str,
    point: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (225, 230, 236),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_tank(image: np.ndarray, project) -> None:
    pixels, depths = project(BOX_CORNERS)
    depth_lo, depth_hi = float(depths.min()), float(depths.max())
    edges = sorted(BOX_EDGES, key=lambda edge: float(depths[list(edge)].mean()))
    for a, b in edges:
        fraction = (float(depths[[a, b]].mean()) - depth_lo) / (depth_hi - depth_lo)
        shade = int(65 + 75 * fraction)
        cv2.line(
            image,
            tuple(np.rint(pixels[a]).astype(int)),
            tuple(np.rint(pixels[b]).astype(int)),
            (shade, shade, shade),
            1,
            cv2.LINE_AA,
        )
    origin = np.array([0.0, 0.0, -6.0])
    axes = ((np.array([42.0, 0.0, -6.0]), "x"), (np.array([0.0, 42.0, -6.0]), "y"),
            (np.array([0.0, 0.0, 36.0]), "z"))
    op, _ = project(origin)
    for endpoint, label in axes:
        ep, _ = project(endpoint)
        cv2.arrowedLine(image, tuple(np.rint(op).astype(int)), tuple(np.rint(ep).astype(int)),
                        (160, 164, 170), 1, cv2.LINE_AA, tipLength=0.12)
        text(image, label, tuple(np.rint(ep + np.array([4, -4])).astype(int)), 0.43,
             (174, 178, 184))


def draw_node(
    image: np.ndarray,
    point: np.ndarray,
    node: int,
    color: tuple[int, int, int],
    observed: bool,
) -> None:
    p = tuple(np.rint(point).astype(int))
    thickness = -1 if observed else 2
    if node == 0:
        cv2.circle(image, p, 6, color, thickness, cv2.LINE_AA)
    elif node == 1:
        cv2.rectangle(image, (p[0] - 6, p[1] - 6), (p[0] + 6, p[1] + 6), color,
                      thickness, cv2.LINE_AA)
    else:
        triangle = np.array([[p[0], p[1] - 7], [p[0] - 7, p[1] + 6],
                             [p[0] + 7, p[1] + 6]], np.int32)
        cv2.fillConvexPoly(image, triangle, color, cv2.LINE_AA) if observed else cv2.polylines(
            image, [triangle], True, color, 2, cv2.LINE_AA
        )


def draw_method(
    image: np.ndarray,
    method: dict[str, np.ndarray],
    index: int,
    panel_index: int,
    label: str,
) -> None:
    rect = PANEL_RECTS[panel_index]
    project = PROJECTORS[panel_index]
    x0, y0, x1, y1 = rect
    cv2.rectangle(image, (x0, y0), (x1, y1), (40, 44, 52), -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (83, 88, 99), 1)
    text(image, label, (x0 + 14, y0 + 27), 0.72, (244, 246, 249), 2)
    text(image, "latent 3D tank, millimetres", (x0 + 14, y0 + 50), 0.43,
         (164, 170, 181))
    draw_tank(image, project)

    pose = method["pose_xyz"][index]
    observed = method["observed"][index]
    points, depths = project(pose)
    history_start = max(0, index - int(SOURCE_FPS))
    for fish in range(3):
        trail = method["pose_xyz"][history_start : index + 1 : 4, fish, 1]
        trail_pixels, _ = project(trail)
        if len(trail_pixels) > 1:
            dim = tuple(int(channel * 0.45) for channel in COLORS[fish])
            cv2.polylines(image, [np.rint(trail_pixels).astype(np.int32)], False, dim, 2,
                          cv2.LINE_AA)

    for fish in np.argsort(depths[:, 1]):
        fish = int(fish)
        color = COLORS[fish]
        for a, b in ((0, 1), (1, 2)):
            cv2.line(image, tuple(np.rint(points[fish, a]).astype(int)),
                     tuple(np.rint(points[fish, b]).astype(int)), color, 3, cv2.LINE_AA)
        for node in range(3):
            draw_node(image, points[fish, node], node, color, bool(observed[fish, node]))
        chest = np.rint(points[fish, 1]).astype(int)
        text(image, f"S{fish}", (int(chest[0] + 10), int(chest[1] - 9)), 0.55, color, 2)

    direct = int(observed.sum())
    review = np.flatnonzero(method["identity_review"][index])
    review_label = "none" if not len(review) else ",".join(f"S{int(v)}" for v in review)
    text(image, f"direct nodes {direct}/9 | ID-review flag {review_label}",
         (x0 + 14, y1 - 17), 0.48, (205, 211, 221))


def render_frame(
    sparse: dict[str, np.ndarray],
    nodewise: dict[str, np.ndarray],
    index: int,
) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), (19, 22, 28), np.uint8)
    native = int(sparse["frame_idx"][index])
    elapsed = (native - START) / SOURCE_FPS
    recording = native / SOURCE_FPS
    text(image, "Three-fish generative 3D tracking: fixed Sparse vs Nodewise", (26, 34),
         0.82, (246, 247, 250), 2)
    text(
        image,
        f"comparison {elapsed:06.2f} / 300.00 s | recording {recording // 60:02.0f}:"
        f"{recording % 60:05.2f} | native frame {native}",
        (26, 64),
        0.62,
        (202, 209, 221),
        1,
    )
    text(image, "S0/S1/S2 are temporal slots, not verified female/male identities.",
         (26, 91), 0.54, (129, 174, 245), 1)

    draw_method(image, sparse, index, 0, "Sparse (fixed baseline)")
    draw_method(image, nodewise, index, 1, "Nodewise three-body tracker")

    difference = np.linalg.norm(
        sparse["pose_xyz"][index] - nodewise["pose_xyz"][index], axis=-1
    )
    text(image, f"same-slot Sparse/Nodewise RMS: {np.sqrt(np.mean(difference ** 2)):.3f} mm",
         (26, 842), 0.53, (210, 216, 226))
    legend_x = 910
    for fish, color in enumerate(COLORS):
        cv2.circle(image, (legend_x + fish * 92, 838), 6, color, -1, cv2.LINE_AA)
        text(image, f"S{fish}", (legend_x + 11 + fish * 92, 844), 0.48, color, 1)
    text(image, "M circle | C square | T triangle | hollow = latent (not directly supported)",
         (26, 875), 0.48, (174, 181, 192))
    progress_x0, progress_x1, progress_y = 1045, 1565, 870
    cv2.line(image, (progress_x0, progress_y), (progress_x1, progress_y), (72, 77, 87), 5)
    progress = index / (len(sparse["frame_idx"]) - 1)
    marker = round(progress_x0 + progress * (progress_x1 - progress_x0))
    cv2.line(image, (progress_x0, progress_y), (marker, progress_y), (105, 170, 238), 5)
    return image


def decode_qa(path: Path, expected_frames: int, expected_fps: float) -> dict:
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot decode output video: {path}")
        reported = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (HEIGHT, WIDTH, 3):
                raise RuntimeError(f"decoded shape differs: {frame.shape}")
            decoded += 1
    finally:
        capture.release()
    if decoded != expected_frames or reported != expected_frames or not np.isclose(
        fps, expected_fps, atol=0.02
    ):
        raise RuntimeError(
            f"video QA failed: decoded={decoded}, reported={reported}, fps={fps}"
        )
    return {
        "decode_ok": True,
        "decoded_frames": decoded,
        "reported_frames": reported,
        "fps": fps,
        "width": WIDTH,
        "height": HEIGHT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--sparse", type=Path, default=DEFAULT_SPARSE)
    parser.add_argument("--nodewise", type=Path, default=DEFAULT_NODEWISE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--encoder", choices=("nvenc", "x264"), default="nvenc")
    args = parser.parse_args()
    args.comparison = args.comparison.resolve()
    args.sparse = args.sparse.resolve()
    args.nodewise = args.nodewise.resolve()
    args.output = args.output.resolve()
    if args.step < 1:
        raise ValueError("step must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    seal_path = args.comparison / "comparison.provenance.json"
    seal = read_json(seal_path)
    if (
        seal.get("status") != "ok"
        or seal.get("stage") != "resumed_nodewise_five_minute_comparison_v1"
    ):
        raise ValueError("the source comparison is not the sealed successful five-minute result")
    input_hashes = {args.sparse: sha256(args.sparse), args.nodewise: sha256(args.nodewise)}
    for path, digest in input_hashes.items():
        if seal.get("inputs", {}).get(str(path)) != digest:
            raise ValueError(f"candidate is not the one bound by the comparison seal: {path}")

    sparse = load_candidate(args.sparse)
    nodewise = load_candidate(args.nodewise)
    if not np.array_equal(sparse["frame_idx"], nodewise["frame_idx"]):
        raise ValueError("Sparse and Nodewise axes differ")
    indices = np.arange(0, len(sparse["frame_idx"]), args.step, dtype=int)
    output_fps = SOURCE_FPS / args.step
    args.output.mkdir(parents=True)
    video = args.output / "sparse_vs_nodewise_3d_5min.mp4"
    log = args.output / "ffmpeg.log"
    codec = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21"] if args.encoder == "nvenc" else [
        "-c:v", "libx264", "-preset", "fast", "-crf", "20"
    ]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", f"{output_fps:.12g}", "-i", "-",
        "-an", *codec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ]
    preview_sequences = set(
        int(value) for value in np.linspace(0, len(indices) - 1, 6)
    )
    previews: list[str] = []
    started = time.perf_counter()
    with log.open("wb") as log_stream:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=log_stream, stderr=log_stream
        )
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin unavailable")
        try:
            for sequence, index in enumerate(indices):
                frame = render_frame(sparse, nodewise, int(index))
                process.stdin.write(frame.tobytes())
                if sequence in preview_sequences:
                    path = args.output / f"preview_{int(sparse['frame_idx'][index]):09d}.jpg"
                    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                        raise RuntimeError(f"cannot write preview: {path}")
                    previews.append(path.name)
                if sequence % 500 == 0:
                    print(
                        json.dumps(
                            {
                                "rendered": sequence + 1,
                                "total": len(indices),
                                "native_frame": int(sparse["frame_idx"][index]),
                                "elapsed_s": round(time.perf_counter() - started, 1),
                            }
                        ),
                        flush=True,
                    )
            process.stdin.close()
            return_code = process.wait()
        except BaseException:
            process.stdin.close()
            process.kill()
            process.wait()
            raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit {return_code}; see {log}")

    qa = decode_qa(video, len(indices), output_fps)
    metadata = {
        "schema": "nodewise_five_minute_3d_comparison_video_v1",
        "status": "ok",
        "scientific_interval": [START, STOP],
        "recording_time_seconds": [START / SOURCE_FPS, STOP / SOURCE_FPS],
        "source_fps": SOURCE_FPS,
        "sample_step": args.step,
        "output_fps": output_fps,
        "playback_duration_seconds": len(indices) / output_fps,
        "rendered_frames": len(indices),
        "encoder": args.encoder,
        "inputs": {
            str(args.sparse): input_hashes[args.sparse],
            str(args.nodewise): input_hashes[args.nodewise],
            str(seal_path): sha256(seal_path),
        },
        "outputs": {
            video.name: {"sha256": sha256(video), "size_bytes": video.stat().st_size},
            **{
                name: {
                    "sha256": sha256(args.output / name),
                    "size_bytes": (args.output / name).stat().st_size,
                }
                for name in previews
            },
        },
        "qa": qa,
        "interpretation": (
            "The left and right panels share source frames and temporal slot colours. "
            "S0/S1/S2 are not certified female/male identities. Hollow nodes are latent "
            "poses without direct support in that frame. This video is a visualization, "
            "not human identity ground truth."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    metadata_path = args.output / "video_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    viewer = (
        "<!doctype html><meta charset='utf-8'><title>Five-minute 3D comparison</title>"
        "<style>body{background:#14171d;color:#eef;font-family:system-ui;margin:24px}"
        "video,img{display:block;width:min(100%,1600px);margin:16px 0}</style>"
        "<h1>Sparse vs Nodewise: five-minute 3D comparison</h1>"
        "<p>Recording 6:40-11:40. S0/S1/S2 are temporal slots and are not certified "
        "female/male identities. Hollow nodes lack direct observation support.</p>"
        f"<video controls preload='metadata' src='{html.escape(video.name)}'></video>"
        + "".join(f"<img src='{html.escape(name)}' alt='{html.escape(name)}'>" for name in previews)
    )
    (args.output / "index.html").write_text(viewer, encoding="utf-8")
    print(json.dumps({"output": str(video), "metadata": str(metadata_path), "qa": qa}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
