"""Three-native-frame raw crop sheets for inspecting local 2D label changes."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/clean2d_video_comparison_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIDEOS = ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4")


def text(image, message, x, y, scale=0.5):
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA
    )


def main():
    cv2.setNumThreads(1)
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    target = OUT / "inspection"
    target.mkdir(exist_ok=True)
    entries = []
    with provenance("direct_2d_raw_inspection_sheets", target) as rec:
        for case in protocol["cases"]:
            if not case["name"].startswith("known_"):
                continue
            center = case["centers"][0]
            for view, filename in zip(VIEWS, VIDEOS, strict=True):
                data = {}
                for arm in ("SLEAP", "P", "F"):
                    path = OUT / case["name"] / f"tracking_{arm}_{view}.npz"
                    with np.load(path, allow_pickle=False) as z:
                        data[arm] = {key: z[key] for key in ("points", "frame_idx", "labels")}
                    rec.input(case["name"] + "_" + view + "_" + arm, path, sha256=sha256(path))
                frames, points = data["SLEAP"]["frame_idx"], data["SLEAP"]["points"]
                nearby = points[(frames >= center - 8) & (frames < center + 9)].reshape(-1, 2)
                nearby = nearby[np.isfinite(nearby).all(axis=-1)]
                low, high = nearby.min(axis=0) - 80, nearby.max(axis=0) + 80
                length = min(1200, int(np.ceil(max(high - low))))
                corner = np.clip(np.floor((low + high - length) / 2), 0, 1200 - length).astype(int)
                x, y = corner
                video = ROOT / "work/three_body/source_cache/medaka_202512"
                video = video / protocol["experiment"] / view / filename
                cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
                cap.set(cv2.CAP_PROP_POS_FRAMES, center - 1)
                canvas = np.full((730, 1500, 3), 20, np.uint8)
                for column, frame in enumerate(range(center - 1, center + 2)):
                    ok, raw = cap.read()
                    if not ok:
                        raise RuntimeError(f"Failed to decode {view} f{frame}")
                    crop = cv2.resize(raw[y : y + length, x : x + length], (500, 500))
                    canvas[38:538, column * 500 : (column + 1) * 500] = crop
                    text(canvas, f"{view} f{frame}", column * 500 + 8, 25, 0.62)
                    rows = np.flatnonzero(frames == frame)
                    for order, row in enumerate(rows):
                        nodes = (points[row] - corner) * 500 / length
                        nodes[:, 0] += column * 500
                        nodes[:, 1] += 38
                        for node in nodes:
                            if np.isfinite(node).all():
                                cv2.circle(
                                    canvas,
                                    tuple(np.rint(node).astype(int)),
                                    3,
                                    (230, 210, 70),
                                    1,
                                    cv2.LINE_AA,
                                )
                        pectoral = nodes[1]
                        if np.isfinite(pectoral).all():
                            label = "/".join(
                                str(int(data[a]["labels"][row])) for a in ("SLEAP", "P", "F")
                            )
                            text(
                                canvas,
                                f"r{order}",
                                int(pectoral[0]) + 5,
                                int(pectoral[1]) - 8 - 14 * (order % 2),
                                0.52,
                            )
                            present = "".join(
                                letter if np.isfinite(point).all() else "-"
                                for letter, point in zip("MPT", points[row], strict=True)
                            )
                            text(
                                canvas,
                                f"r{order}: S/P/F = {label}    nodes:{present}",
                                column * 500 + 10,
                                560 + 20 * order,
                                0.49,
                            )
                cap.release()
                text(
                    canvas,
                    "S/P/F = saved SLEAP / P / F. r = row within each frame, NOT an identity.",
                    10,
                    690,
                    0.56,
                )
                text(
                    canvas,
                    "Common crop from nearby detections + 80 px; full aquarium is in the videos.",
                    10,
                    717,
                    0.5,
                )
                path = target / f"{view}__f{center}__three_frames.png"
                if not cv2.imwrite(str(path), canvas):
                    raise RuntimeError("PNG encoding failed")
                entry = dict(
                    case=case["name"],
                    view=view,
                    anchor=center,
                    native_frames=[center - 1, center, center + 1],
                    crop_xywh=[int(x), int(y), length, length],
                    path=str(path.relative_to(ROOT)),
                    sha256=sha256(path),
                )
                entries.append(entry)
                rec.output(case["name"] + "_" + view, path, sha256=entry["sha256"])
        manifest = target / "manifest.json"
        manifest.write_text(json.dumps(dict(assets=entries), indent=2) + "\n", encoding="utf-8")
        rec.output("manifest", manifest, sha256=sha256(manifest))
    print(f"Created {len(entries)} inspection sheets")


if __name__ == "__main__":
    main()
