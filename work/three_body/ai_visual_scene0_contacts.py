"""Deterministic raw-image contact sheets for a separate provisional AI review."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

WORK = Path(__file__).resolve().parent
SOURCE = WORK / "long_reviews/scene0_65859-66223_v2"
OUTPUT = WORK / "ai_visual_scene0_provisional_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
ANCHORS = (65931, 66151, 66217)
ZOOMS = {
    65931: ((670, 460, 870, 660), (330, 990, 490, 1150), (450, 960, 630, 1140)),
    66151: ((840, 470, 1100, 890), (100, 940, 320, 1160), (450, 910, 760, 1060)),
    66217: ((950, 610, 1160, 900), (100, 880, 320, 1060), (540, 850, 900, 1020)),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--zoom", action="store_true")
    args = parser.parse_args()
    assert args.name.isidentifier()
    frames = [int(x) for x in args.frames.split(",")]
    assert frames and all(65859 <= x < 66223 for x in frames)
    assert not args.zoom or set(frames) <= set(ANCHORS)
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / (args.name + ".png")
    manifest = OUTPUT / (args.name + ".inputs.json")
    assert not target.exists() and not manifest.exists()
    canvas = Image.new("RGB", (1200, 440 * len(frames)), (230, 230, 230))
    draw = ImageDraw.Draw(canvas)
    inputs, panels = {}, []
    for row, frame in enumerate(frames):
        for column, view in enumerate(VIEWS):
            path = (
                SOURCE / f"f{frame}_{view}.png"
                if frame in ANCHORS
                else SOURCE / "frames" / f"f{frame}_{view}.jpg"
            )
            inputs[str(path)] = sha(path)
            with Image.open(path) as opened:
                raw = opened.convert("RGB")
                assert raw.size == (1200, 1200)
                box = ZOOMS[frame][column] if args.zoom else (0, 0, 1200, 1200)
                panel = raw.crop(box)
                panel.thumbnail((400, 400), Image.Resampling.NEAREST)
                if args.zoom:
                    scale = min(400 / panel.width, 400 / panel.height)
                    panel = panel.resize(
                        (round(panel.width * scale), round(panel.height * scale)),
                        Image.Resampling.NEAREST,
                    )
                canvas.paste(panel, (400 * column, 440 * row + 40))
            draw.text((400 * column + 4, 440 * row + 3), f"{frame} {view}", fill="black")
            draw.text((400 * column + 4, 440 * row + 19), f"native crop {box}", fill="black")
            panels.append(dict(frame=frame, view=view, source=str(path), crop=list(box)))
    canvas.save(target)
    for path, digest in inputs.items():
        assert sha(Path(path)) == digest
    result = dict(
        schema="ai_only_raw_contact_inputs_v1",
        raw_inputs=inputs,
        panels=panels,
        source_path=str(Path(__file__).resolve()),
        source_sha256=sha(Path(__file__)),
        output_path=str(target),
        output_sha256=sha(target),
        resampling="nearest, deterministic; no enhancement or inferred overlays",
        human_review=False,
        ground_truth=False,
    )
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
