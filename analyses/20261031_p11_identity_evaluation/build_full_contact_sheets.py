"""Preserve initial review exports and repair contact-sheet label placement."""

import json
import os
import shutil
import tomllib
from pathlib import Path

import cv2
import numpy as np
from build_video_review import contact_tile
from run_pipeline import HERE, settings, sha256

from medaka3d.provenance import provenance


def main():
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / "config_row_mapping_v1.toml")
    cfg, paths, _, run = settings()
    policy = tomllib.loads((HERE / "config_video_review_v1.toml").read_text())
    pack = run / policy["version"]
    output = pack / "full_contact_sheets_v2"
    output.mkdir(exist_ok=False)
    video = paths.raw_data_dir / cfg["experiment"] / "top_cam/cam0_top.mp4"
    inventory = run / "source_frame_preview_v1/sources.json"
    expected = next(
        row["video_sha256"] for row in json.loads(inventory.read_text()) if row["view"] == "top_cam"
    )
    if sha256(video) != expected:
        raise ValueError("The top-camera source video changed")
    with provenance("p11_full_contact_sheets", output) as rec:
        for source in (
            Path(__file__),
            HERE / "build_video_review.py",
            HERE / "config_video_review_v1.toml",
            inventory,
            run / "native_timing.npz",
            pack / "examples.json",
            pack / "index.html",
        ):
            rec.input(source.name, source, sha256=sha256(source))
        rec.input("top_video", video, sha256=expected)
        rec.param(
            "change",
            "Move labels outside source images; original clips and v1 previews remain unchanged",
        )
        shutil.copy2(Path(__file__), output / Path(__file__).name)
        shutil.copy2(HERE / "build_video_review.py", output / "build_video_review.py")
        with np.load(run / "native_timing.npz") as timing:
            times = timing["top_cam"]
        capture = cv2.VideoCapture(str(video))
        try:
            for example in policy["examples"]:
                center = times[example["boundary_frame"]]
                tiles = []
                for delta in (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0):
                    frame = int(np.searchsorted(times, center + delta))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = capture.read()
                    if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                        raise ValueError(f"Native frame seek failed: {frame}")
                    panel = policy["panel_pixels"]
                    resized = cv2.resize(image, (panel, panel), interpolation=cv2.INTER_AREA)
                    tiles.append(contact_tile(resized, frame, times[frame], center))
                sheet = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, 9, 3)])
                target = output / f"{example['name']}_top_contact.png"
                if not cv2.imwrite(str(target), sheet):
                    raise OSError("Could not save repaired contact sheet")
                rec.output(example["name"], target, sha256=sha256(target))
        finally:
            capture.release()
        page = (pack / "index.html").read_text()
        for example in policy["examples"]:
            filename = f"{example['name']}_top_contact.png"
            page = page.replace(f'href="{filename}"', f'href="{output.name}/{filename}"')
        index = pack / "index_full_frame.html"
        if index.exists():
            raise FileExistsError(index)
        index.write_text(page, encoding="utf-8")
        rec.output("full_frame_review_page", index, sha256=sha256(index))
    print(index, flush=True)


if __name__ == "__main__":
    main()
