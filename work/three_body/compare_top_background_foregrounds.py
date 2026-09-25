"""Three saved top photographs: fixed contrast detector with old/new backgrounds.

Full-image foreground disagreement is descriptive, not fish accuracy or ID truth.
Only five existing PNG images are decoded; no video, D2 mask, SL, fit or GPU input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import prepare_top_background_5to6_minutes as prep

REPO, WORK = prep.REPO, prep.WORK
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
import observations  # noqa: E402

require = prep.require
STAGE = "top_background_foreground_comparison_v1"
FRAMES = (90141, 90258, 90259)
OLD = WORK / "saved_top_foreground_image_term_comparison_v1"
OLD_SHA = "fe2b4cd02f4ceb064c853a0d206d2fd5520d08a6adbde885438cb81dc3343a3b"
NEW = WORK / "top_background_5to6_minutes_v1"
NEW_SHA = "07de226b3ad2fe35b533bdd90650c98fbe9e46ba9cada9542699892b9bea142b"
PHOTOS = WORK / "endpoint_negative_native_photos_v1"
PHOTO_SHA = "dac90d016b249e45fd63d0030ec0f4b041a1b7f414f46cd93049ba29acf67256"
PINS = {
    "work/three_body/prepare_top_background_5to6_minutes.py": (
        "3c4bd35a83c192cc61082ef783fe5d3859438943d2ea23f62b4b25b6e8b6175c"
    ),
    "labs/20260909_three_body/observations.py": (
        "f1efcd27b31e63ce693229c9e765152acfc7984b3b8839f1792bcc2efd9cb5df"
    ),
    "labs/20260909_three_body/prepare_sequence.py": (
        "959e862cc58c5840a00a763d8a63b635de134492a637ac2f1e8a7131f25178b8"
    ),
    "src/medaka3d/provenance.py": (
        "1e2c260c959be0f240f244d064ec0f2b4c7f82cd7ecf0d9f29a82bdb1b2150c1"
    ),
    "src/medaka3d/types.py": "71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a",
}


def member(root, seal, name):
    path = (root / name).resolve()
    ref = seal["outputs"][name]
    require(Path(ref["path"]).resolve() == path, "output path differs")
    return ref


def document(root, seal, name, evidence):
    ref = member(root, seal, name)
    return evidence.read(ref["path"], ref["sha256"])


def load_context(evidence):
    old = evidence.read(OLD / "comparison.provenance.json", OLD_SHA)
    new = evidence.read(NEW / "preparation.provenance.json", NEW_SHA)
    photo = evidence.read(PHOTOS / "preparation.provenance.json", PHOTO_SHA)
    for seal, stage in ((old, OLD.name), (new, NEW.name), (photo, PHOTOS.name)):
        require(
            seal["status"] == "ok" and seal["stage"] == stage, "successful exact producer required"
        )
    old_plan = document(OLD, old, "plan.json", evidence)
    new_plan = document(NEW, new, "plan.json", evidence)
    new_frames = document(NEW, new, "frame_evidence.json", evidence)
    photo_frames = document(PHOTOS, photo, "frame_evidence.json", evidence)
    require(
        old_plan["frames"] == list(FRAMES) and old_plan["photo_sha256"] == PHOTO_SHA,
        "old comparison frame/photo lineage differs",
    )
    for name in ("preparation.provenance.json", "frame_evidence.json"):
        path = str((PHOTOS / name).resolve())
        require(
            old["inputs"].get(path) == evidence.records[path]["sha256"],
            "photo metadata was not consumed by old comparison",
        )
    old_background = old_plan["top_background"]
    require(
        old["inputs"].get(str(Path(old_background["path"]).resolve())) == old_background["sha256"],
        "old background consumption edge differs",
    )
    require(
        old_plan["background_median_native_frames"] == list(range(5000, 45001, 5000)),
        "old background times differ",
    )
    require(
        new_plan["native_frames"] == list(prep.FRAMES)
        and new_plan["all_background_samples_before_science"] is True
        and new_plan["source_sha256"]
        == PINS["work/three_body/prepare_top_background_5to6_minutes.py"],
        "new background generation contract differs",
    )
    new_background = new_frames["background"]
    require(
        new_background["image"] == member(NEW, new, "background_top_cam.png"),
        "new background output differs",
    )
    selected = []
    for frame in FRAMES:
        matches = [
            r for r in photo_frames["records"] if (r["frame_idx"], r["view"]) == (frame, "top_cam")
        ]
        require(len(matches) == 1, "unique top photograph required")
        row = matches[0]
        require(
            row["image"] == member(PHOTOS, photo, f"native/frame_{frame}_top_cam.png")
            and old["inputs"].get(row["image"]["path"]) == row["image"]["sha256"],
            "photo was not consumed by old comparison",
        )
        selected.append(row)
    obs_path = str((LAB / "observations.py").resolve())
    require(
        old["inputs"].get(obs_path) == PINS["labs/20260909_three_body/observations.py"],
        "foreground function differs from compared source",
    )
    return dict(
        old_background=old_background,
        new_background=new_background,
        photos=selected,
        old_background_frames=old_plan["background_median_native_frames"],
        new_background_frames=list(prep.FRAMES),
    )


def decode(ref, evidence, expected=None):
    data = evidence.capture(ref["path"], ref["sha256"])
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    require(
        image is not None and image.shape == (1200, 1200, 3) and image.dtype == np.uint8,
        "native BGR PNG required",
    )
    require(expected is None or prep.signature(image) == expected, "decoded BGR signature differs")
    image.flags.writeable = False
    return image


def component_summary(mask):
    require(mask.dtype == bool and mask.ndim == 2, "binary foreground required")
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = [
        dict(label=index, bbox_xywh=stats[index, :4].tolist(), area_pixels=int(stats[index, 4]))
        for index in range(1, count)
    ]
    total = int(np.count_nonzero(mask))
    require(sum(c["area_pixels"] for c in components) == total, "component area sum differs")
    return dict(
        pixel_count=total,
        component_count=len(components),
        components=components,
        mask_signature=prep.signature(mask),
        connectivity=8,
        bbox_convention="x,y,width,height; right/bottom excluded",
    )


def compare_masks(old, new):
    require(
        old.shape == new.shape and old.dtype == new.dtype == bool, "equal binary shapes required"
    )
    both, added, removed = old & new, new & ~old, old & ~new
    counts = dict(
        intersection_pixels=int(both.sum()),
        added_pixels=int(added.sum()),
        removed_pixels=int(removed.sum()),
        union_pixels=int((old | new).sum()),
    )
    require(
        int(old.sum()) == counts["intersection_pixels"] + counts["removed_pixels"]
        and int(new.sum()) == counts["intersection_pixels"] + counts["added_pixels"],
        "pixel partition differs",
    )
    return counts


def comparison_image(image, old, new, frame):
    panels = [image.copy()]
    for mask, color in ((old, (255, 220, 0)), (new, (230, 0, 255))):
        shown = image.copy()
        shown[mask] = np.rint(0.6 * image[mask] + 0.4 * np.asarray(color)).astype(np.uint8)
        panels.append(shown)
    height, width = image.shape[:2]
    canvas = np.zeros((height + 112, width * 3, 3), np.uint8)
    labels = [
        f"Native {frame} | {frame / 166:.3f} seconds",
        f"Old background | FG {int(old.sum())} pixels",
        f"5:00-6:00 background | FG {int(new.sum())} pixels",
    ]
    for column, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        x = width * column
        canvas[112:, x : x + width] = panel
        cv2.putText(canvas, label, (x + 18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(
            canvas,
            "Full image, same scale; foreground color alpha=0.4",
            (x + 18, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (230, 230, 230),
            1,
        )
        cv2.putText(
            canvas,
            "No fish labels / accuracy / identity truth",
            (x + 18, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (230, 230, 230),
            1,
        )
    return canvas


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output required",
    )
    output.mkdir(parents=True)
    started, evidence, outputs = prep.utc(), prep.Evidence(), {}

    def write(name, value):
        data = (
            value
            if isinstance(value, bytes)
            else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        )
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        outputs[name] = dict(path=str(path), sha256=prep.sha(data), size_bytes=len(data))
        return outputs[name]

    def png(name, value):
        okay, encoded = cv2.imencode(".png", value)
        require(okay, "PNG encode failed")
        return write(name, encoded.tobytes())

    try:
        for relative, digest in PINS.items():
            evidence.capture(REPO / relative, digest)
        require(
            Path(observations.__file__).resolve() == LAB / "observations.py",
            "foreground module import differs",
        )
        source = evidence.capture(Path(__file__))
        write("source_snapshot.py", source)
        context = load_context(evidence)
        write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                sources=PINS,
                inputs=context,
                old_comparison_sha256=OLD_SHA,
                new_background_sha256=NEW_SHA,
                detector=(
                    "frozen observations.foreground_contrast, median channel shift, "
                    "abs>=12, 8-connected area>=6"
                ),
                only_background_image_changes=True,
                fish_reference_regions_available=False,
            ),
        )
        old_bg = decode(context["old_background"], evidence)
        new_bg = decode(
            context["new_background"]["image"], evidence, context["new_background"]["bgr_signature"]
        )
        before = [prep.signature(value) for value in (old_bg, new_bg)]
        results = []
        for record in context["photos"]:
            frame = record["frame_idx"]
            image = decode(record["image"], evidence, record["bgr_signature"])
            old = observations.foreground_contrast(image, old_bg)
            new = observations.foreground_contrast(image, new_bg)
            require(prep.signature(image) == record["bgr_signature"], "photograph mutated")
            entry = dict(
                frame_idx=frame,
                recording_seconds=frame / 166,
                old=component_summary(old),
                new=component_summary(new),
                disagreement=compare_masks(old, new),
            )
            entry["old_mask_png"] = png(f"masks/f{frame}_old.png", old.astype(np.uint8) * 255)
            entry["new_mask_png"] = png(f"masks/f{frame}_new.png", new.astype(np.uint8) * 255)
            entry["comparison_png"] = png(
                f"frame_{frame}_comparison.png", comparison_image(image, old, new, frame)
            )
            write(f"cases/f{frame}.json", entry)
            results.append(entry)
        require([prep.signature(v) for v in (old_bg, new_bg)] == before, "background mutated")
        summary = dict(
            schema=STAGE,
            status="ok",
            frames=list(FRAMES),
            results=results,
            decoded_original_PNG_count=5,
            original_video_mask_SL_candidate_checkpoint_reads=0,
            GPU_calls=0,
            new_fit_or_trajectory=False,
            background_adopted=False,
            limits=[
                "No annotated fish reference; counts are foreground disagreement, not accuracy.",
                "Background timing changed; fish retention and causes need visual review.",
                "Three selected stills do not establish identity or continuous tracking.",
            ],
        )
        write("summary.json", summary)
        evidence.final_guard()
        for record in outputs.values():
            data = Path(record["path"]).read_bytes()
            require(
                prep.sha(data) == record["sha256"] and len(data) == record["size_bytes"],
                "saved output changed",
            )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=prep.utc(),
            inputs=evidence.records,
            outputs=dict(outputs),
        )
    except BaseException as error:
        write(
            "comparison.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=prep.utc(),
                inputs=evidence.records,
                outputs=dict(outputs),
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise
    write("comparison.provenance.json", seal)
    print(
        json.dumps(
            {
                "status": "ok",
                "frames": [
                    dict(
                        frame_idx=r["frame_idx"],
                        old_pixels=r["old"]["pixel_count"],
                        new_pixels=r["new"]["pixel_count"],
                        old_components=r["old"]["component_count"],
                        new_components=r["new"]["component_count"],
                        **r["disagreement"],
                    )
                    for r in results
                ],
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
