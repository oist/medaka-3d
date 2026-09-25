"""Separate exact anchor neighborhoods from the longer case summaries."""

from __future__ import annotations

import json

import numpy as np
from run import OUTPUT, ROOT, VIEWS, measure, predecessors, write_json

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance


def main():
    protocol_path = OUTPUT / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    human_path = ROOT / "docs/reviews/2026-09-16_multiview_identity_review.json"
    human = json.loads(human_path.read_text(encoding="utf-8"))
    recent = {row["frame"]: row for row in human["events"]}
    result = []
    with provenance("direct_2d_review_windows", OUTPUT) as rec:
        rec.input("protocol", protocol_path, sha256=sha256(protocol_path))
        rec.input("human_recent", human_path, sha256=sha256(human_path))
        for case in protocol["cases"]:
            for view in VIEWS:
                if not case["centers"]:
                    continue
                arrays, events = {}, {}
                for arm in ("SLEAP", "P", "N", "F", "F_all_loose"):
                    path = OUTPUT / case["name"] / f"tracking_{arm}_{view}.npz"
                    with np.load(path, allow_pickle=False) as z:
                        arrays[arm] = {key: z[key] for key in z.files}
                    events[arm] = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))[
                        "events"
                    ]
                    rec.input(case["name"] + "_" + view + "_" + arm, path, sha256=sha256(path))
                baseline = arrays["SLEAP"]
                previous, _ = predecessors(baseline["frame_idx"], baseline["labels"])
                for center in case["centers"]:
                    windows = [
                        ("anchor_17_native_frames", center - 8, center + 9),
                        ("video_166_native_frames", center - 83, center + 83),
                    ]
                    for kind, start, stop in windows:
                        stats, differences = {}, {}
                        for arm, data in arrays.items():
                            stats[arm], differences[arm] = measure(
                                data["frame_idx"],
                                data["points"],
                                data["labels"],
                                start,
                                stop,
                                previous if arm != "SLEAP" else None,
                                events[arm],
                            )
                        result.append(
                            dict(
                                case=case["name"],
                                view=view,
                                anchor=center,
                                window_kind=kind,
                                frame_start=start,
                                frame_stop=stop,
                                historical_note=case.get("note"),
                                recent_human_note=recent.get(center),
                                note_scope=(
                                    "Original annotations concern earlier 3D panels, "
                                    "not per-view 2D identity GT"
                                ),
                                arms=stats,
                                differences=differences,
                            )
                        )
        path = OUTPUT / "review_windows.json"
        write_json(
            path,
            dict(
                records=result,
                exact_swap_boundaries_known=False,
                anchor_17_window="Diagnostic vicinity, not human-annotated event interval",
            ),
        )
        rec.output("review_windows", path, sha256=sha256(path))
        print(f"Measured {len(result)} anchor/view/window combinations")


if __name__ == "__main__":
    main()
