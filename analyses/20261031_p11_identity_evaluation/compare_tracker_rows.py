"""Locate every changed C27 detection row between the frozen baseline and repair."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from run_pipeline import HERE, settings, sha256, write_json

from medaka3d.provenance import provenance


def main():
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / "config_row_mapping_v1.toml")
    cfg, _, root, repair = settings()
    baseline = root / cfg["output"]["relative_to_data_parent"] / cfg["comparison"]["baseline_run"]
    relative = (
        Path("pipeline/c27_uncapped_tracklets_any")
        / cfg["experiment"]
        / "mouth-pectoral_oks_serial"
    )
    if not (repair / relative / "c27_summary.json").is_file():
        raise ValueError("Wait for all three C27 cameras to finish")
    destination = repair / "c27_row_comparison"
    destination.mkdir(exist_ok=False)
    rows, counts = [], {}
    with provenance("p11_c27_row_comparison", destination) as rec:
        rec.input("code", Path(__file__), sha256=sha256(Path(__file__)))
        rec.param("baseline_run", baseline.name)
        rec.param("repair_run", repair.name)
        with np.load(repair / "native_timing.npz") as timing:
            top_times = timing["top_cam"]
            for view in ("top_cam", "side1_cam", "side2_cam"):
                old_path, new_path = (
                    run / relative / f"track2d__{view}.npy" for run in (baseline, repair)
                )
                old, new = np.load(old_path), np.load(new_path)
                if old.shape != new.shape:
                    raise ValueError(f"C27 input row grids differ: {view}")
                cached = (
                    repair
                    / "pipeline/c01_s16_predictions/bottomup"
                    / (f"{cfg['experiment']}__{view}.npz")
                )
                with np.load(cached) as data:
                    frames = data["frame_idx"]
                if frames.shape != new.shape:
                    raise ValueError("Detection and label row counts differ")
                changed = np.flatnonzero(old != new)
                changed_frames = frames[changed]
                starts = np.searchsorted(frames, changed_frames, side="left")
                counts[view] = {
                    "total_detection_rows": len(new),
                    "changed_rows": len(changed),
                    "changed_frames": len(np.unique(changed_frames)),
                    "changed_post_preparation_rows": int(
                        np.count_nonzero(
                            top_times[changed_frames] >= cfg["evaluation"]["preparation_seconds"]
                        )
                    ),
                }
                for i, row in enumerate(changed):
                    frame = int(changed_frames[i])
                    rows.append(
                        {
                            "view": view,
                            "frame": frame,
                            "global_row": int(row),
                            "frame_local_row": int(row - starts[i]),
                            "recording_time_s": float(top_times[frame]),
                            "camera_time_s": float(timing[view][frame]),
                            "baseline_label": int(old[row]),
                            "repair_label": int(new[row]),
                        }
                    )
                for name, path in (("baseline", old_path), ("repair", new_path), ("cache", cached)):
                    rec.input(f"{view}_{name}", path, sha256=sha256(path))
        table = pd.DataFrame(
            rows,
            columns=[
                "view",
                "frame",
                "global_row",
                "frame_local_row",
                "recording_time_s",
                "camera_time_s",
                "baseline_label",
                "repair_label",
            ],
        )
        table.to_csv(destination / "changed_rows.csv", index=False)
        write_json(destination / "summary.json", counts)
        rec.output("changed_rows", destination / "changed_rows.csv")
        rec.output("summary", destination / "summary.json")
    print(json.dumps(counts, indent=2))
    print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
