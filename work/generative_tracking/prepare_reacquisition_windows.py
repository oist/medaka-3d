"""Select feasible first frames by calibration and frozen seed only, never new tracking."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "labs/20260908_generative_tracking"))
from benchmark import BASELINE_DIR, CALIBRATION_DIR, EXPERIMENT, CachedDetections
from benchmark import sha256, skeleton_from_sources
from medaka3d.calib import load_rig

root = repo / "work/generative_tracking_inputs"
rig, detections = load_rig(root / CALIBRATION_DIR), CachedDetections(root, EXPERIMENT)
seed_path = root / BASELINE_DIR / f"C34-label__{EXPERIMENT}__lam0.767.npz"
with np.load(seed_path) as arrays:
    sources, slot_track = arrays["source"], arrays["slot_track"]
female_path = root / "assets/groundtruth/fish_sex_gt" / EXPERIMENT / "female_sex_gt.csv"
with female_path.open(encoding="utf-8-sig", newline="") as stream:
    female = [int(r["frame"]) for r in csv.DictReader(stream) if r["gt_status"] == "confirmed"]
requested = [
    ("development_interval", 76620, 77286, None),
    ("development_transient", 333519, 333719, None),
    ("validation_swap_a", 99070, 99900, [99209, 99213]),
    ("validation_swap_b", 105250, 106080, [105541, 105541]),
    ("validation_occlusion", 323660, 324490, [323819, 323825]),
    ("validation_temporal_control", 165950, 166780, None),
    ("continuous_one_minute", 76620, 86580, [76930, 76930]),
]
windows = []
for name, requested_start, stop, event in requested:
    selected = None
    for frame in range(requested_start, max(0, requested_start - 166) - 1, -1):
        xyz, observed, residual = skeleton_from_sources(
            rig, detections, np.array([frame]), sources[frame:frame + 1])
        good = (observed.all() and (slot_track[frame] >= 0).all()
                and np.isfinite(xyz).all() and (xyz >= 0).all()
                and (xyz <= [192, 192, 186]).all()
                and (np.linalg.norm(np.diff(xyz[0], axis=1), axis=-1) >= .1).all())
        if good:
            selected = {
                "name": name, "requested_start": requested_start, "start": frame,
                "stop": stop, "n_frames": stop - frame, "seconds": (stop - frame) / 166,
                "event": event, "female_anchors": [f for f in female if frame <= f < stop],
                "seed_source": sources[frame].tolist(), "seed_pose_xyz": xyz[0].tolist(),
                "seed_segment_lengths_mm": np.linalg.norm(
                    np.diff(xyz[0], axis=1), axis=-1).tolist(),
                "seed_max_ray_residual_mm": float(np.nanmax(residual)),
                "adjustment_frames": requested_start - frame,
            }
            break
    if selected is None:
        raise ValueError(f"no complete in-bounds accepted seed within 166 preceding frames: {name}")
    windows.append(selected)
out = repo / "labs/20260908_generative_tracking/reacquisition_windows.json"
if out.exists():
    raise FileExistsError(out)
payload = {
    "experiment": EXPERIMENT, "fps": 166, "frame_intervals": "half-open",
    "selection_rule": "Preselected locations; latest accepted complete in-bounds .767 seed at or "
                      "within 166 preceding frames; stop fixed; no candidate tracking results used.",
    "seed_baseline_relative": seed_path.relative_to(root).as_posix(),
    "seed_baseline_sha256": sha256(seed_path),
    "female_annotations_relative": female_path.relative_to(root).as_posix(),
    "female_annotations_sha256": sha256(female_path),
    "windows": windows,
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps([{k:v for k,v in w.items() if not k.startswith("seed_")} for w in windows], indent=2))
