"""Inspect only a prospective first frame; no later trajectory or labels as seed."""
from pathlib import Path
import json
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
from run_experiment import (BASELINE_DIR, CALIBRATION_DIR, EXPERIMENT, sha256,
                            skeleton_from_sources, staging_catalog, verified_input)
from observations import CachedNodes
from node_geometry import RayObservations, supported_nodes
from medaka3d.calib import load_rig
from medaka3d.types import VIEWS


def main():
    start, stop = 49800, 348600
    frames = np.array([start], np.int64)
    input_root = REPO / "work/generative_tracking_inputs"
    output = REPO / "work/three_body/long_window_first_frame_audit"
    if output.exists():
        raise FileExistsError(output)
    catalog, manifest_path = staging_catalog(input_root)
    relative = f"{BASELINE_DIR}/C34-label__{EXPERIMENT}__lam0.767.npz"
    baseline = verified_input(input_root, relative, catalog)
    hashes = {str(baseline): sha256(baseline), str(manifest_path): sha256(manifest_path),
              str(Path(__file__)): sha256(__file__)}
    for key in catalog:
        if key.startswith(CALIBRATION_DIR) or key.startswith("work/c01_s16_predictions/"):
            path = verified_input(input_root, key, catalog)
            hashes[str(path)] = sha256(path)
    with np.load(baseline, allow_pickle=False) as saved:
        source = saved["source"][frames].copy()
        valid = saved["slot_track"][frames] >= 0
    nodes = CachedNodes(input_root, EXPERIMENT, frames)
    rig = load_rig(input_root / CALIBRATION_DIR)
    pose, observed = skeleton_from_sources(rig, nodes, frames, source)
    rays = RayObservations.from_points(rig, {view: nodes.at(view, start) for view in VIEWS})
    plan = np.repeat(source[0, ..., None], 3, axis=-1)
    accepted_source, strict_observed, _ = supported_nodes(pose[0], plan, rays, rig)
    bounds = np.isfinite(pose).all() and bool(
        np.all(pose >= [0, 0, -6]) and np.all(pose <= [192, 192, 186]))
    report = {
        "scope": "prospective first-frame evidence only; no long run is started or certified",
        "start": start, "stop_exclusive": stop, "frames": stop-start, "fps": 166,
        "duration_seconds": (stop-start)/166,
        "reason_for_interval": "last exact 30 minutes include both known development intervals",
        "seed_source": source[0].tolist(), "baseline_slots_available": valid[0].tolist(),
        "seed_pose_xyz": pose[0].tolist(), "triangulated_nodes": observed[0].tolist(),
        "strict_direct_support": strict_observed.tolist(),
        "strict_sources": accepted_source.tolist(), "within_existing_engineering_bounds": bounds,
        "segment_lengths_mm": np.linalg.norm(np.diff(pose[0], axis=1), axis=-1).tolist(),
        "prospective_background_frames": list(range(5000, 45001, 5000)),
        "background_policy_status": "draft; requires separate registered protocol and implementation",
        "human_labels_opened": False, "later_baseline_frames_used_for_seeding": False,
        "first_pose_optimistic": "source row labels originated from whole-recording C34 baseline",
        "input_sha256": hashes,
    }
    if any(sha256(path) != value for path, value in hashes.items()):
        raise ValueError("prospective seed inputs changed")
    output.mkdir()
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                        encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "start", "duration_seconds", "baseline_slots_available", "strict_direct_support",
        "within_existing_engineering_bounds", "segment_lengths_mm")}))


if __name__ == "__main__":
    main()
