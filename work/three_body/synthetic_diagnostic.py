"""Reproduce the first three-body synthetic findings without loading real data."""
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path("tests").resolve()))
import conftest

path = Path("tests/test_three_body_tracker.py")
spec = importlib.util.spec_from_file_location("tb_test_diagnostic", path)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
torch.set_num_threads(1)
out = Path("work/three_body")
calibration = out / "synthetic_calibration"
calibration.mkdir(parents=True, exist_ok=True)
base_rig = conftest.rig.__wrapped__(calibration)
pose = m.pose.__wrapped__()
rig, frame = m.mask_case.__wrapped__(base_rig, pose)
initial = pose + np.array([1.5, -1., .5])
base = m.config(opt_steps=14, learning_rate_mm=.2, motion_weight=0,
                repulsion_weight=0, review_gap_frames=1)
model = m.ThreeBodyTracker(rig, initial, replace(base, mode="detectron2", mask_weight=10))
result = model.track([frame])
final = result["pose_xyz"][0]
targets = m._OBSERVATIONS.sample_pixels(frame, n_positive=48, n_near=32, n_global=16)
renderer = m._MODULES["renderer"].RefractiveFishRenderer(rig, targets.pixels, dtype=torch.float64)


def losses(renderer, target, weight):
    output = {}
    for name, value in (("initial", initial), ("final", final), ("true_nodes", pose)):
        alpha = renderer.render(value).union_alpha
        output[name] = float(((alpha - torch.tensor(target)) ** 2 * torch.tensor(weight)).sum() / 3)
    return output


fit_losses = losses(renderer, targets.target, targets.weights)
heldout_pixels, heldout_target, heldout_weight = {}, [], []
for vi, view in enumerate(m.VIEWS):
    union = np.any(frame.masks[view], axis=0)
    height, width = union.shape
    used = np.zeros(height * width, bool)
    xy = targets.pixels[view].astype(int)
    used[xy[:, 1] * width + xy[:, 0]] = True
    band = cv2.dilate(union.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~union
    pools = [np.flatnonzero(union.ravel() & ~used), np.flatnonzero(band.ravel() & ~used)]
    rng = np.random.default_rng(629 + vi)
    selected = [rng.choice(pool, 96, replace=len(pool) < 96) for pool in pools]
    flat = np.concatenate(selected)
    assert not used[flat].any()
    yy, xx = np.unravel_index(flat, (height, width))
    heldout_pixels[view] = np.column_stack([xx, yy])
    heldout_target.append(np.r_[np.ones(96), np.zeros(96)])
    heldout_weight.append(np.full(192, 1 / 192))
heldout_renderer = m._MODULES["renderer"].RefractiveFishRenderer(
    rig, heldout_pixels, dtype=torch.float64)
heldout_losses = losses(heldout_renderer, np.array(heldout_target), np.array(heldout_weight))
mask_record = {
    "target_model": "Binary analytic ray/ellipsoid intersection; axial radius mouth-tail "
                    "distance/2, radial radius 0.095*distance, center midpoint mouth/tail; "
                    "not renderer output.",
    "initial_node_error_mm": np.linalg.norm(initial - pose, axis=-1).tolist(),
    "final_node_error_mm": np.linalg.norm(final - pose, axis=-1).tolist(),
    "initial_mean_pectoral_error_mm": float(
        np.linalg.norm(initial[:, 1] - pose[:, 1], axis=-1).mean()),
    "final_mean_pectoral_error_mm": float(np.linalg.norm(final[:, 1] - pose[:, 1], axis=-1).mean()),
    "fit_silhouette_loss": fit_losses,
    "within_image_disjoint_pixel_silhouette_loss": heldout_losses,
    "observed_count": int(result["observed"].sum()),
    "finite_accepted_xyz_count": int(np.isfinite(result["accepted_xyz"]).all(-1).sum()),
    "interpretation": "Mask fit improvement does not guarantee anatomical pectoral accuracy "
                      "under a different body-shape model. Disjoint pixels are within-image "
                      "checks, not independent animals or held-out video frames.",
}
print(json.dumps(mask_record), flush=True)
cross_initial = pose.copy()
cross_initial[:2] += np.array([[18., 25., 15.], [-21., -15., -5.]])[:, None]
truth, frames, sources = [], [], []
for index in range(8):
    current = cross_initial.copy()
    current[0] += [2 * index, 0, 0]
    current[1] -= [2 * index, 0, 0]
    observation, source = m.frame_with_nodes(base_rig, current, 100 + index)
    truth.append(current)
    frames.append(observation)
    sources.append(source)
cross = m.ThreeBodyTracker(base_rig, cross_initial,
                          m.config(opt_steps=8, learning_rate_mm=.3)).track(frames)
mismatches = []
for t, fish, view, node in np.argwhere(cross["sources"] != np.array(sources)):
    actual = cross["sources"][t, fish, view, node]
    expected = sources[t][fish, view, node]
    points = frames[t].detections[m.VIEWS[view]]
    separation = np.linalg.norm(points[actual, node] - points[expected, node])
    mismatches.append({"index": int(t), "fish": int(fish), "view": m.VIEWS[view],
                       "node": int(node), "pixel_separation": float(separation)})
cross_record = {
    "max_pectoral_distance_error_mm": float(np.linalg.norm(
        cross["pose_xyz"][:, :, 1] - np.array(truth)[:, :, 1], axis=-1).max()),
    "source_mismatches": mismatches,
    "interpretation": "3D seed-ID positions stayed correct; source mismatch occurs only "
                      "in one near-overlap camera view.",
}
print(json.dumps(cross_record), flush=True)
np.savez_compressed(
    out / "synthetic_findings_targets.npz", true_pose=pose, initial_pose=initial, final_pose=final,
    fit_pixels=np.stack([targets.pixels[v] for v in m.VIEWS]), fit_target=targets.target,
    fit_weight=targets.weights, heldout_pixels=np.stack([heldout_pixels[v] for v in m.VIEWS]),
    heldout_target=np.array(heldout_target), heldout_weight=np.array(heldout_weight),
    instance_masks=np.stack([frame.masks[v] for v in m.VIEWS]), cross_truth=np.array(truth),
    cross_pose=cross["pose_xyz"], cross_sources=cross["sources"],
    cross_expected_sources=np.array(sources),
)
snapshot = out / "synthetic_findings_test_v0.py"
if not snapshot.exists():
    snapshot.write_bytes(path.read_bytes())


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


record = {
    "purpose": "Preserved synthetic failures and distinct invariants; no biological ID evaluation",
    "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        path, Path(__file__), Path("labs/20260909_three_body/tracker.py"),
        Path("labs/20260909_three_body/renderer.py"))},
    "mask_case": mask_record, "crossing_case": cross_record,
    "mask_tracker_config": asdict(model.config),
    "mask_rig_cameras": [asdict(cam) for cam in rig.cameras],
    "initial_failing_test_snapshot": snapshot.name,
    "initial_failing_test_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    "targets_file": "synthetic_findings_targets.npz",
    "targets_sha256": hashlib.sha256(
        (out / "synthetic_findings_targets.npz").read_bytes()).hexdigest(),
}
(out / "synthetic_findings.json").write_text(
    json.dumps(record, indent=2, default=serial, allow_nan=False), encoding="utf-8")
