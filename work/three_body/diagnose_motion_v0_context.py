"""State-conditioned image-motion diagnostic; frozen v0 poses are not truth."""
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, 'labs/20260909_three_body')
from motion import FlowConfig, FlowObservations
from prepare_sequence import decode_frame_masks
from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite(values):
    array = np.asarray(values, float)
    return np.where(np.isfinite(array), array, None).tolist()


root = Path('work/three_body')
sequence = root / 'inputs/transient_333519-333719'
candidate = root / 'v0_transient_detectron2/candidate.npz'
manifest_path = sequence / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
out = root / 'motion_v0_context.json'
if out.exists() or (root / 'motion_v0_context.provenance.json').exists():
    raise FileExistsError(out)
with np.load(candidate, allow_pickle=False) as arrays:
    poses, frames = arrays['pose_xyz'], arrays['frame_idx']
    # Only previous poses condition flow. These two arrays are reported after
    # construction, solely to mark the frozen v0's observation-loss context.
    observed, sources = arrays['observed'], arrays['sources']
assert frames.tolist() == manifest['frames']
calibration = Path('work/generative_tracking_inputs/calibration/fixed_refractive_192x192x186/phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration')
rig = load_rig(calibration)
config = FlowConfig()
mask_data = {}
for view in VIEWS:
    item = manifest['masks'][view]
    assert sha(sequence / item['path']) == item['sha256']
    with np.load(sequence / item['path'], allow_pickle=False) as arrays:
        mask_data[view] = {key: arrays[key] for key in arrays.files}
hashes = {str(path): sha(path) for path in [
    Path(__file__), Path('labs/20260909_three_body/motion.py'),
    Path('labs/20260909_three_body/prepare_sequence.py'), Path('src/medaka3d/calib.py'),
    Path('src/medaka3d/provenance.py'), Path('src/medaka3d/types.py')
]}
captures = {view: cv2.VideoCapture(str(sequence / manifest['raw'][view]['path'])) for view in VIEWS}
rows = []
with provenance('state_conditioned_image_motion_diagnostic', root, name='motion_v0_context') as record:
    record.input('frozen_v0_candidate', candidate, sha256=sha(candidate))
    record.input('raw_sequence_manifest', manifest_path, sha256=sha(manifest_path))
    record.param('flow_config', dataclasses.asdict(config))
    record.param('source_hashes', hashes)
    record.param('opencv_threads', cv2.getNumThreads())
    record.param('opencv', cv2.__version__)
    record.param('gpu_used_by_diagnostic', False)
    started = time.perf_counter()
    previous = None
    try:
        for index, frame in enumerate(frames):
            images, masks = {}, {}
            for view, capture in captures.items():
                ok, image = capture.read()
                assert ok and round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == index + 1
                assert hashlib.sha256(image.tobytes()).hexdigest() == manifest['raw'][view]['frame_bgr_sha256'][index]
                images[view] = image
                masks[view] = decode_frame_masks(mask_data[view], index)
            current = SimpleNamespace(frame_idx=int(frame), images_bgr=images, masks=masks)
            if previous is not None:
                tic = time.perf_counter()
                flow = FlowObservations.from_frames(previous, current, rig, poses[index - 1], config)
                elapsed = time.perf_counter() - tic
                rows.append({
                    'frame': int(frame), 'previous_frame': int(frames[index - 1]),
                    'motion_seconds': elapsed, 'node_valid': flow.node_valid.tolist(),
                    'view_valid': flow.valid.tolist(), 'used_views': flow.used_views.tolist(),
                    'feature_count': flow.feature_count.tolist(), 'fb_error_px': finite(flow.fb_error_px),
                    'quality': flow.quality.tolist(), 'pose_quality': flow.pose_quality.tolist(),
                    'pose_xyz': finite(flow.pose_xyz), 'ray_residual_mm': finite(flow.ray_residual_mm),
                    'node_step_mm': finite(np.linalg.norm(flow.pose_xyz - poses[index - 1], axis=-1)),
                    'pectoral_delta_xyz': finite(flow.pose_xyz[:, 1] - poses[index - 1, :, 1]),
                    'v0_previous_observed': observed[index - 1].tolist(),
                    'v0_current_observed': observed[index].tolist(),
                    'v0_previous_source_counts': (sources[index - 1] >= 0).sum(axis=1).tolist(),
                    'v0_current_source_counts': (sources[index] >= 0).sum(axis=1).tolist(),
                })
                if index % 25 == 0 or index == len(frames) - 1:
                    print(json.dumps({'frame': int(frame), 'processed_pairs': index,
                                      'elapsed_seconds': time.perf_counter() - started,
                                      'flow_node_valid': flow.node_valid.astype(int).tolist()}), flush=True)
            previous = current
    finally:
        for capture in captures.values():
            capture.release()
    times = np.asarray([row['motion_seconds'] for row in rows])
    node_valid = np.asarray([row['node_valid'] for row in rows])
    focus = [row for row in rows if 333531 <= row['frame'] <= 333550]
    all_lost = [row for row in rows if sum(row['v0_current_source_counts'][2]) == 0]
    any_supported = [row for row in rows if sum(row['v0_current_source_counts'][2]) > 0]
    summary = {
        'pairs': len(rows), 'flow_node_valid_counts_by_fish_node': node_valid.sum(axis=0).tolist(),
        'motion_total_s': float(times.sum()), 'motion_mean_s': float(times.mean()),
        'motion_median_s': float(np.median(times)), 'motion_p95_s': float(np.quantile(times, .95)),
        'estimated_six_parent_motion_mean_s': float(times.mean() * 6),
        'runtime_scope': 'CPU-only motion call, excludes raw decode; root GPU experiment concurrently active',
        'full_diagnostic_elapsed_s': time.perf_counter() - started,
        's2_when_any_v0_source_present': {'frames': len(any_supported),
            'flow_any_node_frames': sum(any(row['node_valid'][2]) for row in any_supported),
            'flow_pectoral_frames': sum(row['node_valid'][2][1] for row in any_supported)},
        's2_when_all_v0_sources_lost': {'frames': len(all_lost),
            'flow_any_node_frames': sum(any(row['node_valid'][2]) for row in all_lost),
            'flow_pectoral_frames': sum(row['node_valid'][2][1] for row in all_lost)},
    }
    assert all(sha(path) == digest for path, digest in hashes.items())
    payload = {
        'config': dataclasses.asdict(config), 'source_hashes': hashes,
        'candidate_npz_sha256': sha(candidate), 'sequence_manifest_sha256': sha(manifest_path),
        'calibration_sha256': {path.name: sha(path) for path in calibration.glob('*.json')},
        'summary': summary, 'focus_interval_inclusive': [333531, 333550], 'focus': focus, 'frames': rows,
        'limitations': [
            'Previous pose is frozen v0 output, conditioned on its offline backtrace; it is not ground truth or a live v1 state.',
            'Flow construction reads only previous pose, two raw frames and independent masks; no current SLEAP rows or baseline IDs.',
            'S2 is a frozen tracker slot name, not an independently verified physical animal identity.',
            'Proposed nodes are locally transported projections and require later tracker support; they are not direct SLEAP observations.',
            'No tracking integration, learned threshold tuning or identity accuracy result is evaluated here.'
        ],
    }
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    record.output('motion_context', out, sha256=sha(out))
print(json.dumps(summary, indent=2), flush=True)
