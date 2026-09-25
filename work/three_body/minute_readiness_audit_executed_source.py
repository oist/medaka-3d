"""Read-only resource inspection plus a bounded CPU record-allocation estimate.

No tracking, camera projection, GPU, labels or baseline inference is executed.
The allocation exercise mimics record dtypes/ownership visible in frozen source;
it is not a measurement of the running tracker process's resident memory.
"""

import ctypes
import gc
import hashlib
import json
import shutil
import sys
import tracemalloc
import zipfile
from ctypes import wintypes
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "work/three_body/minute_readiness_audit.json"


def memory_status():
    fields = [("length", wintypes.DWORD), ("load_percent", wintypes.DWORD)]
    fields += [(name, ctypes.c_ulonglong) for name in (
        "total_physical", "available_physical", "total_pagefile", "available_pagefile",
        "total_virtual", "available_virtual", "available_extended_virtual")]
    structure = type("MemoryStatus", (ctypes.Structure,), {"_fields_": fields})
    status = structure()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {name: getattr(status, name) for name, _ in fields}


def retained_history_allocation(frames=1000):
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    history = []
    for t in range(frames):
        poses = np.empty((18, 3, 3, 3), np.float64)
        parts = np.empty((18, 5), np.float64)
        support = np.zeros((18, 3), bool)
        local = np.zeros(18)
        cost = np.zeros(18)
        flow_nodes = np.zeros((3, 3), bool)
        records = []
        for candidate in range(6):
            records.append(dict(
                frame_idx=76620 + t, pose_xyz=poses[candidate],
                observed=np.zeros((3, 3), bool), sources=np.full((3, 3, 3), -1, np.int32),
                identity_review=np.zeros(3, bool), mask_supported=support[candidate],
                body_supported=np.zeros(3, bool), missing=np.zeros(3, np.int64),
                costs=local[candidate], path_costs=cost[candidate], loss_parts=parts[candidate],
                score_margin=float("inf"), parent=candidate, proposal_kind="predict",
                hypotheses=18, proposal_flow_nodes=flow_nodes, proposal_flow_parent=0,
            ))
        history.append(records)
    current, peak = tracemalloc.get_traced_memory()
    result = dict(simulated_frames=frames, retained_beam=6, proposal_batch=18,
                  retained_traced_bytes=current - before, peak_traced_bytes=peak - before,
                  extrapolated_retained_bytes_9960=(current - before) * 9960 / frames)
    del history
    tracemalloc.stop()
    gc.collect()
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    interval = REPO / "work/three_body/inputs/interval_76620-77286"
    manifest = json.loads((interval / "manifest.json").read_text(encoding="utf-8"))
    run = REPO / "work/three_body/v1_interval_detectron2"
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    raw_bytes = sum(item["bytes"] for item in manifest["raw"].values())
    mask_bytes, max_instances = 0, {}
    sleap_members = {}
    for view, item in manifest["masks"].items():
        with np.load(interval / item["path"], allow_pickle=False) as data:
            mask_bytes += sum(data[k].nbytes for k in data.files)
            max_instances[view] = int(np.diff(data["frame_offsets"]).max())
        cache = REPO / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
        with zipfile.ZipFile(cache / f"20251225_K_1_2_FCMC__{view}.npz") as archive:
            sleap_members[view] = {i.filename: i.file_size for i in archive.infolist()
                                   if i.filename in ("points.npy", "frame_idx.npy")}
    with np.load(run / "candidate.npz", allow_pickle=False) as arrays:
        result_bytes = sum(arrays[key].nbytes for key in arrays.files)
    factor = 9960 / 666
    paths = [REPO / "labs/20260909_three_body" / name for name in (
        "prepare_sequence.py", "run_experiment.py", "tracker.py", "observations.py", "motion.py")]
    paths.extend([Path(__file__), interval / "manifest.json", run / "summary.json"])
    hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    record = dict(
        purpose="one uninterrupted 9960-native-frame technical run readiness, read-only",
        python=sys.version, physical_memory_bytes=memory_status(),
        free_disk_bytes=shutil.disk_usage(REPO).free,
        prepare_conservative_preflight_bytes=9960 * 3 * 1200 * 1200 * 3 * 2,
        observed_666_frames=dict(raw_bytes=raw_bytes, packed_mask_array_bytes=mask_bytes,
                                maximum_instances_per_view=max_instances,
                                result_array_bytes=result_bytes,
                                staging_seconds=manifest["staging_elapsed_s"],
                                tracking_seconds=summary["tracking_seconds"]),
        linear_9960_estimate=dict(raw_bytes=raw_bytes * factor,
                                 packed_mask_array_bytes=mask_bytes * factor,
                                 result_array_bytes=result_bytes * factor,
                                 staging_seconds=manifest["staging_elapsed_s"] * factor,
                                 tracking_seconds=summary["tracking_seconds"] * factor),
        history_allocation=retained_history_allocation(),
        retained_sleap_members=sleap_members, source_and_reference_sha256=hashes,
        findings=[
            "No disk/RAM blocker found for one staged/tracked minute with current free resources.",
            "Images and per-frame decoded masks stream; optical flow retains one previous frame.",
            "PixelTargets, renderer and autograd graphs do not enter temporal history.",
            "History retains six small CPU records per time; ndarray views may retain all "
            "18 proposal poses/parts for that frame. The allocation exercise includes this.",
            "CachedNodes slices NPZ points without copying; each view can retain its full "
            "points member through the slice base, approximately 71 MiB for all three views.",
            "Packed mask arrays are resident for the whole interval; full raster masks are not.",
            "progress.jsonl contains counts only. No beam, parent history, flow frame or RNG "
            "checkpoint is saved. Interrupted inference cannot resume at the next frame.",
            "A new output directory and rerun from frame 76620 are required after interruption. "
            "Concatenated restarted runs cannot be described as uninterrupted tracking.",
            "candidate.npz is written only after the single tracker call and result validation. "
            "It can exist after a later evaluation/hash failure; provenance status and summary "
            "must also succeed before treating the experiment as complete.",
            "All run_experiment frozen_code_paths, including README.md and V1_PROTOCOL.md, "
            "must remain unchanged through the final post-evaluation source hash check.",
            "Identity review is sticky; long-run accepted availability is a quality outcome, "
            "not a resource or technical-completion guarantee.",
        ],
        limits=[
            "Extrapolations use a single 666-frame development interval and are not time bounds.",
            "Staging estimate includes the earlier staging environment and source access costs.",
            "Allocation simulation measures record storage, not full process or CUDA peak memory.",
            "The minute's masks were still being staged during this review.",
            "No GPU calls, tracking calls, new identity labels or future baseline arrays were read.",
        ],
    )
    OUTPUT.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: record[k] for k in (
        "free_disk_bytes", "prepare_conservative_preflight_bytes", "linear_9960_estimate",
        "history_allocation")}, indent=2))


if __name__ == "__main__":
    main()
