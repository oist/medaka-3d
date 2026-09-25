"""CPU-only sizing model; never executes tracking or allocates CUDA tensors."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"
TARGET_FRAMES = 298800
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def npz_headers(path):
    rows = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unexpected NPY header version: {version}")
            rows[info.filename.removesuffix(".npy")] = dict(
                shape=list(shape), dtype=str(dtype), nbytes=math.prod(shape) * dtype.itemsize,
                fortran_order=order, compressed_bytes=info.compress_size,
            )
    return rows


def allocation_model(n_frames, batch=18, beam=6):
    """Mirror retained record ownership, not the unknown actual full beam history."""
    history = []
    for frame_idx in range(49800, 49800 + n_frames):
        # CUDA fitter returns .cpu().numpy(); its CPU storage survives through
        # each selected view. Use CPU tensors of those exact result dimensions.
        poses = torch.zeros((batch, 3, 3, 3), dtype=torch.float64, device="cpu").numpy()
        parts = torch.zeros((batch, 5), dtype=torch.float64, device="cpu").numpy()
        masks = torch.zeros((batch, 3), dtype=torch.bool, device="cpu").numpy()
        cost = np.zeros(batch)
        paths = np.zeros(batch)
        flow_nodes = np.zeros((3, 3), bool)
        records = []
        for candidate in range(beam):
            records.append(dict(
                frame_idx=frame_idx, pose_xyz=poses[candidate],
                observed=np.zeros((3, 3), bool), sources=np.zeros((3, 3, 3), np.int32),
                identity_review=np.zeros(3, bool), mask_supported=masks[candidate],
                body_supported=np.zeros(3, bool), missing=np.zeros(3, int),
                costs=cost[candidate], path_costs=paths[candidate], loss_parts=parts[candidate],
                score_margin=float("inf"), parent=0, proposal_kind="image_flow_translation",
                hypotheses=batch, proposal_flow_nodes=flow_nodes, proposal_flow_parent=0,
            ))
        history.append(records)
    return history


def deep_size(value):
    """Deduplicate object identities and CPU tensor storages; count array bases."""
    seen, storages = set(), set()
    totals = Counter()

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        totals[type(obj).__name__ + "_objects"] += 1
        totals["python_and_owned_numpy_bytes"] += sys.getsizeof(obj)
        if isinstance(obj, np.ndarray):
            if obj.base is not None:
                visit(obj.base)
        elif isinstance(obj, torch.Tensor):
            assert obj.device.type == "cpu"
            storage = obj.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in storages:
                storages.add(key)
                totals["cpu_tensor_storage_bytes"] += storage.nbytes()
        elif isinstance(obj, dict):
            for key, item in obj.items():
                visit(key)
                visit(item)
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                visit(item)

    visit(value)
    totals["total_bytes"] = (
        totals["python_and_owned_numpy_bytes"] + totals["cpu_tensor_storage_bytes"]
    )
    return dict(totals)


def memory_status():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "total_page_file", "available_page_file",
                "total_virtual", "available_virtual", "available_extended_virtual",
            )
        ]

    result = MemoryStatus()
    result.length = ctypes.sizeof(result)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(result)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {name: int(getattr(result, name)) for name, _ in result._fields_}


def main():
    output = ROOT / "work/three_body/long_run_feasibility_v3"
    output.mkdir(exist_ok=False)
    code = [Path(__file__), *(LAB / name for name in (
        "tracker.py", "accelerated_tracker.py", "graph_fit.py", "source_sequence.py",
        "observations.py", "run_experiment.py", "chunk_masks.py",
    ))]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in code}
    allocation = {}
    for batch in (6, 18):
        model = allocation_model(1000, batch=batch)
        measured = deep_size(model)
        measured["model_frames"] = len(model)
        measured["projected_298800_frames_bytes"] = measured["total_bytes"] * TARGET_FRAMES / 1000
        allocation[str(batch)] = measured
        del model

    inputs = {}
    for sequence in ("transient_333519-333719", "continuous_one_minute_76620-86580"):
        views = {}
        for view in VIEWS:
            path = ROOT / f"work/three_body/inputs/{sequence}/masks/{view}.npz"
            headers = npz_headers(path)
            frames = headers["frame_idx"]["shape"][0]
            total = sum(item["nbytes"] for item in headers.values())
            views[view] = dict(
                path=str(path.relative_to(ROOT)), sha256=digest(path), headers=headers,
                frames=frames, resident_array_bytes=total,
                extrapolated_298800_frames_bytes=total * TARGET_FRAMES / frames,
            )
        inputs[sequence] = views
    sleap = {}
    for view in VIEWS:
        path = ROOT / (
            "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup/"
            f"20251225_K_1_2_FCMC__{view}.npz")
        sleap[view] = dict(headers=npz_headers(path), sha256=digest(path))
    candidate = ROOT / "work/three_body/graph_accelerated_transient_200_v1/candidate.npz"
    outputs = npz_headers(candidate)
    per_frame = sum(item["nbytes"] for item in outputs.values()) / 200
    summary_path = candidate.parent / "summary.json"
    timing = json.loads(summary_path.read_text())
    progress_path = ROOT / "work/three_body/v1_continuous_one_minute_detectron2/progress.jsonl"
    # This file is being appended by the main run. Save the actual read snapshot;
    # do not claim that this live progress file remains unchanged after the audit.
    progress = progress_path.read_bytes()
    complete_lines = progress.splitlines()
    if progress and not progress.endswith(b"\n"):
        complete_lines.pop()
    events = [json.loads(line) for line in complete_lines if line.strip()]
    latest = events[-1]
    (output / "minute_progress_snapshot.jsonl").write_bytes(b"\n".join(complete_lines) + b"\n")
    result = dict(
        created_utc=datetime.now(UTC).isoformat(),
        cpu_only=True, tracking_executed=False, cuda_allocations_requested=False,
        target_frames=TARGET_FRAMES, target_source_seconds=TARGET_FRAMES / 166,
        physical_memory_snapshot=memory_status(),
        allocation_model=allocation, packed_masks=inputs, sleap_full_archives=sleap,
        output_arrays=outputs, output_bytes_per_frame=per_frame,
        output_298800_frames_bytes=per_frame * TARGET_FRAMES,
        raw_fullframe_two_observations_bytes=2 * 3 * 1200 * 1200 * 3,
        mask_fullframe_two_observations_three_instances_per_view_bytes=2 * 3 * 3 * 1200 * 1200,
        accelerated_200_timing=dict(
            seconds=timing["tracking_seconds"], frames=200, sha256=digest(summary_path),
            simple_30minute_extrapolation_hours=(
                timing["tracking_seconds"] / 200 * TARGET_FRAMES / 3600),
        ),
        live_minute_progress_snapshot=dict(
            latest=latest,
            snapshot_sha256=hashlib.sha256(b"\n".join(complete_lines) + b"\n").hexdigest(),
            simple_30minute_extrapolation_hours=latest["elapsed_seconds"] / latest["processed"]
            * TARGET_FRAMES / 3600,
        ),
        code_sha256_before=hashes,
        limitations=[
            "Allocation model recreates ownership/dimensions without tracking; it is not RSS.",
            "1000 frames are allocated on CPU only; target memory is a linear estimate.",
            "Long masks and beam evolution are not measured; short-window rates can change.",
            "Runtime extrapolations exclude detector staging, pre/post hashes and evaluation.",
            "Minute progress is a snapshot of incomplete inference, not final timing.",
        ],
    )
    result["code_sha256_after"] = {str(path.relative_to(ROOT)): digest(path) for path in code}
    assert result["code_sha256_after"] == hashes
    (output / "audit.py").write_bytes(Path(__file__).read_bytes())
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(
        allocation=allocation,
        masks_gib={name: sum(v["extrapolated_298800_frames_bytes"] for v in views.values()) / 2**30
                   for name, views in inputs.items()},
        output_gib=per_frame * TARGET_FRAMES / 2**30,
        runtime_hours=[result["accelerated_200_timing"]["simple_30minute_extrapolation_hours"],
                       result["live_minute_progress_snapshot"]["simple_30minute_extrapolation_hours"]],
        memory=result["physical_memory_snapshot"],
    ), indent=2))


if __name__ == "__main__":
    main()
