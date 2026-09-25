"""CPU-only development dry run of six additional spatial-mask starts."""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(REPO / "src"))
from mask_initialization import FrameMaskVoxels, build_mask_initializations  # noqa: E402
from run_temporal_refine import sha256, verify_file  # noqa: E402
from temporal_refine import boundary_starts  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402


def run():
    output = REPO / "work/mask_initialization_v3_dry_run"
    if output.exists():
        raise FileExistsError(output)
    grid_root = REPO / "work/mask_grid_v3"
    diagnostic_path = grid_root / "diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    candidate = REPO / "work/three_body/v1_transient_detectron2/candidate.npz"
    forward_provenance = candidate.parent / "experiment.provenance.json"
    frozen = json.loads(forward_provenance.read_text())
    if frozen["status"] != "ok":
        raise ValueError("forward experiment did not complete")
    verify_file(candidate, frozen["outputs"]["candidate"]["sha256"])
    grid_path = grid_root / "fixed_grid.npz"
    verify_file(grid_path, diagnostic["grid_sha256"])
    input_hashes = {
        str(path): sha256(path)
        for path in (candidate, forward_provenance, grid_path, diagnostic_path)
    }
    with np.load(grid_path, allow_pickle=False) as grid:
        xyz = grid["xyz"]
    with np.load(candidate, allow_pickle=False) as data:
        selected = (data["frame_idx"] >= 333528) & (data["frame_idx"] < 333559)
        pose, observed, frames = (
            data["pose_xyz"][selected],
            data["observed"][selected],
            data["frame_idx"][selected],
        )
    evidence = []
    for row in diagnostic["rows"]:
        path = grid_root / row["output"]
        verify_file(path, row["sha256"])
        input_hashes[str(path)] = row["sha256"]
        with np.load(path, allow_pickle=False) as data:
            evidence.append(FrameMaskVoxels.from_saved(data))
    np.testing.assert_array_equal([e.frame_idx for e in evidence], frames)
    source_paths = [
        Path(__file__),
        LAB / "mask_initialization.py",
        LAB / "temporal_refine.py",
        LAB / "node_geometry.py",
        LAB / "renderer.py",
        LAB / "support_batch.py",
        LAB / "temporal_renderer.py",
        REPO / "src/medaka3d/provenance.py",
    ]
    source_hashes = {p.relative_to(REPO).as_posix(): sha256(p) for p in source_paths}
    output.mkdir()
    with provenance("mask_initialization_cpu_dry_run", output, name="initialization") as record:
        record.param("gpu_used", False)
        record.param("source_sha256", source_hashes)
        for path, digest in input_hashes.items():
            record.input(path, path, sha256=digest)
        for source in source_paths:
            target = output / "source_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        before_pose, before_observed = pose.copy(), observed.copy()
        started = time.perf_counter()
        result = build_mask_initializations(pose, observed, xyz, evidence)
        elapsed = time.perf_counter() - started
        np.testing.assert_array_equal(pose, before_pose)
        np.testing.assert_array_equal(observed, before_observed)
        expected = boundary_starts(pose, observed)
        source_rows = np.flatnonzero(expected[2] == "scene")
        for b, row in enumerate(source_rows):
            np.testing.assert_array_equal(
                result.pose_xyz[b, expected[3]], expected[0][row, expected[3]]
            )
        arrays_path = output / "initializations.npz"
        np.savez_compressed(arrays_path, **result.arrays)
        record.output("initializations", arrays_path, sha256=sha256(arrays_path))
        rows = []
        for b, permutation in enumerate(result.permutations):
            lo, hi = result.arrays["candidate_offsets"][[b * len(frames), (b + 1) * len(frames)]]
            rejected_lo, rejected_hi = result.arrays["rejected_offsets"][
                [b * len(frames), (b + 1) * len(frames)]
            ]
            rows.append(
                {
                    "permutation": permutation.tolist(),
                    "path_cost": float(result.path_costs[b]),
                    "missing_frames": int(np.sum(result.missing_slot[b] >= 0)),
                    "grid_selected_frames": int(np.sum(result.selected_grid_indices[b] >= 0)),
                    "candidate_scene_count": int(hi - lo),
                    "bounds_rejected_count": int(rejected_hi - rejected_lo),
                    "boundary_scenes_exact": True,
                }
            )
        for path, digest in input_hashes.items():
            verify_file(Path(path), digest)
        if any(sha256(REPO / p) != digest for p, digest in source_hashes.items()):
            raise RuntimeError("initializer source changed during dry run")
        summary = {
            "status": "ok",
            "elapsed_seconds": elapsed,
            "gpu_used": False,
            "frames": [333528, 333559],
            "rows": rows,
            "metadata": result.metadata,
            "source_sha256": source_hashes,
            "input_sha256": input_hashes,
        }
        path = output / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(json.dumps({"seconds": elapsed, "rows": rows}), flush=True)


if __name__ == "__main__":
    run()
