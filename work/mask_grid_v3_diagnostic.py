"""Development-only image-volume diagnostic; does not consume minute results or GT."""

import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(REPO / "src"))
from mask_grid import GridSupport, MaskGrid, MaskGridConfig  # noqa: E402
from observations import SequenceReader  # noqa: E402
from run_temporal_refine import (  # noqa: E402
    CALIBRATION_DIR,
    load_sealed_forward,
    sha256,
    verify_file,
)

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402


def finite_list(array):
    value = np.asarray(array)
    if value.ndim == 0:
        return float(value) if np.isfinite(value) else None
    return [finite_list(entry) for entry in value]


def pool_arrays(pool, prefix):
    return {prefix + field.name: getattr(pool, field.name) for field in dataclasses.fields(pool)}


def diagnose():
    out = REPO / "work/mask_grid_v3"
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    run = REPO / "work/three_body/v1_transient_detectron2"
    sequence = REPO / "work/three_body/inputs/transient_333519-333719"
    input_root = REPO / "work/generative_tracking_inputs"
    refinement = REPO / "work/three_body/temporal_v2_transient_333528-333559"
    _, frames, forward_pose, forward_observed, _, inputs, _ = load_sealed_forward(
        run, sequence, input_root, 333528, 333559
    )
    ref_provenance = refinement / "refinement.provenance.json"
    ref_record = json.loads(ref_provenance.read_text(encoding="utf-8"))
    if ref_record["status"] != "ok":
        raise ValueError("temporal reference has not finished")
    ref_summary_path, ref_array_path = refinement / "summary.json", refinement / "hypotheses.npz"
    for path, key in ((ref_summary_path, "summary"), (ref_array_path, "hypotheses")):
        verify_file(path, ref_record["outputs"][key]["sha256"])
        inputs[str(path)] = sha256(path)
    inputs[str(ref_provenance)] = sha256(ref_provenance)
    ref_summary = json.loads(ref_summary_path.read_text(encoding="utf-8"))
    with np.load(ref_array_path, allow_pickle=False) as arrays:
        best = int(np.argmin(arrays["costs"]))
        refined_pose = arrays["pose_xyz"][best]
        np.testing.assert_array_equal(arrays["frame_idx"], frames)
    if ref_summary["forward_candidate_sha256"] != sha256(run / "candidate.npz"):
        raise ValueError("temporal and forward input binding differ")
    code = [
        Path(__file__),
        LAB / "mask_grid.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        LAB / "run_temporal_refine.py",
        LAB / "run_experiment.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
    ]
    source_hashes = {p.relative_to(REPO).as_posix(): sha256(p) for p in code}
    config = MaskGridConfig()
    out.mkdir(parents=True, exist_ok=False)
    with provenance("development_mask_grid_spatial_proposals", out, name="diagnostic") as record:
        record.param("config", dataclasses.asdict(config))
        record.param("numpy", np.__version__)
        record.param("cpu_blas_threads", 1)
        record.param("gpu_used", False)
        record.param("source_hashes", source_hashes)
        record.param("diagnostic_region_radius_mm", 15.0)
        record.param("diagnostic_healthy_clearance_mm", 4.0)
        record.param("target_position_does_not_gate_candidate_pool", True)
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        for source in code:
            target = out / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        rig = load_rig(input_root / CALIBRATION_DIR)
        started = time.perf_counter()
        grid = MaskGrid.from_rig(rig, config)
        projection_seconds = time.perf_counter() - started
        grid_path = out / "fixed_grid.npz"
        np.savez_compressed(
            grid_path,
            xyz=grid.xyz,
            projected_uv=grid.projected_uv,
            grid_shape=np.asarray(grid.grid_shape),
            views=np.asarray(VIEWS),
        )
        record.output("fixed_grid", grid_path, sha256=sha256(grid_path))
        print(
            json.dumps(
                {"phase": "projected", "voxels": len(grid.xyz), "seconds": projection_seconds}
            ),
            flush=True,
        )
        reader = SequenceReader(sequence, input_root)
        iterator, results = iter(reader), []
        empty_segments = np.empty((0, 2, 3))
        try:
            for observation in iterator:
                frame = observation.frame_idx
                if frame < frames[0]:
                    continue
                if frame > frames[-1]:
                    break
                t = frame - frames[0]
                healthy_fish, segments = [], []
                for fish in (0, 1):
                    if not forward_observed[t, fish, :2].all():
                        continue
                    healthy_fish.append(fish)
                    for n in (0, 1):
                        if forward_observed[t, fish, n : n + 2].all():
                            segments.append(forward_pose[t, fish, n : n + 2])
                segments = np.stack(segments) if segments else empty_segments
                tic = time.perf_counter()
                support = grid.evaluate(observation.masks, segments)
                pool = grid.candidate_centers(support)
                ordinary = GridSupport(
                    support.samples,
                    np.full(len(grid.xyz), np.inf),
                    np.ones(len(grid.xyz)),
                    support.samples.mean_known_support,
                )
                ordinary_pool = grid.candidate_centers(ordinary)
                candidate_seconds = time.perf_counter() - tic
                # The pool above was constructed without either S2 location.
                query = np.concatenate((forward_pose[t, :, 1], refined_pose[t, :, 1]))
                point_support = grid.sample_points(query, observation.masks)
                active = np.flatnonzero(support.samples.at_least_two)
                strict = np.flatnonzero(support.samples.strict_three)
                contradicted = np.flatnonzero(support.samples.third_view_contradiction)
                target_centres = np.stack((query[2], query[5]))
                distances = np.linalg.norm(grid.xyz[:, None] - target_centres[None], axis=-1)
                pool_distance = np.linalg.norm(pool.xyz[:, None] - target_centres[None], axis=-1)
                region_stats = {}
                for k, label in enumerate(("forward_S2", "refined_S2")):
                    near = distances[:, k] <= 15
                    clear = support.healthy_distance_mm >= 4
                    region_stats[label] = {
                        "context_pectoral_xyz": target_centres[k].tolist(),
                        "radius_mm": 15,
                        "two_view_voxels": int(np.sum(near & support.samples.at_least_two)),
                        "strict_three_voxels": int(np.sum(near & support.samples.strict_three)),
                        "known_third_contradiction_voxels": int(
                            np.sum(near & support.samples.third_view_contradiction)
                        ),
                        "healthy_clear_two_view_voxels": int(
                            np.sum(near & clear & support.samples.at_least_two)
                        ),
                        "pool_candidates": int(np.sum(pool_distance[:, k] <= 15)),
                        "nearest_pool_distance_mm": float(pool_distance[:, k].min())
                        if len(pool.xyz)
                        else None,
                    }
                nonlocal_clear = (pool_distance > 15).all(axis=1) & (pool.healthy_distance_mm >= 4)
                path = out / f"f{frame}_support.npz"
                np.savez_compressed(
                    path,
                    frame_idx=np.asarray(frame),
                    two_view_or_more_voxel_indices=active,
                    strict_three_voxel_indices=strict,
                    known_third_contradiction_voxel_indices=contradicted,
                    active_view_support=support.samples.view_support[active],
                    active_known_view_count=support.samples.known_view_count[active],
                    active_healthy_distance_mm=support.healthy_distance_mm[active],
                    active_novelty_weight=support.novelty_weight[active],
                    active_ranking_score=support.ranking_score[active],
                    view_available=support.samples.view_available,
                    view_informative=support.samples.view_informative,
                    healthy_segments_xyz=segments,
                    healthy_forward_slots=np.asarray(healthy_fish),
                    query_pectoral_xyz=query,
                    query_view_support=point_support.view_support,
                    **pool_arrays(pool, "pool_"),
                    **pool_arrays(ordinary_pool, "unweighted_pool_"),
                )
                record.output(path.stem, path, sha256=sha256(path))
                row = {
                    "frame": frame,
                    "two_view_voxels": len(active),
                    "strict_three_voxels": len(strict),
                    "known_third_contradiction_voxels": len(contradicted),
                    "view_available": support.samples.view_available.tolist(),
                    "view_informative": support.samples.view_informative.tolist(),
                    "healthy_forward_slots": healthy_fish,
                    "healthy_segments": len(segments),
                    "pool_count": len(pool.xyz),
                    "pool_strict_three": int(np.sum(pool.count_supported == 3)),
                    "pool_known_third_contradiction": int(np.sum(pool.third_view_contradiction)),
                    "pool_clear_nonlocal_from_both_S2_contexts": int(nonlocal_clear.sum()),
                    "forward_pectoral_view_support": finite_list(point_support.view_support[:3]),
                    "refined_pectoral_view_support": finite_list(point_support.view_support[3:]),
                    "S2_context_regions": region_stats,
                    "candidate_generation_seconds": candidate_seconds,
                    "output": path.name,
                    "sha256": sha256(path),
                }
                results.append(row)
                print(
                    json.dumps(
                        {
                            "frame": frame,
                            "two": len(active),
                            "strict3": len(strict),
                            "pool": len(pool.xyz),
                            "seconds": candidate_seconds,
                        }
                    ),
                    flush=True,
                )
        finally:
            iterator.close()
        np.testing.assert_array_equal([r["frame"] for r in results], frames)
        if any(sha256(Path(p)) != digest for p, digest in inputs.items()):
            raise RuntimeError("diagnostic source data changed")
        if any(sha256(REPO / p) != digest for p, digest in source_hashes.items()):
            raise RuntimeError("diagnostic source code changed")
        result = {
            "status": "ok",
            "frames": [int(frames[0]), int(frames[-1]) + 1],
            "native_fps": 166,
            "temporal_subsampling": False,
            "gpu_used": False,
            "projection_seconds": projection_seconds,
            "grid_shape": grid.grid_shape,
            "grid_cell_count": len(grid.xyz),
            "grid_sha256": sha256(grid_path),
            "configuration": dataclasses.asdict(config),
            "source_sha256": source_hashes,
            "v2_context_hypothesis": best,
            "rows": results,
            "mean_candidate_generation_seconds": float(
                np.mean([r["candidate_generation_seconds"] for r in results])
            ),
            "limitations": [
                "This known 31-frame development failure is not held-out identity validation.",
                "S2 locations are used only after global pool construction for diagnostic regions.",
                "Healthy S0/S1 segments require current directly observed mouth/pectoral nodes.",
                "Soft novelty changes ranking only; all supported voxels remain saved.",
                "Two-view support explicitly includes known third-view contradiction.",
                "Even strict-three support can form nonanimal silhouette-intersection ghosts.",
                "Spatial centres are not anatomical pectoral observations or accepted IDs.",
                "The running minute experiment and its results are not inspected or changed.",
            ],
        }
        path = out / "diagnostic.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("diagnostic", path, sha256=sha256(path))


if __name__ == "__main__":
    with threadpool_limits(limits=1, user_api="blas"):
        diagnose()
