"""Diagnose whole-row 3D body candidates on the seven registered development frames.

Uses saved SL point JSON and the registered refractive rig. This creates new
geometric candidates; it does not fit images, initialize a tracker, use future
poses as predictions, or assign identities. Two depth bounds are engineering
conditions matching the existing comparison, not measured tank surfaces.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import compare_partial_body_saved_banks as saved
import numpy as np
from body_row_proposals import BodyRowConfig, body_row_pool
from node_geometry import RayObservations

proof, require, codec = saved.proof, saved.require, saved.codec
WORK = saved.WORK
ROOT = WORK / "partial_body_saved_bank_comparison_v1"
INPUT_SHA = "10d29aad72ca98674abbcbaf386ce48244087cb4abae552b51e592b5dbb9910f"
CALLER_SHA = "7e069ea6b727c05c951f1e40d70ea8ad54276db2a0a344fa8b4dca534f7d2dd8"
STAGE = "whole_row_body_geometry_diagnostic_v1"
CONFIG = BodyRowConfig()
VIEWS = saved.VIEWS


def unordered_scenes(pool, keep=4):
    """Enumerate row-disjoint triples of complete bodies, without temporal ID labels.

    Rank first by the number of supplied rays, then geometric RMS; both are
    proposal heuristics. Count all compatible triples; retain only a few for
    inspection. Same-fish duplicate detector rows remain a possible error.
    """
    indices = [i for i, body in enumerate(pool) if body["triangulated_nodes"].all()]
    rows = np.stack([pool[i]["view_rows"] for i in indices]) if indices else np.empty((0, 3))
    conflicts = np.any(
        (rows[:, None] >= 0) & (rows[None] >= 0) & (rows[:, None] == rows[None]), axis=-1
    )
    total, best = 0, []
    for ai, bi in combinations(range(len(indices)), 2):
        if conflicts[ai, bi]:
            continue
        available = np.flatnonzero(~conflicts[ai] & ~conflicts[bi] & (np.arange(len(indices)) > bi))
        for ci in available:
            chosen = [indices[ai], indices[bi], indices[int(ci)]]
            bodies = [pool[i] for i in chosen]
            count = sum(b["used_ray_count"] for b in bodies)
            rms = float(sum(np.sum(b["ray_rms_mm"] ** 2) for b in bodies))
            total += 1
            best.append(((-count, rms, tuple(chosen)), chosen))
            best.sort(key=lambda item: item[0])
            del best[keep:]
    selected = []
    for rank, (_, chosen) in enumerate(best):
        bodies = [pool[i] for i in chosen]
        pose = np.stack([b["pose_xyz"] for b in bodies])
        selected.append(
            dict(
                rank=rank,
                pool_indices=chosen,
                pose_xyz=pose,
                proposed_sources=np.stack([b["proposed_sources"] for b in bodies]),
                source_rows=np.stack([b["view_rows"] for b in bodies]),
                total_used_rays=sum(b["used_ray_count"] for b in bodies),
                sum_squared_ray_rms_mm=float(sum(np.sum(b["ray_rms_mm"] ** 2) for b in bodies)),
                minimum_chest_separation_mm=float(
                    min(
                        np.linalg.norm(pose[a, 1] - pose[b, 1])
                        for a, b in combinations(range(3), 2)
                    )
                ),
                body_order_is_anonymous=True,
                physical_IDs_assigned=False,
            )
        )
    return selected, dict(
        complete_bodies=len(indices),
        row_disjoint_complete_triples=total,
        displayed_scene_limit=keep,
        scenes_returned=len(selected),
        ranking_is_proposal_heuristic=True,
        ranking_order=["more_assigned_rays", "lower_sum_squared_ray_rms", "pool_index_tuple"],
        duplicate_detector_rows_can_still_represent_one_fish=True,
    )


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        evidence.add(ROOT / "comparison.provenance.json", INPUT_SHA)
        seal = proof.read_json(ROOT / "comparison.provenance.json")
        require(
            seal["status"] == "ok" and seal["stage"] == saved.STAGE, "successful input required"
        )
        path = saved.member(ROOT, seal, "plan.json", evidence)
        input_plan = proof.read_json(path)
        require(
            input_plan["source_code"][str(WORK / "compare_partial_body_saved_banks.py")]
            == CALLER_SHA,
            "registered input caller differs",
        )
        context = saved.load_headers(evidence)
        source = saved.source_code(context, evidence)
        for name in ("body_row_proposals.py", Path(__file__).name):
            source[str(WORK / name)] = evidence.add(WORK / name)
        for path, digest in input_plan["source_code"].items():
            require(source.get(path) == digest, "source differs from saved comparison")
        rig = saved.load_rig(context, evidence)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                input_sha256=INPUT_SHA,
                source_code=source,
                body_config=asdict(CONFIG),
                frames=list(saved.ALL_FRAMES),
                engineering_bounds=dict(
                    sparse=[0.0, 0.0, -6.0], depth=[0.0, 0.0, -12.0], upper=[192.0, 192.0, 186.0]
                ),
                new_geometric_candidates=True,
                image_optimization=False,
                tracker_calls=0,
                external_pose_initializations=0,
                ID_solver_calls=0,
                original_candidate_reads=0,
                candidate_bank_payload_reads=0,
                original_SLEAP_cache_reads=0,
                video_reads=0,
                mask_archive_reads=0,
                GPU_used=False,
                physical_ID_truth=False,
            ),
        )
        summaries, render_cases = [], []
        for frame in saved.ALL_FRAMES:
            records = context["records"][frame]["observations"]
            points = {v: np.asarray(records[v]["SLEAP_points"], np.float32) for v in VIEWS}
            for view in VIEWS:
                require(
                    proof.signature(points[view]) == records[view]["SL_points_signature"],
                    "saved SLEAP point signature differs",
                )
            rays = RayObservations.from_points(rig, points)
            for arm in saved.ARMS if frame in saved.FRAMES else saved.ARMS[:1]:
                lower = np.array([0.0, 0.0, -12.0 if arm == saved.ARMS[1] else -6.0])
                upper = np.array([192.0, 192.0, 186.0])
                pool, pool_stats = body_row_pool(rig, rays, lower, upper, CONFIG)
                scenes, scene_stats = unordered_scenes(pool)
                for scene in scenes:
                    scene["projections"] = {
                        v: rig.camera(v).project(scene["pose_xyz"]) for v in VIEWS
                    }
                result = dict(
                    schema=STAGE + "_case",
                    frame_idx=frame,
                    arm=arm,
                    lower=lower,
                    upper=upper,
                    pool=pool,
                    pool_stats=pool_stats,
                    unordered_scenes=scenes,
                    scene_stats=scene_stats,
                    source_points_signatures={v: proof.signature(points[v]) for v in VIEWS},
                    no_temporal_identity_assignment=True,
                    direct_support_not_recomputed=True,
                )
                archive = saved.write_case(out, f"cases/{arm}_{frame}", result)
                summaries.append(
                    dict(
                        frame_idx=frame,
                        arm=arm,
                        archive=archive,
                        pool=pool_stats,
                        scenes=scene_stats,
                    )
                )
                render_cases.append(dict(frame_idx=frame, arm=arm, scenes=scenes))
                print(
                    f"{arm} {frame}: bodies={len(pool)}, "
                    f"complete={scene_stats['complete_bodies']}, "
                    f"row-disjoint triples={scene_stats['row_disjoint_complete_triples']}",
                    flush=True,
                )
        out.write(
            "summary.json",
            saved.small_json(
                dict(
                    schema=STAGE,
                    status="ok",
                    cases=summaries,
                    physical_ID_truth=False,
                    image_or_motion_score_evaluated=False,
                    new_trajectory_created=False,
                    partial_body_scene_completion_tested_on_actual_data=False,
                )
            ),
        )
        out.write(
            "render_payload.json",
            saved.small_json(
                dict(
                    schema=STAGE + "_render_payload",
                    cases=render_cases,
                    body_order_is_anonymous=True,
                    physical_ID_truth=False,
                    new_trajectory_created=False,
                )
            ),
        )
        evidence.final_guard()
        out.final_guard()
        receipt = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        receipt = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "geometry.provenance.json").write_bytes(proof.json_bytes(receipt))
        raise
    (output / "geometry.provenance.json").write_bytes(proof.json_bytes(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
