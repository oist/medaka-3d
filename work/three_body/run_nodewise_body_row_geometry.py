"""Compare legacy and nodewise-view body geometry on seven saved SL frames.

This is a geometry diagnostic, not a temporal tracker or identity evaluation.
Only saved SL JSON, registered calibration JSON and source metadata are read.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import run_body_row_geometry_diagnostic as old
from nodewise_body_row_proposals import nodewise_body_row_pool

saved, proof, require = old.saved, old.proof, old.require
WORK, VIEWS = old.WORK, old.VIEWS
STAGE = "nodewise_body_row_geometry_v1"
LEGACY = WORK / "whole_row_body_geometry_diagnostic_v1"
LEGACY_SHA = "39208ffcd2314098fcf89d944ef9bedbe949fd6c6dbc4c19bc88238acf4a27fb"
LEGACY_CALLER_SHA = "7fc54b4d1bc2f332e4bb40ce9af941dd540ccea25f45b9266a2f6d5743197cc9"
LEGACY_CORE_SHA = "c4dbca4c44c6b2cee2f68a2175c809987e489b5e131a1fb792f9bbd36de918fb"
CONFIG = old.BodyRowConfig()
MAXIMUM_UNORDERED_TRIPLES = 5_000_000


def json_value(value):
    """Diagnostic NaN is null; infinity keeps an explicit sign, never becomes zero."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else {"nonfinite": "+inf" if value > 0 else "-inf"}
    return value


def enumerate_scenes(pool):
    complete = sum(bool(b["triangulated_nodes"].all()) for b in pool)
    upper = math.comb(complete, 3) if complete >= 3 else 0
    require(
        upper <= MAXIMUM_UNORDERED_TRIPLES,
        f"explicit scene enumeration bound exceeded: C({complete},3)={upper}",
    )
    started = perf_counter()
    scenes, stats = old.unordered_scenes(pool, keep=4)
    stats.update(
        unordered_triples_upper_bound_before_row_conflicts=upper,
        maximum_unordered_triples=MAXIMUM_UNORDERED_TRIPLES,
        enumeration_seconds=perf_counter() - started,
        silent_truncation=False,
    )
    return scenes, stats


def compare_case(frame, arm, pool, stats, rig, historical):
    count = stats["legacy_candidate_count"]
    require(type(count) is int and 0 <= count <= len(pool), "invalid legacy prefix length")
    require([b["pool_index"] for b in pool] == list(range(len(pool))), "pool index differs")
    require(
        all(np.asarray(b["triangulated_nodes"]).all() for b in pool[count:]),
        "new body must have all M/C/T triangulated",
    )
    legacy_scenes, legacy_stats = enumerate_scenes(pool[:count])
    new_scenes, new_stats = enumerate_scenes(pool)
    require(count == historical["pool"]["geometry_candidates"], "legacy body count differs")
    for key in ("complete_bodies", "row_disjoint_complete_triples"):
        require(legacy_stats[key] == historical["scenes"][key], "legacy scene count differs")
    for scene in legacy_scenes + new_scenes:
        scene["projections"] = {v: rig.camera(v).project(scene["pose_xyz"]) for v in VIEWS}
        scene["contains_new_body"] = any(i >= count for i in scene["pool_indices"])
    new_bodies = []
    for body in pool[count:]:
        # Preserve the core's per-node source subsets and omitted-view residuals.
        value = dict(body)
        value["projections"] = {v: rig.camera(v).project(body["pose_xyz"]) for v in VIEWS}
        value["physical_IDs_assigned"] = False
        new_bodies.append(value)
    summary = dict(
        frame_idx=frame,
        arm=arm,
        legacy_candidate_count=count,
        added_complete_bodies=len(pool) - count,
        total_body_count=len(pool),
        legacy_scenes=legacy_stats,
        nodewise_scenes=new_stats,
        pool_stats=stats,
        added_row_disjoint_complete_triples=(
            new_stats["row_disjoint_complete_triples"]
            - legacy_stats["row_disjoint_complete_triples"]
        ),
    )
    render = dict(
        frame_idx=frame,
        arm=arm,
        legacy_candidate_count=count,
        legacy_scenes=legacy_scenes,
        nodewise_scenes=new_scenes,
        new_bodies=new_bodies,
        physical_IDs_assigned=False,
    )
    return summary, render


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        evidence.add(LEGACY / "geometry.provenance.json", LEGACY_SHA)
        legacy_seal = proof.read_json(LEGACY / "geometry.provenance.json")
        require(
            legacy_seal["status"] == "ok" and legacy_seal["stage"] == old.STAGE,
            "successful legacy diagnostic required",
        )
        legacy_plan = proof.read_json(saved.member(LEGACY, legacy_seal, "plan.json", evidence))
        legacy_summary = proof.read_json(
            saved.member(LEGACY, legacy_seal, "summary.json", evidence)
        )
        historical = {(c["arm"], c["frame_idx"]): c for c in legacy_summary["cases"]}
        require(set(historical) == set(saved.case_keys()), "same eleven cases required")
        context = saved.load_headers(evidence)
        source = saved.source_code(context, evidence)
        for name, digest in (
            ("body_row_proposals.py", LEGACY_CORE_SHA),
            ("run_body_row_geometry_diagnostic.py", LEGACY_CALLER_SHA),
        ):
            source[str(WORK / name)] = evidence.add(WORK / name, digest)
        for name in ("nodewise_body_row_proposals.py", Path(__file__).name):
            source[str(WORK / name)] = evidence.add(WORK / name)
        for path, digest in legacy_plan["source_code"].items():
            require(source.get(path) == digest, "legacy source differs")
        require(legacy_plan["body_config"] == asdict(CONFIG), "unchanged body config required")
        rig = saved.load_rig(context, evidence)
        plan = dict(
            schema=STAGE,
            source_code=source,
            legacy_geometry_sha256=LEGACY_SHA,
            frames=list(saved.ALL_FRAMES),
            case_count=11,
            body_config=asdict(CONFIG),
            legacy_prefix_recomputed_inside_new_core_once=True,
            maximum_unordered_triples=MAXIMUM_UNORDERED_TRIPLES,
            scene_limit_for_rendering=4,
            enumerated_counts_are_complete_below_bound=True,
            geometry_only=True,
            engineering_depth_bounds=[-6.0, -12.0],
            nan_encoding="null, with triangulated-node flags and source subsets retained",
            original_SL_cache_reads=0,
            original_candidate_or_state_reads=0,
            mask_or_video_reads=0,
            tracker_or_fit_calls=0,
            GPU_used=False,
            physical_ID_truth=False,
        )
        out.write("plan.json", plan)
        summaries, renders = [], []
        for frame in saved.ALL_FRAMES:
            records = context["records"][frame]["observations"]
            points = {v: np.asarray(records[v]["SLEAP_points"], np.float32) for v in VIEWS}
            for view in VIEWS:
                require(
                    proof.signature(points[view]) == records[view]["SL_points_signature"],
                    "saved SL point signature differs",
                )
            rays = old.RayObservations.from_points(rig, points)
            for arm in saved.ARMS if frame in saved.FRAMES else saved.ARMS[:1]:
                lower = np.array([0.0, 0.0, -12.0 if arm == saved.ARMS[1] else -6.0])
                upper = np.array([192.0, 192.0, 186.0])
                then = perf_counter()
                pool, stats = nodewise_body_row_pool(rig, rays, lower, upper, CONFIG)
                generation_seconds = perf_counter() - then
                summary, render = compare_case(
                    frame, arm, pool, stats, rig, historical[(arm, frame)]
                )
                summary["pool_generation_seconds"] = generation_seconds
                record = out.write(
                    f"cases/{arm}_{frame}.json",
                    json_value(
                        dict(
                            schema=STAGE + "_case",
                            frame_idx=frame,
                            arm=arm,
                            lower=lower,
                            upper=upper,
                            pool=pool,
                            pool_stats=stats,
                            summary=summary,
                            source_points_signatures={v: proof.signature(points[v]) for v in VIEWS},
                            no_temporal_identity_assignment=True,
                            direct_support_not_recomputed=True,
                        )
                    ),
                )
                summary["case_json"] = record
                render["case_json"] = record
                summaries.append(summary)
                renders.append(render)
                print(
                    f"{arm} {frame}: legacy={summary['legacy_candidate_count']}, "
                    f"added={summary['added_complete_bodies']}, complete triples="
                    f"{summary['legacy_scenes']['row_disjoint_complete_triples']} -> "
                    f"{summary['nodewise_scenes']['row_disjoint_complete_triples']}",
                    flush=True,
                )
        out.write(
            "summary.json",
            json_value(
                dict(
                    schema=STAGE,
                    status="ok",
                    cases=summaries,
                    physical_ID_truth=False,
                    new_trajectory_created=False,
                    new_nodes_completed_from_prediction=False,
                )
            ),
        )
        out.write(
            "render_payload.json",
            json_value(
                dict(
                    schema=STAGE + "_render_payload",
                    cases=renders,
                    physical_ID_truth=False,
                    body_order_is_anonymous=True,
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
