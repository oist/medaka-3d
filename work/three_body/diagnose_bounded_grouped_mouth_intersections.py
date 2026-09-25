"""Compare the same eight saved mouth ray subsets inside the unchanged engineering box."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bounded_ray_seed as bounded
import diagnose_grouped_mouth_intersections as free
import numpy as np

prior, proof, require = free.prior, free.proof, free.require
WORK, REPO, VIEWS = free.WORK, free.REPO, free.VIEWS
STAGE = "bounded_grouped_mouth_two_frame_feasibility_v1"
PARENT = WORK / free.STAGE
PARENT_SHA = "08c5204744c671acde28cfc25ee9f3ff1cc04c95674fa67be21b2ca6e1399ef7"
FREE_SHA = "45d501725195fd6596ae4f8ff13b60ef5b8a560234a5a6553f9be00d3156ab76"
BOUNDED_SHA = "f8a173486394fffb96f9fb8726a981b37101f3957d9ee00ec88752893f860728"


def assess_subset(base, trial, origins, directions, rays, rig, config):
    """Change only S2 mouth and its contributing source rows; retain every other node."""
    pose, plan = base["base_pose_xyz"].copy(), base["base_plan"].copy()
    before = {k: proof.signature(v) for k, v in base.items()}
    source, observed, _ = free.supported_nodes(pose, plan, rays, rig)
    require(np.array_equal(source, plan), "base source gate differs")
    require(np.array_equal(observed, base["base_observed"]), "base observed gate differs")
    indices = [VIEWS.index(v) for v in trial["views"]]
    for index, vi in enumerate(indices):
        row = trial["rows"][index]
        require(row == plan[2, vi, 1], "mouth row is not the direct chest row")
        require(np.array_equal(origins[index], rays.origins[vi][row, 0]), "ray origin differs")
        require(np.array_equal(directions[index], rays.directions[vi][row, 0]), "ray differs")
    result = bounded.solve_bounded_ray_seed(origins, directions, config.lower_mm, config.upper_mm)
    pose[2, 0] = result.point_xyz
    plan[2, :, 0] = -1
    plan[2, indices, 0] = trial["rows"]
    new_source, direct, distance = free.supported_nodes(pose, plan, rays, rig)
    require(np.array_equal(pose[:2], base["base_pose_xyz"][:2]), "other fish pose changed")
    require(np.array_equal(pose[:, 1:], base["base_pose_xyz"][:, 1:]), "chest/tail pose changed")
    require(np.array_equal(plan[:2], base["base_plan"][:2]), "other fish plan changed")
    require(np.array_equal(plan[:, :, 1:], base["base_plan"][:, :, 1:]), "chest/tail plan changed")
    require(np.array_equal(new_source[:2], source[:2]), "other fish camera support changed")
    require(np.array_equal(new_source[:, :, 1:], source[:, :, 1:]), "chest/tail support changed")
    projected = np.stack([rig.camera(v).project(pose) for v in VIEWS])
    pixel = np.full(3, np.nan)
    for vi, view in enumerate(VIEWS):
        row = base["base_plan"][2, vi, 1]
        if row >= 0 and np.isfinite(rays.points[view][row, 0]).all():
            pixel[vi] = np.linalg.norm(projected[vi, 2, 0] - rays.points[view][row, 0])
    lengths = np.linalg.norm(np.diff(pose, axis=1), axis=-1)
    ratio = lengths[2, 0] / base["initial_segment_lengths_mm"][2, 0]
    losses = np.argwhere(observed & ~direct)
    losses = losses[(losses[:, 0] != 2) | (losses[:, 1] != 0)]
    gates = dict(
        finite=bool(np.isfinite(result.point_xyz).all() and np.isfinite(result.rms_mm)),
        forward=bool(result.forward_ray.all()),
        ray_rms=result.rms_mm <= config.ray_rms_mm,
        box=bool(
            np.all(result.point_xyz >= config.lower_mm)
            and np.all(result.point_xyz <= config.upper_mm)
        ),
        anatomy=bool(config.minimum_length_ratio <= ratio <= config.maximum_length_ratio),
        source_exclusive=bool(free._injective(plan)),
        mouth_direct=bool(direct[2, 0]),
        chest_direct=bool(direct[2, 1]),
        no_other_direct_support_loss=len(losses) == 0,
    )
    arrays = dict(
        pose_xyz=pose,
        plan=plan,
        direct_sources=new_source,
        direct_observed=direct,
        assigned_ray_distance_mm=distance,
        projected_nodes=projected,
        mouth_pixel_error_to_chest_SL_rows=pixel,
        segment_lengths_mm=lengths,
        original_ray_origins=origins.copy(),
        original_ray_directions=directions.copy(),
    )
    arrays.update({"solver__" + k: np.asarray(v) for k, v in vars(result).items()})
    require(before == {k: proof.signature(v) for k, v in base.items()}, "saved input mutated")
    row = dict(
        index=trial["index"],
        views=trial["views"],
        rows=trial["rows"],
        free_point_xyz=trial["mouth_xyz"],
        free_ray_rms_mm=trial["ray_rms_mm"],
        free_gates=trial["gates"],
        free_mouth_pixel_error_by_view=trial["mouth_pixel_error_by_view"],
        point_xyz=result.point_xyz.tolist(),
        ray_rms_mm=result.rms_mm,
        mouth_chest_mm=float(lengths[2, 0]),
        seed_length_ratio=float(ratio),
        mouth_pixel_error_by_view=dict(zip(VIEWS, pixel.tolist(), strict=True)),
        mouth_supported_views=int((new_source[2, :, 0] >= 0).sum()),
        mouth_plan_by_view=plan[2, :, 0].tolist(),
        mouth_direct_sources_by_view=new_source[2, :, 0].tolist(),
        other_direct_support_losses=losses.tolist(),
        gates=gates,
        passes_existing_spatial_gates=all(gates.values()),
        kkt_violation=result.kkt_violation,
        kkt_tolerance=result.kkt_tolerance,
        active_bounds=result.active_bounds.tolist(),
        solver_status=result.solver_status,
        solver_iterations=result.solver_iterations,
        unconstrained_recompute_delta_mm=float(
            np.linalg.norm(result.unconstrained_point_xyz - trial["mouth_xyz"])
        ),
    )
    return row, arrays


def execute(output):
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output")
    out, evidence = proof.Output(output), proof.Evidence()
    started = proof.utc()
    try:
        seal = prior.read(PARENT / "feasibility.provenance.json", evidence, PARENT_SHA)
        require(seal["status"] == "ok" and seal["stage"] == free.STAGE, "successful source stage")
        plan = json.loads(prior.member(PARENT, seal, "plan.json", evidence).read_text("utf8"))
        require(
            plan["frames"] == list(free.FRAMES) and plan["target_raw_slot"] == 2, "fixed scenes"
        )
        config = free.NodeRecoveryConfig(
            ray_rms_mm=plan["maximum_ray_rms_mm"],
            minimum_length_ratio=plan["length_ratio"][0],
            maximum_length_ratio=plan["length_ratio"][1],
            lower_mm=tuple(plan["lower_mm"]),
            upper_mm=tuple(plan["upper_mm"]),
        )
        require(
            config.lower_mm == (0.0, 0.0, -6.0) and config.upper_mm == (192.0, 192.0, 186.0),
            "unchanged engineering box",
        )
        require(
            plan["strict_ray_mm"] == 2.0 and plan["strict_pixel_px"] == 12.0, "same support gates"
        )
        data_path = prior.REVIEW / "source_and_frame_evidence.json"
        data = prior.read(data_path, evidence, seal["inputs"][str(data_path.resolve())])
        require(
            data["frames"] == [77709, 77723, 77880] and data["horizon_stop"] == 78020,
            "same review frames/horizon",
        )
        paths = [Path(p) for p in seal["inputs"] if Path(p).name == "calibration_manifest.json"]
        require(len(paths) == 1, "unique calibration")
        calibration = prior.read(paths[0], evidence, seal["inputs"][str(paths[0])])
        for row in calibration["model_files"]:
            p = (paths[0].parent / row["path"]).resolve(strict=True)
            evidence.add(p, seal["inputs"][str(p)])
        rig = prior.load_rig(paths[0].parent)
        require(tuple(rig.views) == VIEWS and rig.world_units == "mm", "rig order and mm")
        explicit = {
            str(Path(__file__).resolve()): proof.sha(__file__),
            str(Path(free.__file__).resolve()): FREE_SHA,
            str(Path(bounded.__file__).resolve()): BOUNDED_SHA,
        }
        for module in list(sys.modules.values()):
            file = getattr(module, "__file__", None)
            if file is None:
                continue
            p = Path(file).resolve()
            if (
                p.suffix == ".py"
                and p.is_relative_to(REPO)
                and ".venv" not in p.parts
                and p.is_file()
            ):
                digest = explicit.get(str(p), seal["inputs"].get(str(p)))
                require(digest is not None, "unbound source " + str(p))
                evidence.add(p, digest)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_seal_sha256=PARENT_SHA,
                frames=list(free.FRAMES),
                ray_subsets_per_frame=4,
                lower_mm=config.lower_mm,
                upper_mm=config.upper_mm,
                ray_rms_mm=config.ray_rms_mm,
                length_ratio=[config.minimum_length_ratio, config.maximum_length_ratio],
                support_ray_mm=2.0,
                support_pixel_px=12.0,
                normalization="solver normalizes owned directions",
                objective="box-constrained sum squared distances to the same infinite ray lines",
                solver_tolerance=bounded.SOLVER_TOLERANCE,
                solver_max_iterations=bounded.MAX_ITERATIONS,
                forward_ray_and_anatomy="post-solve gates",
                automatic_adoption=False,
                temporal_age_gate_applied=False,
                fitter_used=False,
                physical_floor_survey=False,
            ),
        )
        records = []
        for ti, frame in enumerate(free.FRAMES):
            row = json.loads(
                prior.member(PARENT, seal, f"f{frame}.json", evidence).read_text("utf8")
            )
            p = prior.member(PARENT, seal, f"f{frame}.npz", evidence)
            with np.load(p, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            require(
                {k: proof.signature(v) for k, v in arrays.items()} == row["array_signatures"],
                "source array signatures",
            )
            require(
                len(row["trials"]) == 4
                and all(r["allowed_for_intersection"] for r in row["chest_group_rows"]),
                "same four eligible subsets",
            )
            raw = data["records"][ti]
            require(raw["frame"] == frame, "review frame")
            points = {
                v: np.asarray(raw["observations"][v]["SLEAP_points"], dtype=np.float32)
                for v in VIEWS
            }
            require(
                all(
                    np.where(np.isfinite(x), x.astype(np.float64), None).tolist()
                    == raw["observations"][v]["SLEAP_points"]
                    for v, x in points.items()
                ),
                "SL float32 roundtrip",
            )
            rays = free.RayObservations.from_points(rig, points)
            outputs, trials = (
                {
                    k: v.copy()
                    for k, v in arrays.items()
                    if k.startswith("base_") or k == "initial_segment_lengths_mm"
                },
                [],
            )
            before = {k: proof.signature(v) for k, v in arrays.items()}
            for trial in row["trials"]:
                prefix = f"trial_{trial['index']:02d}__"
                result, values = assess_subset(
                    arrays,
                    trial,
                    arrays[prefix + "mouth_ray_origins"],
                    arrays[prefix + "mouth_ray_directions"],
                    rays,
                    rig,
                    config,
                )
                trials.append(result)
                outputs.update({prefix + k: v for k, v in values.items()})
            require(
                before == {k: proof.signature(v) for k, v in arrays.items()},
                "parent arrays changed",
            )
            record = dict(
                frame_idx=frame,
                trials=trials,
                base_mouth_pixel_error_by_view=row["base_mouth_pixel_error_by_view"],
                feasible_subsets=sum(t["passes_existing_spatial_gates"] for t in trials),
                array_signatures={k: proof.signature(v) for k, v in outputs.items()},
            )
            out.write(f"f{frame}.npz", outputs)
            with np.load(output / f"f{frame}.npz", allow_pickle=False) as z:
                require(
                    {k: proof.signature(z[k]) for k in z.files} == record["array_signatures"],
                    "exact output NPZ readback",
                )
            out.write(f"f{frame}.json", record)
            records.append(record)
        report = [
            "# Bounded mouth intersection diagnostic",
            "",
            "The engineering box and every acceptance threshold are unchanged. "
            "This is a spatial proposal diagnostic, not a rollout or physical ID truth.",
            "",
            "| Frame | Views | Free RMS | Bounded RMS | Mouth z | Support views | Failed gates |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for row in records:
            for t in row["trials"]:
                failed = ", ".join(k for k, v in t["gates"].items() if not v) or "none"
                report.append(
                    f"| {row['frame_idx']} | {','.join(t['views'])} "
                    f"| {t['free_ray_rms_mm']:.5f} | {t['ray_rms_mm']:.5f} "
                    f"| {t['point_xyz'][2]:.5f} | {t['mouth_supported_views']} | {failed} |"
                )
        out.write("report.md", "\n".join(report).encode("utf8"))
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                records=records,
                bound_change=False,
                calibration_change=False,
                checkpoint39_reads=0,
                raw_video_reads=0,
                mask_decodes=0,
                GPU_used=False,
                fitter_used=False,
                physical_ID_truth=False,
                tracking_modified=False,
                automatic_adoption=False,
            ),
        )
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        evidence.final_guard()
        out.final_guard()
        final = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        final = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            error=repr(error),
            inputs=evidence.paths,
            outputs=out.records,
        )
        (output / "feasibility.provenance.json").write_bytes(proof.json_bytes(final))
        raise
    (output / "feasibility.provenance.json").write_bytes(proof.json_bytes(final))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.execute, "explicit execution required")
    execute(args.output)
