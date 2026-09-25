"""Two saved scenes: intersect mouths from directly supported chest SL rows on CPU."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import diagnose_instance_mouth_candidates_v2 as prior
import numpy as np
from node_geometry import RayObservations, supported_nodes
from node_recovery import NodeRecoveryConfig, _injective

from medaka3d.calib import triangulate

proof, require, WORK, REPO, VIEWS = prior.proof, prior.require, prior.WORK, prior.REPO, prior.VIEWS
STAGE = "grouped_mouth_two_frame_intersection_feasibility_v1"
PARENT = WORK / "instance_mouth_saved_candidate_diagnosis_v2"
PARENT_SHA = "9aed5bd919f6287ef0e3774c784e4349500e016a79529f25883391d404fbda79"
PRIOR_SOURCE_SHA = "b6e86a2576748d88accd89a2a85cd0af4f70bf53dd220e4d618e23ef1936f3f3"
FRAMES, FISH = (77709, 77723), 2


def chest_rows(plan, rays, reference_available, reference_rows, fish):
    """A neutral mask view is unknown; direct chest support still supplies a row."""
    rows = []
    for vi, view in enumerate(VIEWS):
        row = int(plan[fish, vi, 1])
        present = row >= 0
        finite = present and np.isfinite(rays.points[view][row, 0]).all()
        valid_ray = present and np.isfinite(rays.origins[vi][row, 0]).all()
        valid_ray = valid_ray and np.isfinite(rays.directions[vi][row, 0]).all()
        expected = int(reference_rows[vi, fish]) if reference_available[vi] else None
        group_matches = None if expected is None else row == expected
        protected = [f for f in range(3) if f != fish and row >= 0 and plan[f, vi, 0] == row]
        allowed = bool(
            present and finite and valid_ray and not protected and group_matches is not False
        )
        rows.append(
            dict(
                view=view,
                chest_direct_row=row,
                mouth_SL_uv=rays.points[view][row, 0].tolist() if finite else None,
                mouth_finite=bool(finite),
                mouth_ray_finite=bool(valid_ray),
                mask_group_available=bool(reference_available[vi]),
                mask_group_SL_row=expected,
                chest_row_matches_mask_group=group_matches,
                competing_other_fish_direct_mouth=protected,
                allowed_for_intersection=allowed,
            )
        )
    return rows


def assess_scene(
    pose,
    base_plan,
    expected_observed,
    rays,
    rig,
    initial_lengths,
    config,
    reference_available,
    reference_rows,
    fish=FISH,
):
    source, observed, _ = supported_nodes(pose, base_plan, rays, rig)
    require(
        np.array_equal(source, base_plan) and np.array_equal(observed, expected_observed),
        "base direct source/support differs",
    )
    rows = chest_rows(base_plan, rays, reference_available, reference_rows, fish)
    base_projected = np.stack([rig.camera(v).project(pose) for v in VIEWS])
    base_pixel = np.full(3, np.nan)
    for vi, row in enumerate(rows):
        if row["mouth_finite"]:
            base_pixel[vi] = np.linalg.norm(
                base_projected[vi, fish, 0] - rays.points[VIEWS[vi]][row["chest_direct_row"], 0]
            )
    usable = [vi for vi, r in enumerate(rows) if r["allowed_for_intersection"]]
    trials, arrays = (
        [],
        dict(
            base_pose_xyz=pose.copy(),
            base_plan=base_plan.copy(),
            base_observed=observed.copy(),
            base_projected_nodes=base_projected,
            base_mouth_pixel_error_to_chest_SL_rows=base_pixel,
            initial_segment_lengths_mm=initial_lengths.copy(),
        ),
    )
    for count in (3, 2):
        for views in itertools.combinations(usable, count):
            selected_rows = [rows[v]["chest_direct_row"] for v in views]
            origins = np.array(
                [rays.origins[v][r, 0] for v, r in zip(views, selected_rows, strict=True)]
            )
            directions = np.array(
                [rays.directions[v][r, 0] for v, r in zip(views, selected_rows, strict=True)]
            )
            xyz, rms = triangulate(origins, directions)
            candidate, plan = pose.copy(), base_plan.copy()
            candidate[fish, 0] = xyz
            plan[fish, :, 0] = -1
            plan[fish, list(views), 0] = selected_rows
            depth = ((xyz - origins) * directions).sum(-1)
            lengths = np.linalg.norm(np.diff(candidate, axis=1), axis=-1)
            ratio = lengths[fish, 0] / initial_lengths[fish, 0]
            finite = bool(np.isfinite(xyz).all() and np.isfinite(rms))
            forward = bool(finite and np.all(depth >= 0))
            box = bool(finite and np.all(xyz >= config.lower_mm) and np.all(xyz <= config.upper_mm))
            anatomy = bool(
                np.isfinite(ratio)
                and config.minimum_length_ratio <= ratio <= config.maximum_length_ratio
            )
            new_sources, direct, distances = supported_nodes(candidate, plan, rays, rig)
            other = np.arange(3) != fish
            unchanged = bool(
                np.array_equal(candidate[other], pose[other])
                and np.array_equal(candidate[:, 1:], pose[:, 1:])
                and np.array_equal(plan[other], base_plan[other])
                and np.array_equal(plan[:, :, 1:], base_plan[:, :, 1:])
            )
            require(unchanged, "mouth-only replacement altered another coordinate or plan")
            require(
                np.array_equal(new_sources[other], source[other])
                and np.array_equal(new_sources[:, :, 1:], source[:, :, 1:]),
                "unchanged nodes lost a supported camera source",
            )
            other_loss = np.argwhere(observed & ~direct)
            other_loss = other_loss[(other_loss[:, 0] != fish) | (other_loss[:, 1] != 0)]
            projected = np.stack([rig.camera(v).project(candidate) for v in VIEWS])
            pixel = np.full(3, np.nan)
            for vi, row in enumerate(rows):
                if row["chest_direct_row"] >= 0 and row["mouth_finite"]:
                    pixel[vi] = np.linalg.norm(
                        projected[vi, fish, 0] - rays.points[VIEWS[vi]][row["chest_direct_row"], 0]
                    )
            gates = dict(
                finite=finite,
                forward=forward,
                ray_rms=bool(finite and rms <= config.ray_rms_mm),
                box=box,
                anatomy=anatomy,
                source_exclusive=bool(_injective(plan)),
                mouth_direct=bool(direct[fish, 0]),
                chest_direct=bool(direct[fish, 1]),
                no_other_direct_support_loss=len(other_loss) == 0,
            )
            key = f"trial_{len(trials):02d}"
            values = dict(
                pose_xyz=candidate,
                plan=plan,
                direct_sources=new_sources,
                direct_observed=direct,
                ray_distance_mm=distances,
                mouth_ray_origins=origins,
                mouth_ray_directions=directions,
                mouth_forward_depth_mm=depth,
                projected_nodes=projected,
                mouth_pixel_error_to_chest_SL_rows=pixel,
                segment_lengths_mm=lengths,
                mouth_xyz=np.asarray(xyz),
                ray_rms_mm=np.asarray(rms),
            )
            arrays.update({key + "__" + k: a for k, a in values.items()})
            trials.append(
                dict(
                    index=len(trials),
                    views=[VIEWS[v] for v in views],
                    rows=selected_rows,
                    mouth_xyz=xyz.tolist(),
                    ray_rms_mm=float(rms),
                    mouth_chest_mm=float(lengths[fish, 0]),
                    seed_length_ratio=float(ratio),
                    mouth_displacement_mm=float(np.linalg.norm(xyz - pose[fish, 0])),
                    mouth_pixel_error_by_view=dict(zip(VIEWS, pixel.tolist(), strict=True)),
                    mouth_plan_by_view=plan[fish, :, 0].tolist(),
                    mouth_supported_views=int((new_sources[fish, :, 0] >= 0).sum()),
                    unchanged_other_pose_and_plan=unchanged,
                    other_direct_support_losses=other_loss.tolist(),
                    gates=gates,
                    passes_existing_spatial_gates=all(gates.values()),
                )
            )
    return dict(
        chest_group_rows=rows,
        base_mouth_pixel_error_by_view=dict(zip(VIEWS, base_pixel.tolist(), strict=True)),
        trials=trials,
        feasible_subsets=sum(t["passes_existing_spatial_gates"] for t in trials),
    ), arrays


def execute(output):
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output")
    out, evidence = proof.Output(output), proof.Evidence()
    started = proof.utc()
    try:
        parent = prior.read(PARENT / "diagnosis.provenance.json", evidence, PARENT_SHA)
        tracking = prior.read(
            prior.TRACKING / "tracking.provenance.json", evidence, prior.TRACKING_SHA
        )
        review = prior.read(prior.REVIEW / "render.provenance.json", evidence, prior.REVIEW_SHA)
        comparison = prior.read(
            prior.COMPARISON / "comparison.provenance.json", evidence, prior.COMPARISON_SHA
        )
        require(
            all(s["status"] == "ok" for s in (parent, tracking, review, comparison)),
            "successful sources",
        )
        require(parent["stage"] == prior.STAGE, "parent diagnosis stage")
        for path, digest in (
            (prior.TRACKING / "tracking.provenance.json", prior.TRACKING_SHA),
            (prior.REVIEW / "render.provenance.json", prior.REVIEW_SHA),
            (prior.COMPARISON / "comparison.provenance.json", prior.COMPARISON_SHA),
        ):
            require(parent["inputs"][str(path.resolve())] == digest, "parent source lineage")
        evidence.add(prior.__file__, PRIOR_SOURCE_SHA)
        data = json.loads(
            prior.member(
                prior.REVIEW, review, "source_and_frame_evidence.json", evidence
            ).read_text("utf8")
        )
        require(
            data["frames"] == [77709, 77723, 77880] and data["horizon_stop"] == 78020,
            "same78020 review",
        )
        metadata = json.loads(
            prior.member(
                prior.TRACKING, tracking, "checkpoints/through_078020/metadata.json", evidence
            ).read_text("utf8")
        )
        config = NodeRecoveryConfig(**metadata["context"]["recovery_config"])
        shape_path = prior.member(
            prior.COMPARISON, comparison, "geometry/instance_rescore.npz", evidence
        )
        with np.load(shape_path, allow_pickle=False) as z:
            lengths = z["seed_segment_length_mm"]
        require(
            proof.signature(lengths) == metadata["context"]["initial_lengths"],
            "registered seed lengths",
        )
        paths = [Path(p) for p in review["inputs"] if Path(p).name == "calibration_manifest.json"]
        require(len(paths) == 1, "unique calibration")
        calibration = prior.read(paths[0], evidence, review["inputs"][str(paths[0])])
        for row in calibration["model_files"]:
            p = (paths[0].parent / row["path"]).resolve(strict=True)
            evidence.add(p, review["inputs"][str(p)])
        rig = prior.load_rig(paths[0].parent)
        require(tuple(rig.views) == VIEWS and rig.world_units == "mm", "rig order and units")
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
                digest = (
                    proof.sha(p)
                    if p == Path(__file__).resolve()
                    else (
                        PRIOR_SOURCE_SHA
                        if p == Path(prior.__file__).resolve()
                        else tracking["inputs"].get(str(p), review["inputs"].get(str(p)))
                    )
                )
                require(digest is not None, "unbound code: " + str(p))
                evidence.add(p, digest)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                target_raw_slot=FISH,
                mouth_rows=(
                    "same SL row as the selected direct chest; "
                    "reject conflicts and eligible-mask disagreement"
                ),
                neutral_mask_view="unknown mask grouping; direct chest row is still permitted",
                source_candidate_diagnosis_sha256=PARENT_SHA,
                source_tracking_sha256=prior.TRACKING_SHA,
                maximum_ray_rms_mm=config.ray_rms_mm,
                length_ratio=[config.minimum_length_ratio, config.maximum_length_ratio],
                lower_mm=config.lower_mm,
                upper_mm=config.upper_mm,
                strict_ray_mm=2.0,
                strict_pixel_px=12.0,
                applied_temporal_age_gate=False,
                applied_proposal_budget=False,
                fitting=False,
                GPU_used=False,
                physical_ID_truth=False,
                adoption=False,
                nonfinite_numeric_evidence=(
                    "preserved exactly in signed NPZ arrays; JSON display uses null"
                ),
                protection_scope=(
                    "other fish's filtered direct mouth rows; "
                    "not the old generator's full unfiltered plan"
                ),
            ),
        )
        records = []
        for t, frame in enumerate(FRAMES):
            record = json.loads(
                prior.member(PARENT, parent, f"f{frame}.json", evidence).read_text("utf8")
            )
            path = prior.member(PARENT, parent, f"f{frame}.npz", evidence)
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            require(
                {k: proof.signature(v) for k, v in a.items()} == record["array_signatures"],
                "candidate diagnosis arrays",
            )
            selected = record["selected_candidate_index"]
            pose = a["candidate_pose_xyz"][selected]
            plan = a["direct_sources"][selected]
            expected = a["direct_observed"][selected]
            raw = data["records"][t]
            require(raw["frame"] == frame, "frame index")
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
            rays = RayObservations.from_points(rig, points)
            before = {k: proof.signature(x) for k, x in a.items()}
            result, arrays = assess_scene(
                pose,
                plan,
                expected,
                rays,
                rig,
                lengths,
                config,
                a["reference_available"],
                a["reference_SL_rows"],
            )
            require(before == {k: proof.signature(x) for k, x in a.items()}, "input arrays mutated")
            result.update(
                frame_idx=frame,
                selected_candidate_index=selected,
                array_signatures={k: proof.signature(x) for k, x in arrays.items()},
            )
            out.write(f"f{frame}.npz", arrays)
            with np.load(output / f"f{frame}.npz", allow_pickle=False) as z:
                require(
                    {k: proof.signature(z[k]) for k in z.files} == result["array_signatures"],
                    "output readback",
                )
            out.write(f"f{frame}.json", result)
            records.append(result)
        lines = [
            "# Chest-group mouth intersection feasibility",
            "",
            "Saved selected poses; only the S2 mouth is replaced. "
            "These are spatial proposals, not tracking or ID truth.",
            "",
            "| Frame | Views | Rows | RMS mm | Mouth-chest mm | Seed ratio "
            "| Mouth support views | Spatial gates |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        for r in records:
            for trial in r["trials"]:
                lines.append(
                    f"| {r['frame_idx']} | {','.join(trial['views'])} | {trial['rows']} "
                    f"| {trial['ray_rms_mm']:.4f} | {trial['mouth_chest_mm']:.4f} "
                    f"| {trial['seed_length_ratio']:.4f} | {trial['mouth_supported_views']} "
                    f"| {trial['passes_existing_spatial_gates']} |"
                )
        lines += [
            "",
            "No temporal age gate, fitter, proposal budget, mask coverage score "
            "or future rollout was applied.",
            "Other poses/plans are unchanged; per-view SL grouping "
            "and conflicts are in the JSON records.",
        ]
        out.write("report.md", "\n".join(lines).encode("utf8"))
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                records=records,
                GPU_used=False,
                checkpoint39_reads=0,
                raw_video_reads=0,
                mask_decodes=0,
                fitting=False,
                tracking_modified=False,
                physical_ID_truth=False,
                automatic_adoption=False,
            ),
        )
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        evidence.final_guard()
        out.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "feasibility.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "feasibility.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.execute, "explicit execution required")
    execute(args.output)
