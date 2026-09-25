"""Two saved post-fit pools per arm; no tracking, raw/mask/39/GPU reads.

Strict support is recomputed from already saved two-frame SL rows and the same
sealed rig. Current-layer selection is reconstructed, never a future rollout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import render_expanded_depth_10s_review as render

proof, require, prior = render.proof, render.require, render.prior
WORK, ARMS, VIEWS = render.WORK, render.ARMS, render.VIEWS
codec = render.observations.codec
FRAMES = (77677, 77808)
STAGE = "expanded_depth_two_frame_regression_pool_v1"
RENDER_SHA = "aaba4d2fe231d53b4c35966af9e0561f84872053f7f501fc5756792c2cdb7697"
QA_SHA = "984f9c0efeb688930a42a3c2ea25d058b1754f4d7e3864e27340c542f7198d8c"
SELECTED_SHA = "96f6e9164b2f84384207bdf809a52772396813cc0b4b81cdbcdf827c6954472c"


def doc(root, seal, name, evidence):
    return json.loads(render.previous.member(root, seal, name, evidence).read_text("utf8"))


def retained_layer(d, width):
    """The frozen stable rank / same-parent-permutation / <0.5mm dedup rule."""
    selected, reason = [], {}
    for candidate in np.argsort(d["adjusted_path_cost"], kind="stable"):
        candidate = int(candidate)
        if len(selected) == width:
            reason[candidate] = "not_reached_after_beam_filled"
            continue
        duplicate = next(
            (
                j
                for j in selected
                if d["candidate_parent"][candidate] == d["candidate_parent"][j]
                and np.array_equal(
                    d["candidate_permutation"][candidate], d["candidate_permutation"][j]
                )
                and np.max(
                    np.linalg.norm(
                        d["candidate_pose_xyz"][candidate] - d["candidate_pose_xyz"][j], axis=-1
                    )
                )
                < 0.5
            ),
            None,
        )
        if duplicate is not None:
            reason[candidate] = f"same_parent_permutation_pose_duplicate_of_{duplicate}"
        else:
            selected.append(candidate)
            reason[candidate] = "retained_at_this_frame"
    return selected, reason


def analyze(d, final, rays, rig, cfg, selected_index):
    count = len(d["candidate_index"])
    require(np.array_equal(d["candidate_index"], np.arange(count)), "pool row order differs")
    expected = np.asarray(final["pose_xyz_mm"], dtype=d["candidate_pose_xyz"].dtype)
    matches = [
        i
        for i, pose in enumerate(d["candidate_pose_xyz"])
        if proof.signature(pose) == proof.signature(expected)
    ]
    require(len(matches) == 1, f"final saved pose must identify unique pool row: {matches}")
    selected = matches[0]
    require(
        selected_index is None or selected == selected_index, "independent selected index differs"
    )
    require(
        np.array_equal(d["base_local_cost"] + d["sixth_instance_term"], d["adjusted_local_cost"])
        and np.array_equal(
            d["parent_path_cost"] + d["adjusted_local_cost"], d["adjusted_path_cost"]
        ),
        "saved objective arithmetic differs",
    )
    kept, reasons = retained_layer(d, cfg["beam_size"])
    require(selected in kept, "final path candidate was not retained at this frame")
    support, sources, ray_distances, pixel_errors = [], [], [], []
    for pose, plan in zip(d["candidate_pose_xyz"], d["candidate_plan"], strict=True):
        source, observed, distance = prior.node_geometry.supported_nodes(
            pose, plan, rays, rig, maximum_ray_mm=2.0, maximum_pixel_error=12.0
        )
        support.append(observed)
        sources.append(source)
        ray_distances.append(distance)
        error = np.full((3, 3, 3), np.nan)
        for vi, view in enumerate(VIEWS):
            uv = rig.camera(view).project(pose)
            for body, node in np.argwhere(plan[:, vi] >= 0):
                error[body, vi, node] = np.linalg.norm(
                    uv[body, node] - rays.points[view][plan[body, vi, node], node]
                )
        pixel_errors.append(error)
    support, sources = np.asarray(support), np.asarray(sources)
    require(
        np.array_equal(support[selected], final["observed"])
        and np.array_equal(sources[selected], final["sources"]),
        "selected strict support differs",
    )
    weighted = d["original_five_loss_parts"] * np.array(
        [1, cfg["shape_weight"], cfg["motion_weight"], cfg["repulsion_weight"], cfg["mask_weight"]]
    )
    remainder = d["base_local_cost"] - weighted.sum(1)
    ranks = {}
    for key in ("base_local_cost", "adjusted_local_cost", "adjusted_path_cost"):
        rank = np.empty(count, int)
        rank[np.argsort(d[key], kind="stable")] = np.arange(1, count + 1)
        ranks[key] = rank
    pool = []
    for i in range(count):
        pool.append(
            dict(
                index=i,
                selected_on_final_78020_path=i == selected,
                parent=int(d["candidate_parent"][i]),
                kind=d["candidate_kind"][i],
                permutation=d["candidate_permutation"][i].tolist(),
                retained_at_this_frame=i in kept,
                selection_reason=reasons[i],
                ranks={k: int(v[i]) for k, v in ranks.items()},
                s0_plan_views_by_node=d["candidate_plan"][i, 0].T.tolist(),
                s0_strict_sources_by_node=sources[i, 0].T.tolist(),
                s0_direct_MCT=support[i, 0].tolist(),
                s0_finite_latent_MCT=(
                    np.isfinite(d["candidate_pose_xyz"][i, 0]).all(-1) & ~support[i, 0]
                ).tolist(),
                s0_assigned_ray_distance_mm=prior.finite_json(np.asarray(ray_distances)[i, 0].T),
                s0_assigned_pixel_error_px=prior.finite_json(np.asarray(pixel_errors)[i, 0].T),
                base_local=float(d["base_local_cost"][i]),
                sixth=float(d["sixth_instance_term"][i]),
                adjusted_local=float(d["adjusted_local_cost"][i]),
                parent_path=float(d["parent_path_cost"][i]),
                path=float(d["adjusted_path_cost"][i]),
                weighted_reported_five_parts=weighted[i].tolist(),
                unreported_orientation_and_roundoff_remainder=float(remainder[i]),
                delta_vs_selected=dict(
                    base=float(d["base_local_cost"][i] - d["base_local_cost"][selected]),
                    sixth=float(d["sixth_instance_term"][i] - d["sixth_instance_term"][selected]),
                    parent_path=float(d["parent_path_cost"][i] - d["parent_path_cost"][selected]),
                    path=float(d["adjusted_path_cost"][i] - d["adjusted_path_cost"][selected]),
                    weighted_reported_five_parts=(weighted[i] - weighted[selected]).tolist(),
                ),
            )
        )
    targets = {}
    for name, nodes in {"chest": [1], "tail": [2], "chest_and_tail": [1, 2]}.items():
        valid = np.flatnonzero(support[:, 0][:, nodes].all(1)).tolist()
        plan_two = ((d["candidate_plan"][:, 0] >= 0).sum(1)[:, nodes] >= 2).all(1)
        targets[name] = dict(
            postfit_direct_indices=valid,
            assigned_two_or_more_views_indices=np.flatnonzero(plan_two).tolist(),
            retained_direct_indices=[i for i in valid if i in kept],
            best_path_direct_index=min(valid, key=lambda i: d["adjusted_path_cost"][i])
            if valid
            else None,
            conclusion=(
                "Direct-support candidate exists in saved post-fit pool; inspect rank/retention."
                if valid
                else "No directly supported candidate in saved post-fit pool. "
                + (
                    "Some plans assign >=2 views, but their fitted geometry fails support."
                    if plan_two.any()
                    else "No saved plan assigns >=2 views for this target."
                )
            ),
        )
    return dict(
        candidate_count=count,
        selected_index=selected,
        retained_indices=kept,
        selected_local_rank=int(ranks["adjusted_local_cost"][selected]),
        selected_path_rank=int(ranks["adjusted_path_cost"][selected]),
        targets=targets,
        view_gates={
            v: dict(eligible=row["eligible"], reason=row["reason"], neutral=row["neutral"])
            for v, row in d["views"].items()
        },
        pool=pool,
    )


def main():
    evidence = proof.Evidence()
    read = render.previous.read
    render_root = WORK / render.STAGE
    render_seal = read(render_root / "render.provenance.json", evidence, RENDER_SHA)
    saved = doc(render_root, render_seal, "source_and_frame_evidence.json", evidence)
    comp = read(
        render.COMPARISON_ROOT / "comparison.provenance.json", evidence, render.COMPARISON_SEAL_SHA
    )
    require(
        comp["status"] == render_seal["status"] == "ok", "successful comparison/render required"
    )
    comparison_summary = doc(render.COMPARISON_ROOT, comp, "summary.json", evidence)
    qa_root = WORK / "expanded_depth_ten_second_independent_qa_v1"
    qa = read(qa_root / "qa.json", evidence, QA_SHA)
    mapping_path = qa_root / "selected_candidates.npz"
    require(
        qa["selected_candidates"]["sha256"] == SELECTED_SHA
        and qa["comparison_seal_sha256"] == render.COMPARISON_SEAL_SHA,
        "selected mapping scope differs",
    )
    evidence.add(mapping_path, SELECTED_SHA)
    with np.load(mapping_path, allow_pickle=False) as z:
        require(set(z.files) == {"frame_idx", "selected_candidate_index"}, "mapping schema differs")
        mapping = {k: z[k] for k in z.files}
    require(
        {k: proof.signature(v) for k, v in mapping.items()} == qa["selected_candidates"]["arrays"],
        "mapping numeric signatures differ",
    )
    frames = {r["frame"]: r for r in saved["records"] if r["frame"] in FRAMES}
    require(set(frames) == set(FRAMES), "two saved render rows required")
    seals, configs = {}, {}
    for arm, stage in zip(
        ARMS, (render.comparison.CONTROL_STAGE, render.comparison.VARIANT_STAGE), strict=True
    ):
        root = WORK / stage
        ref = comparison_summary["candidate_references"][arm]
        require(
            saved["candidate_references"][arm]["original_final"] == ref,
            "render/comparison candidate reference differs",
        )
        seal_path = root / "tracking.provenance.json"
        require(
            comp["inputs"][str(seal_path)] == ref["tracking_seal_sha256"], "producer edge differs"
        )
        seal = read(seal_path, evidence, ref["tracking_seal_sha256"])
        require(seal["status"] == "ok" and seal["stage"] == stage, "producer success required")
        seals[arm] = seal
        plan = doc(root, seal, "plan.json", evidence)
        protocol_sha = plan["original_binding"]["tracking_protocol_sha256"]
        paths = [Path(p) for p, digest in seal["inputs"].items() if digest == protocol_sha]
        require(len(paths) == 1, "one actual tracking protocol required")
        configs[arm] = read(paths[0], evidence, protocol_sha)["tracker_config"]
    require(configs[ARMS[0]] == configs[ARMS[1]], "tracker cost/beam configurations differ")
    calibration_paths = [
        Path(p) for p in render_seal["inputs"] if Path(p).name == "calibration_manifest.json"
    ]
    require(len(calibration_paths) == 1, "one saved rig required")
    path = calibration_paths[0]
    calibration = read(path, evidence, render_seal["inputs"][str(path)])
    for row in calibration["model_files"]:
        model = prior.native.checked_relative(path.parent, row["path"])
        require(
            all(s["inputs"].get(str(model)) == row["sha256"] for s in seals.values()),
            "same producer rig required",
        )
        evidence.add(model, row["sha256"])
    rig = render.observations.calib.load_rig(path.parent)
    for module in (codec, prior.node_geometry, render.observations.calib):
        path = Path(module.__file__).resolve()
        require(
            seals[ARMS[0]]["inputs"].get(str(path)) == seals[ARMS[1]]["inputs"].get(str(path)),
            "shared scientific helper source required",
        )
        evidence.add(path, seals[ARMS[0]]["inputs"][str(path)])
    for path in (
        Path(__file__),
        Path(render.__file__),
        Path(prior.__file__),
        Path(render.previous.__file__),
    ):
        evidence.add(path)
    results = []
    for frame, row in frames.items():
        points = {v: np.asarray(row["observations"][v]["SLEAP_points"], np.float32) for v in VIEWS}
        require(
            all(
                proof.signature(points[v]) == row["observations"][v]["SL_points_signature"]
                for v in VIEWS
            ),
            "saved SL float32 bytes differ",
        )
        rays = prior.node_geometry.RayObservations.from_points(rig, points)
        methods = {}
        for arm, stage in zip(
            ARMS, (render.comparison.CONTROL_STAGE, render.comparison.VARIANT_STAGE), strict=True
        ):
            root, seal = WORK / stage, seals[arm]
            meta = doc(root, seal, f"diagnostics/f{frame}.json", evidence)
            name = f"diagnostics/f{frame}.npz"
            require(meta["archive"] == seal["outputs"][name], "companion archive binding differs")
            path = render.previous.member(root, seal, name, evidence)
            with np.load(path, allow_pickle=False) as z:
                d = codec.restore_diagnostic(meta["tree"], {k: z[k] for k in z.files})
            require(d["frame_idx"] == frame, "companion frame differs")
            sel = (
                int(mapping["selected_candidate_index"][frame - 76360]) if arm == ARMS[1] else None
            )
            methods[arm] = analyze(d, row["methods"][arm], rays, rig, configs[arm], sel)
        results.append(dict(frame=frame, methods=methods))
    require(
        not render.sys.modules["torch"].cuda.is_initialized(), "CPU diagnostic initialized CUDA"
    )
    evidence.final_guard()
    output = WORK / f"{STAGE}.json"
    require(not output.exists(), "fresh diagnostic output required")
    report = dict(
        schema=STAGE,
        status="ok",
        inputs=evidence.paths,
        frames=list(FRAMES),
        results=results,
        tracker_config=configs[ARMS[0]],
        strict_ray_mm=2.0,
        strict_pixel_px=12.0,
        weighted_five_part_order=[
            "node_and_missing_ray",
            "shape",
            "directional_center_motion",
            "repulsion",
            "union_silhouette",
        ],
        limits=[
            "All costs describe the saved post-fit pool; no objective or candidate fit rerun.",
            "Original five parts omit orientation; reported remainder includes orientation and arithmetic roundoff.",
            "A retained current-layer candidate can lose on future frames; no future loss attribution without further history.",
            "Pre-fit poses and all rejected pre-pool proposals are not retained here. Assignment presence does not prove pre-fit support.",
            "Finite latent nodes are estimated coordinates without >=2-view direct support, not absent coordinates or confirmed physical IDs.",
        ],
        companion_frames_read=4,
        raw_or_mask_payload_reads=0,
        checkpoint39_reads=0,
        new_tracking=False,
        GPU_used=False,
        physical_ID_truth=False,
        thresholds_changed=False,
    )
    output.write_bytes(proof.json_bytes(report))
    print(
        json.dumps(
            dict(
                path=str(output),
                sha256=proof.sha(output),
                results=[
                    dict(
                        frame=r["frame"],
                        methods={
                            a: {
                                k: m[k]
                                for k in (
                                    "candidate_count",
                                    "selected_index",
                                    "retained_indices",
                                    "targets",
                                    "view_gates",
                                )
                            }
                            for a, m in r["methods"].items()
                        },
                    )
                    for r in results
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
