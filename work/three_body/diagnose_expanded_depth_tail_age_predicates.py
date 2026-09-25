"""Eight saved companions: two state0 four-ancestor tail/chest age predicates.

No actual age integers are inferred. No checkpoint/raw/mask, fitting or GPU.
The preceding retained-first state differs from the final78020 path.
"""

from __future__ import annotations

import json
from pathlib import Path

import diagnose_expanded_depth_regression_pool as pool
import numpy as np

proof, require, render = pool.proof, pool.require, pool.render
WORK, VIEWS = pool.WORK, pool.VIEWS
TARGETS = (77677, 77808)
FRAMES = tuple(n for target in TARGETS for n in range(target - 4, target))
STAGE = "expanded_depth_two_frame_tail_age_predicates_v1"
PRIOR_SHA = "0abbd3399f0f4a149a9b01d6d854ec18a8911cfa594da4a42b35365f17544a09"
POOL_SOURCE_SHA = "87109c8fc1eba5b0d85d309b10062c4471e13c5b6a482b752c82fcb1383da3a0"
PRODUCER_SHA = "0ff5f6a31f020a2f3773176e5af60990ffb6bbf5fc7c67cbf505dd208d6e1d8c"


def selected_SL(shared, seal, evidence):
    """One archive decode per view; retain only the eight requested native rows.

    The flat sorted-frame selector is identical to CachedNodes' searchsorted
    interval + at behavior. Native within-frame detection order is preserved.
    NPZ compression requires decoding each complete points member once.
    """
    points, paths = {frame: {} for frame in FRAMES}, {}
    for view in VIEWS:
        matches = [Path(p) for p in shared if Path(p).name.endswith(f"__{view}.npz")]
        require(len(matches) == 1, "one registered SL archive per view required")
        path = matches[0]
        require(seal["inputs"].get(str(path)) == shared[str(path)], "SL producer input differs")
        evidence.add(path, shared[str(path)])
        with np.load(path, allow_pickle=False) as z:
            require(
                str(z["view"]) == view
                and str(z["experiment"]) == render.prior.legacy.baseline.EXPERIMENT
                and tuple(z["nodes"]) == ("mouth", "pectoral", "tail"),
                "SL schema differs",
            )
            index, values = z["frame_idx"], z["points"]
            require(
                np.all(np.diff(index) >= 0) and values.dtype == np.float32,
                "SL ordering/dtype differs",
            )
            for frame in FRAMES:
                lo, hi = (
                    np.searchsorted(index, frame, side="left"),
                    np.searchsorted(index, frame, side="right"),
                )
                points[frame][view] = values[lo:hi].copy()
                points[frame][view].flags.writeable = False
        paths[view] = dict(
            path=str(path),
            sha256=shared[str(path)],
            archive_decodes=1,
            selected_native_frames=list(FRAMES),
        )
    return points, paths


def main():
    evidence = proof.Evidence()
    read = render.previous.read
    old = read(WORK / f"{pool.STAGE}.json", evidence, PRIOR_SHA)
    require(old["status"] == "ok", "successful earlier pool diagnosis required")
    evidence.add(Path(pool.__file__), POOL_SOURCE_SHA)
    evidence.add(Path(__file__))
    root = WORK / render.comparison.VARIANT_STAGE
    seal = read(root / "tracking.provenance.json", evidence, PRODUCER_SHA)
    require(seal["status"] == "ok", "successful producer required")
    plan = pool.doc(root, seal, "plan.json", evidence)
    protocol_sha = plan["original_binding"]["tracking_protocol_sha256"]
    paths = [Path(p) for p, d in seal["inputs"].items() if d == protocol_sha]
    require(len(paths) == 1, "one bound tracking protocol required")
    protocol = read(paths[0], evidence, protocol_sha)
    cfg = protocol["persistence_config"]
    require(
        cfg["minimum_missing_frames"] == cfg["minimum_pectoral_frames"] == 4,
        "this bounded predicate diagnostic supports exactly four-frame thresholds",
    )
    require(protocol["tracker_config"] == old["tracker_config"], "same tracker config required")
    shared = plan["original_binding"]["inputs"]["seed"]["SL_and_calibration_sha256"]
    require(len(shared) == 7, "registered three SL/four rig inputs required")
    manifests = [Path(p) for p in shared if Path(p).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one bound rig required")
    path = manifests[0]
    calibration = read(path, evidence, shared[str(path)])
    for row in calibration["model_files"]:
        model = render.prior.native.checked_relative(path.parent, row["path"])
        require(
            shared.get(str(model)) == seal["inputs"].get(str(model)) == row["sha256"],
            "rig model binding differs",
        )
        evidence.add(model, row["sha256"])
    rig = render.observations.calib.load_rig(path.parent)
    points, sl_records = selected_SL(shared, seal, evidence)
    before_points = {
        str(n): {v: proof.signature(p) for v, p in byview.items()} for n, byview in points.items()
    }
    layers = {}
    for frame in FRAMES:
        meta = pool.doc(root, seal, f"diagnostics/f{frame}.json", evidence)
        name = f"diagnostics/f{frame}.npz"
        require(meta["archive"] == seal["outputs"][name], "companion binding differs")
        path = render.previous.member(root, seal, name, evidence)
        with np.load(path, allow_pickle=False) as z:
            d = pool.codec.restore_diagnostic(meta["tree"], {k: z[k] for k in z.files})
        require(
            d["frame_idx"] == frame and np.isfinite(d["adjusted_path_cost"]).all(),
            "finite saved native pool required",
        )
        require(
            np.array_equal(d["candidate_index"], np.arange(len(d["candidate_index"]))),
            "candidate order differs",
        )
        kept, _ = pool.retained_layer(d, protocol["tracker_config"]["beam_size"])
        require(len(kept) == protocol["tracker_config"]["beam_size"], "complete layer required")
        layers[frame] = (d, kept)
    records = []
    for target in TARGETS:
        current = next(r for r in old["results"] if r["frame"] == target)["methods"][pool.ARMS[1]]
        main = current["pool"][0]
        require(
            main["index"] == main["parent"] == 0
            and main["kind"] == "predict"
            and main["permutation"] == [0, 1, 2],
            "current first proposal must use state0",
        )
        state_index, expected_cost, chain = 0, main["parent_path"], []
        for frame in range(target - 1, target - 5, -1):
            d, kept = layers[frame]
            require(0 <= state_index < len(kept), "ancestor parent outside reconstructed beam")
            candidate = kept[state_index]
            require(
                float(d["adjusted_path_cost"][candidate]) == expected_cost,
                "ancestor cost link differs; stop without invented state",
            )
            rays = render.prior.node_geometry.RayObservations.from_points(rig, points[frame])
            sources, observed, _ = render.prior.node_geometry.supported_nodes(
                d["candidate_pose_xyz"][candidate],
                d["candidate_plan"][candidate],
                rays,
                rig,
                maximum_ray_mm=2.0,
                maximum_pixel_error=12.0,
            )
            chain.append(
                dict(
                    frame=frame,
                    retained_state_index=state_index,
                    retained_pool_indices=kept,
                    candidate_index=int(candidate),
                    parent_index=int(d["candidate_parent"][candidate]),
                    cumulative_path_cost=float(d["adjusted_path_cost"][candidate]),
                    parent_path_cost=float(d["parent_path_cost"][candidate]),
                    observed_MCT_by_slot=observed.tolist(),
                    strict_sources=sources.tolist(),
                )
            )
            state_index = int(d["candidate_parent"][candidate])
            expected_cost = float(d["parent_path_cost"][candidate])
        observed = np.asarray([r["observed_MCT_by_slot"] for r in chain], bool)
        missing_four = (~observed[:, :, 2]).all(0)
        chest_four = observed[:, :, 1].all(0)
        eligible = missing_four & chest_four & cfg["enabled"]
        records.append(
            dict(
                proposal_native=target,
                state0_native=target - 1,
                ancestry_order="newest_to_oldest",
                ancestors=chain,
                consecutive_four_tail_unsupported=missing_four.tolist(),
                consecutive_four_chest_direct=chest_four.tolist(),
                age_gate_predicate_by_slot=eligible.tolist(),
                age_gate_any_slot=bool(eligible.any()),
                exact_age_values_reconstructed=False,
                final78020_selected_candidate=current["selected_index"],
                final78020_selected_parent=current["pool"][current["selected_index"]]["parent"],
                predicate_uses_current_state0_not_final_path=True,
                helper_not_called_proven_by_all_slots_age_ineligible=not bool(eligible.any()),
            )
        )
    for module in (pool.codec, render.prior.node_geometry, render.observations.calib):
        path = Path(module.__file__).resolve()
        evidence.add(path, seal["inputs"][str(path)])
    for name in (
        "labs/20260909_three_body/continuous_tracker.py",
        "labs/20260909_three_body/persistent_node_tracker.py",
        "work/three_body/tail_recovery_tracker.py",
    ):
        path = render.REPO / name
        evidence.add(path, seal["inputs"][str(path)])
    require(
        before_points
        == {
            str(n): {v: proof.signature(p) for v, p in byview.items()}
            for n, byview in points.items()
        },
        "selected SL changed",
    )
    require(not render.sys.modules["torch"].cuda.is_initialized(), "CPU-only diagnostic required")
    evidence.final_guard()
    output = WORK / f"{STAGE}.json"
    require(not output.exists(), "fresh report required")
    result = dict(
        schema=STAGE,
        status="ok",
        inputs=evidence.paths,
        records=records,
        persistence_config=cfg,
        selected_SL=sl_records,
        selected_SL_signatures=before_points,
        companion_native_frames=list(FRAMES),
        companion_frames_read=8,
        ancestry_frames_per_target=4,
        maximum_authorized_per_target=8,
        strict_ray_mm=2.0,
        strict_pixel_px=12.0,
        exact_ages_reconstructed=False,
        final_path_used_as_state0=False,
        raw_mask_checkpoint_payload_reads=0,
        GPU_used=False,
        new_fit_or_tracking=False,
        thresholds_changed=False,
        physical_ID_truth=False,
        limitation=(
            "Four-frame predicates only. "
            "No pre-fit poses, helper free-gate replay, or exact age values."
        ),
    )
    output.write_bytes(proof.json_bytes(result))
    print(json.dumps(dict(path=str(output), sha256=proof.sha(output), records=records), indent=2))


if __name__ == "__main__":
    main()
