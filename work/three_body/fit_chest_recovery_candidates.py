"""Same-objective CPU fitting of every admitted chest-recovery development candidate.

All original proposals and the new proposals share one independent batch per
saved frame. No tracker is started, no temporal state is advanced, and no output
is adopted. The original subset is also compared with its saved CUDA fit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import summarize_chest_candidate_replay as evidence_tools
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]
from chest_candidate_diagnostics import selected_indices, support_snapshot  # noqa: E402
from graph_fit import FitInputs, objective  # noqa: E402
from node_geometry import RayObservations, assigned_rays  # noqa: E402
from observations import PixelTargets  # noqa: E402
from persistent_node_tracker import PersistentNodeState  # noqa: E402
from renderer import RefractiveFishRenderer, RendererConfig  # noqa: E402
from run_experiment import CALIBRATION_DIR  # noqa: E402
from tracker import Proposal, ThreeBodyTracker, TrackerConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

require, sha, read = evidence_tools.require, evidence_tools.sha, evidence_tools.read
PLAN = Path(__file__).with_name("CHEST_RECOVERY_LOCAL_FIT_PLAN.md")
ATOL = 1e-10


def local_code():
    paths = {Path(__file__).resolve(), PLAN}
    scopes = (REPO / "labs/20260909_three_body", REPO / "src", REPO / "work/three_body")
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and any(path.is_relative_to(scope) for scope in scopes):
                paths.add(path)
    return {str(path): sha(path) for path in paths}


def load_arrays(path, hashes):
    require(str(path) in hashes and sha(path) == hashes[str(path)], "unbound or changed arrays")
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def six_weighted_parts(pose, predicted, parts, cfg):
    relative = pose - pose[:, :, 1:2]
    old_relative = predicted - predicted[:, :, 1:2]
    orientation = ((relative - old_relative) ** 2).sum(-1).mean((1, 2)) / 25
    return np.column_stack(
        (
            parts[:, 0],
            cfg.shape_weight * parts[:, 1],
            cfg.motion_weight * parts[:, 2],
            0.05 * cfg.motion_weight * orientation,
            cfg.repulsion_weight * parts[:, 3],
            cfg.mask_weight * parts[:, 4],
        )
    )


def initial_objective(tracker, proposals, states, rays, targets):
    """One evaluation of the already frozen public objective, without optimizer steps."""
    tensor, cfg = tracker._tensor, tracker.config
    origins, directions, valid = assigned_rays(rays, np.stack([p.plan for p in proposals]))
    predicted = tensor(np.stack([p.predicted for p in proposals]))
    missing = tensor(np.stack([states[p.parent].missing for p in proposals]))
    lengths = tensor(tracker.lengths)[None]
    weights = tensor([1.0, 1.0, 0.35])[None, None, None]
    renderer = RefractiveFishRenderer(
        tracker.rig,
        targets.pixels,
        RendererConfig(width_ratio=cfg.width_ratio),
        device="cpu",
        dtype=torch.float64,
    )
    inputs = FitInputs(
        tensor(origins),
        tensor(directions),
        tensor(valid),
        predicted,
        cfg.motion_sigma_mm + 0.3 * torch.clamp(missing, max=20),
        lengths,
        torch.clamp(0.2 * lengths, min=1.0),
        weights,
        tensor(targets.target),
        tensor(targets.weights) * renderer.ray_valid,
    )
    with torch.no_grad():
        scores, parts, _ = objective(
            tensor(np.stack([p.pose for p in proposals])),
            cfg,
            inputs,
            9 * float(weights.sum()),
            renderer,
            diagnostics=True,
        )
    return scores.numpy().copy(), parts.numpy().copy()


def frame_fit(tracker, a, recovery, kinds):
    old_count, new_count = len(a["proposal_pose"]), len(recovery["pose_xyz"])
    states = [
        PersistentNodeState(
            a["parent_pose"][i].copy(),
            a["parent_velocity"][i].copy(),
            a["parent_missing"][i].copy(),
            a["parent_review"][i].copy(),
            float(a["parent_cost"][i]),
            a["parent_node_missing"][i].copy(),
            a["parent_direct_pectoral_age"][i].copy(),
        )
        for i in range(6)
    ]
    proposals = [
        Proposal(
            a["proposal_pose"][i].copy(),
            a["proposal_predicted"][i].copy(),
            a["proposal_plan"][i].copy(),
            int(a["proposal_parent"][i]),
            kinds[i],
            tuple(a["proposal_permutation"][i].tolist()),
        )
        for i in range(old_count)
    ]
    for pose, plan in zip(recovery["pose_xyz"], recovery["plan"], strict=True):
        proposals.append(
            Proposal(
                pose.copy(),
                a["proposal_predicted"][0].copy(),
                plan.copy(),
                0,
                "chest_recovery_counterfactual",
                (0, 1, 2),
            )
        )
    rays = RayObservations.from_points(tracker.rig, {v: a[v + "_points"] for v in VIEWS})
    for i, view in enumerate(VIEWS):
        require(
            evidence_tools.equal_bytes(rays.origins[i], a[view + "_ray_origins"])
            and evidence_tools.equal_bytes(rays.directions[i], a[view + "_ray_directions"]),
            "calibrated rays differ from original exact replay",
        )
    targets = PixelTargets(
        {v: a[v + "_fit_pixels"].copy() for v in VIEWS},
        a["fit_target"].copy(),
        a["fit_weights"].copy(),
        a["fit_foreground"].copy(),
    )
    plans = np.stack([p.plan for p in proposals])
    initial = np.stack([p.pose for p in proposals])
    predicted = np.stack([p.predicted for p in proposals])
    parents = np.asarray([p.parent for p in proposals], np.int64)
    permutations = np.asarray([p.permutation for p in proposals], np.int64)
    before = support_snapshot(initial, plans, rays, tracker.rig)
    require(
        np.array_equal(before["strict_observed"][old_count:], recovery["strict_observed"]),
        "new candidate's previously certified geometric support changed",
    )
    initial_cost, initial_parts = initial_objective(tracker, proposals, states, rays, targets)
    pose, cost, parts, masks = tracker._fit(proposals, states, rays, targets)
    require(np.all(cost <= initial_cost + ATOL), "best retained fit is worse than its initial pose")
    after = support_snapshot(pose, plans, rays, tracker.rig)
    require(
        not before["plan_source_collision"].any() and not after["plan_source_collision"].any(),
        "fit proposal reused an anatomical source",
    )
    weighted_initial = six_weighted_parts(initial, predicted, initial_parts, tracker.config)
    weighted_final = six_weighted_parts(pose, predicted, parts, tracker.config)
    require(
        np.allclose(weighted_initial.sum(1), initial_cost, atol=ATOL, rtol=0)
        and np.allclose(weighted_final.sum(1), cost, atol=ATOL, rtol=0),
        "six weighted objective terms do not reproduce total cost",
    )
    path_cost = cost + np.asarray([states[p].cost for p in parents])
    selected, order = selected_indices(
        pose, path_cost, parents, permutations, tracker.config.beam_size
    )
    old_selected, old_order = selected_indices(
        pose[:old_count],
        path_cost[:old_count],
        parents[:old_count],
        permutations[:old_count],
        tracker.config.beam_size,
    )
    parity = {}
    for field, actual in (("fitted_pose", pose), ("local_cost", cost), ("loss_parts", parts)):
        expected = a[field]
        actual = actual[:old_count]
        parity[field] = dict(
            byte_exact=evidence_tools.equal_bytes(actual, expected),
            within_fixed_atol=bool(np.allclose(actual, expected, atol=ATOL, rtol=0)),
            maximum_absolute_difference=float(np.max(np.abs(actual - expected))),
        )
    parity["mask_supported_exact"] = bool(np.array_equal(masks[:old_count], a["mask_supported"]))
    parity["strict_support_exact"] = bool(
        np.array_equal(after["strict_observed"][:old_count], a["fitted_strict_observed"])
    )
    all_chests = after["strict_observed"][:, :, 1].all(1)
    new_mask = np.arange(len(proposals)) >= old_count
    info = dict(
        original_candidates=old_count,
        new_candidates=new_count,
        original_cpu_gpu=parity,
        new_all_three_chests_before=int(before["strict_observed"][new_mask, :, 1].all(1).sum()),
        new_all_three_chests_after=int(all_chests[new_mask].sum()),
        selected_new_candidates=int(new_mask[selected].sum()),
        selected_new_all_three_chests=int((new_mask[selected] & all_chests[selected]).sum()),
        original_cpu_best=int(old_selected[0]),
        combined_cpu_best=int(selected[0]),
        best_is_new=bool(new_mask[selected[0]]),
        original_best_all_three_chests=bool(all_chests[old_selected[0]]),
        combined_best_all_three_chests=bool(all_chests[selected[0]]),
    )
    arrays = dict(
        initial_pose_xyz=initial,
        fitted_pose_xyz=pose,
        predicted_pose_xyz=predicted,
        proposal_plan=plans,
        proposal_parent=parents,
        proposal_permutation=permutations,
        parent_identity_review=np.stack([states[p].review for p in parents]),
        new_candidate=new_mask,
        initial_cost=initial_cost,
        local_cost=cost,
        path_cost=path_cost,
        initial_loss_parts=initial_parts,
        final_loss_parts=parts,
        initial_weighted_six_parts=weighted_initial,
        final_weighted_six_parts=weighted_final,
        mask_supported=masks,
        selected_indices=selected,
        stable_path_order=order,
        original_selected_indices=old_selected,
        original_stable_path_order=old_order,
    )
    arrays.update({"before_" + k: v for k, v in before.items()})
    arrays.update({"after_" + k: v for k, v in after.items()})
    return arrays, info


def main(replay, recovery, output):
    code = local_code()
    hashes = dict(code)
    run = evidence_tools.verify(
        replay, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
    )
    counter = evidence_tools.verify(
        recovery, "probe.provenance.json", "chest_recovery_counterfactual_proposals", hashes
    )
    require(
        counter["inputs"][str(replay / "replay.provenance.json")]["sha256"]
        == hashes[str(replay / "replay.provenance.json")],
        "counterfactual uses a different replay",
    )
    counter_rows = read(recovery / "frames.json")
    native_frames = [r["frame_idx"] for r in counter_rows if r["returned_candidates"]]
    require(
        len(native_frames) == 20 and sum(r["returned_candidates"] for r in counter_rows) == 31,
        "fixed all20/all31 candidate scope differs",
    )
    registration = read(replay / "registration/plan.json")
    binding = registration["original_binding"]

    def bind(path):
        path = Path(path).resolve()
        expected = run["inputs"][str(path)]["sha256"]
        require(sha(path) == expected, "original replay input changed")
        hashes[str(path)] = expected
        return path

    for source in binding["code_sha256"]:
        bind(source)
    calibration = Path(binding["inputs"]["seed"]["input_root"]) / CALIBRATION_DIR
    for path in calibration.iterdir():
        if path.is_file():
            bind(path)
    seed_root = Path(binding["inputs"]["seed"]["registration_path"]).parent
    initial = np.load(bind(seed_root / "initial_pose_xyz.npy"), allow_pickle=False)
    cfg = TrackerConfig(**{**registration["original_context"]["tracker_config"], "device": "cpu"})
    torch.set_num_threads(2)
    tracker = ThreeBodyTracker(load_rig(calibration), initial, cfg)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with provenance("chest_recovery_same_objective_local_CPU_fit", output, name="fit") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        record.param("native_frames", native_frames)
        record.param("tracker_config", dataclasses.asdict(cfg))
        record.param("numeric_comparison", dict(atol=ATOL, rtol=0, byte_exact_also_reported=True))
        record.param("fit_samples_are_not_held_out", True)
        record.param(
            "runtime",
            dict(
                torch=str(torch.__version__),
                numpy=np.__version__,
                device="cpu",
                dtype="float64",
                torch_cpu_threads=2,
            ),
        )
        start = time.perf_counter()
        for native in native_frames:
            diagnostic_root = replay / "diagnostics" / f"frame_{native:06d}"
            a = load_arrays(diagnostic_root / "arrays.npz", hashes)
            new = load_arrays(recovery / f"frame_{native:06d}.npz", hashes)
            metadata = read(diagnostic_root / "metadata.json")
            frame_start = time.perf_counter()
            arrays, info = frame_fit(tracker, a, new, metadata["proposal_kind"])
            info.update(frame_idx=native, seconds=time.perf_counter() - frame_start)
            path = output / f"frame_{native:06d}.npz"
            np.savez_compressed(path, **arrays)
            record.output(path.name, path, sha256=sha(path))
            rows.append(info)
            print(json.dumps(info), flush=True)
        require(tracker.initialize_count == 0, "local fitter unexpectedly started tracking")
        result = dict(
            schema="chest_recovery_same_objective_local_CPU_fit_v1",
            status="ok",
            frames=len(rows),
            new_candidates=sum(r["new_candidates"] for r in rows),
            new_all_three_before=sum(r["new_all_three_chests_before"] for r in rows),
            new_all_three_after=sum(r["new_all_three_chests_after"] for r in rows),
            frames_with_supported_new_selected=sum(
                r["selected_new_all_three_chests"] > 0 for r in rows
            ),
            frames_new_best=sum(r["best_is_new"] for r in rows),
            frames_combined_best_all_three=sum(r["combined_best_all_three_chests"] for r in rows),
            frames_original_best_all_three=sum(r["original_best_all_three_chests"] for r in rows),
            original_cpu_gpu_support_all_exact=all(
                r["original_cpu_gpu"]["strict_support_exact"] for r in rows
            ),
            seconds=time.perf_counter() - start,
            additional_tracking=False,
            adopted=False,
            independent_ID_truth=False,
            held_out_image_evaluation=False,
        )
        for name, content in (("summary.json", result), ("frames.json", rows)):
            path = output / name
            path.write_text(json.dumps(content, indent=2), encoding="utf-8")
            record.output(name, path, sha256=sha(path))
        require(code == local_code(), "local fitter sources changed or imported late")
        for path, digest in hashes.items():
            require(sha(path) == digest, "bound evidence changed during local fitting")
        for item in record.outputs.values():
            require(sha(item["path"]) == item["sha256"], "local fitting output changed")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    require(
        target.is_relative_to(REPO / "work") and not target.exists(),
        "new workspace output required",
    )
    main(args.replay.resolve(), args.recovery.resolve(), target)
