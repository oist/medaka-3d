"""Qualify four fixed directional CUDA fits in a read-only production scheduling window."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fit_chest_recovery_candidates as base
import numpy as np
import torch
from directional_device_fitter import DirectionalDeviceFitter
from node_geometry import RayObservations
from observations import PixelTargets
from persistent_node_tracker import PersistentNodeState
from replay_gpu_window import ReplayGPUWindow
from run_directional_chest_local_fit import verify_control
from tracker import Proposal

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks = base.evidence_tools
REPO = Path(__file__).resolve().parents[2]
FRAMES = (66660, 66664, 66667, 66696)
PLAN = Path(__file__).with_name("DIRECTIONAL_CHEST_CONTINUOUS_PLAN.md")


def fit_inputs(a, new, kinds, rig):
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
        for i in range(len(kinds))
    ]
    proposals.extend(
        Proposal(
            pose.copy(),
            a["proposal_predicted"][0].copy(),
            plan.copy(),
            0,
            "chest_recovery_counterfactual",
            (0, 1, 2),
        )
        for pose, plan in zip(new["pose_xyz"], new["plan"], strict=True)
    )
    rays = RayObservations.from_points(rig, {v: a[v + "_points"] for v in VIEWS})
    for i, v in enumerate(VIEWS):
        checks.require(
            checks.equal_bytes(rays.origins[i], a[v + "_ray_origins"])
            and checks.equal_bytes(rays.directions[i], a[v + "_ray_directions"]),
            "ray input differs from exact replay",
        )
    targets = PixelTargets(
        {v: a[v + "_fit_pixels"].copy() for v in VIEWS},
        a["fit_target"].copy(),
        a["fit_weights"].copy(),
        a["fit_foreground"].copy(),
    )
    return proposals, states, rays, targets


def main(args):
    replay, recovery, cpu_fit, cpu_qualification, output = (
        getattr(args, k).resolve()
        for k in ("replay", "recovery", "cpu_fit", "cpu_qualification", "output")
    )
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(),
        "new workspace output required",
    )
    code = {**base.local_code(), str(PLAN): checks.sha(PLAN)}
    hashes = dict(code)
    output.mkdir(parents=True)
    window = ReplayGPUWindow(args.supervisor_root, args.child_pid, output)
    rows = []
    with provenance(
        "directional_CUDA_window_qualification", output, name="qualification"
    ) as record:
        try:
            window.attach()
            run = checks.verify(
                replay, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
            )
            checks.verify(
                recovery, "probe.provenance.json", "chest_recovery_counterfactual_proposals", hashes
            )
            cpu_fit_seal = checks.verify(
                cpu_fit, "fit.provenance.json", "directional_chest_local_CPU_fit", hashes
            )
            for root, name in (
                (replay, "replay.provenance.json"),
                (recovery, "probe.provenance.json"),
            ):
                key = str(root / name)
                checks.require(
                    cpu_fit_seal["inputs"][key]["sha256"] == hashes[key],
                    "CPU fit used different observations or proposals",
                )
            qualification = checks.verify(
                cpu_qualification,
                "qualification.provenance.json",
                "directional_device_CPU_qualification",
                hashes,
            )
            checks.require(
                qualification["inputs"][str(cpu_fit / "fit.provenance.json")]["sha256"]
                == hashes[str(cpu_fit / "fit.provenance.json")],
                "different CPU comparison",
            )
            for p in (
                Path(__file__).with_name("directional_device_fitter.py"),
                Path(__file__).with_name("directional_chest_fitter.py"),
            ):
                checks.require(
                    qualification["inputs"][str(p)]["sha256"] == code[str(p)],
                    "fitter changed after CPU qualification",
                )
            registration = checks.read(replay / "registration/plan.json")
            binding = registration["original_binding"]
            calibration = Path(binding["inputs"]["seed"]["input_root"]) / base.CALIBRATION_DIR
            seed = (
                Path(binding["inputs"]["seed"]["registration_path"]).parent / "initial_pose_xyz.npy"
            )
            for p in [seed, *[p for p in calibration.iterdir() if p.is_file()]]:
                digest = run["inputs"][str(p)]["sha256"]
                checks.require(checks.sha(p) == digest, "seed/calibration differs")
                hashes[str(p)] = digest
            rig, prepared = load_rig(calibration), []
            for native in FRAMES:
                folder = replay / "diagnostics" / f"frame_{native:06d}"
                a = base.load_arrays(folder / "arrays.npz", hashes)
                new = base.load_arrays(recovery / f"frame_{native:06d}.npz", hashes)
                saved = base.load_arrays(cpu_fit / f"frame_{native:06d}.npz", hashes)
                prepared.append(
                    (
                        native,
                        fit_inputs(
                            a, new, checks.read(folder / "metadata.json")["proposal_kind"], rig
                        ),
                        saved,
                        len(a["proposal_pose"]),
                    )
                )
            for path, digest in hashes.items():
                checks.require(checks.sha(path) == digest, "input changed before waiting")
            record.param("fixed_frames", FRAMES)
            record.param(
                "numeric_comparison", dict(atol=base.ATOL, rtol=0, boolean_and_selection_exact=True)
            )
            print(
                json.dumps(
                    dict(event="CPU_prepared_waiting_for_GPU_window", child_pid=args.child_pid)
                ),
                flush=True,
            )
            window.wait_until_open()
            checks.require(
                code == {**base.local_code(), str(PLAN): checks.sha(PLAN)},
                "source set changed while waiting",
            )
            checks.require(torch.cuda.is_available(), "CUDA unavailable")
            checks.require(
                torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5070 Ti", "unexpected GPU"
            )
            torch.set_num_threads(2)
            cfg = base.TrackerConfig(**registration["original_context"]["tracker_config"])
            checks.require(cfg.device == "cuda", "original CUDA config required")
            fitter = DirectionalDeviceFitter(rig, np.load(seed, allow_pickle=False), cfg)
            started = time.perf_counter()
            for native, inputs, saved, old_count in prepared:
                window.assert_open()
                proposals, states, rays, _targets = inputs
                poses, costs, parts, masks = fitter._fit(*inputs)
                torch.cuda.synchronize()
                window.assert_open()
                plans = np.stack([p.plan for p in proposals])
                parents = np.array([p.parent for p in proposals], np.int64)
                permutations = np.array([p.permutation for p in proposals], np.int64)
                checks.require(
                    checks.equal_bytes(plans, saved["proposal_plan"])
                    and checks.equal_bytes(
                        np.stack([p.pose for p in proposals]), saved["initial_pose_xyz"]
                    ),
                    "GPU candidate input differs",
                )
                support = base.support_snapshot(poses, plans, rays, rig)
                path_cost = costs + np.asarray([states[p.parent].cost for p in proposals])
                selected, order = base.selected_indices(
                    poses, path_cost, parents, permutations, cfg.beam_size
                )
                old_selected, _ = base.selected_indices(
                    poses[:old_count],
                    path_cost[:old_count],
                    parents[:old_count],
                    permutations[:old_count],
                    cfg.beam_size,
                )
                arrays = dict(
                    fitted_pose_xyz=poses,
                    local_cost=costs,
                    final_loss_parts=parts,
                    mask_supported=masks,
                    after_strict_sources=support["strict_sources"],
                    after_strict_observed=support["strict_observed"],
                    selected_indices=selected,
                    original_selected_indices=old_selected,
                    stable_path_order=order,
                )
                comparisons = verify_control(arrays, saved)
                for key in ("axes", "enabled", "sigma", "parallel_sigma", "sources", "chest_age"):
                    checks.require(
                        checks.equal_bytes(
                            fitter.last_diagnostics[key], saved["parent_prior_" + key]
                        ),
                        "GPU prior policy differs",
                    )
                rows.append(dict(frame_idx=native, comparisons=comparisons))
                np.savez_compressed(output / f"frame_{native:06d}.npz", **arrays)
                print(json.dumps(dict(event="CUDA_frame_passed", frame=native)), flush=True)
            window.assert_open()
            checks.require(
                fitter.initialize_count == 0, "qualification unexpectedly initialized tracking"
            )
            summary = dict(
                schema="directional_CUDA_window_qualification_v1",
                status="ok",
                frames=FRAMES,
                seconds=time.perf_counter() - started,
                comparisons=rows,
                GPU=torch.cuda.get_device_name(0),
                torch_version=str(torch.__version__),
                CUDA_qualified_for_fixed_four_frames=True,
                temporal_run=False,
                independent_ID_truth=False,
            )
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        finally:
            window.close()
            for path, digest in hashes.items():
                checks.require(checks.sha(path) == digest, "bound input changed")
                record.input(path, path, sha256=digest)
            checks.require(
                code == {**base.local_code(), str(PLAN): checks.sha(PLAN)}, "source set changed"
            )
            for p in output.iterdir():
                if p.is_file() and p.name != "qualification.provenance.json":
                    record.output(p.name, p, sha256=checks.sha(p))
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "recovery", "cpu-fit", "cpu-qualification", "supervisor-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--child-pid", type=int, required=True)
    main(p.parse_args())
