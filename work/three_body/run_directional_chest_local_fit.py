"""Qualify an off control, then fit all20/all31 with the fixed directional policy."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import fit_chest_recovery_candidates as base
import numpy as np
import torch
from directional_chest_fitter import LocalDirectionalChestFitter

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

REPO = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name("DIRECTIONAL_CHEST_PRIOR_PLAN.md")
checks = base.evidence_tools


def verify_control(actual, expected):
    checked = {}
    for key in (
        "fitted_pose_xyz",
        "local_cost",
        "final_loss_parts",
        "mask_supported",
        "after_strict_sources",
        "after_strict_observed",
        "selected_indices",
        "original_selected_indices",
        "stable_path_order",
    ):
        a, b = actual[key], expected[key]
        checks.require(a.shape == b.shape and a.dtype == b.dtype, "control dtype/shape differs")
        exact = checks.equal_bytes(a, b)
        close = (
            bool(np.allclose(a, b, rtol=0, atol=base.ATOL, equal_nan=True))
            if a.dtype.kind == "f"
            else exact
        )
        checks.require(close, "disabled directional control differs: " + key)
        checked[key] = dict(
            byte_exact=exact,
            within_fixed_comparison=close,
            maximum_absolute_difference=(
                float(np.max(np.abs(a - b))) if a.dtype.kind == "f" else 0.0
            ),
        )
    return checked


def main(replay, recovery, prior_fit, output):
    code = {**base.local_code(), str(PLAN): checks.sha(PLAN)}
    hashes = dict(code)
    run = checks.verify(
        replay, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
    )
    checks.verify(
        recovery, "probe.provenance.json", "chest_recovery_counterfactual_proposals", hashes
    )
    original_fit = checks.verify(
        prior_fit, "fit.provenance.json", "chest_recovery_same_objective_local_CPU_fit", hashes
    )
    for root, name in ((replay, "replay.provenance.json"), (recovery, "probe.provenance.json")):
        path = str(root / name)
        checks.require(
            original_fit["inputs"][path]["sha256"] == hashes[path], "different prior inputs"
        )
    frames = [row["frame_idx"] for row in checks.read(prior_fit / "frames.json")]
    checks.require(
        len(frames) == 20 and frames[0] == 66660 and frames[-1] == 66696,
        "fixed all20 scope changed",
    )
    registration = checks.read(replay / "registration/plan.json")
    binding = registration["original_binding"]

    def bind(path):
        path = Path(path).resolve()
        expected = run["inputs"][str(path)]["sha256"]
        checks.require(checks.sha(path) == expected, "original calibration/seed/code changed")
        hashes[str(path)] = expected
        return path

    for path in binding["code_sha256"]:
        bind(path)
    calibration = Path(binding["inputs"]["seed"]["input_root"]) / base.CALIBRATION_DIR
    for path in calibration.iterdir():
        if path.is_file():
            bind(path)
    seed = Path(binding["inputs"]["seed"]["registration_path"]).parent / "initial_pose_xyz.npy"
    initial = np.load(bind(seed), allow_pickle=False)
    config = base.TrackerConfig(
        **{**registration["original_context"]["tracker_config"], "device": "cpu"}
    )
    torch.set_num_threads(2)
    rig = load_rig(calibration)
    output.mkdir(parents=True, exist_ok=False)
    rows, controls = [], []

    def frame_inputs(native):
        folder = replay / "diagnostics" / f"frame_{native:06d}"
        return (
            base.load_arrays(folder / "arrays.npz", hashes),
            base.load_arrays(recovery / f"frame_{native:06d}.npz", hashes),
            checks.read(folder / "metadata.json")["proposal_kind"],
        )

    with provenance("directional_chest_local_CPU_fit", output, name="fit") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        record.param("native_frames", frames)
        record.param("disabled_control_frames", [frames[0], frames[-1]])
        record.param("tracker_config", dataclasses.asdict(config))
        record.param("atol", base.ATOL)
        record.param(
            "prior_policy",
            dict(
                minimum_chest_age=4,
                current_predict_views=1,
                sigma_increment=0.3,
                age_cap=20,
                same_policy_for_every_proposal_of_a_parent=True,
            ),
        )
        record.param("local_only_not_tracking", True)
        start = time.perf_counter()
        control = LocalDirectionalChestFitter(rig, initial, config, directional=False)
        for native in (frames[0], frames[-1]):
            a, new, kinds = frame_inputs(native)
            arrays, _ = base.frame_fit(control, a, new, kinds)
            saved = base.load_arrays(prior_fit / f"frame_{native:06d}.npz", hashes)
            checked = verify_control(arrays, saved)
            controls.append(dict(frame_idx=native, comparisons=checked))
            print(json.dumps(dict(event="disabled_control_passed", frame=native)), flush=True)
        path = output / "disabled_controls.json"
        path.write_text(json.dumps(controls, indent=2), encoding="utf-8")
        record.output(path.name, path, sha256=checks.sha(path))
        fitter = LocalDirectionalChestFitter(rig, initial, config, directional=True)
        for native in frames:
            a, new, kinds = frame_inputs(native)
            frame_start = time.perf_counter()
            arrays, info = base.frame_fit(fitter, a, new, kinds)
            policy = fitter.last_diagnostics
            # base.frame_fit also evaluates the unchanged isotropic start as a
            # diagnostic. The fitter uses its own directional initial score.
            arrays["isotropic_initial_cost"] = arrays["initial_cost"].copy()
            arrays["initial_cost"] = policy["initial_score"].copy()
            arrays["initial_loss_parts"] = policy["initial_parts"].copy()
            arrays["initial_weighted_six_parts"] = base.six_weighted_parts(
                arrays["initial_pose_xyz"],
                arrays["predicted_pose_xyz"],
                arrays["initial_loss_parts"],
                config,
            )
            checks.require(
                np.allclose(
                    arrays["initial_weighted_six_parts"].sum(1),
                    arrays["initial_cost"],
                    atol=base.ATOL,
                    rtol=0,
                ),
                "directional initial cost decomposition differs",
            )
            checks.require(
                np.all(arrays["local_cost"] <= arrays["initial_cost"] + base.ATOL),
                "directional best fit is worse than its directional start",
            )
            for key in ("axes", "enabled", "sigma", "parallel_sigma", "sources", "chest_age"):
                arrays["parent_prior_" + key] = policy[key].copy()
            info["original_proposals_vs_isotropic_GPU"] = info.pop("original_cpu_gpu")
            info.update(
                frame_idx=native,
                seconds=time.perf_counter() - frame_start,
                directional_parent_fish=int(policy["enabled"].sum()),
            )
            path = output / f"frame_{native:06d}.npz"
            np.savez_compressed(path, **arrays)
            record.output(path.name, path, sha256=checks.sha(path))
            rows.append(info)
            print(json.dumps(info), flush=True)
        checks.require(
            control.initialize_count == fitter.initialize_count == 0, "unexpected tracking"
        )
        result = dict(
            schema="directional_chest_local_CPU_fit_v1",
            status="ok",
            frames=len(rows),
            new_candidates=sum(r["new_candidates"] for r in rows),
            new_all_three_before=sum(r["new_all_three_chests_before"] for r in rows),
            new_all_three_after=sum(r["new_all_three_chests_after"] for r in rows),
            frames_new_best=sum(r["best_is_new"] for r in rows),
            frames_combined_best_all_three=sum(r["combined_best_all_three_chests"] for r in rows),
            frames_supported_new_selected=sum(r["selected_new_all_three_chests"] > 0 for r in rows),
            disabled_controls_passed=len(controls),
            seconds=time.perf_counter() - start,
            adopted=False,
            tracking_performed=False,
            held_out_image_evaluation=False,
            independent_ID_truth=False,
        )
        for name, data in (("summary.json", result), ("frames.json", rows)):
            path = output / name
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            record.output(path.name, path, sha256=checks.sha(path))
        checks.require(
            code == {**base.local_code(), str(PLAN): checks.sha(PLAN)}, "source set changed"
        )
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "bound input changed")
        for item in record.outputs.values():
            checks.require(checks.sha(item["path"]) == item["sha256"], "fit output changed")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--prior-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    checks.require(
        target.is_relative_to(REPO / "work") and not target.exists(), "new output required"
    )
    main(args.replay.resolve(), args.recovery.resolve(), args.prior_fit.resolve(), target)
