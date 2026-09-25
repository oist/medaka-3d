"""Four fixed actual CPU fits qualify the device fitter before any CUDA launch."""

import argparse
import json
import time
from pathlib import Path

import fit_chest_recovery_candidates as base
import numpy as np
import torch
from directional_device_fitter import DirectionalDeviceFitter
from run_directional_chest_local_fit import verify_control

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

checks = base.evidence_tools
REPO = Path(__file__).resolve().parents[2]
FRAMES = (66660, 66664, 66667, 66696)  # early, disabled policy, first recovery winner, last


def main(replay, recovery, local_fit, output):
    code = base.local_code()
    hashes = dict(code)
    run = checks.verify(
        replay, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
    )
    checks.verify(
        recovery, "probe.provenance.json", "chest_recovery_counterfactual_proposals", hashes
    )
    checks.verify(local_fit, "fit.provenance.json", "directional_chest_local_CPU_fit", hashes)
    registration = checks.read(replay / "registration/plan.json")
    binding = registration["original_binding"]
    calibration = Path(binding["inputs"]["seed"]["input_root"]) / base.CALIBRATION_DIR
    seed = Path(binding["inputs"]["seed"]["registration_path"]).parent / "initial_pose_xyz.npy"
    for path in [seed, *[p for p in calibration.iterdir() if p.is_file()]]:
        digest = run["inputs"][str(path)]["sha256"]
        checks.require(checks.sha(path) == digest, "original input changed")
        hashes[str(path)] = digest
    config = base.TrackerConfig(
        **{**registration["original_context"]["tracker_config"], "device": "cpu"}
    )
    torch.set_num_threads(2)
    fitter = DirectionalDeviceFitter(
        load_rig(calibration), np.load(seed, allow_pickle=False), config
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    start = time.perf_counter()
    with provenance("directional_device_CPU_qualification", output, name="qualification") as record:
        record.param("fixed_frames", FRAMES)
        record.param("comparison_atol", base.ATOL)
        record.param("CUDA_launched", False)
        for native in FRAMES:
            folder = replay / "diagnostics" / f"frame_{native:06d}"
            arrays, _ = base.frame_fit(
                fitter,
                base.load_arrays(folder / "arrays.npz", hashes),
                base.load_arrays(recovery / f"frame_{native:06d}.npz", hashes),
                checks.read(folder / "metadata.json")["proposal_kind"],
            )
            saved = base.load_arrays(local_fit / f"frame_{native:06d}.npz", hashes)
            compared = verify_control(arrays, saved)
            for key in ("axes", "enabled", "sigma", "parallel_sigma", "sources", "chest_age"):
                checks.require(
                    checks.equal_bytes(fitter.last_diagnostics[key], saved["parent_prior_" + key]),
                    "prior policy differs",
                )
            rows.append(dict(frame_idx=native, comparisons=compared))
            path = output / f"frame_{native:06d}.npz"
            np.savez_compressed(
                path,
                fitted_pose_xyz=arrays["fitted_pose_xyz"],
                local_cost=arrays["local_cost"],
                final_loss_parts=arrays["final_loss_parts"],
                selected_indices=arrays["selected_indices"],
            )
            record.output(path.name, path, sha256=checks.sha(path))
            print(json.dumps(dict(frame_idx=native, passed=True)), flush=True)
        checks.require(fitter.initialize_count == 0, "unexpected temporal initialization")
        result = dict(
            status="ok",
            schema="directional_device_CPU_qualification_v1",
            frames=list(FRAMES),
            seconds=time.perf_counter() - start,
            comparisons=rows,
            CUDA_qualified=False,
            temporal_run=False,
        )
        path = output / "summary.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.output(path.name, path, sha256=checks.sha(path))
        checks.require(code == base.local_code(), "source set changed")
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "bound input changed")
            record.input(path, path, sha256=digest)
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--recovery", type=Path, required=True)
    p.add_argument("--local-fit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(),
        "new workspace output required",
    )
    main(args.replay.resolve(), args.recovery.resolve(), args.local_fit.resolve(), output)
