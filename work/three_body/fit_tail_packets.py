"""Frozen-objective local comparison of original and additional free-intersection tail seeds.

Every frame starts from its saved six original parents. No selected result is
propagated. The original prefix must reproduce all 19 retained history fields;
the unrecorded original pre-fit pool is not claimed byte-certified. CUDA also
compares every candidate on four fixed frames with this driver's CPU reference.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import fit_chest_recovery_candidates as base
import numpy as np
import torch
from bounded_fit_record_check import compare_records, reconstruct_records
from continuous_tracker import RECORD_FIELDS
from directional_device_fitter import DirectionalDeviceFitter
from node_geometry import RayObservations
from observations import PixelTargets
from persistent_node_tracker import PersistentNodeState
from qualify_concurrent_directional_gpu import active_progress, gpu_sample
from replay_gpu_window import ReplayGPUWindow
from tracker import Proposal

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks = base.evidence_tools
REPO = base.REPO
FRAMES = (66524, *range(66620, 66920))
CPU_FRAMES = (66524, 66628, 66703, 66907)
STAGE = "tail_same_objective_local_fit"
PACKET_STAGE = "tail_local_fit_packet_preparation_v1"
QUALIFIED = REPO / "work/three_body/directional_cuda_qualification_v1"
QUALIFIED_SHA = "bea4025f748990b240418d57977209c51e17284fdac8e68d479779ca9adccc24"


def validate_tail_metadata(metadata):
    """Bind the predeclared free-only, original-parent local experiment."""
    checks.require(
        metadata["schema"] == PACKET_STAGE
        and tuple(metadata["frames"]) == FRAMES
        and tuple(metadata["cpu_fit_frames"]) == CPU_FRAMES
        and metadata["allow_bounded"] is False
        and metadata["proposal_kind"] == "tail_recovery"
        and metadata["original_prefix_preserved"] is True
        and metadata["serial_flow"] is True
        and metadata["fitted"] is False
        and metadata["tracked"] is False,
        "fixed free-only tail packet policy differs",
    )
    persistence = metadata["persistence_config"]
    checks.require(
        persistence["enabled"] is True
        and persistence["minimum_missing_frames"] == 4
        and persistence["minimum_pectoral_frames"] == 4,
        "tail/chest age4 and enabled policy required",
    )


def member(root, name, hashes):
    path = (root / name).resolve(strict=True)
    checks.require(str(path) in hashes, "required file is not a sealed member: " + str(path))
    checks.require(checks.sha(path) == hashes[str(path)], "sealed member changed")
    return path


def bind_input(seal, path, hashes):
    path = Path(path).resolve(strict=True)
    item = seal["inputs"].get(str(path))
    checks.require(item is not None, "producer did not bind consumed input: " + str(path))
    checks.require(checks.sha(path) == item["sha256"], "producer input changed")
    if str(path) in hashes:
        checks.require(hashes[str(path)] == item["sha256"], "conflicting input hashes")
    hashes[str(path)] = item["sha256"]
    return path


def fit_inputs(packet, rig, cfg):
    count, original = len(packet["proposal_pose_xyz"]), int(packet["original_count"])
    checks.require(6 <= original <= count <= cfg.max_hypotheses + 6, "invalid candidate budget")
    checks.require(
        checks.equal_bytes(packet["proposal_is_added"], np.arange(count) >= original),
        "original candidates must be an unchanged prefix",
    )
    checks.require(
        np.all(packet["proposal_kind"][original:] == "tail_recovery"),
        "unexpected added candidate kind",
    )
    states = [
        PersistentNodeState(
            packet["parent_pose"][i].copy(),
            packet["parent_velocity"][i].copy(),
            packet["parent_missing"][i].copy(),
            packet["parent_review"][i].copy(),
            float(packet["parent_cost"][i]),
            packet["parent_node_missing"][i].copy(),
            packet["parent_direct_pectoral_age"][i].copy(),
        )
        for i in range(6)
    ]
    proposals = [
        Proposal(
            packet["proposal_pose_xyz"][i].copy(),
            packet["proposal_predicted"][i].copy(),
            packet["proposal_plan"][i].copy(),
            int(packet["proposal_parent"][i]),
            str(packet["proposal_kind"][i]),
            tuple(int(v) for v in packet["proposal_permutation"][i]),
        )
        for i in range(count)
    ]
    rays = RayObservations.from_points(rig, {v: packet[v + "_points"].copy() for v in VIEWS})
    for i, view in enumerate(VIEWS):
        checks.require(
            checks.equal_bytes(rays.origins[i], packet[view + "_ray_origins"])
            and checks.equal_bytes(rays.directions[i], packet[view + "_ray_directions"]),
            "recalibrated rays differ from sealed packet",
        )
    checks.require(packet["fit_pixels"].shape == (3, 1024, 2), "fixed 1024 samples per view")
    targets = PixelTargets(
        {v: packet["fit_pixels"][i].copy() for i, v in enumerate(VIEWS)},
        packet["fit_target"].copy(),
        packet["fit_weights"].copy(),
        packet["fit_foreground"].copy(),
    )
    return proposals, states, rays, targets


def fitted_frame(fitter, packet):
    inputs = fit_inputs(packet, fitter.rig, fitter.config)
    proposals, states, rays, _ = inputs
    original = int(packet["original_count"])
    initial = packet["proposal_pose_xyz"]
    plans = packet["proposal_plan"]
    before = base.support_snapshot(initial, plans, rays, fitter.rig)
    for source, saved in (
        ("strict_sources", "generated_strict_sources"),
        ("strict_observed", "generated_observed"),
        ("ray_distance_mm", "generated_ray_distance_mm"),
    ):
        checks.require(
            checks.equal_bytes(before[source], packet[saved]),
            "pre-fit geometric evidence differs: " + source,
        )
    started = time.perf_counter()
    pose, cost, parts, masks = fitter._fit(*inputs)
    if fitter.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    after = base.support_snapshot(pose, plans, rays, fitter.rig)
    checks.require(
        not before["plan_source_collision"].any() and not after["strict_source_collision"].any(),
        "candidate reused a source",
    )
    checks.require(
        np.all(cost <= fitter.last_diagnostics["initial_score"] + base.ATOL),
        "best retained fit worsened its initial objective",
    )
    parents, permutations = packet["proposal_parent"], packet["proposal_permutation"]
    path_cost = cost + np.array([states[p.parent].cost for p in proposals])
    selected, order = base.selected_indices(
        pose, path_cost, parents, permutations, fitter.config.beam_size
    )
    control_selected, control_order = base.selected_indices(
        pose[:original],
        path_cost[:original],
        parents[:original],
        permutations[:original],
        fitter.config.beam_size,
    )
    arrays = dict(
        fitted_pose_xyz=pose,
        local_cost=cost,
        path_cost=path_cost,
        final_loss_parts=parts,
        mask_supported=masks,
        selected_indices=selected,
        original_selected_indices=control_selected,
        stable_path_order=order,
        original_stable_path_order=control_order,
        initial_score=fitter.last_diagnostics["initial_score"].copy(),
        initial_parts=fitter.last_diagnostics["initial_parts"].copy(),
    )
    arrays.update({"before_" + k: v for k, v in before.items()})
    arrays.update({"after_" + k: v for k, v in after.items()})
    for key in ("axes", "enabled", "sigma", "parallel_sigma", "sources", "chest_age"):
        arrays["parent_prior_" + key] = fitter.last_diagnostics[key].copy()
    for label, chosen, ranked, count in (
        ("control", control_selected, control_order, original),
        ("combined", selected, order, len(proposals)),
    ):
        records = reconstruct_records(
            packet,
            pose,
            cost,
            parts,
            masks,
            after["strict_sources"],
            after["strict_observed"],
            chosen,
            ranked,
            fitter.config,
            proposal_count=count,
        )
        arrays.update({label + "_" + k: v for k, v in records.items()})
    return arrays, seconds


def compare_fit_arrays(actual, expected):
    """CPU/CUDA parity includes the full pool, stable decisions and all records."""
    checks.require(actual.keys() == expected.keys(), "CPU/CUDA saved array fields differ")
    result = {}
    for key in actual:
        a, b = actual[key], expected[key]
        checks.require(a.shape == b.shape and a.dtype == b.dtype, "CPU/CUDA schema: " + key)
        exact = checks.equal_bytes(a, b)
        if a.dtype.kind == "f":
            checks.require(
                np.array_equal(np.isnan(a), np.isnan(b))
                and np.array_equal(np.isposinf(a), np.isposinf(b))
                and np.array_equal(np.isneginf(a), np.isneginf(b)),
                "CPU/CUDA nonfinite layout differs: " + key,
            )
            finite = np.isfinite(a) & np.isfinite(b)
            difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            checks.require(difference <= base.ATOL, "CPU/CUDA numerical mismatch: " + key)
        else:
            difference = 0.0
            checks.require(exact, "CPU/CUDA decision mismatch: " + key)
        result[key] = dict(byte_exact=exact, maximum_absolute_difference=difference)
    return result


def frame_summary(native, packet, arrays, seconds):
    original = int(packet["original_count"])
    old, new = int(arrays["original_selected_indices"][0]), int(arrays["selected_indices"][0])
    selected = arrays["selected_indices"]
    observed = arrays["after_strict_observed"]
    return dict(
        frame_idx=native,
        original_candidates=original,
        additional_candidates=len(packet["proposal_parent"]) - original,
        fit_seconds=seconds,
        control_best=old,
        combined_best=new,
        best_is_added=new >= original,
        selected_added_count=int(np.count_nonzero(selected >= original)),
        added_strict_tails_before=int(arrays["before_strict_observed"][original:, :, 2].sum()),
        added_strict_tails_after=int(observed[original:, :, 2].sum()),
        control_best_observed=observed[old].tolist(),
        combined_best_observed=observed[new].tolist(),
        control_best_path_cost=float(arrays["path_cost"][old]),
        combined_best_path_cost=float(arrays["path_cost"][new]),
        local_score_change=float(arrays["local_cost"][new] - arrays["local_cost"][old]),
        mask_error_change=float(
            arrays["final_loss_parts"][new, 4] - arrays["final_loss_parts"][old, 4]
        ),
        control_best_review=arrays["control_identity_review"][0].tolist(),
        combined_best_review=arrays["combined_identity_review"][0].tolist(),
    )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main(args):
    packets, output = args.packets.resolve(strict=True), args.output.resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    frames = CPU_FRAMES if args.device == "cpu" else FRAMES
    code, hashes = base.local_code(), {}
    hashes.update(code)
    output.mkdir(parents=True)
    guard = None
    rows, controls, parity, telemetry = [], [], [], []
    with provenance(STAGE, output, name="fit") as record:
        try:
            seal_path = packets / "preparation.provenance.json"
            checks.require(checks.sha(seal_path) == args.packet_sha, "packet seal differs")
            producer = checks.verify(packets, seal_path.name, PACKET_STAGE, hashes)
            metadata = checks.read(member(packets, "metadata.json", hashes))
            checks.require(tuple(metadata["frames"]) == FRAMES, "fixed 301-frame packet scope")
            validate_tail_metadata(metadata)
            cfg = base.TrackerConfig(**metadata["tracker_config"])
            checks.require(cfg.device == "cuda" and cfg.opt_steps == 14, "historical CUDA/14 steps")
            cfg = dataclasses.replace(cfg, device=args.device)
            calibration = Path(metadata["calibration_directory"])
            calibration_files = [p for p in calibration.iterdir() if p.is_file()]
            checks.require(len(calibration_files) == 4, "four fixed calibration files required")
            for path in calibration_files:
                bind_input(producer, path, hashes)
            shared = set(code) & set(producer["inputs"])
            for path in shared:
                bind_input(producer, path, hashes)
            checks.require(
                str(Path(__file__).with_name("directional_device_fitter.py")) in shared,
                "packet did not bind original fitter",
            )
            qpath = QUALIFIED / "qualification.provenance.json"
            checks.require(
                QUALIFIED_SHA and checks.sha(qpath) == QUALIFIED_SHA, "qualified seal changed"
            )
            qualified = checks.verify(
                QUALIFIED, qpath.name, "directional_CUDA_window_qualification", hashes
            )
            for name in ("directional_device_fitter.py", "directional_chest_fitter.py"):
                bind_input(qualified, Path(__file__).with_name(name), hashes)
            rig = load_rig(calibration)
            prepared = {}
            for native in FRAMES:
                packet = base.load_arrays(member(packets, f"f{native}_packet.npz", hashes), hashes)
                checks.require(int(packet["frame_idx"]) == native, "packet frame differs")
                prepared[native] = packet
            initial = prepared[FRAMES[0]]["initial_pose_xyz"]
            for packet in prepared.values():
                checks.require(
                    checks.equal_bytes(initial, packet["initial_pose_xyz"]), "seed changed"
                )
            cpu_expected = {}
            if args.device == "cuda":
                checks.require(args.cpu_reference is not None, "four-frame CPU reference required")
                cpu = args.cpu_reference.resolve(strict=True)
                cpu_seal = checks.verify(cpu, "fit.provenance.json", STAGE, hashes)
                status = checks.read(member(cpu, "summary.json", hashes))
                checks.require(
                    status["status"] == "ok"
                    and status["device"] == "cpu"
                    and tuple(status["frames"]) == CPU_FRAMES,
                    "successful fixed four-frame CPU reference required",
                )
                for path in (Path(__file__), packets / seal_path.name):
                    bind_input(cpu_seal, path, hashes)
                for path in set(code) & set(cpu_seal["inputs"]):
                    bind_input(cpu_seal, path, hashes)
                cpu_expected = {
                    native: base.load_arrays(member(cpu, f"f{native}_fit.npz", hashes), hashes)
                    for native in CPU_FRAMES
                }
                checks.require(args.supervisor_root and args.child_pid, "owned reference required")
                observed = output / "registered_child"
                observed.mkdir()
                guard = ReplayGPUWindow(args.supervisor_root, args.child_pid, observed)
                guard.attach()
                active_progress(guard)
                telemetry.append(gpu_sample())
                checks.require(
                    telemetry[-1]["total_MiB"] - telemetry[-1]["used_MiB"] >= 6144,
                    "less than 6GiB initial global GPU headroom",
                )
                torch.cuda.set_per_process_memory_fraction(0.25, 0)
                torch.cuda.reset_peak_memory_stats()
            torch.set_num_threads(2)
            fitter = DirectionalDeviceFitter(rig, initial.copy(), cfg)
            for packet in prepared.values():
                checks.require(
                    checks.equal_bytes(fitter.lengths, packet["initial_lengths"]),
                    "seed lengths differ",
                )
            for path, digest in code.items():
                destination = output / "source_snapshot" / Path(path).relative_to(REPO)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(Path(path).read_bytes())
                checks.require(checks.sha(destination) == digest, "source snapshot differs")
            record.param("frames", frames)
            record.param("tracker_config", dataclasses.asdict(cfg))
            record.param("comparison_atol", base.ATOL)
            record.param("rtol", 0)
            record.param("propagate_selected_results", False)
            record.param("allocator_fraction_cap", 0.25 if args.device == "cuda" else None)
            for native in frames:
                if guard:
                    active_progress(guard)
                    telemetry.append(gpu_sample())
                    checks.require(
                        telemetry[-1]["total_MiB"] - telemetry[-1]["used_MiB"] >= 2048,
                        "less than 2GiB remaining global GPU headroom",
                    )
                packet = prepared[native]
                arrays, seconds = fitted_frame(fitter, packet)
                np.savez_compressed(output / f"f{native}_fit.npz", **arrays)
                actual = {key: arrays["control_" + key] for key in RECORD_FIELDS}
                expected = {key: packet["retained_" + key] for key in RECORD_FIELDS}
                control = compare_records(actual, expected)
                controls.append(dict(frame_idx=native, comparisons=control))
                if native in cpu_expected:
                    parity.append(
                        dict(
                            frame_idx=native,
                            comparisons=compare_fit_arrays(arrays, cpu_expected[native]),
                        )
                    )
                rows.append(frame_summary(native, packet, arrays, seconds))
                print(json.dumps(rows[-1]), flush=True)
            checks.require(fitter.initialize_count == 0, "unexpected temporal initialization")
            if guard:
                active_progress(guard)
                telemetry.append(gpu_sample())
            summary = dict(
                schema="tail_same_objective_local_fit_v1",
                status="ok",
                device=args.device,
                frames=frames,
                control_all19_records_reproduced=True,
                all_prefit_original_candidates_byte_certified=False,
                independent_frames=True,
                allow_bounded=False,
                target_node=2,
                temporal_tracking_performed=False,
                independent_ID_truth=False,
                original_pool_preserved=True,
                frames_best_is_added=sum(r["best_is_added"] for r in rows),
                frames_any_added_selected=sum(r["selected_added_count"] > 0 for r in rows),
                all_fit_seconds=sum(r["fit_seconds"] for r in rows),
                CPU_CUDA_all_candidate_frames=[r["frame_idx"] for r in parity],
                control_best_direct_nodes=np.sum(
                    [r["control_best_observed"] for r in rows], axis=(0, 1)
                ).tolist(),
                combined_best_direct_nodes=np.sum(
                    [r["combined_best_observed"] for r in rows], axis=(0, 1)
                ).tolist(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if guard else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if guard else None,
            )
            write_json(output / "summary.json", summary)
        finally:
            if guard:
                guard.close()
            for name, value in (
                ("frames.json", rows),
                ("historical_control.json", controls),
                ("CPU_CUDA_parity.json", parity),
                ("GPU_telemetry.json", telemetry),
            ):
                write_json(output / name, value)
            for path, digest in hashes.items():
                checks.require(checks.sha(path) == digest, "bound input changed")
                record.input(path, path, sha256=digest)
            checks.require(code == base.local_code(), "loaded source set changed")
            for path in output.rglob("*"):
                if path.is_file() and path.name != "fit.provenance.json":
                    record.output(str(path.relative_to(output)), path, sha256=checks.sha(path))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--packet-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cpu-reference", type=Path)
    parser.add_argument("--supervisor-root", type=Path)
    parser.add_argument("--child-pid", type=int)
    main(parser.parse_args())
