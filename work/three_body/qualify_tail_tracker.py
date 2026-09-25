"""CPU proposal-only equivalence on the sealed301 exact historical parent packets."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPO / "labs/20260909_three_body"), str(_REPO / "src")]

import cv2
import numpy as np
import prepare_bounded_mouth_fit_packets as packets
import prepare_tail_fit_packets as producer
import tail_recovery_tracker as variant
import torch
from motion import FlowObservations
from node_geometry import RayObservations
from node_recovery import NodeRecoveryConfig
from persistent_node_tracker import PersistentNodeConfig, PersistentNodeState
from tracker import TrackerConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks, REPO = packets.checks, packets.REPO
STAGE = "tail_tracker_generation_qualification_301_v1"
ROOT = REPO / "work/three_body/tail_local_fit_packets_301_v1"
PACKET_SHA = "6e0e3a8d7d006a9f3afc673af829f54de77567d1346a7cb613179114f1769cdc"
TRACKER_SHA = "b5a98085c6b1b03bd17e2f0b627ffc3eebdd569cf5add545146d9322761868a2"
HELPER_SHA = "636509cf37bb485a5e8efd34bbd55320d928df35fce59e4617addfee4b75a201"
PRODUCER_SHA = "900183e8978f150cf9f4a7fa09c2de1799a17a556ade06da8af57d516479d7f3"
FRAMES = (66524, *range(66620, 66920))
PROPOSAL_KEYS = (
    "proposal_pose_xyz",
    "proposal_predicted",
    "proposal_plan",
    "proposal_parent",
    "proposal_kind",
    "proposal_permutation",
)


def validate_proposals(actual, expected, count):
    values = packets.packet_proposals(actual)
    checks.require(len(actual) == count, "proposal count differs")
    for key in PROPOSAL_KEYS:
        checks.require(
            checks.equal_bytes(values[key], expected[key][:count]),
            "proposal dtype/shape/bytes differ: " + key,
        )
    return values


def validate_context(context, expected, enabled):
    """Only the declared tail policy and implementation label may be added."""
    adapted = copy.deepcopy(context)
    policy = adapted.pop("tail_recovery_policy")
    checks.require(
        policy["schema"] == "tail_recovery_continuous_policy_v1"
        and policy["enabled"] is enabled
        and policy["target_node"] == 2
        and policy["allow_bounded"] is False
        and policy["clears_identity_review"] is False,
        "explicit free-only tail context differs",
    )
    adapted["implementation"] = expected["implementation"]
    checks.require(adapted == expected, "core CPU context differs")
    return policy


def context_input_signature(parents, rays, flow):
    arrays = {
        "parent_" + k: np.stack([getattr(p, k) for p in parents]) for k in packets.STATE_FIELDS
    }
    arrays.update(
        {"flow_" + f.name: np.asarray(getattr(flow, f.name)) for f in dataclasses.fields(flow)}
    )
    for vi, v in enumerate(VIEWS):
        arrays[v + "_points"] = rays.points[v]
        arrays[v + "_origins"] = rays.origins[vi]
        arrays[v + "_directions"] = rays.directions[vi]
    return packets.signature(arrays)


def validate_totals(rows):
    checks.require(
        [r["frame_idx"] for r in rows] == list(FRAMES), "fixed301 frame sequence differs"
    )
    original = sum(r["original_count"] for r in rows)
    total = sum(r["total_count"] for r in rows)
    checks.require(
        original == 4273 and total == 4949 and total - original == 676,
        "fixed proposal totals differ",
    )
    return original, total - original, total


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = packets.comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    checks.require(checks.sha(variant.__file__) == TRACKER_SHA, "tracker freeze differs")
    checks.require(
        checks.sha(variant.tail_recovery.__file__) == HELPER_SHA, "helper freeze differs"
    )
    checks.require(checks.sha(producer.__file__) == PRODUCER_SHA, "packet producer changed")
    output.mkdir(parents=True)
    threads = torch.get_num_threads(), cv2.getNumThreads()
    with provenance(STAGE, output, name="qualification") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            digest = checks.sha(path)
            outputs[str(path)] = digest
            record.output(name, path, sha256=digest)

        try:
            packets.prior.bind(ROOT / "preparation.provenance.json", PACKET_SHA, hashes)
            seal = checks.verify(ROOT, "preparation.provenance.json", producer.STAGE, hashes)
            for path, digest in code.items():
                if path in seal["inputs"]:
                    checks.require(
                        digest == seal["inputs"][path]["sha256"], "historical code changed"
                    )
                target = output / "code_snapshot" / Path(path).relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(path).read_bytes())
                checks.require(checks.sha(target) == digest, "source copy changed")
                emit(str(target.relative_to(output)))
            for name in ("metadata.json", "summary.json"):
                checks.require(str(ROOT / name) in hashes, "unsealed packet metadata")
            meta = checks.read(ROOT / "metadata.json")
            summary = checks.read(ROOT / "summary.json")
            checks.require(
                tuple(meta["frames"]) == FRAMES
                and summary["packets"] == 301
                and summary["originals"] == 4273
                and summary["added"] == 676
                and meta["allow_bounded"] is False
                and meta["serial_flow"] is True
                and summary["fit_calls"]
                == summary["advance_calls"]
                == summary["restore_calls"]
                == 0,
                "wrong fixed proposal packets",
            )
            shared = meta["checkpoint_binding"]["original_binding"]["inputs"]["seed"][
                "SL_and_calibration_sha256"
            ]
            checks.require(len(shared) == 7, "same SL3/calibration4 required")
            for path, digest in shared.items():
                packets.prior.audit_input(seal, path, hashes)
                checks.require(hashes[str(Path(path).resolve())] == digest, "shared input differs")
            rig = load_rig(meta["calibration_directory"])
            torch.set_num_threads(meta["torch_cpu_threads"])
            cv2.setNumThreads(meta["opencv_threads"])
            cfg = copy.deepcopy(meta["tracker_config"])
            checks.require(cfg["device"] == "cuda", "original CUDA context required")
            cfg["device"] = "cpu"
            expected_context = copy.deepcopy(meta["original_context"])
            expected_context["tracker_config"]["device"] = "cpu"
            checks.require(
                expected_context == meta["generation_context"],
                "packet CPU context differs beyond device",
            )
            rows = []
            contexts = {}
            for native in FRAMES:
                path = ROOT / f"f{native}_packet.npz"
                checks.require(str(path) in hashes, "unsealed native packet")
                keys = (
                    *PROPOSAL_KEYS,
                    "frame_idx",
                    "original_count",
                    "initial_pose_xyz",
                    "initial_lengths",
                    "flow_present",
                    *("parent_" + k for k in packets.STATE_FIELDS),
                    *("flow_" + f.name for f in dataclasses.fields(FlowObservations)),
                    *(v + s for v in VIEWS for s in ("_points", "_ray_origins", "_ray_directions")),
                )
                with np.load(path, allow_pickle=False) as archive:
                    data = {k: archive[k].copy() for k in keys}
                frozen = packets.signature(data)
                checks.require(
                    int(data["frame_idx"]) == native and bool(data["flow_present"]),
                    "native flow context missing",
                )
                flow = FlowObservations(
                    **{
                        f.name: int(data["flow_" + f.name])
                        if f.name in ("frame_idx", "previous_frame_idx")
                        else data["flow_" + f.name].copy()
                        for f in dataclasses.fields(FlowObservations)
                    }
                )
                checks.require(
                    flow.frame_idx == native and flow.previous_frame_idx == native - 1,
                    "flow native offset differs",
                )
                rays = RayObservations.from_points(rig, {v: data[v + "_points"] for v in VIEWS})
                for vi, view in enumerate(VIEWS):
                    checks.require(
                        checks.equal_bytes(rays.origins[vi], data[view + "_ray_origins"])
                        and checks.equal_bytes(rays.directions[vi], data[view + "_ray_directions"]),
                        "actual calibrated rays differ",
                    )
                values = {}
                diagnostic = None
                for enabled in (True, False):
                    tracker = variant.TailRecoveryContinuousTracker(
                        rig,
                        data["initial_pose_xyz"],
                        TrackerConfig(**cfg),
                        recovery_config=NodeRecoveryConfig(**meta["recovery_config"]),
                        persistence_config=PersistentNodeConfig(**meta["persistence_config"]),
                        tail_recovery_enabled=enabled,
                    )
                    context = tracker._context()
                    validate_context(context, expected_context, enabled)
                    checks.require(
                        checks.equal_bytes(tracker.lengths, data["initial_lengths"]),
                        "initial shape lengths differ",
                    )
                    parents = [
                        PersistentNodeState(
                            **{k: data["parent_" + k][i].copy() for k in packets.STATE_FIELDS}
                        )
                        for i in range(6)
                    ]
                    parent_before = packets.signature(
                        {
                            k: np.stack([getattr(s, k) for s in parents])
                            for k in packets.STATE_FIELDS
                        }
                    )
                    context_before = context_input_signature(parents, rays, flow)
                    actual = tracker._proposals(parents, rays, flow)
                    count = len(data["proposal_parent"]) if enabled else int(data["original_count"])
                    result = validate_proposals(actual, data, count)
                    label = "enabled" if enabled else "disabled"
                    values.update({label + "_" + k: v for k, v in result.items()})
                    contexts[label] = context
                    checks.require(
                        context_before == context_input_signature(parents, rays, flow),
                        "ray/flow/parent mutation",
                    )
                    checks.require(
                        parent_before
                        == packets.signature(
                            {
                                k: np.stack([getattr(s, k) for s in parents])
                                for k in packets.STATE_FIELDS
                            }
                        ),
                        "parent mutation",
                    )
                    checks.require(
                        tracker.initialize_count
                        == tracker.processed_frames
                        == tracker.restore_count
                        == 0
                        and tracker._states is None
                        and tracker._previous_observation is None,
                        "qualification advanced a tracker",
                    )
                    if enabled:
                        diagnostic = tracker.last_tail_recovery_diagnostic
                    else:
                        checks.require(
                            not tracker.last_tail_recovery_diagnostic["enabled"],
                            "off control invoked new policy",
                        )
                checks.require(frozen == packets.signature(data), "packet mutation")
                np.savez_compressed(
                    output / f"f{native}_proposals.npz", frame_idx=np.asarray(native), **values
                )
                emit(f"f{native}_proposals.npz")
                rows.append(
                    dict(
                        frame_idx=native,
                        original_count=int(data["original_count"]),
                        total_count=len(data["proposal_parent"]),
                        diagnostic=diagnostic,
                    )
                )
                if len(rows) % 25 == 0 or len(rows) == 1:
                    print(json.dumps(dict(qualified=len(rows), native=native)), flush=True)
            originals, added, total = validate_totals(rows)
            result = dict(
                schema=STAGE,
                passed=True,
                frames=list(FRAMES),
                native_interval=[66524, 66920],
                frame_count=301,
                original_count=originals,
                added_count=added,
                originals=originals,
                added=added,
                total_count=total,
                all_proposals_byte_exact=True,
                off_original_prefix_byte_exact=True,
                input_arrays_unchanged=True,
                calibrated_rays_and_flow15_unchanged=True,
                core_context_equal_except_explicit_tail_policy=True,
                packet_seal_sha256=PACKET_SHA,
                tracker_sha256=TRACKER_SHA,
                helper_sha256=HELPER_SHA,
                device="cpu",
                no_fit=True,
                no_advance=True,
                fit_calls=0,
                advance_calls=0,
                initialize_calls=0,
                restore_calls=0,
                no_raw_decode=True,
                no_GPU=True,
                independent_ID_truth=False,
            )
            emit("frames.json", rows)
            emit("contexts.json", contexts)
            emit("summary.json", result)
            checks.require(
                code == packets.comparison.source_registry.local_code(), "loaded code changed"
            )
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            packets.final_guard(hashes, outputs)
            record.param("all_source_input_snapshot_output_SHA_unchanged", True)
        finally:
            torch.set_num_threads(threads[0])
            cv2.setNumThreads(threads[1])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
