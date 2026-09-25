"""Posthoc all-eight support losses in the fixed301 local tail experiment.

Only successful saved packets/fits and registered calibration are used. All301
best observations determine the denominator; eight cases receive CPU geometric
checks. No proposal generation, image decoding, fit, GPU, ID or threshold change.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import fit_tail_packets as fit
import numpy as np
from continuous_tracker import _canonical
from node_geometry import RayObservations

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks, REPO = fit.checks, fit.REPO
WORK = REPO / "work/three_body"
FITS = WORK / "tail_local_cuda301_v1"
PACKETS = WORK / "tail_local_fit_packets_301_v1"
FIT_SHA = "efc3a4ef51c56cbea058fa67a9091c804a1778f004bc6d66c2b926a3c57b4a22"
PACKET_SHA = "6e0e3a8d7d006a9f3afc673af829f54de77567d1346a7cb613179114f1769cdc"
STAGE = "tail_local_other_node_loss_diagnosis_v2"


def finite_json(x):
    if isinstance(x, np.ndarray):
        return finite_json(x.tolist())
    if isinstance(x, np.generic):
        return finite_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    return x


def load_npz(path, hashes):
    checks.require(str(path) in hashes and checks.sha(path) == hashes[str(path)], "unbound NPZ")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def bind_shared_inputs(shared, packet_seal, fit_seal, hashes):
    checks.require(
        len(shared) == 7
        and sum(Path(p).suffix == ".npz" for p in shared) == 3
        and sum(Path(p).suffix == ".json" for p in shared) == 4,
        "sameSL3calibration4",
    )
    for path, digest in shared.items():
        fit.bind_input(packet_seal, path, hashes)
        if Path(path).suffix == ".json":
            fit.bind_input(fit_seal, path, hashes)
        checks.require(hashes[str(Path(path).resolve())] == digest, "same registered inputs")


def geometry(packet, arrays, rig):
    rays = RayObservations.from_points(rig, {v: packet[v + "_points"] for v in VIEWS})
    for vi, view in enumerate(VIEWS):
        checks.require(
            checks.equal_bytes(rays.origins[vi], packet[view + "_ray_origins"])
            and checks.equal_bytes(rays.directions[vi], packet[view + "_ray_directions"]),
            "registered packet rays differ",
        )
    for prefix, poses in (
        ("before", packet["proposal_pose_xyz"]),
        ("after", arrays["fitted_pose_xyz"]),
    ):
        actual = fit.base.support_snapshot(poses, packet["proposal_plan"], rays, rig)
        for key, value in actual.items():
            checks.require(
                checks.equal_bytes(value, arrays[prefix + "_" + key]),
                "saved CPU geometry differs: " + prefix + "_" + key,
            )
    for key, field in (
        ("strict_sources", "generated_strict_sources"),
        ("strict_observed", "generated_observed"),
        ("ray_distance_mm", "generated_ray_distance_mm"),
    ):
        checks.require(
            checks.equal_bytes(arrays["before_" + key], packet[field]), "packet prefit differs"
        )
    return rays


def case_details(native, slot, node, packet, a, rig):
    rays = geometry(packet, a, rig)
    old, new = int(a["original_selected_indices"][0]), int(a["selected_indices"][0])
    count = int(packet["original_count"])
    checks.require(
        old < count
        and a["after_strict_observed"][old, slot, node]
        and not a["after_strict_observed"][new, slot, node],
        "not a selected support loss",
    )
    cost = a["local_cost"] + packet["parent_cost"][packet["proposal_parent"]]
    checks.require(checks.equal_bytes(cost, a["path_cost"]), "parent plus local path differs")
    for label, n in (("", len(cost)), ("original_", count)):
        selected, order = fit.base.selected_indices(
            a["fitted_pose_xyz"][:n],
            cost[:n],
            packet["proposal_parent"][:n],
            packet["proposal_permutation"][:n],
            6,
        )
        checks.require(
            checks.equal_bytes(selected, a[label + "selected_indices"])
            and checks.equal_bytes(order, a[label + "stable_path_order"]),
            "saved selection differs",
        )
    cross = fit.base.support_snapshot(
        a["fitted_pose_xyz"][new : new + 1], packet["proposal_plan"][old : old + 1], rays, rig
    )
    zero = np.flatnonzero((packet["proposal_kind"] == "predict") & (packet["proposal_parent"] == 0))
    checks.require(len(zero) > 0, "state0 predict absent")
    predict = int(zero[0])
    views = []
    for vi, view in enumerate(VIEWS):
        row = dict(view=view)
        for label, index in (("control", old), ("combined", new)):
            row[label] = dict(plan_row=int(packet["proposal_plan"][index, slot, vi, node]))
            for stage in ("before", "after"):
                row[label][stage] = {
                    k: a[stage + "_" + k][index, slot, vi, node]
                    for k in (
                        "strict_sources",
                        "ray_distance_mm",
                        "native_pixel_error",
                        "ray_gate_pass",
                        "pixel_gate_pass",
                        "plan_source_collision",
                        "strict_source_collision",
                    )
                }
        row["new_pose_control_plan"] = {
            k: v[0, slot, vi, node] for k, v in cross.items() if v.ndim == 4
        }
        views.append(row)
    candidates = []
    for index in range(len(cost)):
        candidates.append(
            dict(
                index=index,
                original=index < count,
                kind=str(packet["proposal_kind"][index]),
                parent=int(packet["proposal_parent"][index]),
                recovery_fish=int(packet["proposal_recovery_fish"][index]),
                permutation=packet["proposal_permutation"][index],
                lost_node_observed_before=bool(a["before_strict_observed"][index, slot, node]),
                lost_node_observed_after=bool(a["after_strict_observed"][index, slot, node]),
                all_nodes_observed_before=a["before_strict_observed"][index],
                all_nodes_observed_after=a["after_strict_observed"][index],
                local_cost=float(a["local_cost"][index]),
                path_cost=float(cost[index]),
                loss_parts=a["final_loss_parts"][index],
            )
        )
    before = bool(a["before_strict_observed"][new, slot, node])
    row = dict(
        frame_idx=native,
        slot=slot,
        node=node,
        node_name=("mouth", "pectoral", "tail")[node],
        control_index=old,
        combined_index=new,
        original_count=count,
        classification="lost_during_fit" if before else "unsupported_in_combined_prefit",
        control=candidates[old],
        combined=candidates[new],
        candidates=candidates,
        new_pose_with_control_plan_supported=bool(cross["strict_observed"][0, slot, node]),
        new_pose_control_plan_is_counterfactual=True,
        prefit_pose_displacement_by_slot_node=np.linalg.norm(
            packet["proposal_pose_xyz"][new] - packet["proposal_pose_xyz"][old], axis=-1
        ),
        fitted_pose_displacement_by_slot_node=np.linalg.norm(
            a["fitted_pose_xyz"][new] - a["fitted_pose_xyz"][old], axis=-1
        ),
        combined_fit_displacement_by_slot_node=np.linalg.norm(
            a["fitted_pose_xyz"][new] - packet["proposal_pose_xyz"][new], axis=-1
        ),
        combined_prefit_same_state0_predict_by_slot_node=np.all(
            packet["proposal_pose_xyz"][new] == packet["proposal_pose_xyz"][predict], axis=-1
        ),
        combined_plan_same_state0_predict_by_slot_node=np.all(
            packet["proposal_plan"][new] == packet["proposal_plan"][predict], axis=1
        ),
        state0_predict_index=predict,
        views=views,
        all_slot_node_gains=~a["after_strict_observed"][old] & a["after_strict_observed"][new],
        all_slot_node_losses=a["after_strict_observed"][old] & ~a["after_strict_observed"][new],
        local_cost_change=float(a["local_cost"][new] - a["local_cost"][old]),
        path_cost_change=float(cost[new] - cost[old]),
        proposed_cause_is_not_proven=True,
        biological_ID_truth=False,
    )
    return row, cross


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(WORK) and not output.exists(), "new work output required")
    code = fit.base.local_code()
    hashes, outputs = dict(code), {}
    for path, digest in (
        (
            WORK / "diagnose_tail_local_other_losses.py",
            "dff0ff157ab3b87df3306706e9672d77cce610fcc459613f9ff3fb1c047cd049",
        ),
        (
            WORK / "tail_local_other_node_loss_diagnosis_v1/diagnosis.provenance.json",
            "dca84caa604f1ef1b244c1ea3f050f11295a2fbc33c74ba450e90018ef0dbb45",
        ),
    ):
        checks.require(checks.sha(path) == digest, "preserved failed revision differs")
        hashes[str(path)] = digest
        if path.suffix == ".py":
            code[str(path)] = digest
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="diagnosis") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(
                    json.dumps(finite_json(value), indent=2, allow_nan=False), encoding="utf-8"
                )
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        emit(
            "plan.json",
            dict(
                selection="all301 control observed AND NOT combined observed; expected8",
                frame_set=list(fit.FRAMES),
                fit_seal=FIT_SHA,
                packet_seal=PACKET_SHA,
                read_only_posthoc=True,
                GT_read=False,
                ID_assignment=False,
                raw_decode=False,
                fit=False,
                GPU=False,
                thresholds=[2.0, 12.0],
            ),
        )
        for root, name, stage, expected in (
            (FITS, "fit.provenance.json", fit.STAGE, FIT_SHA),
            (PACKETS, "preparation.provenance.json", fit.PACKET_STAGE, PACKET_SHA),
        ):
            checks.require(checks.sha(root / name) == expected, "fixed seal differs")
            seal = checks.verify(root, name, stage, hashes)
            checks.require(hashes[str(root / name)] == expected, "seal changed while reading")
            for path, digest in code.items():
                if path in seal["inputs"]:
                    checks.require(
                        seal["inputs"][path]["sha256"] == digest, "historical code differs"
                    )
            if root == FITS:
                fit_seal = seal
            else:
                packet_seal = seal
        fit.bind_input(fit_seal, PACKETS / "preparation.provenance.json", hashes)
        meta = checks.read(fit.member(PACKETS, "metadata.json", hashes))
        fit.bind_input(fit_seal, PACKETS / "metadata.json", hashes)
        shared = meta["checkpoint_binding"]["original_binding"]["inputs"]["seed"][
            "SL_and_calibration_sha256"
        ]
        bind_shared_inputs(shared, packet_seal, fit_seal, hashes)
        rig = load_rig(meta["calibration_directory"])
        checks.require(
            meta["original_context"]["cameras"]
            == json.loads(_canonical([dataclasses.asdict(c) for c in rig.cameras])),
            "camera context differs",
        )
        frames = checks.read(fit.member(FITS, "frames.json", hashes))
        checks.require([r["frame_idx"] for r in frames] == list(fit.FRAMES), "all301 native order")
        cases, original, combined = [], [], []
        for row in frames:
            native = row["frame_idx"]
            path = fit.member(FITS, f"f{native}_fit.npz", hashes)
            with np.load(path, allow_pickle=False) as z:
                a, b = z["control_observed"][0], z["combined_observed"][0]
                old, new = int(z["original_selected_indices"][0]), int(z["selected_indices"][0])
                observed = z["after_strict_observed"]
                checks.require(
                    np.array_equal(a, observed[old])
                    and np.array_equal(b, observed[new])
                    and np.array_equal(a, row["control_best_observed"])
                    and np.array_equal(b, row["combined_best_observed"]),
                    "best observations disagree",
                )
                original.append(a)
                combined.append(b)
                cases.extend((native, int(s), int(n)) for s, n in np.argwhere(a & ~b))
        checks.require(len(cases) == 8, "fixed loss denominator differs")
        emit("fixed_cases.json", cases)
        original, combined = np.stack(original), np.stack(combined)
        declared = checks.read(fit.member(FITS, "summary.json", hashes))
        checks.require(
            declared["status"] == "ok"
            and declared["control_all19_records_reproduced"]
            and declared["control_best_direct_nodes"] == original.sum((0, 1)).tolist()
            and declared["combined_best_direct_nodes"] == combined.sum((0, 1)).tolist(),
            "all301 recomputed counts differ from successful producer",
        )
        result = []
        for native, slot, node in cases:
            ppath = fit.member(PACKETS, f"f{native}_packet.npz", hashes)
            fit.bind_input(fit_seal, ppath, hashes)
            packet = load_npz(ppath, hashes)
            arrays = load_npz(fit.member(FITS, f"f{native}_fit.npz", hashes), hashes)
            row, cross = case_details(native, slot, node, packet, arrays, rig)
            result.append(row)
            payload = {
                "packet_" + k: v
                for k, v in packet.items()
                if k.startswith(("proposal_", "parent_", "generated_"))
            }
            payload.update({"fit_" + k: v for k, v in arrays.items()})
            payload.update({"counterfactual_newpose_controlplan_" + k: v for k, v in cross.items()})
            np.savez_compressed(output / f"f{native}_evidence.npz", **payload)
            emit(f"f{native}_evidence.npz")
        emit("cases.json", result)
        summary = dict(
            status="ok",
            schema=STAGE,
            scanned_frames=len(frames),
            lost_node_cases=len(cases),
            control_nodes=original.sum((0, 1)),
            combined_nodes=combined.sum((0, 1)),
            gains=(~original & combined).sum(0),
            losses=(original & ~combined).sum(0),
            classification=dict(Counter(r["classification"] for r in result)),
            control_kind=dict(Counter(r["control"]["kind"] for r in result)),
            combined_kind=dict(Counter(r["combined"]["kind"] for r in result)),
            newpose_controlplan_supported=sum(
                r["new_pose_with_control_plan_supported"] for r in result
            ),
            raw_decode=0,
            fit_calls=0,
            GPU=False,
            independent_ID_truth=False,
            causal_conclusion=False,
            preserved_failed_v1="extra direct SL membership requirement; no numerical measurement",
            source_binding="SL3 through sealed packet; calibration4 direct in packet and fit",
        )
        emit("summary.json", summary)
        for path, digest in code.items():
            dest = output / "code_snapshot" / Path(path).relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(dest) == digest, "source snapshot changed")
            emit(str(dest.relative_to(output)))
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "input/source changed")
            record.input(path, path, sha256=digest)
        for path, digest in outputs.items():
            checks.require(checks.sha(path) == digest, "output changed")
    print(json.dumps(finite_json(summary)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
