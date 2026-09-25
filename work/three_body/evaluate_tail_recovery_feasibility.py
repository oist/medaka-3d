"""Fixed lost215 tail feasibility at saved fitted poses and filtered sources.

This is not reconstruction of a pre-fit proposal pool. No age or tracker-wide
budget is applied; old poses enter only posthoc output annotations. No fit,
tracking, raw pixels, candidate selection by images, angle gate or GPU operation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

import diagnose_bounded_tail_first_chunk as evidence
import numpy as np
import tail_recovery as helper
from node_geometry import RayObservations, supported_nodes
from node_recovery import NodeRecoveryConfig, _injective

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks, comparison, REPO = evidence.checks, evidence.comparison, evidence.REPO
ROOT = REPO / "work/three_body/bounded_mouth_first_chunk_tail_diagnosis_v1"
PARENT_SHA = "6a331bdde8f67fc46fd4e4284ff22196ea2000bd2965db93b2bcdaaadf1c0d32"
EVIDENCE_SOURCE_SHA = "6b9e43056debe0025d7bbf4287342dd5d79cdf4d2631434b93a247eff367b450"
HELPER_SHA = "636509cf37bb485a5e8efd34bbd55320d928df35fce59e4617addfee4b75a201"
STAGE = "tail_recovery_lost215_feasibility_v1"
ARMS = ("free_only", "free_and_bounded")


def signature(scene, plan, lengths, rays):
    arrays = dict(scene=scene, plan=plan, initial_lengths=lengths)
    for vi, view in enumerate(VIEWS):
        arrays.update(
            {
                view + "_points": rays.points[view],
                view + "_origins": rays.origins[vi],
                view + "_directions": rays.directions[vi],
            }
        )
    return evidence.interim.array_signature(arrays)


def pose_shape(pose):
    a, b = pose[:, 0] - pose[:, 1], pose[:, 2] - pose[:, 1]
    lengths = np.stack((np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)), -1)
    denominator = lengths.prod(-1)
    cosine = np.full(3, np.nan)
    np.divide((a * b).sum(-1), denominator, out=cosine, where=denominator > 1e-24)
    return lengths, np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def validate_candidate(candidate, scene, plan, rays, rig, config):
    fish = candidate.fish
    checks.require(
        type(fish) is int and 0 <= fish < 3 and candidate.node == 2, "native tail proposal required"
    )
    others = [i for i in range(3) if i != fish]
    checks.require(
        checks.equal_bytes(candidate.pose_xyz[:, :2], scene[:, :2])
        and checks.equal_bytes(candidate.plan[:, :, :2], plan[:, :, :2])
        and checks.equal_bytes(candidate.pose_xyz[others], scene[others])
        and checks.equal_bytes(candidate.plan[others], plan[others]),
        "candidate changed mouth/chest or another fish",
    )
    checks.require(_injective(candidate.plan), "candidate source not exclusive")
    checks.require(
        np.isfinite(candidate.pose_xyz).all()
        and np.all(candidate.pose_xyz >= config.lower_mm)
        and np.all(candidate.pose_xyz <= config.upper_mm),
        "candidate violates box",
    )
    sources, observed, distance = supported_nodes(candidate.pose_xyz, candidate.plan, rays, rig)
    checks.require(
        observed[fish, 1:].all()
        and int((sources[fish, :, 2] >= 0).sum()) == candidate.supported_views,
        "candidate strict chest/tail support differs",
    )
    checks.require(
        candidate.seed_method in ("free_intersection", "bounded_intersection")
        and np.isfinite(candidate.ray_rms_mm)
        and candidate.ray_rms_mm <= config.ray_rms_mm
        and config.minimum_length_ratio <= candidate.length_ratio <= config.maximum_length_ratio,
        "candidate raw-ray or anatomy diagnostics differ",
    )
    pixel = np.full((3, 3, 3), np.nan)
    for vi, view in enumerate(VIEWS):
        projected = rig.camera(view).project(candidate.pose_xyz)
        for f, node in np.argwhere(candidate.plan[:, vi] >= 0):
            row = int(candidate.plan[f, vi, node])
            pixel[f, vi, node] = np.linalg.norm(projected[f, node] - rays.points[view][row, node])
    lengths, angles = pose_shape(candidate.pose_xyz)
    checks.require(
        np.isclose(lengths[fish, 1], candidate.tail_pectoral_mm, rtol=0, atol=1e-12),
        "tail length field differs",
    )
    return dict(
        sources=sources,
        observed=observed,
        ray_distance_mm=distance,
        native_pixel_error=pixel,
        segment_lengths_mm=lengths,
        angle_deg=angles,
    )


def run_current_frame(scene, plan, expected_observed, rays, rig, lengths, config):
    """Only current inputs; no old geometry or review/age enters either generator."""
    before = signature(scene, plan, lengths, rays)
    sources, observed, _ = supported_nodes(scene, plan, rays, rig)
    checks.require(
        checks.equal_bytes(sources, plan) and checks.equal_bytes(observed, expected_observed),
        "saved filtered base source/observation differs",
    )
    pools, diagnostics, timings = {}, {}, {}
    for arm, allow in zip(ARMS, (False, True), strict=True):
        started = time.perf_counter()
        result = helper.tail_recovery_proposals(
            scene, plan, rays, rig, lengths, config, allow_bounded=allow
        )
        timings[arm] = time.perf_counter() - started
        counts = Counter(p.fish for p in result.proposals)
        checks.require(all(n <= config.proposals_per_fish for n in counts.values()), "per-fish cap")
        checked = [validate_candidate(p, scene, plan, rays, rig, config) for p in result.proposals]
        keys = [
            (
                p.fish,
                -p.supported_views,
                p.ray_rms_mm,
                p.displacement_mm,
                tuple(p.plan[p.fish, :, 2]),
            )
            for p in result.proposals
        ]
        checks.require(keys == sorted(keys), "native merged order differs")
        pools[arm] = (result.proposals, checked)
        diagnostics[arm] = result.diagnostics
        checks.require(before == signature(scene, plan, lengths, rays), "generator mutated input")
    return pools, diagnostics, timings


def pool_arrays(proposals, checked):
    arrays = dict(
        pose_xyz=np.array([p.pose_xyz for p in proposals], np.float64).reshape(-1, 3, 3, 3),
        plan=np.array([p.plan for p in proposals], np.int32).reshape(-1, 3, 3, 3),
        fish=np.array([p.fish for p in proposals], np.int64),
        node=np.array([p.node for p in proposals], np.int64),
        seed_method=np.array([p.seed_method for p in proposals], dtype="U24"),
    )
    for key in ("displacement_mm", "tail_pectoral_mm", "length_ratio", "ray_rms_mm"):
        arrays[key] = np.array([getattr(p, key) for p in proposals], np.float64)
    arrays["supported_views"] = np.array([p.supported_views for p in proposals], np.int64)
    for key, shape, dtype in (
        ("sources", (3, 3, 3), np.int32),
        ("observed", (3, 3), bool),
        ("ray_distance_mm", (3, 3, 3), np.float64),
        ("native_pixel_error", (3, 3, 3), np.float64),
        ("segment_lengths_mm", (3, 2), np.float64),
        ("angle_deg", (3,), np.float64),
    ):
        arrays[key] = np.array([c[key] for c in checked], dtype=dtype).reshape(-1, *shape)
    return arrays


def posthoc_rows(native, proposals, checked, lost_slots, old_pose, old_sources):
    """Annotations only, called after both generator arms have finished."""
    rows = []
    for index, (p, detail) in enumerate(zip(proposals, checked, strict=True)):
        fish = p.fish
        rows.append(
            dict(
                frame_idx=int(native),
                proposal_index=index,
                fish=fish,
                target_lost_fish=fish in lost_slots,
                seed_method=p.seed_method,
                supported_views=p.supported_views,
                ray_rms_mm=p.ray_rms_mm,
                displacement_from_current_mm=p.displacement_mm,
                tail_pectoral_mm=p.tail_pectoral_mm,
                length_ratio=p.length_ratio,
                angle_deg=float(detail["angle_deg"][fish]),
                old_tail_distance_mm=float(np.linalg.norm(p.pose_xyz[fish, 2] - old_pose[fish, 2])),
                old_strict_tail_source_rows_equal=bool(
                    np.array_equal(detail["sources"][fish, :, 2], old_sources[fish, :, 2])
                ),
                plan_tail_rows=p.plan[fish, :, 2].tolist(),
                strict_tail_rows=detail["sources"][fish, :, 2].tolist(),
                old_pose_used_by_generator=False,
                ID_confirmation=False,
            )
        )
    return rows


def summarize_cases(cases, proposal_rows):
    checks.require(
        len(cases) == 215 and len({(x["frame_idx"], x["slot"]) for x in cases}) == 215,
        "fixed215 denominator required",
    )
    result = {}
    for arm in ARMS:
        rows = [r for r in proposal_rows if r["arm"] == arm]
        target = [r for r in rows if r["target_lost_fish"]]
        other = [r for r in rows if not r["target_lost_fish"]]
        result[arm] = dict(
            cases_with_candidates=sum(r[arm + "_count"] > 0 for r in cases),
            cases_without_candidates=sum(r[arm + "_count"] == 0 for r in cases),
            by_slot={
                str(s): dict(
                    cases=sum(r["slot"] == s for r in cases),
                    with_candidates=sum(r["slot"] == s and r[arm + "_count"] > 0 for r in cases),
                )
                for s in range(3)
            },
            target_lost_proposals=len(target),
            other_fish_proposals=len(other),
            target_seed_methods=dict(Counter(r["seed_method"] for r in target)),
            other_seed_methods=dict(Counter(r["seed_method"] for r in other)),
        )
    result["incremental_case_availability"] = dict(
        only_with_bounded=sum(
            r["free_only_count"] == 0 and r["free_and_bounded_count"] > 0 for r in cases
        ),
        both=sum(r["free_only_count"] > 0 and r["free_and_bounded_count"] > 0 for r in cases),
        neither=sum(r["free_only_count"] == r["free_and_bounded_count"] == 0 for r in cases),
    )
    return result


def load_inputs(hashes):
    evidence.interim.bind(ROOT / "diagnosis.provenance.json", PARENT_SHA, hashes)
    sealed = checks.verify(ROOT, "diagnosis.provenance.json", evidence.STAGE, hashes)
    evidence.interim.bind(evidence.__file__, EVIDENCE_SOURCE_SHA, hashes)
    evidence.interim.bind(helper.__file__, HELPER_SHA, hashes)
    evidence.interim.input_member(sealed, evidence.AUDIT / "audit.provenance.json", hashes)
    checks.require(
        hashes[str(evidence.AUDIT / "audit.provenance.json")] == evidence.AUDIT_SHA,
        "interim audit origin changed",
    )
    audit = checks.verify(evidence.AUDIT, "audit.provenance.json", evidence.interim.STAGE, hashes)
    prior = checks.read(evidence.AUDIT / "summary.json")
    checks.require(
        prior["status"] == "ok"
        and prior["native_interval"] == [evidence.START, evidence.STOP]
        and prior["complete_minute"] is False
        and prior["all21_arrays_backtraced"] is True
        and prior["expected_manifest_sha256"] == evidence.MANIFEST_SHA,
        "fixed successful interim audit required",
    )
    traces, contexts = {}, []
    for name, root in (("directional", evidence.interim.OLD), ("bounded", evidence.BRANCH)):
        path = evidence.interim.input_member(
            sealed, evidence.AUDIT / f"{name}_same68060_candidate.npz", hashes
        )
        with np.load(path, allow_pickle=False) as z:
            traces[name] = {k: z[k].copy() for k in z.files}
        checks.require(
            len(traces[name]) == 21
            and np.array_equal(traces[name]["frame_idx"], np.arange(evidence.START, evidence.STOP)),
            "fixed all21 native trace",
        )
        path = evidence.interim.input_member(
            sealed, root / "checkpoints/through_068060/metadata.json", hashes
        )
        contexts.append(checks.read(path)["context"])
    for path, digest in list(hashes.items()):
        if path in sealed["inputs"]:
            checks.require(
                digest == sealed["inputs"][path]["sha256"], "historical code/input differs"
            )
    old, new = traces["directional"], traces["bounded"]
    lost = old["observed"][..., 2] & ~new["observed"][..., 2]
    with np.load(ROOT / "tail_classification.npz", allow_pickle=False) as z:
        checks.require(
            checks.equal_bytes(lost, z["lost"])
            and checks.equal_bytes(new["frame_idx"], z["frame_idx"])
            and int(lost.sum()) == 215,
            "fixed lost215 differs",
        )
    context = contexts[1]
    seed_info = prior["shared_input_binding"]["inputs"]["seed"]
    seed_path = evidence.interim.input_member(audit, seed_info["registration_path"], hashes)
    checks.require(hashes[str(seed_path)] == seed_info["registration_sha256"], "seed changed")
    seed_pose = np.asarray(checks.read(seed_path)["pose_xyz"], np.float64)
    lengths = np.linalg.norm(np.diff(seed_pose, axis=1), axis=-1)
    checks.require(
        evidence.interim.array_signature({"x": lengths})["x"] == context["initial_lengths"],
        "initial tail/head lengths changed",
    )
    shared = seed_info["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "same SL3/calibration4 required")
    for path, digest in shared.items():
        actual = evidence.interim.input_member(sealed, path, hashes)
        checks.require(hashes[str(actual)] == digest, "SL/calibration origin differs")
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    paths = {
        v: next(
            Path(p)
            for p in shared
            if Path(p).name == evidence.baseline.EXPERIMENT + "__" + v + ".npz"
        )
        for v in VIEWS
    }
    rig = load_rig(calibration)
    normalized = comparison.support_audit.COMMON.normalize(
        [dataclasses.asdict(c) for c in rig.cameras]
    )
    for ctx in contexts:
        checks.require(
            ctx["cameras"] == normalized
            and ctx["views"] == list(rig.views)
            and ctx["world_units"] == rig.world_units == "mm",
            "actual camera context differs",
        )
    nodes = evidence.baseline.RegisteredNodes(paths, np.arange(evidence.START, evidence.STOP))
    return traces, lost, nodes, rig, lengths, NodeRecoveryConfig(**context["recovery_config"])


def run(output):
    output = Path(output).resolve()
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(), "new output required"
    )
    code = comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="feasibility") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(
                    json.dumps(evidence.finite_json(value), indent=2, allow_nan=False),
                    encoding="utf-8",
                )
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        for path, digest in code.items():
            dest = output / "code_snapshot" / Path(path).relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(dest) == digest, "snapshot changed")
            emit(str(dest.relative_to(output)))
        emit(
            "plan.json",
            dict(
                native_interval=[evidence.START, evidence.STOP],
                same_horizon=evidence.STOP,
                selection="all215 old-tail-direct/new-tail-unsupported fish frames",
                expected_cases=215,
                helper_sha256=HELPER_SHA,
                parent_diagnosis_sha256=PARENT_SHA,
                arms=ARMS,
                supplied_scene="current fitted bounded pose",
                supplied_plan="saved post-filter sources; not original pre-fit plan",
                age_gate_applied=False,
                total_tracker_budget_applied=False,
                old_pose_only_posthoc=True,
                added_angle_gate=False,
                raw=False,
                fit=False,
                GPU=False,
            ),
        )
        traces, lost, nodes, rig, lengths, config = load_inputs(hashes)
        frames = traces["bounded"]["frame_idx"]
        cases, proposal_rows, timing_rows = [], [], []
        expected_cases = [
            {"frame_idx": int(frames[t]), "slot": int(f)} for t, f in np.argwhere(lost)
        ]
        emit("fixed_cases.json", expected_cases)
        for number, t in enumerate(np.flatnonzero(lost.any(1))):
            native = int(frames[t])
            scene, plan = (traces["bounded"][key][t].copy() for key in ("pose_xyz", "sources"))
            points = {v: nodes.at(v, native).copy() for v in VIEWS}
            rays = RayObservations.from_points(rig, points)
            expected = traces["bounded"]["observed"][t]
            pools, diagnostics, times = run_current_frame(
                scene, plan, expected, rays, rig, lengths, config
            )
            slots = np.flatnonzero(lost[t]).tolist()
            payload = dict(
                frame_idx=np.asarray(native, np.int64),
                current_fitted_pose_xyz=scene,
                current_filtered_sources=plan,
                current_observed=expected.copy(),
                current_identity_review=traces["bounded"]["identity_review"][t].copy(),
                target_lost_slots=np.asarray(slots, np.int64),
                initial_lengths_mm=lengths.copy(),
            )
            for vi, view in enumerate(VIEWS):
                payload.update(
                    {
                        view + "_points": rays.points[view],
                        view + "_origins": rays.origins[vi],
                        view + "_directions": rays.directions[vi],
                    }
                )
            per_arm_rows = {}
            for arm in ARMS:
                proposals, checked = pools[arm]
                payload.update(
                    {
                        arm + "_" + key: value
                        for key, value in pool_arrays(proposals, checked).items()
                    }
                )
                rows = posthoc_rows(
                    native,
                    proposals,
                    checked,
                    slots,
                    traces["directional"]["pose_xyz"][t],
                    traces["directional"]["sources"][t],
                )
                for row in rows:
                    row["arm"] = arm
                proposal_rows.extend(rows)
                per_arm_rows[arm] = rows
            # Old data is serialized after both current-only arms have completed.
            payload["posthoc_old_tail_xyz"] = traces["directional"]["pose_xyz"][t, :, 2].copy()
            payload["posthoc_old_tail_sources"] = traces["directional"]["sources"][
                t, :, :, 2
            ].copy()
            for slot in slots:
                row = dict(
                    frame_idx=native,
                    slot=slot,
                    current_chest_direct=bool(expected[slot, 1]),
                    current_identity_review=bool(traces["bounded"]["identity_review"][t, slot]),
                )
                for arm in ARMS:
                    row[arm + "_count"] = sum(p.fish == slot for p in pools[arm][0])
                    row[arm + "_seed_methods"] = [
                        p.seed_method for p in pools[arm][0] if p.fish == slot
                    ]
                    row[arm + "_free_gate_status"] = diagnostics[arm]["original_delegate"]["fish"][
                        slot
                    ]["status"]
                    bounded_diag = diagnostics[arm]["bounded_delegate"]
                    row[arm + "_bounded_gate_status"] = (
                        None if bounded_diag is None else bounded_diag["fish"][slot]["status"]
                    )
                cases.append(row)
            np.savez_compressed(output / f"f{native}_candidates.npz", **payload)
            emit(f"f{native}_candidates.npz")
            emit(
                f"f{native}_diagnostics.json",
                dict(diagnostics=diagnostics, proposals=per_arm_rows, timings=times),
            )
            timing_rows.append(dict(frame_idx=native, **times))
            if (number + 1) % 50 == 0:
                print(
                    json.dumps(dict(processed_native_frames=number + 1, cases=len(cases))),
                    flush=True,
                )
        checks.require(
            [{k: r[k] for k in ("frame_idx", "slot")} for r in cases] == expected_cases,
            "fixed denominator/order changed",
        )
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[evidence.START, evidence.STOP],
            same_horizon=evidence.STOP,
            case_count=215,
            unique_native_frames=len(timing_rows),
            **summarize_cases(cases, proposal_rows),
            source_sha256=code[str(Path(__file__).resolve())],
            helper_sha256=HELPER_SHA,
            helper_call_seconds={arm: sum(row[arm] for row in timing_rows) for arm in ARMS},
            all_current_inputs_unchanged=True,
            old_pose_used_by_generator=False,
            native_tail_chest_strict_support_verified=True,
            source_exclusivity_verified=True,
            mouth_chest_other_fish_pose_and_plan_preserved=True,
            new_angle_filter=False,
            age_gate_applied=False,
            total_tracker_budget_applied=False,
            actual_prefit_pool_reconstructed=False,
            actual_tracker_candidate_delivery_proved=False,
            ID_confirmation=False,
            full_minute_complete=False,
            raw_decode=0,
            checkpoint_image_payload_deserialized=False,
            fits=0,
            GPU=False,
        )
        emit("cases.json", cases)
        emit("proposals.json", proposal_rows)
        emit("helper_timings.json", timing_rows)
        evidence.write_csv(output / "cases.csv", cases)
        emit("cases.csv")
        if proposal_rows:
            evidence.write_csv(output / "proposals.csv", proposal_rows)
            emit("proposals.csv")
        emit("summary.json", summary)
        text = (
            "# 尾支持喪失215件・保存姿勢からの候補可解性\n\n"
            f"固定215件中、free-onlyで候補あり {summary['free_only']['cases_with_candidates']}、"
            f"free+boundedで {summary['free_and_bounded']['cases_with_candidates']}。"
            "bounded追加で初めて候補があるケース "
            f"{summary['incremental_case_availability']['only_with_bounded']}。\n\n"
            "同68060終端のcurrent fitted poseと支持filter後sourceを渡した。初期長・同SL3/校正4・"
            "元box/RMS/解剖/2mm12px支持/排他/各魚2候補を維持し、角度閾値は追加していない。"
            "元口/胸・他魚は変更しない。対象lost魚と同時に返った他魚候補を別集計した。\n\n"
            "実prefit plan/全6親/年齢条件/全tracker候補枠を再現しておらず、"
            "実trackerに届く・選ばれる"
            "候補数ではない。旧tail座標は生成・順位へ渡さず、生成後の距離/行比較にだけ使用した。"
            "旧source・slot・候補自体は生物学的正解やID確定ではない。原画/fit/描画/GPUは使用しない。\n"
        )
        (output / "RESULT.md").write_text(text, encoding="utf-8")
        emit("RESULT.md")
        checks.require(code == comparison.source_registry.local_code(), "loaded code changed")
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
            checks.require(checks.sha(path) == digest, "input/source changed")
        for path, digest in outputs.items():
            checks.require(checks.sha(path) == digest, "output/snapshot changed")
        record.param("all_used_inputs_sources_outputs_snapshots_unchanged", True)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
