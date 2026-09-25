"""Exact saved parent recurrence and original mouth generator for fixed lost33.

No tracker advance, flow reconstruction, fit, new ID, GPU or raw decode.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import diagnose_directional_mouth_first_chunk as prior
import measure_directional_mouth_gate_history as gates
import numpy as np
from continuous_tracker import STATE_FIELDS
from node_geometry import RayObservations, assignment_plans, supported_nodes
from node_recovery import NodeRecoveryConfig, node_recovery_proposals
from observations import CachedNodes

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance

checks, comparison = prior.checks, prior.comparison
START, STOP, HORIZON = 67663, 67696, 68060
GATE_ROOT = prior.REPO / "work/three_body/directional_mouth_gate_history_v1"
GATE_SHA = "8581fab9670925aa05474324708c93096d071027c6029aec5cbd0bcb81cbacd0"
STAGE = "directional_lost33_original_mouth_generation_v1"


def reconstruct(parent, state, metadata):
    """The frozen advance's 0.5 mixture, 2mm clip, 0.8 missing decay per parent."""
    comparison.verify_history_prefix(parent, state)
    previous = {key: parent["state_" + key].copy() for key in STATE_FIELDS}
    offset = state["history_offsets"]
    captured, retained = [], []
    for t in range(prior.START - prior.interim.FIRST, HORIZON - prior.interim.FIRST):
        frame = prior.interim.FIRST + t
        rows = np.arange(offset[t], offset[t + 1])
        checks.require(
            len(rows) == 6 and np.all(state["history_frame_idx"][rows] == frame),
            "six native records required",
        )
        if START <= frame < STOP:
            captured.append({key: value.copy() for key, value in previous.items()})
            retained.append(
                {
                    key: state["history_" + key][rows].copy()
                    for key in ("proposal_kind", "hypotheses", "parent")
                }
            )
        pi = state["history_parent"][rows]
        checks.require(np.all((pi >= 0) & (pi < 6)), "invalid history parent")
        pose, source, held = (
            state["history_" + key][rows] for key in ("pose_xyz", "sources", "observed")
        )
        body = held[..., 1] | (
            state["history_mask_supported"][rows] & (source[..., :2] >= 0).any((2, 3))
        )
        missing = np.where(body, 0, previous["missing"][pi] + 1)
        node_age = np.where(held, 0, previous["node_missing"][pi] + 1)
        chest_age = np.where(held[..., 1], previous["direct_pectoral_age"][pi] + 1, 0)
        cost = previous["cost"][pi] + state["history_costs"][rows]
        for key, value in (
            ("body_supported", body),
            ("missing", missing),
            ("node_missing", node_age),
            ("direct_pectoral_age", chest_age),
            ("path_costs", cost),
        ):
            checks.require(
                comparison.same(value, state["history_" + key][rows]),
                "parent recurrence differs: " + key,
            )
        review = state["history_identity_review"][rows]
        checks.require(not np.any(previous["review"][pi] & ~review), "sticky review changed")
        displacement = pose[:, :, 1] - previous["pose"][pi, :, 1]
        velocity = 0.5 * previous["velocity"][pi] + 0.5 * displacement
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity *= np.minimum(1.0, 2.0 / np.maximum(speed, 1e-12))
        velocity = np.where(body[..., None], velocity, 0.8 * previous["velocity"][pi])
        previous = dict(
            pose=pose,
            velocity=velocity,
            missing=missing,
            review=review,
            cost=cost,
            node_missing=node_age,
            direct_pectoral_age=chest_age,
        )
    for key in STATE_FIELDS:
        checks.require(
            comparison.same(previous[key], state["state_" + key]), "terminal state differs: " + key
        )
    checks.require(
        len(captured) == STOP - START and metadata["previous_frame"] == HORIZON - 1,
        "wrong reconstruction scope",
    )
    return {key: np.stack([item[key] for item in captured]) for key in STATE_FIELDS}, retained


def inputs(hashes):
    prior.bind(GATE_ROOT / "gate.provenance.json", GATE_SHA, hashes)
    gate = checks.verify(GATE_ROOT, "gate.provenance.json", gates.STAGE, hashes)
    first = prior.BRANCH / "checkpoints/through_068060"
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        prior.audit_input(gate, first / name, hashes)
    bound = checks.read(first / "manifest.json")["bindings"]
    for path, digest in bound["experimental_code_sha256"].items():
        source = prior.bind(path, digest, hashes)
        prior.bind(prior.BRANCH / "code_snapshot" / source.relative_to(prior.REPO), digest, hashes)
    for path, digest in bound["original_binding"]["code_sha256"].items():
        prior.bind(path, digest, hashes)
    state, meta, _ = comparison.load_bound_checkpoint(first, bound, hashes)
    comparison.validate_lifecycle(meta, HORIZON, 4)
    prior.bind(prior.AUDIT / "audit.provenance.json", prior.AUDIT_SHA, hashes)
    old_audit = checks.read(prior.AUDIT / "audit.provenance.json")
    parent_root = prior.REPO / "work/three_body/long_continuation_through66400_v1"
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        prior.audit_input(old_audit, parent_root / "checkpoint" / name, hashes)
    parent, parent_meta, _ = comparison.load_bound_checkpoint(
        parent_root / "checkpoint", bound["original_binding"], hashes
    )
    comparison.validate_lifecycle(parent_meta, prior.START, 3)
    migration = prior.bind(prior.BRANCH / "state_import.json", bound["state_import_sha256"], hashes)
    comparison.verify_import(checks.read(migration), parent, parent_meta)
    checks.require(
        checks.read(migration)["new_context"] == meta["context"], "imported context differs"
    )
    selected, retained = reconstruct(parent, state, meta)
    del parent, state
    with np.load(GATE_ROOT / "six_beam_gates.npz", allow_pickle=False) as saved:
        use = (saved["frame_idx"] >= START) & (saved["frame_idx"] < STOP)
        for key in ("node_missing", "direct_pectoral_age", "missing"):
            checks.require(
                comparison.same(selected[key], saved["previous_" + key][use]),
                "previous gate state differs",
            )
        old_sources = saved["old_same_horizon_mouth_sources"][use].copy()
        old_observed = saved["old_same_horizon_mouth_observed"][use].copy()
    seed = bound["original_binding"]["inputs"]["seed"]
    registration = prior.bind(seed["registration_path"], seed["registration_sha256"], hashes)
    initial = np.asarray(checks.read(registration)["pose_xyz"], np.float64)
    lengths = np.linalg.norm(np.diff(initial, axis=1), axis=-1)
    normalize = comparison.support_audit.COMMON.normalize
    checks.require(
        normalize(initial) == meta["context"]["initial_pose"]
        and normalize(lengths) == meta["context"]["initial_lengths"],
        "registered anatomical reference differs",
    )
    shared = seed["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "three SL/four calibration inputs required")
    for path, digest in shared.items():
        prior.bind(path, digest, hashes)
    manifest = next(Path(p) for p in shared if Path(p).name == "calibration_manifest.json")
    rig = load_rig(manifest.parent)
    checks.require(
        normalize([dataclasses.asdict(c) for c in rig.cameras]) == meta["context"]["cameras"],
        "rig context differs",
    )
    nodes = CachedNodes(seed["input_root"], prior.baseline.EXPERIMENT, np.arange(START, STOP))
    checks.require(
        set(map(str, nodes.paths.values())) == {p for p in shared if Path(p).suffix == ".npz"},
        "SL cache inputs differ",
    )
    return selected, retained, meta, nodes, rig, lengths, old_sources, old_observed


def run(output):
    output = Path(output).resolve()
    checks.require(
        output.is_relative_to(prior.REPO / "work") and not output.exists(), "new output required"
    )
    codes = comparison.source_registry.local_code()
    hashes = dict(codes)
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="generation") as record:
        try:
            for source, digest in codes.items():
                target = output / "code_snapshot" / Path(source).relative_to(prior.REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(source).read_bytes())
                checks.require(checks.sha(target) == digest, "source copy changed")
                record.output(str(target.relative_to(output)), target, sha256=digest)
            states, retained, metadata, nodes, rig, lengths, old_sources, old_observed = inputs(
                hashes
            )
            cfg = metadata["context"]["tracker_config"]
            persistent = gates.frozen.PersistentNodeConfig(
                **metadata["context"]["persistence_config"]
            )
            recovery = NodeRecoveryConfig(**metadata["context"]["recovery_config"])
            age, _ = gates.age_gates(
                states["node_missing"], states["direct_pectoral_age"], persistent
            )
            np.savez_compressed(
                output / "previous_all6_exact_states.npz",
                frame_idx=np.arange(START, STOP),
                **states,
            )
            all_rows = []
            for i, frame in enumerate(range(START, STOP)):
                points = {view: nodes.at(view, frame).copy() for view in prior.VIEWS}
                rays = RayObservations.from_points(rig, points)
                prediction = np.clip(
                    states["pose"][i, 0] + states["velocity"][i, 0, :, None],
                    recovery.lower_mm,
                    recovery.upper_mm,
                )
                main_plan = assignment_plans(
                    prediction, rays, cfg["ray_sigma_mm"], cfg["missing_cost"], keep=2
                )[0][1]
                source, observed, distance = supported_nodes(prediction, main_plan, rays, rig)
                checks.require(
                    age[i, 0, 0] and age[i, 0].any(), "fixed lost33 age gate unexpectedly closed"
                )
                before = {
                    key: value.copy()
                    for key, value in dict(pose=prediction, plan=main_plan, lengths=lengths).items()
                }
                generated = node_recovery_proposals(
                    prediction, main_plan, rays, rig, lengths, recovery
                )
                for key, value in dict(pose=prediction, plan=main_plan, lengths=lengths).items():
                    checks.require(comparison.same(before[key], value), "generator mutated input")
                emitted = [p for p in generated.proposals if age[i, 0, p.fish]]
                protected = []
                for vi, _view in enumerate(prior.VIEWS):
                    oldrow = int(old_sources[i, 0, vi])
                    protected.append(bool(oldrow >= 0 and oldrow in main_plan[1:, vi, 0]))
                facts = dict(
                    frame_idx=frame,
                    actual_parent=0,
                    age_eligible=age[i, 0].tolist(),
                    previous_body_missing=states["missing"][i, 0].tolist(),
                    current_predicted_strict_support=observed.tolist(),
                    original_helper_diagnostics=generated.diagnostics,
                    returned_before_temporal_filter=len(generated.proposals),
                    emitted_after_temporal_filter=len(emitted),
                    emitted_by_slot=[sum(p.fish == slot for p in emitted) for slot in range(3)],
                    old_same_horizon_S0_mouth_sources=old_sources[i, 0].tolist(),
                    old_same_horizon_S0_mouth_observed=bool(old_observed[i, 0]),
                    old_S0_source_protected_by_current_other_slot=protected,
                    retained_kinds=retained[i]["proposal_kind"].tolist(),
                    recorded_total_hypotheses=int(retained[i]["hypotheses"][0]),
                    original_budget=cfg["max_hypotheses"],
                    mouth_additional_budget=3 * recovery.proposals_per_fish,
                    mouth_dropped_by_post_append_total_cap=0,
                    total_original_pool_and_flow_not_recomputed=True,
                    node_pools_function_used_by_mouth_helper=False,
                    fit_performed=False,
                    identity_confirmed=False,
                )
                all_rows.append(facts)
                arrays = dict(
                    predicted_pose=prediction,
                    predicted_plan=main_plan,
                    predicted_sources=source,
                    predicted_observed=observed,
                    predicted_ray_distance_mm=distance,
                    initial_lengths=lengths,
                    returned_pose=np.stack([p.pose_xyz for p in generated.proposals])
                    if generated.proposals
                    else np.empty((0, 3, 3, 3)),
                    returned_plan=np.stack([p.plan for p in generated.proposals])
                    if generated.proposals
                    else np.empty((0, 3, 3, 3), np.int32),
                    returned_fish=np.array([p.fish for p in generated.proposals], np.int64),
                    emitted_after_age=np.array(
                        [age[i, 0, p.fish] for p in generated.proposals], bool
                    ),
                )
                for vi, view in enumerate(prior.VIEWS):
                    arrays[view + "_points"] = points[view]
                    arrays[view + "_ray_origins"] = rays.origins[vi]
                    arrays[view + "_ray_directions"] = rays.directions[vi]
                np.savez_compressed(output / f"f{frame}_generation.npz", **arrays)
            statuses = Counter(
                row["original_helper_diagnostics"]["fish"][0]["status"] for row in all_rows
            )
            rejection = Counter()
            for row in all_rows:
                rejection.update(
                    row["original_helper_diagnostics"]["fish"][0].get("rejection_counts", {})
                )
            summary = dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                native_frames=33,
                completed_checkpoint_horizon=HORIZON,
                parent_minute_completion_claim=False,
                original_age_gate_artifact_sha256=GATE_SHA,
                state_recurrence_interval=[prior.START, HORIZON],
                reconstructed_frames=HORIZON - prior.START,
                terminal_all7_state_arrays_byte_exact=True,
                terminal_all6_velocities_byte_exact=True,
                captured_previous_all6_age_arrays_match_prior_gate=True,
                generated_pose_only_for33_native_frames=True,
                all33_state0_S0_age_open=True,
                S0_generator_status_counts=dict(statuses),
                S0_rejection_counts=dict(rejection),
                S0_returned_by_original_helper=sum(
                    row["original_helper_diagnostics"]["fish"][0]["returned"] for row in all_rows
                ),
                S0_emitted_after_age_filter=sum(row["emitted_by_slot"][0] for row in all_rows),
                all_slots_emitted_after_age_filter=sum(
                    row["emitted_after_temporal_filter"] for row in all_rows
                ),
                frames_with_S0_generated=[
                    row["frame_idx"] for row in all_rows if row["emitted_by_slot"][0]
                ],
                actual_retained_mouth_recovery_rows=sum(
                    row["retained_kinds"].count("mouth_recovery") for row in all_rows
                ),
                pool_semantics=(
                    "Original node_recovery enumerates unclaimed finite mouth ray pairs/triples; "
                    "it does not call node_pools."
                ),
                cap_semantics=(
                    "Frozen base max18 is applied before mouth append; max6 mouths are preserved. "
                    "Directional max24 truncates only additional chest candidates."
                ),
                no_fit=True,
                no_GPU=True,
                no_raw_decode=True,
                no_ID_truth=True,
                no_algorithm_change=True,
            )
            (output / "frames.jsonl").write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in all_rows),
                encoding="utf-8",
            )
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
            )
            lines = [
                "# lost33フレームの元mouth生成の後段条件",
                "",
                "[67663,67696)の実state0に対して、元の予測・first assignment plan・"
                "node_recovery_proposalsをCPUで再評価した。",
                "66400の6状態から保存された1660フレームの全親再帰をたどり、"
                "68060の全7 state配列と全6 velocityのbyte一致を確認。速度をゼロで近似していない。",
                "",
                f"S0生成status: {dict(statuses)}。候補棄却内訳: {dict(rejection)}。",
                f"S0の元helper返却は{summary['S0_returned_by_original_helper']}候補、"
                f"age filter後も{summary['S0_emitted_after_age_filter']}候補。"
                "実際のretained mouth_recoveryは"
                f"{summary['actual_retained_mouth_recovery_rows']}行。",
                "",
                "|frame|S0 status|予測口支持|予測胸支持|free rows/view|組合せ数|admitted|returned|",
                "|---:|---|---|---|---|---:|---:|---:|",
            ]
            for row in all_rows:
                d = row["original_helper_diagnostics"]["fish"][0]
                lines.append(
                    f"|{row['frame_idx']}|{d['status']}|{d['mouth_direct']}|"
                    f"{d['pectoral_direct']}|{d.get('free_mouth_rows_per_view', '未実行')}|"
                    f"{d['combinations']}|{d['admitted']}|{d['returned']}|"
                )
            lines += [
                "",
                "mouth helperはnode_poolsではなく、"
                "未占有の有限mouth rowsの2/3視点組合せを列挙する。"
                "交差・ray RMS・box・解剖長・厳密支持の棄却理由は元コードのdiagnosticsを保存した。",
                "元のmax18はmouth追加前に適用され、mouth最大6候補は後から保持される。directionalの総capは追加chest候補だけを制限する。"
                "したがって、この実装では生成されたmouthを総capで落とす段階はない。",
                "",
                "生成した候補は保存しただけでfitしていない。もし生成されてretainedに無い場合でも、fit・score・dedup・beamのどこで落ちたかは未判定。"
                "原画・全flow候補・未保持fit履歴を新規に再現せず、生物学的IDや直接因果を断定しない。"
                "既存checkpoint/source/結果と実行中の1分追跡は変更していない。",
                "",
            ]
            (output / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
            for path in output.iterdir():
                if path.is_file():
                    record.output(path.name, path, sha256=checks.sha(path))
        finally:
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            for path, digest in hashes.items():
                checks.require(checks.sha(path) == digest, "generation input changed")
            for item in record.outputs.values():
                checks.require(
                    checks.sha(item["path"]) == item["sha256"], "generation output changed"
                )
        record.param("all_opened_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
