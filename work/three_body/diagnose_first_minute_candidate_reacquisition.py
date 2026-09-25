"""Read selected saved diagnostic members, without fit, tracking, or original SL."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
OLD = W / "tail_one_minute_66400-76360_v1"
NEW = W / "nodewise_body_row_five_minute_continuation_v1"
OUT = W / "nodewise_first_minute_candidate_reacquisition_v1"
FRAMES = (75053, 75054, 75055)
FIELDS = (
    "candidate_index",
    "candidate_pose_xyz",
    "candidate_parent",
    "candidate_plan",
    "candidate_permutation",
    "base_local_cost",
    "original_five_loss_parts",
    "sixth_instance_term",
    "adjusted_local_cost",
    "parent_path_cost",
    "adjusted_path_cost",
)
BODY = (
    "parent_prediction",
    "new_proposal_indices",
    "new_proposal_plans",
    "new_proposal_poses",
    "filled_nodes_by_proposal",
)
OLD_FIELDS = (
    "proposal_pose",
    "proposal_predicted",
    "proposal_plan",
    "proposal_parent",
    "proposal_permutation",
    "parent_cost",
    "parent_missing",
    "parent_node_missing",
    "parent_direct_pectoral_age",
    "fitted_pose",
    "local_cost",
    "loss_parts",
    "path_cost",
    "generated_geometric_observed",
    "generated_strict_observed",
    "fitted_geometric_observed",
    "fitted_strict_observed",
    "fitted_geometric_sources",
    "selected_indices",
    "stable_path_order",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def signature(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), sha256=sha(a.tobytes()))


def normalize(value):
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def retained_indices(pose, path, parent, permutation, count=6):
    order = np.argsort(path, kind="stable")
    selected = []
    for c in order:
        if any(
            parent[c] == parent[j]
            and np.array_equal(permutation[c], permutation[j])
            and np.max(np.linalg.norm(pose[c] - pose[j], axis=-1)) < 0.5
            for j in selected
        ):
            continue
        selected.append(int(c))
        if len(selected) == count:
            break
    return np.array(selected, dtype=np.int64), order


def run():
    assert not OUT.exists(), "fresh output required"
    inputs, stats = {}, {}

    def read(path, digest=None):
        path = Path(path).resolve()
        assert path.is_relative_to(W)
        st = path.stat()
        data = path.read_bytes()
        assert (st.st_size, st.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
        assert digest is None or sha(data) == digest, path
        inputs[str(path)] = dict(sha256=sha(data), size_bytes=len(data))
        stats[path] = (st.st_size, st.st_mtime_ns)
        return data

    def doc(path, digest=None):
        return json.loads(read(path, digest))

    seal = doc(
        OLD / "tracking.provenance.json",
        "50ccc8d2a4bbcc7e09aa2b512b70374fb9b37a4d4ece0d0939322c66bfa59416",
    )
    assert seal["status"] == "ok"

    def old_bytes(relative):
        path = (OLD / relative).resolve()
        refs = [r for r in seal["outputs"].values() if Path(r["path"]).resolve() == path]
        assert len(refs) == 1
        data = read(path, refs[0]["sha256"])
        assert len(data) == refs[0]["size"]
        return data

    old_context = json.loads(old_bytes("checkpoints/through_076360/metadata.json"))["context"]
    new_plan = doc(
        NEW / "plan.json", "e05e174a291d6d79bf2cf3f74b848e7b125b3304f8ad1f0d168165dac3a8ba31"
    )
    cfg = old_context["tracker_config"]
    assert cfg == new_plan["tracker_context"]["tracker_config"] and cfg["beam_size"] == 6
    assert new_plan["sixth_term_effective_from_native"] == 76360
    batch_path = NEW / "diagnostic_batches/f075000-075100.json"
    batch = doc(batch_path, "0583cd38e5888d198e8bb2039b4ef0ae6f6f801ed83a656852a1f3ce23efbcad")
    assert batch["schema"] == "nodewise_lossless_diagnostic_batch_v1"
    assert batch["first_frame"] == 75000 and batch["stop_frame_exclusive"] == 75100
    assert [r["frame_idx"] for r in batch["tree"]["frames"]] == list(range(75000, 75100))
    ref = batch["archive"]
    payload = read(ref["path"], ref["sha256"])
    assert len(payload) == ref["size_bytes"]
    new_data, new_body, kinds, new_declarations = {}, {}, {}, {}
    descriptors = {}

    def collect(value):
        if isinstance(value, dict):
            if "__array__" in value:
                assert value["__array__"] not in descriptors
                descriptors[value["__array__"]] = value
            else:
                for v in value.values():
                    collect(v)
        elif isinstance(value, list):
            for v in value:
                collect(v)

    collect(batch["tree"])
    with np.load(io.BytesIO(payload), allow_pickle=False) as bank:
        assert len(bank.files) == len(descriptors) and set(bank.files) == set(descriptors)
        used = set()
        for frame in FRAMES:
            tree = batch["tree"]["frames"][frame - 75000]
            assert tree["frame_idx"] == frame
            for family, fields, dest in (
                ("instance", FIELDS, new_data),
                ("body_row", BODY, new_body),
            ):
                dest[frame] = {}
                for key in fields:
                    descriptor = tree[family][key]
                    member = descriptor["__array__"]
                    assert member not in used
                    array = bank[member]
                    used.add(member)
                    assert signature(array) == descriptor["signature"]
                    dest[frame][key] = array
                    new_declarations[member] = dict(
                        frame_idx=frame, field=f"{family}.{key}", **descriptor
                    )
            kinds[frame] = tree["instance"]["candidate_kind"]
            assert all(
                row["reason"] == "policy_not_yet_effective" and row["neutral"]
                for row in tree["instance"]["views"].values()
            )
            assert not np.any(new_data[frame]["sixth_instance_term"])
    del payload
    final = doc(
        W / "NODEWISE_FIRST_MINUTE_REACQUISITION_SOURCE_ROWS_V1.json",
        "fbae74148ba4eb2a69c3831ec0d348d2aa92714d442ba36f5fac365ea343f42d",
    )
    geometry = doc(
        W / "nodewise_first_minute_row011_geometry_v1/diagnostics.json",
        "797af289bd0d108aaf8eb70eeb2485e89b7cbd0245408772fb829a824902857c",
    )
    assert geometry["scope"]["source_rows"] == [0, 1, 1]
    references = {
        c["frame_idx"]: np.asarray(
            next(r["pose_xyz"] for r in c["reconstructions"] if r["used_view_indices"] == [0, 1, 2])
        )
        for c in geometry["cases"]
    }
    cases, snapshots, old_signatures = [], {}, {}
    weights = np.array(
        [
            1.0,
            cfg["shape_weight"],
            cfg["motion_weight"],
            cfg["repulsion_weight"],
            cfg["mask_weight"],
        ]
    )
    target_plan = np.repeat(np.array([0, 1, 1])[:, None], 3, axis=1)
    for frame in FRAMES:
        old_meta = json.loads(old_bytes(f"diagnostics/frame_{frame:06d}/metadata.json"))
        payload = old_bytes(f"diagnostics/frame_{frame:06d}/arrays.npz")
        with np.load(io.BytesIO(payload), allow_pickle=False) as bank:
            assert len(bank.files) == len(set(bank.files)) and set(OLD_FIELDS) <= set(bank.files)
            old = {k: bank[k] for k in OLD_FIELDS}
        old_signatures[str(frame)] = {k: signature(v) for k, v in old.items()}
        del payload
        new = new_data[frame]
        for arm, values, meta in (("tail", old, old_meta), ("nodewise", new, None)):
            if arm == "tail":
                pose, parent, plan = (
                    values[k] for k in ("fitted_pose", "proposal_parent", "proposal_plan")
                )
                local, parts, path, permutation = (
                    values[k]
                    for k in ("local_cost", "loss_parts", "path_cost", "proposal_permutation")
                )
                parent_cost = values["parent_cost"][parent]
                candidate_kind = meta["proposal_kind"]
                initial = values["proposal_pose"]
                initial_available = np.ones(len(pose), bool)
                sixth = np.zeros(len(pose))
            else:
                pose, parent, plan = (
                    values[k] for k in ("candidate_pose_xyz", "candidate_parent", "candidate_plan")
                )
                local, parts, path, permutation = (
                    values[k]
                    for k in (
                        "adjusted_local_cost",
                        "original_five_loss_parts",
                        "adjusted_path_cost",
                        "candidate_permutation",
                    )
                )
                parent_cost, sixth, candidate_kind = (
                    values["parent_path_cost"],
                    values["sixth_instance_term"],
                    kinds[frame],
                )
                initial = np.full_like(pose, np.nan)
                initial_available = np.zeros(len(pose), bool)
                indices = new_body[frame]["new_proposal_indices"]
                initial[indices] = new_body[frame]["new_proposal_poses"]
                initial_available[indices] = True
                np.testing.assert_array_equal(plan[indices], new_body[frame]["new_proposal_plans"])
                np.testing.assert_array_equal(values["base_local_cost"] + sixth, local)
            count = len(pose)
            assert plan.shape == pose.shape == (count, 3, 3, 3)
            assert all(np.isfinite(a).all() for a in (pose, local, parts, parent_cost, path))
            np.testing.assert_array_equal(parent_cost + local, path)
            retained, order = retained_indices(pose, path, parent, permutation, cfg["beam_size"])
            if arm == "tail":
                np.testing.assert_array_equal(retained, values["selected_indices"])
                np.testing.assert_array_equal(order, values["stable_path_order"])
            selected = next(r for r in final["arms"][arm] if r["frame_idx"] == frame)
            selected_pose = np.asarray(selected["pose_xyz"])
            selected_sources = np.asarray(selected["sources_by_body_view_node"])
            matches = [
                i
                for i in range(count)
                if np.array_equal(pose[i], selected_pose)
                and candidate_kind[i] == selected["proposal_kind"]
                and np.all((selected_sources < 0) | (selected_sources == plan[i]))
            ]
            assert matches and set(matches) & set(retained), (
                "final saved pose must match retained candidate"
            )
            weighted = parts * weights
            residual = local - sixth - weighted.sum(axis=1)
            exact_tuple = np.all(plan == target_plan, axis=(2, 3))
            compatible_tuple = np.all((plan == -1) | (plan == target_plan), axis=(2, 3)) & np.all(
                (plan >= 0).sum(axis=2) >= 2, axis=2
            )
            distances = np.linalg.norm(pose - references[frame][None, None], axis=-1)
            before_distances = np.linalg.norm(initial - references[frame][None, None], axis=-1)
            rank = np.argsort(order)
            rows = []
            for i in range(count):
                row = dict(
                    index=i,
                    kind=candidate_kind[i],
                    parent=int(parent[i]),
                    permutation=permutation[i],
                    local_cost=float(local[i]),
                    parent_path_cost=float(parent_cost[i]),
                    path_cost=float(path[i]),
                    parts=parts[i],
                    weighted_five_parts=weighted[i],
                    sixth_term=float(sixth[i]),
                    stored_local_minus_weighted_five_and_sixth=float(residual[i]),
                    stable_path_rank=int(rank[i]),
                    retained_at_frame=i in retained,
                    final76360_pose_kind_source_match=i in matches,
                    exact_row011_plan_by_body=exact_tuple[i],
                    compatible_row011_min2views_by_body=compatible_tuple[i],
                    fitted_node_distance_to_row011_reference_by_body_mm=distances[i],
                    initial_pose_available=bool(initial_available[i]),
                    initial_node_distance_to_row011_reference_by_body_mm=before_distances[i]
                    if initial_available[i]
                    else None,
                )
                if arm == "tail":
                    row.update(
                        generated_geometric_observed=values["generated_geometric_observed"][i],
                        generated_strict_observed=values["generated_strict_observed"][i],
                        fitted_geometric_observed=values["fitted_geometric_observed"][i],
                        fitted_strict_observed=values["fitted_strict_observed"][i],
                    )
                rows.append(row)
            cases.append(
                dict(
                    frame_idx=frame,
                    arm=arm,
                    count=count,
                    retained_indices=retained,
                    beam0_candidate_index=int(retained[0]),
                    final76360_match_indices=matches,
                    selected_kind=selected["proposal_kind"],
                    saved_final_observed=selected["saved_observed_by_body_node"],
                    target_row011_S2_exact_plan_indices=np.flatnonzero(exact_tuple[:, 2]),
                    target_row011_any_body_compatible_indices=np.flatnonzero(
                        compatible_tuple.any(axis=1)
                    ),
                    state0_missing=old["parent_missing"][0] if arm == "tail" else None,
                    state0_node_missing=old["parent_node_missing"][0] if arm == "tail" else None,
                    state0_direct_pectoral_age=old["parent_direct_pectoral_age"][0]
                    if arm == "tail"
                    else None,
                    row011_reference_pose=references[frame],
                    candidates=rows,
                )
            )
            for key, a in dict(
                fitted_pose=pose,
                proposal_plan=plan,
                initial_pose=initial,
                initial_pose_available=initial_available,
            ).items():
                snapshots[f"{arm}_{frame}_{key}"] = a
    for path, state in stats.items():
        st = path.stat()
        assert (st.st_size, st.st_mtime_ns) == state
    OUT.mkdir()
    source = Path(__file__).read_bytes()
    report = dict(
        schema=OUT.name,
        status="ok",
        source_sha256=sha(source),
        inputs=inputs,
        config=cfg,
        old_candidate_member_signatures=old_signatures,
        new_target_member_descriptors=new_declarations,
        new_batch_declared_members=len(descriptors),
        new_batch_decoded_members=len(new_declarations),
        cases=cases,
        semantics=dict(
            selected_is_current_retained6_not_final_backtrace=True,
            new_retained_reconstructed_from_frozen_stable_path_and_same_parent_pose_dedup=True,
            final_matching="all exact whole-pose + kind matches with saved supported sources subset of plan; no extra final-cost field read",
            parts_order=["node", "shape", "directional_center", "repulsion", "silhouette"],
            parts_do_not_include_orientation=True,
            residual="stored local minus weighted five and sixth: includes unreported orientation and floating aggregation differences; not a re-evaluated objective",
            new_existing_proposal_initial_pose_and_parent_ages_not_saved=True,
            new_all_candidate_geometric_support_not_recomputed=True,
            candidate_distances_to_row011_reference_not_support_or_truth=True,
            parent_cost_histories_differ_across_arms=True,
            no_original_SL_video_mask_checkpoint39_fit_solver_GPU_reads=True,
            excluded_pre_return_proposals_and_unadmitted_body_scenes_not_evaluated=True,
        ),
    )
    (OUT / "diagnostics.json").write_text(
        json.dumps(normalize(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(OUT / "candidate_samples.npz", **snapshots)
    (OUT / "source_snapshot.py").write_bytes(source)
    outputs = {
        p.name: dict(sha256=sha(p.read_bytes()), size_bytes=p.stat().st_size) for p in OUT.iterdir()
    }
    (OUT / "diagnosis.provenance.json").write_text(
        json.dumps(dict(stage=OUT.name, status="ok", inputs=inputs, outputs=outputs), indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            dict(
                status="ok",
                source_sha256=sha(source),
                outputs=outputs,
                seal_sha256=sha((OUT / "diagnosis.provenance.json").read_bytes()),
                cases=[
                    dict(
                        frame=c["frame_idx"],
                        arm=c["arm"],
                        retained=c["retained_indices"].tolist(),
                        final=c["final76360_match_indices"],
                        target_indices=c["target_row011_S2_exact_plan_indices"].tolist(),
                        candidates=[
                            {
                                k: r[k]
                                for k in (
                                    "index",
                                    "kind",
                                    "local_cost",
                                    "parent_path_cost",
                                    "path_cost",
                                    "stable_path_rank",
                                    "retained_at_frame",
                                )
                            }
                            for r in c["candidates"]
                            if r["exact_row011_plan_by_body"][2]
                            or r["final76360_pose_kind_source_match"]
                        ],
                    )
                    for c in cases
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
