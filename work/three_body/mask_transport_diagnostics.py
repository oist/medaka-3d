"""Pure reporting for all fixed mask-transport arms, rows and initial anchors.

No tracking, rendering, source loading or acceptance decision occurs here.
The caller supplies already verified candidate arrays and decoded producer
frame trees. All arrays returned are independent copies. NaN stays in arrays;
JSON summaries use None for unavailable statistics. Counts are repeated
observations/hypotheses, never independent fish or physical identity truth.

Fold angle is the angle between chest-mouth and tail-chest: 0 degrees is
straight continuation, 180 degrees a reversal. A zero/nonfinite segment has
no angle. Seed ratios use the same body and segment of the registered seed;
zero seed length has no ratio. No anatomical threshold is introduced.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

ARMS = ("original", "lk", "ct_first")
VIEWS = ("top_cam", "side1_cam", "side2_cam")
SCHEMA = "mask_transport_ancillary_diagnostics_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, kind, label):
    require(
        isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in kind,
        label + " schema differs",
    )
    return value


def _native(value):
    require(
        isinstance(value, np.ndarray)
        and value.dtype == np.int64
        and value.ndim == 1
        and len(value) > 0
        and np.all(value >= 0)
        and np.all(np.diff(value) == 1),
        "ordered contiguous native int64 vector required",
    )
    return value


def _same(a, b, label):
    require(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
        label + " bytes differ",
    )


def _stats(values, selected=None):
    values = np.asarray(values)
    chosen = np.ones(values.shape, bool) if selected is None else selected
    finite = chosen & np.isfinite(values)
    measured = values[finite]
    return dict(
        requested_count=int(values.size),
        selected_count=int(chosen.sum()),
        finite_count=int(finite.sum()),
        selected_nonfinite_count=int((chosen & ~np.isfinite(values)).sum()),
        minimum=float(measured.min()) if len(measured) else None,
        median=float(np.median(measured)) if len(measured) else None,
        maximum=float(measured.max()) if len(measured) else None,
        mean=float(measured.mean()) if len(measured) else None,
    )


def _geometry(pose, observed, seed):
    shape = pose.shape[:-2]
    _array(pose, (*shape, 3, 3), "f", "pose")
    _array(observed, (*shape, 3), "b", "observed")
    _array(seed, (3, 3, 3), "f", "registered seed")
    require(np.isfinite(seed).all(), "finite registered seed required")
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        vector = np.diff(pose.astype(np.float64), axis=-2)
        lengths = np.linalg.norm(vector, axis=-1)
        lengths[~np.isfinite(vector).all(axis=-1)] = np.nan
        usable = np.isfinite(lengths).all(axis=-1) & (lengths > 0).all(axis=-1)
        unit = np.zeros_like(vector)
        np.divide(vector, lengths[..., None], out=unit, where=lengths[..., None] > 0)
        cosine = np.sum(unit[..., 0, :] * unit[..., 1, :], axis=-1)
        angle = np.where(usable, np.degrees(np.arccos(np.clip(cosine, -1, 1))), np.nan)
        seed_length = np.linalg.norm(np.diff(seed.astype(np.float64), axis=1), axis=-1)
        ratio = np.full_like(lengths, np.nan)
        np.divide(lengths, seed_length, out=ratio, where=seed_length > 0)
    direct = observed[..., :-1] & observed[..., 1:]
    return dict(
        segment_length_mm=lengths,
        seed_length_ratio=ratio,
        fold_angle_deg=angle,
        direct_segment=direct.copy(),
        direct_fold=observed.all(axis=-1),
        zero_segment=(lengths == 0),
        nonfinite_segment=~np.isfinite(lengths),
        zero_seed_segment=(seed_length == 0),
        seed_segment_length_mm=seed_length,
    )


def _geometry_summary(arrays):
    result = []
    for body in range(3):
        segments = []
        for segment, label in enumerate(("mouth_chest", "chest_tail")):
            direct = arrays["direct_segment"][:, body, segment]
            length = arrays["segment_length_mm"][:, body, segment]
            ratio = arrays["seed_length_ratio"][:, body, segment]
            segments.append(
                dict(
                    segment=label,
                    all_estimated_length_mm=_stats(length),
                    direct_endpoint_length_mm=_stats(length, direct),
                    seed_ratio=_stats(ratio),
                    zero_length_count=int((length == 0).sum()),
                    zero_seed_length=bool(arrays["zero_seed_segment"][body, segment]),
                )
            )
        result.append(
            dict(
                body=body,
                segments=segments,
                all_estimated_fold_angle_deg=_stats(arrays["fold_angle_deg"][:, body]),
                direct_three_node_fold_angle_deg=_stats(
                    arrays["fold_angle_deg"][:, body], arrays["direct_fold"][:, body]
                ),
            )
        )
    return result


def _candidates(candidates):
    require(set(candidates) == set(ARMS), "exact three candidate arms required")
    native = _native(candidates[ARMS[0]]["frame_idx"])
    for arm in ARMS:
        current = candidates[arm]
        _same(_native(current["frame_idx"]), native, "candidate native")
        _array(current["pose_xyz"], (len(native), 3, 3, 3), "f", "candidate pose")
        _array(current["observed"], (len(native), 3, 3), "b", "candidate support")
        _array(current["identity_review"], (len(native), 3), "b", "candidate review")
    return native


def candidate_geometry(candidates_by_arm, seed_pose_xyz):
    """All T x body geometry; direct support subsets have explicit denominators."""
    native = _candidates(candidates_by_arm)
    arrays, summary = {}, {}
    for arm in ARMS:
        candidate = candidates_by_arm[arm]
        values = _geometry(candidate["pose_xyz"], candidate["observed"], seed_pose_xyz)
        values.update(frame_idx=native.copy(), identity_review=candidate["identity_review"].copy())
        arrays[arm] = values
        summary[arm] = dict(
            frames=len(native),
            body_times=3 * len(native),
            review_by_body=values["identity_review"].sum(axis=0).tolist(),
            bodies=_geometry_summary(values),
        )
    return dict(schema=SCHEMA, arrays=arrays, summary=summary, independent_ID_truth=False)


def _trees(trees, native):
    require(set(trees) == set(ARMS), "exact three frame-tree arms required")
    for arm in ARMS:
        require(len(trees[arm]) == len(native), "all native frame trees required")
        for n, tree in zip(native, trees[arm], strict=True):
            require(
                type(tree["fit"]["native"]) is int and tree["fit"]["native"] == int(n),
                "frame-tree native differs",
            )


def retained_history(frame_trees_by_arm, candidates_by_arm, seed_pose_xyz):
    """Flatten every retained row; distinguish causal best from final backtrace.

    Row zero at each native is a causal best. The final candidate uses the
    terminal row-zero parent chain and is labelled retrospective here. Review
    clearing is counted against the actual parent, not adjacent row rank.
    """
    native = _candidates(candidates_by_arm)
    _trees(frame_trees_by_arm, native)
    all_arrays, summaries = {}, {}
    for arm in ARMS:
        trees = frame_trees_by_arm[arm]
        initial_count = len(trees[0]["records"])
        offsets, rows, parent_flat, selected, anchors, cleared = [0], [], [], [], [], []
        for t, tree in enumerate(trees):
            records, fit, ids = tree["records"], tree["fit"], tree["anchor_ids"]
            require(1 <= len(records) <= 6, "one to six retained rows required")
            _array(ids, (len(records),), "i", "retained anchor IDs")
            require(
                ids.dtype == np.int64 and ((ids >= 0) & (ids < initial_count)).all(),
                "anchor IDs outside initial bank",
            )
            sel = _array(fit["selected"], (len(records),), "i", "selected fit indices")
            require(
                sel.dtype == np.int64 and len(np.unique(sel)) == len(sel),
                "unique selected fit indices required",
            )
            parents = fit["parent"]
            require(
                parents.dtype == np.int64
                and parents.ndim == 1
                and ((sel >= 0) & (sel < len(parents))).all(),
                "fit parent/selection scope",
            )
            for rank, (record, anchor) in enumerate(zip(records, ids, strict=True)):
                require(int(record["frame_idx"]) == int(native[t]), "record native differs")
                p = int(record["parent"])
                require(p == int(parents[sel[rank]]), "record/selected parent differs")
                _array(record["pose_xyz"], (3, 3, 3), "f", "retained pose")
                _array(record["observed"], (3, 3), "b", "retained support")
                _array(record["identity_review"], (3,), "b", "retained review")
                if t:
                    previous = trees[t - 1]
                    require(0 <= p < len(previous["records"]), "retained parent outside beam")
                    require(anchor == previous["anchor_ids"][p], "retained anchor lineage differs")
                    parent_flat.append(offsets[t - 1] + p)
                    cleared.append(
                        previous["records"][p]["identity_review"] & ~record["identity_review"]
                    )
                else:
                    require(p == 0 and anchor == rank, "initial retained anchor order differs")
                    parent_flat.append(-1)
                    cleared.append(np.zeros(3, bool))
                rows.append(record)
                selected.append(int(sel[rank]))
                anchors.append(int(anchor))
            offsets.append(len(rows))
        backtrace = np.empty(len(native), np.int64)
        rank = 0
        for t in range(len(native) - 1, -1, -1):
            backtrace[t] = offsets[t] + rank
            record = trees[t]["records"][rank]
            for key in ("pose_xyz", "observed", "identity_review"):
                _same(record[key], candidates_by_arm[arm][key][t], "final backtrace " + key)
            rank = int(record["parent"])
        values = dict(
            frame_idx=np.repeat(native, np.diff(offsets)),
            history_offsets=np.asarray(offsets, np.int64),
            retained_rank=np.concatenate([np.arange(n) for n in np.diff(offsets)]),
            parent_flat_index=np.asarray(parent_flat, np.int64),
            selected_candidate_index=np.asarray(selected, np.int64),
            anchor_id=np.asarray(anchors, np.int64),
            pose_xyz=np.stack([r["pose_xyz"] for r in rows]),
            observed=np.stack([r["observed"] for r in rows]),
            identity_review=np.stack([r["identity_review"] for r in rows]),
            parent_review_cleared=np.stack(cleared),
            local_cost=np.asarray([r["costs"] for r in rows]),
            path_cost=np.asarray([r["path_costs"] for r in rows]),
            causal_best_flat_index=np.asarray(offsets[:-1], np.int64),
            final_backtrace_flat_index=backtrace,
        )
        values.update(_geometry(values["pose_xyz"], values["observed"], seed_pose_xyz))

        def count(indexes, anchor_ids=values["anchor_id"], size=initial_count):
            return np.bincount(anchor_ids[indexes], minlength=size).tolist()

        summaries[arm] = dict(
            native_frames=len(native),
            retained_rows=len(rows),
            initial_anchors=initial_count,
            retained_rows_per_native=np.diff(offsets).tolist(),
            retained_body_times=3 * len(rows),
            review_by_body=values["identity_review"].sum(axis=0).tolist(),
            review_parent_edges=len(rows) - initial_count,
            parent_review_clear_by_body=values["parent_review_cleared"].sum(axis=0).tolist(),
            all_retained_anchor_rows=count(np.arange(len(rows))),
            causal_best_anchor_native_counts=count(values["causal_best_flat_index"]),
            final_backtrace_anchor_native_counts=count(backtrace),
            bodies=_geometry_summary(values),
        )
        all_arrays[arm] = values
    return dict(schema=SCHEMA, arrays=all_arrays, summary=summaries, independent_ID_truth=False)


def _number(value):
    return float(value) if np.isfinite(value) else None


def transport_diagnostics(frame_trees_by_arm, *, frame_idx):
    """All packet decisions, no selection by fitted pose or final anchor.

    Evidence rows are emitted ONCE per arm/native/view/group; assignments and
    group-mask competition are anchor-specific. Original/t0 absent packets
    have explicit policy status, not measured zero. Target SUM T, positive
    draws and pre-ray weight*foreground mass are separate. Actual ray-valid
    used-view counts cannot be inferred from these trees and remain None.
    Full ray-valid mass belongs to the caller's shared-pixel renderer audit.
    """
    native = _native(frame_idx)
    _trees(frame_trees_by_arm, native)
    frames, evidence_rows, group_rows, cell_rows, body_rows = [], [], [], [], []
    summaries = {}
    for arm in ARMS:
        reasons, packet_count, evidence_count, group_count, cells = Counter(), 0, 0, 0, 0
        body_start = len(body_rows)
        bank = frame_trees_by_arm[arm][0]["anchor_ids"]
        require(
            isinstance(bank, np.ndarray)
            and bank.dtype == np.int64
            and bank.ndim == 1
            and 1 <= len(bank) <= 6
            and np.array_equal(bank, np.arange(len(bank))),
            "initial retained anchor bank required",
        )
        for t, (n, tree) in enumerate(zip(native, frame_trees_by_arm[arm], strict=True)):
            fit, packet = tree["fit"], tree["fit"]["packet"]
            if packet is None:
                require(t == 0 or arm == "original", "active later packet missing")
                frames.append(
                    dict(
                        arm=arm,
                        native=int(n),
                        packet_present=False,
                        reason="initial_zero_policy" if t == 0 else "coefficient_zero",
                    )
                )
                continue
            require(arm != "original" and t > 0, "packet violates zero coefficient policy")
            ids = packet["anchor_ids"]
            require(
                ids.dtype == np.int64
                and ids.ndim == 1
                and 1 <= len(ids) <= 6
                and np.array_equal(ids, np.arange(len(ids))),
                "all initial anchors required",
            )
            _same(bank, ids, "packet initial anchor bank")
            require(
                packet["native"] == int(n) and packet["initial_native"] == int(native[0]),
                "packet native/initial binding differs",
            )
            target = packet["target"]
            require(
                target.dtype == np.float64
                and target.ndim == 4
                and target.shape[:2] == (len(ids), 3)
                and target.shape[-1] == 3
                and np.isfinite(target).all()
                and ((target >= 0) & (target <= 1)).all(),
                "all-anchor finite target schema",
            )
            _same(target.sum(axis=2), packet["target_mass"], "unweighted target mass")
            _same(
                np.count_nonzero(target, axis=2),
                packet["target_nonzero_count"],
                "nonzero target draws",
            )
            shape = target.shape[1:3]
            weights = _array(fit["fit_weights"], shape, "f", "original pixel weights")
            foreground = _array(fit["fit_target"], shape, "f", "original foreground")
            require(
                np.isfinite(weights).all()
                and (weights >= 0).all()
                and np.isin(foreground, [0, 1]).all(),
                "finite original pixel evidence",
            )
            # Cast before multiplication, matching the float64 fitter evidence.
            pre_ray = (target * (weights.astype(np.float64) * foreground)[None, :, :, None]).sum(2)
            require(len(packet["assignments"]) == len(ids), "all-anchor assignments required")
            packet_count += 1
            frames.append(
                dict(
                    arm=arm, native=int(n), packet_present=True, anchor_count=len(ids), reason=None
                )
            )
            baseline_evidence = packet["assignments"][0]
            require(len(baseline_evidence) == 3, "three evidence views required")
            for a, anchor in enumerate(ids):
                require(len(packet["assignments"][a]) == 3, "three assignment views required")
                for v, view in enumerate(VIEWS):
                    assignment = packet["assignments"][a][v]
                    ev = assignment["evidence"]
                    require(ev["native"] == int(n), "vote native differs")
                    g, masks = ev["votes"].shape
                    require(
                        g == ev["group_count"] and ev["single_votes"].shape == (g, masks),
                        "all group/mask votes required",
                    )
                    if a:
                        old = baseline_evidence[v]["evidence"]
                        require(set(ev) == set(old), "shared evidence fields differ")
                        for key in ev:
                            if isinstance(ev[key], np.ndarray):
                                _same(ev[key], old[key], "shared anchor evidence " + key)
                            else:
                                require(ev[key] == old[key], "shared anchor evidence differs")
                    for group in range(g):
                        where = dict(arm=arm, native=int(n), view=view, group=group)
                        if a == 0:
                            evidence_count += 1
                            available = ev["mask_available"]
                            evidence_rows.append(
                                dict(
                                    **where,
                                    available=available,
                                    current_mask_count=masks,
                                    reported_count=int(ev["reported_count"][group]),
                                    single_count=int(ev["single_count"][group])
                                    if available
                                    else None,
                                    multi_count=int(ev["multi_count"][group])
                                    if available
                                    else None,
                                    unassigned_count=int(ev["unassigned_count"][group])
                                    if available
                                    else None,
                                    release_horizon=int(ev["release_horizon"]),
                                )
                            )
                        why = list(assignment["group_reasons"][group])
                        reasons.update(why)
                        group_count += 1
                        group_rows.append(
                            dict(
                                **where,
                                anchor_id=int(anchor),
                                body=int(assignment["group_body"][group]),
                                candidate_mask_row=int(assignment["candidate_mask_row"][group]),
                                admitted=bool(assignment["group_admitted"][group]),
                                reasons=why,
                            )
                        )
                        for mask in range(masks):
                            cells += 1
                            cell_rows.append(
                                dict(
                                    **where,
                                    anchor_id=int(anchor),
                                    mask_row=mask,
                                    votes=int(ev["votes"][group, mask]),
                                    single_votes=int(ev["single_votes"][group, mask]),
                                    single_fraction_of_all_reported=_number(
                                        ev["weights"][group, mask]
                                    ),
                                    vote_maximum_tie=bool(ev["vote_maximum_ties"][group, mask]),
                                    single_maximum_tie=bool(ev["single_maximum_ties"][group, mask]),
                                    competing=bool(assignment["competing_group_mask"][group, mask]),
                                    unknown_competing=bool(
                                        assignment["unknown_competing_group_mask"][group, mask]
                                    ),
                                    linked_competing=bool(
                                        assignment["linked_competing_group_mask"][group, mask]
                                    ),
                                )
                            )
                    for body in range(3):
                        body_rows.append(
                            dict(
                                arm=arm,
                                native=int(n),
                                anchor_id=int(anchor),
                                view=view,
                                body=body,
                                mask_row=int(assignment["body_mask_row"][body]),
                                body_weight=float(assignment["body_weight"][body]),
                                reasons=list(assignment["body_reasons"][body]),
                                target_sum=float(packet["target_mass"][a, v, body]),
                                target_nonzero_draws=int(
                                    packet["target_nonzero_count"][a, v, body]
                                ),
                                pre_ray_weighted_foreground_mass=float(pre_ray[a, v, body]),
                                assigned_view=bool(assignment["body_mask_row"][body] >= 0),
                                positive_target_view=bool(packet["target_mass"][a, v, body] > 0),
                                positive_pre_ray_weighted_view=bool(pre_ray[a, v, body] > 0),
                                ray_valid_used_view=None,
                            )
                        )
        arm_bodies = body_rows[body_start:]
        summaries[arm] = dict(
            frames=len(native),
            packet_frames=packet_count,
            disabled_packet_frames=len(native) - packet_count,
            initial_anchors=len(bank),
            scheduled_anchor_body_view_times=len(native) * len(bank) * 9,
            policy_disabled_anchor_body_view_times=(len(native) - packet_count) * len(bank) * 9,
            evidence_group_view_times=evidence_count,
            anchor_group_view_times=group_count,
            anchor_group_mask_cells=cells,
            anchor_body_view_times=len(arm_bodies),
            admitted_anchor_body_view_times=sum(r["assigned_view"] for r in arm_bodies),
            positive_target_anchor_body_view_times=sum(
                r["positive_target_view"] for r in arm_bodies
            ),
            positive_pre_ray_anchor_body_view_times=sum(
                r["positive_pre_ray_weighted_view"] for r in arm_bodies
            ),
            abstention_reason_counts=dict(sorted(reasons.items())),
            reasons_are_nonexclusive=True,
            ray_valid_used_views=None,
        )
    return dict(
        schema=SCHEMA,
        frames=frames,
        evidence_rows=evidence_rows,
        group_rows=group_rows,
        group_mask_rows=cell_rows,
        body_view_rows=body_rows,
        summary=summaries,
        independent_ID_truth=False,
    )
