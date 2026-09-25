"""Pure same-slot reporting; no pose fitting, ID matching or review modification."""

from __future__ import annotations

import compare_tail_one_minute as old
import mask_transport_diagnostics as geometry
import numpy as np

checks = old.checks
SOURCE_CLASSES = ("both_absent", "same_present_row", "added", "removed", "different_present_row")


def validate_candidate(candidate, first, stop):
    keys = set(old.prior.RECORD_FIELDS) - {"parent"}
    keys.update(("xyz", "accepted_xyz", "body_xyz"))
    checks.require(set(candidate) == keys and len(keys) == 21, "all21 candidate fields required")
    count = stop - first
    shapes = {
        "frame_idx": ((), "i"),
        "pose_xyz": ((3, 3, 3), "f"),
        "observed": ((3, 3), "b"),
        "sources": ((3, 3, 3), "i"),
        "identity_review": ((3,), "b"),
        "mask_supported": ((3,), "b"),
        "body_supported": ((3,), "b"),
        "missing": ((3,), "i"),
        "costs": ((), "f"),
        "path_costs": ((), "f"),
        "loss_parts": ((5,), "f"),
        "score_margin": ((), "f"),
        "proposal_kind": ((), "U"),
        "hypotheses": ((), "i"),
        "proposal_flow_nodes": ((3, 3), "b"),
        "proposal_flow_parent": ((), "i"),
        "node_missing": ((3, 3), "i"),
        "direct_pectoral_age": ((3,), "i"),
        "xyz": ((3, 3), "f"),
        "accepted_xyz": ((3, 3), "f"),
        "body_xyz": ((3, 3), "f"),
    }
    for key, (suffix, kind) in shapes.items():
        a = candidate[key]
        checks.require(
            isinstance(a, np.ndarray) and a.shape == (count, *suffix) and a.dtype.kind == kind,
            "candidate shape/dtype: " + key,
        )
        if kind == "f":
            checks.require(a.dtype == np.float64, "original FP64 field required: " + key)
    checks.require(
        candidate["frame_idx"].dtype == np.int64
        and np.array_equal(candidate["frame_idx"], np.arange(first, stop, dtype=np.int64)),
        "exact native candidate extent required",
    )
    old.prior.support_audit.support_input(candidate)
    observed, pose, review = (candidate[k] for k in ("observed", "pose_xyz", "identity_review"))
    xyz = np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan)
    derived = dict(
        xyz=xyz,
        accepted_xyz=np.where(review[:, :, None], np.nan, xyz),
        body_xyz=np.where(candidate["body_supported"][:, :, None], pose[:, :, 1], np.nan),
    )
    for key, value in derived.items():
        checks.require(
            checks.equal_bytes(candidate[key], value), "derived support/review field: " + key
        )


def interval_metrics(candidate, seed, start, stop):
    frames = candidate["frame_idx"]
    keep = (frames >= start) & (frames < stop)
    checks.require(int(keep.sum()) == stop - start, "complete metric interval required")
    observed, review = candidate["observed"][keep], candidate["identity_review"][keep]
    accepted = observed & ~review[:, :, None]
    values = geometry._geometry(candidate["pose_xyz"][keep], observed, seed)
    motion = old.old.node_metrics(candidate, start=start, stop=stop)
    summary = dict(
        native_interval=[start, stop],
        native_frames=stop - start,
        body_times=3 * (stop - start),
        node_times=9 * (stop - start),
        direct_by_slot_node=observed.sum(0).tolist(),
        direct_by_node=observed.sum((0, 1)).tolist(),
        accepted_by_slot_node=accepted.sum(0).tolist(),
        accepted_by_node=accepted.sum((0, 1)).tolist(),
        all_three_direct_chest_frames=int(observed[:, :, 1].all(1).sum()),
        all_three_accepted_chest_frames=int(accepted[:, :, 1].all(1).sum()),
        review_by_slot=review.sum(0).tolist(),
        review_fish_frames=int(review.sum()),
        mask_supported_by_slot=candidate["mask_supported"][keep].sum(0).tolist(),
        body_supported_by_slot=candidate["body_supported"][keep].sum(0).tolist(),
        selected_kind_counts={
            str(v): int(n)
            for v, n in zip(
                *np.unique(candidate["proposal_kind"][keep], return_counts=True), strict=True
            )
        },
        node_metrics=motion["by_slot_node"],
        body_geometry=geometry._geometry_summary(values),
        geometry_scope=(
            "All estimated poses; direct-endpoint subsets separately counted, no new shape gate"
        ),
    )
    values["frame_idx"] = frames[keep].copy()
    return summary, motion, values


def differences(original, tail, start, stop):
    checks.require(
        np.array_equal(original["frame_idx"], tail["frame_idx"]), "same-horizon frames required"
    )
    keep = (original["frame_idx"] >= start) & (original["frame_idx"] < stop)
    checks.require(int(keep.sum()) == stop - start, "complete difference interval required")
    x, y = original["observed"][keep], tail["observed"][keep]
    gain, loss = y & ~x, x & ~y
    a, b = original["sources"][keep], tail["sources"][keep]
    classes = np.zeros(a.shape, np.uint8)
    classes[(a >= 0) & (b >= 0) & (a == b)] = 1
    classes[(a < 0) & (b >= 0)] = 2
    classes[(a >= 0) & (b < 0)] = 3
    classes[(a >= 0) & (b >= 0) & (a != b)] = 4
    review_a, review_b = original["identity_review"][keep], tail["identity_review"][keep]
    accept_a, accept_b = x & ~review_a[:, :, None], y & ~review_b[:, :, None]
    source_counts = {name: (classes == i).sum(0).tolist() for i, name in enumerate(SOURCE_CLASSES)}
    rows = []
    frames = original["frame_idx"][keep]
    for t, body, node in np.argwhere(gain | loss):
        rows.append(
            dict(
                frame_idx=int(frames[t]),
                slot=int(body),
                node=old.old.NODES[node],
                change="gain" if gain[t, body, node] else "loss",
                original_sources=a[t, body, :, node].tolist(),
                tail_sources=b[t, body, :, node].tolist(),
                original_review=bool(review_a[t, body]),
                tail_review=bool(review_b[t, body]),
            )
        )
    arrays = dict(
        frame_idx=frames.copy(),
        support_gain=gain,
        support_loss=loss,
        accepted_gain=accept_b & ~accept_a,
        accepted_loss=accept_a & ~accept_b,
        source_transition_class=classes,
        review_changed=review_a != review_b,
    )
    summary = dict(
        native_interval=[start, stop],
        node_denominator=int(x.size),
        source_slot_denominator=int(a.size),
        gained_by_slot_node=gain.sum(0).tolist(),
        lost_by_slot_node=loss.sum(0).tolist(),
        gained_by_node=gain.sum((0, 1)).tolist(),
        lost_by_node=loss.sum((0, 1)).tolist(),
        accepted_gain_by_slot_node=arrays["accepted_gain"].sum(0).tolist(),
        accepted_loss_by_slot_node=arrays["accepted_loss"].sum(0).tolist(),
        source_transition_by_body_view_node=source_counts,
        source_transition_totals={
            name: int((classes == i).sum()) for i, name in enumerate(SOURCE_CLASSES)
        },
        review_changed_by_slot=arrays["review_changed"].sum(0).tolist(),
        slot_scope=(
            "Same numeric body slots; source rows are within-native/view/node, not physical IDs"
        ),
    )
    return summary, rows, arrays


def horizon_changes(snapshot, final, start, stop):
    a_keep = (snapshot["frame_idx"] >= start) & (snapshot["frame_idx"] < stop)
    b_keep = (final["frame_idx"] >= start) & (final["frame_idx"] < stop)
    frames = snapshot["frame_idx"][a_keep]
    checks.require(
        set(snapshot) == set(final)
        and len(snapshot) == 21
        and len(frames) == stop - start
        and np.array_equal(frames, final["frame_idx"][b_keep]),
        "same first-minute rows required",
    )
    fields, arrays = {}, {"frame_idx": frames.copy()}
    for key in sorted(snapshot):
        a, b = np.ascontiguousarray(snapshot[key][a_keep]), np.ascontiguousarray(final[key][b_keep])
        checks.require(a.shape == b.shape, "horizon field shape differs")
        unicode = a.dtype.kind == b.dtype.kind == "U"
        if unicode:
            equal = (a == b).reshape(len(frames), -1).all(1)
        else:
            checks.require(a.dtype == b.dtype, "numeric horizon dtype differs")
            equal = (
                a.view(np.uint8).reshape(len(frames), -1)
                == b.view(np.uint8).reshape(len(frames), -1)
            ).all(1)
        changed = ~equal
        arrays[key + "_row_changed"] = changed
        fields[key] = dict(
            changed_rows=int(changed.sum()),
            first_changed_frame=int(frames[changed][0]) if changed.any() else None,
            original_dtype=a.dtype.str,
            final_dtype=b.dtype.str,
            unicode_values_ignoring_capacity=unicode,
        )
    return dict(
        compared_interval=[start, stop],
        snapshot_horizon=stop,
        final_horizon=int(final["frame_idx"][-1]) + 1,
        fields=fields,
        interpretation=(
            "Different terminal backtrace; not a same-horizon numeric failure or online change"
        ),
    ), arrays
