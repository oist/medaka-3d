"""Pure reconstruction of the frozen continuous tracker's 19 retained fields.

No fit, raw geometry validation, file I/O, tracker state update, or GPU occurs.
The caller supplies already checked strict sources/observed and the full fitted
pool. ``proposal_count`` selects its original prefix for control, or the whole
pool for treatment. Review/margins always inspect every proposal in that prefix,
including alternatives removed by deduplication. No future selected row is used.
"""

from itertools import permutations

import numpy as np

VERSION = "bounded_fit_record_check_v1"
ATOL = 1e-10
RECORD_FIELDS = (
    "frame_idx",
    "pose_xyz",
    "observed",
    "sources",
    "identity_review",
    "mask_supported",
    "body_supported",
    "missing",
    "costs",
    "path_costs",
    "loss_parts",
    "score_margin",
    "parent",
    "proposal_kind",
    "hypotheses",
    "proposal_flow_nodes",
    "proposal_flow_parent",
    "node_missing",
    "direct_pectoral_age",
)
FLOAT_FIELDS = {"pose_xyz", "costs", "path_costs", "loss_parts", "score_margin"}
BOOL_FIELDS = {
    "observed",
    "identity_review",
    "mask_supported",
    "body_supported",
    "proposal_flow_nodes",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def array(value, shape, kind, name):
    result = np.asarray(value)
    require(result.shape == shape and result.dtype.kind in kind, "invalid " + name)
    if result.dtype.kind == "f":
        require(np.isfinite(result).all(), "nonfinite " + name)
    return result


def integer(value, name):
    scalar = np.asarray(value)
    require(scalar.shape == () and scalar.dtype.kind in "iu", name + " must be an integer scalar")
    return int(scalar)


def different_identity(first, second):
    """Exact frozen chest-permutation comparison, including lexical spatial ties."""
    costs = [
        (sum(np.linalg.norm(first[f, 1] - second[p[f], 1]) ** 2 for f in range(3)), p)
        for p in permutations(range(3))
    ]
    _, best = min(costs)
    return np.asarray(best) != np.arange(3)


def reconstruct_records(
    packet,
    fitted_pose,
    local_cost,
    loss_parts,
    mask_supported,
    strict_sources,
    strict_observed,
    selected,
    full_order,
    cfg,
    *,
    proposal_count,
):
    """Return owned K-leading arrays with exactly ``RECORD_FIELDS`` keys.

    Packet keys: frame_idx; proposal_parent/permutation/kind for all B proposals;
    parent_pose/missing/review/cost/node_missing/direct_pectoral_age for all P
    parents; flow_present (bool), flow_node_valid (3,3 bool). Parent pose is
    validated as context but velocity is unnecessary for these 19 record fields.
    Source rows retain caller dtype and must be exclusive per view/node. Actual
    2mm/12px geometry must have been checked by the caller. Kind values retain
    full Unicode width; comparison reports harmless width differences separately.
    Stable order and all selected indices are independently recomputed/checked.
    """
    native = integer(packet["frame_idx"], "frame_idx")
    require(0 <= native < np.iinfo(np.int64).max, "native frame outside int64 range")
    count = integer(proposal_count, "proposal_count")
    parent = np.asarray(packet["proposal_parent"])
    require(parent.ndim == 1 and parent.dtype.kind == "i", "invalid proposal_parent")
    batch = len(parent)
    require(0 < count <= batch, "proposal_count outside fitted pool")
    pp = np.asarray(packet["parent_pose"])
    require(pp.ndim == 4 and pp.shape[1:] == (3, 3, 3), "invalid parent_pose")
    parents = len(pp)
    require(parents > 0 and np.all((parent >= 0) & (parent < parents)), "parent index outside pool")
    array(pp, (parents, 3, 3, 3), "f", "parent_pose")
    missing = array(packet["parent_missing"], (parents, 3), "i", "parent_missing")
    review = array(packet["parent_review"], (parents, 3), "b", "parent_review")
    cost = array(packet["parent_cost"], (parents,), "f", "parent_cost")
    age = array(packet["parent_node_missing"], (parents, 3, 3), "i", "parent_node_missing")
    pect_age = array(
        packet["parent_direct_pectoral_age"], (parents, 3), "i", "parent_direct_pectoral_age"
    )
    for values in (missing, age, pect_age):
        require(
            np.all(values >= 0) and np.all(values < np.iinfo(values.dtype).max),
            "invalid or overflowing parent age",
        )
    permutation = array(packet["proposal_permutation"], (batch, 3), "i", "proposal_permutation")
    require(np.all(np.sort(permutation, axis=1) == np.arange(3)), "not a fish permutation")
    kind = array(packet["proposal_kind"], (batch,), "U", "proposal_kind")
    flow_scalar = np.asarray(packet["flow_present"])
    require(
        flow_scalar.shape == () and flow_scalar.dtype.kind == "b",
        "flow_present must be bool scalar",
    )
    flow_present = bool(flow_scalar)
    flow = array(packet["flow_node_valid"], (3, 3), "b", "flow_node_valid")
    require(flow_present or not flow.any(), "absent flow cannot declare valid nodes")
    pose = array(fitted_pose, (batch, 3, 3, 3), "f", "fitted_pose")
    local = array(local_cost, (batch,), "f", "local_cost")
    parts = array(loss_parts, (batch, 5), "f", "loss_parts")
    masks = array(mask_supported, (batch, 3), "b", "mask_supported")
    sources = array(strict_sources, (batch, 3, 3, 3), "i", "strict_sources")
    observed = array(strict_observed, (batch, 3, 3), "b", "strict_observed")
    require(np.all(sources >= -1), "source rows below -1")
    require(
        np.array_equal(observed, (sources >= 0).sum(axis=2) >= 2), "strict support count differs"
    )
    for one in sources:
        for view in range(3):
            for node in range(3):
                used = one[:, view, node]
                used = used[used >= 0]
                require(len(np.unique(used)) == len(used), "strict source collision")
    beam = integer(cfg.beam_size, "beam_size")
    review_gap = integer(cfg.review_gap_frames, "review_gap_frames")
    require(beam > 0 and review_gap > 0 and np.isfinite(cfg.ambiguity_margin), "invalid config")
    with np.errstate(over="ignore", invalid="ignore"):
        path = local + cost[parent]
    require(np.isfinite(path).all(), "nonfinite accumulated path cost")
    expected_order = np.argsort(path[:count], kind="stable")
    order = array(full_order, (count,), "i", "full_order")
    require(np.array_equal(order, expected_order), "full stable path order differs")
    chosen = []
    for candidate in order:
        if any(
            parent[candidate] == parent[j]
            and np.array_equal(permutation[candidate], permutation[j])
            and np.max(np.linalg.norm(pose[candidate] - pose[j], axis=-1)) < 0.5
            for j in chosen
        ):
            continue
        chosen.append(int(candidate))
        if len(chosen) == beam:
            break
    selected = array(selected, (len(chosen),), "i", "selected")
    require(np.array_equal(selected, chosen), "selected stable dedup beam differs")
    records = []
    for candidate in chosen:
        pi = parent[candidate]
        direct = observed[candidate]
        held_sources = sources[candidate]
        has_head_ray = (held_sources[:, :, :2] >= 0).any(axis=(1, 2))
        body = direct[:, 1] | (masks[candidate] & has_head_ray)
        next_missing = np.where(body, 0, missing[pi] + 1)
        next_review = review[pi] | (next_missing >= review_gap)
        margin = float("inf")
        for other in order:
            if other == candidate:
                continue
            affected = different_identity(pose[candidate], pose[other])
            if parent[other] == pi:
                tied = (
                    np.max(np.linalg.norm(pose[candidate, :, 1] - pose[other, :, 1], axis=-1)) < 0.5
                )
                if tied:
                    affected |= permutation[candidate] != permutation[other]
            if not affected.any():
                continue
            gap = float(path[other] - path[candidate])
            require(np.isfinite(gap), "nonfinite alternative path gap")
            margin = min(margin, gap)
            if gap <= cfg.ambiguity_margin:
                next_review |= affected
        records.append(
            dict(
                frame_idx=native,
                pose_xyz=pose[candidate],
                observed=direct,
                sources=held_sources,
                identity_review=next_review,
                mask_supported=masks[candidate],
                body_supported=body,
                missing=next_missing,
                costs=local[candidate],
                path_costs=path[candidate],
                loss_parts=parts[candidate],
                score_margin=margin,
                parent=pi,
                proposal_kind=kind[candidate],
                hypotheses=count,
                proposal_flow_nodes=flow if flow_present else np.zeros((3, 3), bool),
                proposal_flow_parent=0 if flow_present else -1,
                node_missing=np.where(direct, 0, age[pi] + 1),
                direct_pectoral_age=np.where(direct[:, 1], pect_age[pi] + 1, 0),
            )
        )
    # Scalar extraction of Unicode would otherwise narrow its storage width.
    result = {key: np.asarray([r[key] for r in records]).copy() for key in RECORD_FIELDS}
    result["proposal_kind"] = kind[selected].copy()
    return result


def compare_records(actual, expected):
    """Check all19 arrays; fixed atol=1e-10/rtol=0, signed infinities exact.

    Numeric/bool dtypes and all shapes must agree. Unicode values must agree;
    only storage width may differ, and that difference is explicitly reported.
    NaNs are forbidden, including when both arrays contain NaN. Returned numeric
    error summaries compare finite entries only and never hide an infinity.
    """
    require(set(actual) == set(expected) == set(RECORD_FIELDS), "all19 record keys required")
    floats, widths = {}, {}
    batch = None
    shapes = {
        "pose_xyz": (3, 3, 3),
        "observed": (3, 3),
        "sources": (3, 3, 3),
        "identity_review": (3,),
        "mask_supported": (3,),
        "body_supported": (3,),
        "missing": (3,),
        "loss_parts": (5,),
        "proposal_flow_nodes": (3, 3),
        "node_missing": (3, 3),
        "direct_pectoral_age": (3,),
    }
    for key in RECORD_FIELDS:
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        require(
            a.ndim >= 1 and a.shape == b.shape and a.shape[1:] == shapes.get(key, ()),
            "record shape mismatch: " + key,
        )
        if batch is None:
            batch = len(a)
        require(len(a) == batch and batch > 0, "record batch mismatch: " + key)
        if key == "proposal_kind":
            require(
                a.dtype.kind == b.dtype.kind == "U" and np.array_equal(a, b),
                "record string mismatch: " + key,
            )
            if a.dtype != b.dtype:
                widths[key] = dict(actual_dtype=a.dtype.str, expected_dtype=b.dtype.str)
            continue
        kind = "f" if key in FLOAT_FIELDS else "b" if key in BOOL_FIELDS else "i"
        require(a.dtype == b.dtype and a.dtype.kind == kind, "record dtype mismatch: " + key)
        if key in FLOAT_FIELDS:
            require(not np.isnan(a).any() and not np.isnan(b).any(), "NaN record: " + key)
            require(
                np.array_equal(np.isposinf(a), np.isposinf(b))
                and np.array_equal(np.isneginf(a), np.isneginf(b)),
                "record infinity mismatch: " + key,
            )
            finite = np.isfinite(a)
            require(key == "score_margin" or finite.all(), "nonfinite record: " + key)
            with np.errstate(over="ignore", invalid="ignore"):
                difference = np.abs(a[finite] - b[finite])
            maximum = float(difference.max(initial=0))
            require(maximum <= ATOL, "record numeric mismatch: " + key)
            floats[key] = dict(
                maximum_absolute_difference=maximum, byte_exact=a.tobytes() == b.tobytes()
            )
        else:
            require(np.array_equal(a, b), "record discrete mismatch: " + key)
    return dict(
        status="ok",
        records=batch,
        fields=19,
        atol=ATOL,
        rtol=0,
        floating_fields=floats,
        unicode_width_differences=widths,
        all_discrete_values_exact=True,
    )
