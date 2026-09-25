"""Candidate-independent, conservative whole-mask transport (pure NumPy CPU).

Initial geometry belongs to the caller: at its causal t0 retained pose, use the
frozen scene9/body33 projection, coherence_metrics and
diagnose_identity_components.geometry_reference with original strict sources
AND global observed. build_initial_anchor only consumes that decision through
the frozen initial_link_matrix; it never uses the current-view-only diagnostic.
Neither current pose, body coverage, candidate rank nor permutation is an input
to target construction. Targets are conditional observations, not physical IDs.

All arrays returned here own immutable bytes backing; changing caller arrays or
turning WRITEABLE back on cannot change the packet. File/provenance sealing and
checking point membership against native masks remain the stream caller's job.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from temporal_body_mask_evidence import initial_link_matrix

SCHEMA = "object_mask_transport_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
INITIAL_LINK_HELPER_SHA256 = "3e1ae044ee73972dfda2c4250aa50a6370e60181cd37ec49b0fec084fdd6d216"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    require(
        type(value) is int and value >= minimum, f"{name}: Python integer >= {minimum} required"
    )


def _sha(value):
    require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "lowercase SHA256 required",
    )


def _array(value, name, kind):
    require(isinstance(value, np.ndarray) and value.dtype.kind in kind, f"{name}: ndarray {kind}")
    return value


def _freeze(value):
    a = np.asarray(value)
    require(not a.dtype.hasobject, "object arrays forbidden")
    return np.frombuffer(a.tobytes(order="C"), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, slots=True)
class ArraySignature:
    dtype: str
    shape: tuple[int, ...]
    sha256: str


def signature(value):
    a = np.asarray(value)
    require(not a.dtype.hasobject, "object arrays forbidden")
    return ArraySignature(a.dtype.str, a.shape, hashlib.sha256(a.tobytes(order="C")).hexdigest())


@dataclass(frozen=True, slots=True)
class VoteEvidence:
    native: int
    release_horizon: int
    group_count: int
    mask_available: bool
    query_sha256: str
    query_group: np.ndarray
    reported: np.ndarray
    membership: np.ndarray
    reported_count: np.ndarray
    membership_degree: np.ndarray
    single_count: np.ndarray
    multi_count: np.ndarray
    unassigned_count: np.ndarray
    votes: np.ndarray
    single_votes: np.ndarray
    weights: np.ndarray
    vote_maximum_ties: np.ndarray
    single_maximum_ties: np.ndarray


def transport_votes(
    query_group,
    reported,
    membership,
    *,
    group_count,
    mask_available,
    native,
    release_horizon,
    query_sha256,
):
    """Retain all groups/N, all reported denominators and exact zero/max ties.

    membership is all-N native point membership, including multi membership.
    Nonfinite/outside points have no membership but remain in reported_count.
    Unavailable has N=0 and degree/single/multi/unassigned=-1 (unknown), whereas
    available N=0 has degree/single/multi=0 and unassigned=reported_count.
    Weights have NaN for R=0; unavailable/empty rows keep their zero-width shape.
    """
    _integer(group_count, "group_count")
    _integer(native, "native")
    _integer(release_horizon, "release_horizon", native)
    _sha(query_sha256)
    require(type(mask_available) is bool, "explicit mask availability required")
    groups = _array(query_group, "query_group", "iu")
    report = _array(reported, "reported", "b")
    member = _array(membership, "membership", "b")
    require(groups.ndim == 1 and report.shape == groups.shape, "query/report shape mismatch")
    require(member.ndim == 2 and len(member) == len(groups), "all-N membership shape mismatch")
    require(
        np.all(groups < group_count) and np.all(groups >= 0), "query group outside initial rows"
    )
    n = member.shape[1]
    require(mask_available or n == 0, "unavailable masks cannot contain rows")
    r = np.bincount(groups[report].astype(np.int64), minlength=group_count).astype(np.int64)
    degree = member.sum(axis=1, dtype=np.int64)
    votes = np.zeros((group_count, n), np.int64)
    singles = np.zeros_like(votes)
    single_count = np.zeros(group_count, np.int64)
    multi = np.zeros_like(single_count)
    unassigned = np.zeros_like(single_count)
    for g in range(group_count):
        selected = report & (groups == g)
        one = selected & (degree == 1)
        votes[g] = member[selected].sum(axis=0, dtype=np.int64)
        singles[g] = member[one].sum(axis=0, dtype=np.int64)
        single_count[g] = int(one.sum())
        multi[g] = int(np.count_nonzero(selected & (degree > 1)))
        unassigned[g] = int(np.count_nonzero(selected & (degree == 0)))
    weights = np.full((group_count, n), np.nan, np.float64)
    np.divide(singles, r[:, None], out=weights, where=r[:, None] > 0)
    vote_ties = votes == votes.max(axis=1, keepdims=True) if n else np.zeros_like(votes, bool)
    single_ties = (
        singles == singles.max(axis=1, keepdims=True) if n else np.zeros_like(singles, bool)
    )
    if not mask_available:
        degree.fill(-1)
        single_count.fill(-1)
        multi.fill(-1)
        unassigned.fill(-1)
    return VoteEvidence(
        native,
        release_horizon,
        group_count,
        mask_available,
        query_sha256,
        *map(
            _freeze,
            (
                groups,
                report,
                member,
                r,
                degree,
                single_count,
                multi,
                unassigned,
                votes,
                singles,
                weights,
                vote_ties,
                single_ties,
            ),
        ),
    )


def _links(value, group_count):
    links = _array(value, "initial_links", "b")
    require(links.shape == (group_count, 3), "initial links must retain G,3 cells")
    require(np.all(links.sum(axis=1) <= 1), "initial group linked to multiple bodies")
    require(np.all(links.sum(axis=0) <= 1), "initial body linked to multiple groups")
    return links


@dataclass(frozen=True, slots=True)
class Assignment:
    evidence: VoteEvidence
    initial_links: np.ndarray
    assignment: np.ndarray
    group_body: np.ndarray
    candidate_mask_row: np.ndarray
    group_admitted: np.ndarray
    group_reasons: tuple[tuple[str, ...], ...]
    body_mask_row: np.ndarray
    body_weight: np.ndarray
    body_reasons: tuple[tuple[str, ...], ...]
    competing_group_mask: np.ndarray
    unknown_competing_group_mask: np.ndarray
    linked_competing_group_mask: np.ndarray


def assign_transport(votes, initial_links):
    """One positive single-vote mask with >=2 votes, no other-group ANY votes.

    A split is never rescued by argmax. Unknown groups and multi memberships
    veto alongside known groups. Raw evidence is retained even after abstention.
    body_weight=0 is an inactive coefficient, never a replacement for raw NaN.
    """
    require(type(votes) is VoteEvidence, "VoteEvidence required")
    links = _links(initial_links, votes.group_count)
    g, n = votes.votes.shape
    known = links.any(axis=1)
    groups = np.where(known, links.argmax(axis=1), -1).astype(np.int64)
    other = votes.votes.sum(axis=0, dtype=np.int64)[None, :] - votes.votes
    unknown = (
        votes.votes[~known].sum(axis=0, dtype=np.int64)[None, :] - votes.votes * (~known)[:, None]
    )
    linked = votes.votes[known].sum(axis=0, dtype=np.int64)[None, :] - votes.votes * known[:, None]
    candidates = np.full(g, -1, np.int64)
    admitted = np.zeros(g, bool)
    claimed = np.zeros((n, 3), bool)
    body_rows = np.full(3, -1, np.int64)
    body_weights = np.zeros(3, np.float64)
    group_reasons = []
    body_reasons = [("no_strict_initial_link",)] * 3
    for group in range(g):
        reasons = []
        if not known[group]:
            reasons.append("no_strict_initial_link")
        if not votes.mask_available:
            reasons.append("masks_unavailable")
        elif n == 0:
            reasons.append("no_current_masks")
        if votes.reported_count[group] < 2:
            reasons.append("fewer_than_two_reported_points")
        if votes.mask_available and votes.single_count[group] < 2:
            reasons.append("fewer_than_two_single_members")
        positive = np.flatnonzero(votes.single_votes[group] > 0)
        if not len(positive):
            reasons.append("no_positive_single_mask")
        elif len(positive) > 1:
            reasons.append("split_positive_masks")
        else:
            row = int(positive[0])
            candidates[group] = row
            if votes.single_votes[group, row] < 2:
                reasons.append("fewer_than_two_single_votes_in_mask")
            if unknown[group, row] > 0:
                reasons.append("mask_has_unknown_group_votes")
            if linked[group, row] > 0:
                reasons.append("mask_has_other_linked_group_votes")
        why = tuple(reasons)
        group_reasons.append(why)
        if known[group]:
            body_reasons[int(groups[group])] = why
        if not reasons:
            row, body = candidates[group], groups[group]
            admitted[group] = True
            claimed[row, body] = True
            body_rows[body] = row
            body_weights[body] = votes.weights[group, row]
    require(np.all(claimed.sum(axis=1) <= 1), "current mask claimed by competing bodies")
    require(np.all(claimed.sum(axis=0) <= 1), "body claims multiple current masks")
    return Assignment(
        votes,
        _freeze(links),
        _freeze(claimed),
        _freeze(groups),
        _freeze(candidates),
        _freeze(admitted),
        tuple(group_reasons),
        _freeze(body_rows),
        _freeze(body_weights),
        tuple(body_reasons),
        _freeze(other > 0),
        _freeze(unknown > 0),
        _freeze(linked > 0),
    )


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        s = signature(value)
        return dict(dtype=s.dtype, shape=s.shape, sha256=s.sha256)
    raise TypeError(f"unsupported initial reference value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class InitialAnchor:
    anchor_id: int
    initial_native: int
    links: tuple[np.ndarray, ...]
    references_json: tuple[str, ...]
    reference_sha256: tuple[str, ...]


def build_initial_anchor(references, group_counts, *, anchor_id, initial_native):
    """Freeze already-qualified causal t0 references; do not reproject a pose.

    The caller must supply strict geometry_reference outputs in VIEWS order,
    not current single-view reference outputs or retrospective winner labels.
    Full ambiguous references remain in references_json; links keep all Gx3.
    """
    _integer(anchor_id, "anchor_id")
    _integer(initial_native, "initial_native")
    require(len(references) == len(group_counts) == 3, "three initial views required")
    result, encoded = [], []
    for ref, count in zip(references, group_counts, strict=True):
        require(type(ref) is dict, "geometry reference dict required")
        require(
            ref.get("scope") != "single_current_view_geometry",
            "current-view reference is not a t0 link",
        )
        matrix = initial_link_matrix(ref, count)
        result.append(_freeze(_links(matrix, count)))
        encoded.append(
            json.dumps(ref, sort_keys=True, separators=(",", ":"), default=_json_default)
        )
    return InitialAnchor(
        anchor_id,
        initial_native,
        tuple(result),
        tuple(encoded),
        tuple(hashlib.sha256(item.encode("utf-8")).hexdigest() for item in encoded),
    )


@dataclass(frozen=True, slots=True)
class TargetPacket:
    schema: str
    native: int
    release_horizons: np.ndarray
    initial_native: int
    anchor_ids: np.ndarray
    target: np.ndarray
    target_mass: np.ndarray
    target_nonzero_count: np.ndarray
    assignments: tuple[tuple[Assignment, ...], ...]
    anchors: tuple[InitialAnchor, ...]
    pixels: np.ndarray
    pixels_signature: ArraySignature
    mask_signatures: tuple[ArraySignature | None, ...]
    query_sha256: tuple[str, ...]
    query_group_signatures: tuple[ArraySignature, ...]
    reported_signatures: tuple[ArraySignature, ...]
    membership_signatures: tuple[ArraySignature, ...]
    sampled_mask_membership: tuple[np.ndarray, ...]
    sampled_mask_degree: np.ndarray
    native_shape_hw: tuple[int, int]


def packet_for_pixels(
    votes_by_view,
    anchors,
    masks_by_view,
    pixels,
    *,
    native_shape_hw=(1200, 1200),
):
    """T[A,V,P,body], retaining all anchor/view decisions and all-N pixel overlap.

    Pixels must be the exact integer native coordinates supplied to the fitter;
    repeats are retained. target_mass[A,V,body] is SUM T, not renderer-weighted
    mass. The fitter additionally multiplies old weights/foreground/ray_valid.
    Native masks are None (unavailable), or bool[N,H,W] including N=0.
    t0 and lambda=0 dispatch are the tracker/fitter caller's explicit policy.
    This packet is identical for all candidates sharing an initial anchor.
    """
    require(len(votes_by_view) == len(masks_by_view) == 3, "three current views required")
    require(1 <= len(anchors) <= 6, "one to six initial anchors required")
    require(all(type(v) is VoteEvidence for v in votes_by_view), "VoteEvidence per view required")
    require(all(type(a) is InitialAnchor for a in anchors), "InitialAnchor required")
    require(len(set(a.anchor_id for a in anchors)) == len(anchors), "duplicate anchor ids")
    native = votes_by_view[0].native
    initial = anchors[0].initial_native
    require(all(v.native == native for v in votes_by_view), "current native differs across views")
    require(
        all(a.initial_native == initial for a in anchors) and native >= initial,
        "initial native mismatch",
    )
    require(len(native_shape_hw) == 2, "native H,W required")
    for value in native_shape_hw:
        _integer(value, "native shape", 1)
    height, width = native_shape_hw
    px = _array(pixels, "pixels", "fiu")
    require(px.ndim == 3 and px.shape[0] == 3 and px.shape[2] == 2, "pixels must be V,P,2")
    require(
        np.isfinite(px).all() and np.equal(px, np.floor(px)).all(),
        "integer finite native pixels required",
    )
    require(np.all((px[..., 0] >= 0) & (px[..., 0] < width)), "native pixel x outside image")
    require(np.all((px[..., 1] >= 0) & (px[..., 1] < height)), "native pixel y outside image")
    sampled, degrees, mask_sigs = [], [], []
    for v, (ev, masks) in enumerate(zip(votes_by_view, masks_by_view, strict=True)):
        n = ev.membership.shape[1]
        if masks is None:
            require(not ev.mask_available and n == 0, "None masks differ from availability")
            member = np.zeros((px.shape[1], 0), bool)
            degree = np.full(px.shape[1], -1, np.int64)
            mask_sigs.append(None)
        else:
            mask = _array(masks, "masks", "b")
            require(ev.mask_available, "unavailable masks must be None")
            require(mask.shape == (n, height, width), "all-N mask shape/count mismatch")
            xy = px[v].astype(np.int64)
            member = mask[:, xy[:, 1], xy[:, 0]].T
            degree = member.sum(axis=1, dtype=np.int64)
            mask_sigs.append(signature(mask))
        sampled.append(_freeze(member))
        degrees.append(degree)
    target = np.zeros((len(anchors), 3, px.shape[1], 3), np.float64)
    assignments = []
    for a, anchor in enumerate(anchors):
        require(len(anchor.links) == 3, "anchor must retain three views")
        decisions = []
        for v, ev in enumerate(votes_by_view):
            decision = assign_transport(ev, anchor.links[v])
            decisions.append(decision)
            for body, row in enumerate(decision.body_mask_row):
                if row >= 0:
                    chosen = sampled[v][:, row] & (degrees[v] == 1)
                    target[a, v, chosen, body] = decision.body_weight[body]
        assignments.append(tuple(decisions))
    require(np.all(np.count_nonzero(target, axis=3) <= 1), "pixel assigned to multiple bodies")
    require(np.isfinite(target).all() and np.all(target >= 0), "invalid effective target")
    return TargetPacket(
        SCHEMA,
        native,
        _freeze(np.array([v.release_horizon for v in votes_by_view], np.int64)),
        initial,
        _freeze(np.array([a.anchor_id for a in anchors], np.int64)),
        _freeze(target),
        _freeze(target.sum(axis=2)),
        _freeze(np.count_nonzero(target, axis=2)),
        tuple(assignments),
        tuple(anchors),
        _freeze(px),
        signature(px),
        tuple(mask_sigs),
        tuple(v.query_sha256 for v in votes_by_view),
        tuple(signature(v.query_group) for v in votes_by_view),
        tuple(signature(v.reported) for v in votes_by_view),
        tuple(signature(v.membership) for v in votes_by_view),
        tuple(sampled),
        _freeze(np.stack(degrees)),
        tuple(native_shape_hw),
    )
