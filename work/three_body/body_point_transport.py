"""Mass-preserving transport from admitted masks to their own initial group's points.

Pure NumPy; no pose, renderer, model, file or device execution is an input.
The frozen whole-mask packet decides admission. Only the spatial support of
owner+gap changes: each body/view's old W*y*T mass is divided by its original
single-member count, and invalid points receive zero without renormalization.
Checkpoint validation proves saved algebra; native mask/file provenance is the
sealed caller's responsibility. With rig supplied, saved query rays are checked
against the original renderer's CPU ray construction, without rendering a pose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass

import numpy as np
import object_mask_transport as old
from temporal_point_core import mask_memberships

SCHEMA = "body_point_transport_v1"
PLAN_SHA256 = "1409a0214ac2c8c2c3c17e13e6497b004290b9b844c8963f74fb258bcbbc15cb"
VIEWS, require = old.VIEWS, old.require
freeze, signature = old._freeze, old.signature


def _same(left, right, label):
    if is_dataclass(left):
        left = asdict(left)
    if is_dataclass(right):
        right = asdict(right)
    if isinstance(left, np.ndarray):
        require(isinstance(right, np.ndarray), label + " array missing")
        require(signature(left) == signature(right), label + " array bytes differ")
    elif isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), label + " keys differ")
        for key in left:
            _same(left[key], right[key], label + "/" + str(key))
    elif isinstance(left, (list, tuple)):
        require(
            isinstance(right, (list, tuple)) and len(left) == len(right), label + " length differs"
        )
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            _same(a, b, label + "/" + str(i))
    else:
        require(type(left) is type(right) and left == right, label + " value differs")


def _record(value, cls):
    value = asdict(value) if is_dataclass(value) else value
    require(
        isinstance(value, dict) and set(value) == {f.name for f in fields(cls)},
        cls.__name__ + " fields differ",
    )
    return value


def _vote(value):
    record = _record(value, old.VoteEvidence)
    result = old.transport_votes(
        record["query_group"],
        record["reported"],
        record["membership"],
        group_count=record["group_count"],
        mask_available=record["mask_available"],
        native=record["native"],
        release_horizon=record["release_horizon"],
        query_sha256=record["query_sha256"],
    )
    _same(result, record, "whole-mask vote")
    return result


@dataclass(frozen=True, slots=True)
class PointEvidence:
    vote: old.VoteEvidence
    xy: np.ndarray

    def __post_init__(self):
        vote = _vote(self.vote)
        require(
            isinstance(self.xy, np.ndarray)
            and self.xy.dtype == np.float32
            and self.xy.shape == (len(vote.query_group), 2),
            "all original float32 query coordinates required",
        )
        object.__setattr__(self, "vote", vote)
        object.__setattr__(self, "xy", freeze(self.xy))


class SavedBodyPointEvidence:
    """Wrap the frozen SavedPointEvidence, preserving its verification and cursor."""

    def __init__(self, saved):
        self.saved = saved
        self.group_counts = tuple(saved.group_counts)
        self.query_sha256 = tuple(saved.query_sha256)
        self._current = None

    def prepare(self, frame):
        self._current = None
        self.saved.prepare(frame)
        self._current = int(frame.frame_idx)

    def provider(self, method):
        require(method in ("lk", "ct_first"), "body point evidence arm differs")
        provider = self.saved.provider(method)

        def provide(native):
            require(type(native) is int and self._current == native, "body point cursor differs")
            votes = provider(native)
            result = []
            for vi, view in enumerate(VIEWS):
                query = self.saved.queries[view]
                first = int(query["frame_idx"])
                offset = native - first
                coords = self.saved.coords[method, view]
                flags = self.saved.flags[method, view]
                require(0 <= offset < len(coords), "body point native outside saved coordinates")
                vote = votes[vi]
                require(
                    vote.native == native
                    and vote.query_sha256 == self.query_sha256[vi]
                    and vote.group_count == self.group_counts[vi],
                    "body point query/native binding differs",
                )
                _same(query["mask_index"], vote.query_group, "original query group")
                _same(flags[offset], vote.reported, "same-native reported")
                result.append(PointEvidence(vote, coords[offset]))
            return tuple(result)

        return provide


def _point_layout(evidence):
    require(len(evidence) == 3, "three PointEvidence views required")
    require(all(type(e) is PointEvidence for e in evidence), "PointEvidence required")
    counts = np.array([len(e.xy) for e in evidence], np.int64)
    pixels = np.full((3, int(counts.max(initial=0)), 2), np.nan, np.float64)
    for v, ev in enumerate(evidence):
        pixels[v, : counts[v]] = ev.xy.astype(np.float64)
    return counts, pixels


def _rays(rig, pixels, counts):
    """Original RefractiveFishRenderer.__init__ ray order and invalid defaults."""
    require(tuple(rig.views) == VIEWS and rig.world_units == "mm", "same ordered mm rig required")
    origins = np.zeros((*pixels.shape[:2], 3), np.float64)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1
    valid = np.zeros(pixels.shape[:2], bool)
    for index, view in enumerate(VIEWS):
        supplied = pixels[index, : counts[index]]
        finite = np.isfinite(supplied).all(axis=1)
        if finite.any():
            ray_o, ray_d = rig.camera(view).rays(supplied[finite])
            require(
                np.shape(ray_o) == np.shape(ray_d) == (int(finite.sum()), 3),
                "calibration ray shape differs",
            )
            length = np.linalg.norm(ray_d, axis=1)
            good = np.isfinite(ray_o).all(axis=1) & np.isfinite(ray_d).all(axis=1)
            good &= length > 1e-12
            locations = np.flatnonzero(finite)[good]
            origins[index, locations] = ray_o[good]
            directions[index, locations] = ray_d[good] / length[good, None]
            valid[index, locations] = True
    return origins, directions, valid


@dataclass(frozen=True, slots=True)
class BodyPointPacket:
    schema: str
    plan_sha256: str
    native: int
    initial_native: int
    anchor_ids: np.ndarray
    release_horizons: np.ndarray
    query_sha256: tuple[str, ...]
    pixels: np.ndarray
    pixels_signature: old.ArraySignature
    whole_mask_packet: old.TargetPacket
    point_evidence: tuple[PointEvidence, ...]
    fit_foreground: np.ndarray
    fit_weights: np.ndarray
    fit_ray_valid: np.ndarray
    point_pixels: np.ndarray
    point_weight: np.ndarray
    ray_origins: np.ndarray
    ray_directions: np.ndarray
    ray_valid: np.ndarray
    query_counts: np.ndarray
    native_inside: np.ndarray
    point_selected: np.ndarray
    single_count: np.ndarray
    valid_single_count: np.ndarray
    old_mass: np.ndarray
    effective_mass: np.ndarray
    array_signatures: tuple[tuple[str, old.ArraySignature], ...]


def _assemble(mask_packet, evidence, foreground, weights, fit_valid, ray_bundle):
    counts, pixels = _point_layout(evidence)
    origins, directions, valid = ray_bundle
    shape = mask_packet.target.shape[1:3]
    require(
        isinstance(foreground, np.ndarray)
        and isinstance(weights, np.ndarray)
        and isinstance(fit_valid, np.ndarray)
        and foreground.shape == weights.shape == fit_valid.shape == shape
        and fit_valid.dtype == bool,
        "same old fit foreground/weight/ray schema required",
    )
    require(
        foreground.dtype.kind in "biuf"
        and weights.dtype.kind in "iuf"
        and np.isfinite(foreground).all()
        and np.isfinite(weights).all()
        and ((foreground == 0) | (foreground == 1)).all()
        and (weights >= 0).all(),
        "finite binary foreground and nonnegative old weights required",
    )
    require(
        origins.dtype == directions.dtype == np.float64
        and origins.shape == directions.shape == (*pixels.shape[:2], 3)
        and valid.dtype == bool
        and valid.shape == pixels.shape[:2]
        and np.isfinite(origins).all()
        and np.isfinite(directions).all(),
        "query ray bundle schema differs",
    )
    defaults = np.zeros((int((~valid).sum()), 3), np.float64)
    defaults[:, 2] = 1
    require(
        not origins[~valid].any() and np.array_equal(directions[~valid], defaults),
        "invalid query ray sentinel differs",
    )
    finite = np.isfinite(pixels).all(axis=-1)
    require(not np.any(valid & ~finite), "nonfinite query declared ray-valid")
    h, w = mask_packet.native_shape_hw
    inside = finite & (pixels[..., 0] >= 0) & (pixels[..., 0] < w)
    inside &= (pixels[..., 1] >= 0) & (pixels[..., 1] < h)
    # Exactly the independent old NumPy mass expression; no upper-mass bound.
    weight = weights.astype(np.float64) * fit_valid
    mass = (mask_packet.target * (weight * foreground)[None, :, :, None]).sum(axis=2)
    require(np.isfinite(mass).all() and (mass >= 0).all(), "nonfinite old mass")
    selected = np.zeros((len(mask_packet.anchors), 3, pixels.shape[1], 3), bool)
    single = np.zeros(mass.shape, np.int64)
    for a, decisions in enumerate(mask_packet.assignments):
        for v, decision in enumerate(decisions):
            ev = evidence[v].vote
            _same(ev, decision.evidence, "point/whole-mask evidence")
            for b, row in enumerate(decision.body_mask_row):
                if row < 0:
                    continue
                groups = np.flatnonzero(decision.initial_links[:, b])
                require(len(groups) == 1, "admitted body lacks unique initial group")
                group = int(groups[0])
                chosen = (
                    (ev.query_group == group)
                    & ev.reported
                    & (ev.membership_degree == 1)
                    & ev.membership[:, row]
                )
                count = int(ev.single_votes[group, row])
                require(count >= 2 and int(chosen.sum()) == count, "original single count differs")
                selected[a, v, : counts[v], b] = chosen
                single[a, v, b] = count
    require((selected.sum(axis=-1) <= 1).all(), "query assigned to multiple bodies")
    unit = np.zeros(mass.shape, np.float64)
    np.divide(mass, single, out=unit, where=single > 0)
    usable = selected & (valid & inside)[None, :, :, None]
    point_weight = np.where(usable, unit[:, :, None, :], 0.0)
    require(
        np.isfinite(point_weight).all() and (point_weight >= 0).all(),
        "point weights must be finite and nonnegative; no upper-one restriction",
    )
    arrays = dict(
        anchor_ids=mask_packet.anchor_ids,
        release_horizons=mask_packet.release_horizons,
        pixels=mask_packet.pixels,
        fit_foreground=foreground,
        fit_weights=weights,
        fit_ray_valid=fit_valid,
        point_pixels=pixels,
        point_weight=point_weight,
        ray_origins=origins,
        ray_directions=directions,
        ray_valid=valid,
        query_counts=counts,
        native_inside=inside,
        point_selected=selected,
        single_count=single,
        valid_single_count=usable.sum(axis=2, dtype=np.int64),
        old_mass=mass,
        effective_mass=point_weight.sum(axis=2),
    )
    return BodyPointPacket(
        schema=SCHEMA,
        plan_sha256=PLAN_SHA256,
        native=mask_packet.native,
        initial_native=mask_packet.initial_native,
        query_sha256=mask_packet.query_sha256,
        pixels_signature=mask_packet.pixels_signature,
        whole_mask_packet=mask_packet,
        point_evidence=tuple(evidence),
        array_signatures=tuple((key, signature(value)) for key, value in arrays.items()),
        **{key: freeze(value) for key, value in arrays.items()},
    )


def build_packet(
    point_evidence,
    anchors,
    masks_by_view,
    fit_pixels,
    *,
    fit_foreground,
    fit_weights,
    fit_ray_valid,
    rig,
    native_shape_hw=(1200, 1200),
):
    """Build the frozen old packet first, then distribute exactly its mass m/S."""
    evidence = tuple(PointEvidence(e.vote, e.xy) for e in point_evidence)
    counts, pixels = _point_layout(evidence)
    require(len(masks_by_view) == 3, "all native masks required")
    for ev, masks in zip(evidence, masks_by_view, strict=True):
        actual = mask_memberships(ev.xy, masks, native_shape_hw)
        require(actual["available"] is ev.vote.mask_available, "point mask availability differs")
        _same(actual["membership"], ev.vote.membership, "current point/native membership")
    mask_packet = old.packet_for_pixels(
        tuple(e.vote for e in evidence),
        anchors,
        masks_by_view,
        fit_pixels,
        native_shape_hw=native_shape_hw,
    )
    # Canonical t0 anchors are independently reconstructed from their saved raw
    # strict references. Full pose/rig qualification remains the tracker caller.
    mask_packet = _canonical_mask(mask_packet)
    return _assemble(
        mask_packet,
        evidence,
        fit_foreground,
        fit_weights,
        fit_ray_valid,
        _rays(rig, pixels, counts),
    )


def _canonical_mask(value):
    """Rebuild saved assignment/T from raw votes, strict refs and sampled masks."""
    record = _record(value, old.TargetPacket)
    anchors = []
    for saved in record["anchors"]:
        saved = _record(saved, old.InitialAnchor)
        anchor = old.build_initial_anchor(
            [json.loads(text) for text in saved["references_json"]],
            [len(x) for x in saved["links"]],
            anchor_id=saved["anchor_id"],
            initial_native=saved["initial_native"],
        )
        _same(anchor, saved, "initial anchor")
        anchors.append(anchor)
    require(1 <= len(anchors) <= 6, "all initial anchors required")
    require(len(record["assignments"]) == len(anchors), "initial assignment count differs")
    decisions, votes = [], None
    for anchor, saved_views in zip(anchors, record["assignments"], strict=True):
        require(len(saved_views) == 3, "three saved assignments required")
        current_votes, current_decisions = [], []
        for v, saved in enumerate(saved_views):
            saved = _record(saved, old.Assignment)
            vote = _vote(saved["evidence"])
            decision = old.assign_transport(vote, anchor.links[v])
            _same(decision, saved, "frozen mask admission")
            current_votes.append(vote)
            current_decisions.append(decision)
        if votes is not None:
            _same(current_votes, votes, "same raw evidence across anchors")
        votes = current_votes
        decisions.append(tuple(current_decisions))
    pixels = record["pixels"]
    h, w = record["native_shape_hw"]
    require(
        isinstance(pixels, np.ndarray)
        and pixels.ndim == 3
        and pixels.shape[0] == 3
        and pixels.shape[2] == 2
        and np.isfinite(pixels).all()
        and np.equal(pixels, np.floor(pixels)).all()
        and (pixels[..., 0] >= 0).all()
        and (pixels[..., 0] < w).all()
        and (pixels[..., 1] >= 0).all()
        and (pixels[..., 1] < h).all(),
        "saved original sample pixels differ",
    )
    require(
        len(record["sampled_mask_membership"]) == len(record["mask_signatures"]) == 3,
        "sampled all-mask views differ",
    )
    sampled, degrees, mask_sigs = [], [], []
    for v, ev in enumerate(votes):
        member = record["sampled_mask_membership"][v]
        require(
            isinstance(member, np.ndarray)
            and member.dtype == bool
            and member.shape == (pixels.shape[1], ev.membership.shape[1]),
            "sampled all-N membership differs",
        )
        degree = member.sum(axis=1, dtype=np.int64)
        saved_sig = record["mask_signatures"][v]
        if not ev.mask_available:
            require(saved_sig is None and member.shape[1] == 0, "unavailable mask differs")
            degree.fill(-1)
            mask_sigs.append(None)
        else:
            saved_sig = _record(saved_sig, old.ArraySignature)
            old._sha(saved_sig["sha256"])
            sig = old.ArraySignature(
                saved_sig["dtype"], tuple(saved_sig["shape"]), saved_sig["sha256"]
            )
            require(
                sig.dtype == np.dtype(bool).str and sig.shape == (member.shape[1], h, w),
                "native mask signature differs",
            )
            mask_sigs.append(sig)
        sampled.append(freeze(member))
        degrees.append(degree)
    target = np.zeros((len(anchors), 3, pixels.shape[1], 3), np.float64)
    for a, views in enumerate(decisions):
        for v, decision in enumerate(views):
            for b, row in enumerate(decision.body_mask_row):
                if row >= 0:
                    chosen = sampled[v][:, row] & (degrees[v] == 1)
                    target[a, v, chosen, b] = decision.body_weight[b]
    require((np.count_nonzero(target, axis=-1) <= 1).all(), "sampled pixel collision")
    native, initial = votes[0].native, anchors[0].initial_native
    require(
        all(ev.native == native for ev in votes)
        and all(a.initial_native == initial for a in anchors)
        and native >= initial
        and len({a.anchor_id for a in anchors}) == len(anchors),
        "saved native/anchor lineage differs",
    )
    result = old.TargetPacket(
        old.SCHEMA,
        native,
        freeze(np.array([ev.release_horizon for ev in votes], np.int64)),
        initial,
        freeze(np.array([a.anchor_id for a in anchors], np.int64)),
        freeze(target),
        freeze(target.sum(axis=2)),
        freeze(np.count_nonzero(target, axis=2)),
        tuple(decisions),
        tuple(anchors),
        freeze(pixels),
        signature(pixels),
        tuple(mask_sigs),
        tuple(ev.query_sha256 for ev in votes),
        tuple(signature(ev.query_group) for ev in votes),
        tuple(signature(ev.reported) for ev in votes),
        tuple(signature(ev.membership) for ev in votes),
        tuple(sampled),
        freeze(np.array(degrees)),
        (h, w),
    )
    _same(result, record, "whole-mask packet")
    return result


def validate_packet(value, *, rig=None):
    """Return owned immutable canonical packet; no fit/state restore is performed.

    Without rig this checks saved ray signatures/defaults and full packet algebra.
    With rig, additionally recompute query rays byte-for-byte. Native mask bytes
    are deliberately not duplicated here: caller must bind their sealed SHA.
    """
    record = _record(value, BodyPointPacket)
    mask_packet = _canonical_mask(record["whole_mask_packet"])
    evidence = []
    for saved in record["point_evidence"]:
        saved = _record(saved, PointEvidence)
        evidence.append(PointEvidence(saved["vote"], saved["xy"]))
    counts, pixels = _point_layout(evidence)
    bundle = tuple(record[key] for key in ("ray_origins", "ray_directions", "ray_valid"))
    if rig is not None:
        _same(_rays(rig, pixels, counts), bundle, "calibrated query rays")
    result = _assemble(
        mask_packet,
        evidence,
        record["fit_foreground"],
        record["fit_weights"],
        record["fit_ray_valid"],
        bundle,
    )
    _same(result, record, "body point packet")
    return result
