"""Bounded200 original tracker plus immutable t0 mask-transport anchors.

All proposals and beam/state updates use the frozen continuous tracker. Each
initial retained hypothesis gets one anchor; later target body axes follow that
parent lineage and are never relabelled by a proposal permutation. A provider
receives only the native index, not live poses, candidates or retained winners.
The initial and terminal bundle APIs bind the original state to the sidecar.
No file, model, process or GPU execution is initiated by importing this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, is_dataclass

import diagnose_temporal_refractive_geometry_v2 as initial_geometry
import mask_transport_fitter as fitter
import numpy as np
import object_mask_transport as transport
from chest_candidate_diagnostics import selected_indices
from continuous_tracker import (
    RECORD_FIELDS,
    AcceleratedContinuousPersistentNodeTracker,
    ContinuousPersistentNodeTracker,
)
from node_geometry import RayObservations, supported_nodes

require = fitter.require
VIEWS = initial_geometry.VIEWS
SCHEMA = "mask_transport_tracker_sidecar_v1"


def signature(value):
    a = np.asarray(value)
    return dict(
        dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest()
    )


def same(a, b, name):
    require(signature(a) == signature(b), name + " differs")


def same_record(a, b, name):
    """Flattened history can widen Unicode storage; textual values must agree."""
    if np.asarray(a).dtype.kind == np.asarray(b).dtype.kind == "U":
        require(np.array_equal(a, b), name + " differs")
    else:
        same(a, b, name)


def json_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _pack(value, arrays, key="root"):
    """Small lossless sidecar codec; original state arrays keep their own schema."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject and key not in arrays, "invalid sidecar array")
        arrays[key] = value.copy()
        return {"__array__": key}
    if isinstance(value, np.generic):
        return _pack(np.asarray(value), arrays, key)
    if isinstance(value, dict):
        require(all(isinstance(k, str) and k != "__array__" for k in value), "sidecar keys")
        return {k: _pack(v, arrays, f"{key}/{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(v, arrays, f"{key}/{i}") for i, v in enumerate(value)]
    require(value is None or type(value) in (str, int, bool, float), "sidecar scalar type")
    require(type(value) is not float or np.isfinite(value), "nonfinite sidecar scalar")
    return value


def _unpack(value, arrays, used):
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            key = value["__array__"]
            require(key in arrays and key not in used, "sidecar array reference differs")
            used.add(key)
            return arrays[key].copy()
        return {k: _unpack(v, arrays, used) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v, arrays, used) for v in value]
    return value


def decode_sidecar(arrays, metadata):
    """Lossless read for a caller after joint core/sidecar validation and sealing.

    This checks the sidecar's byte bindings, not scientific core-state validity;
    restore_transport_state supplies that independent semantic validation.
    """
    header = {k: v for k, v in metadata.items() if k != "content_sha256"}
    require(
        metadata.get("schema") == SCHEMA and json_hash(header) == metadata["content_sha256"],
        "sidecar content changed",
    )
    require(
        header["array_signatures"] == {k: signature(v) for k, v in arrays.items()},
        "sidecar arrays changed",
    )
    used = set()
    content = _unpack(header["tree"], arrays, used)
    require(used == set(arrays), "sidecar exact array membership differs")
    return content


def propagate_anchor_ids(previous, parents):
    require(
        previous.ndim == parents.ndim == 1 and previous.dtype == parents.dtype == np.int64,
        "anchor/parent int64 vectors required",
    )
    require(
        len(previous) >= 1
        and len(parents) >= 1
        and np.all(parents >= 0)
        and np.all(parents < len(previous)),
        "anchor parent index outside previous beam",
    )
    return previous[parents].copy()


def initial_anchors(records, frame, rig, group_counts):
    """Only causal t0 retained records and t0 masks/SL; no future path is read."""
    require(1 <= len(records) <= 6 and len(group_counts) == 3, "initial B0/group scope")
    rays = RayObservations.from_points(rig, frame.detections)
    anchors, facts = [], []
    for anchor_id, record in enumerate(records):
        require(int(record["frame_idx"]) == frame.frame_idx, "anchor native differs")
        pose, sources, observed = record["pose_xyz"], record["sources"], record["observed"]
        strict, repeated, _ = supported_nodes(pose, sources, rays, rig)
        same(strict, sources, "t0 strict source rows")
        same(repeated, observed, "t0 strict observed flags")
        references, arrays = [], []
        for vi, view in enumerate(VIEWS):
            masks = frame.masks[view]
            available = masks is not None
            if not available:
                masks = np.zeros((0, 1200, 1200), bool)
            require(len(masks) == group_counts[vi], "t0 mask groups differ from fixed queries")
            data, info = initial_geometry.geometry_frame(
                pose,
                observed,
                sources[:, vi],
                masks,
                available,
                rig.camera(view),
            )
            references.append(info["reference"])
            arrays.append(data)
        anchor = transport.build_initial_anchor(
            tuple(references),
            tuple(group_counts),
            anchor_id=anchor_id,
            initial_native=int(frame.frame_idx),
        )
        anchors.append(anchor)
        facts.append(
            dict(
                anchor_id=anchor_id,
                retained_index=anchor_id,
                record={k: np.asarray(record[k]).copy() for k in RECORD_FIELDS},
                references=initial_geometry.coh.encode_nonfinite(references),
                geometry=arrays,
            )
        )
    return tuple(anchors), facts


def check_retained_fit(records, capture, beam_size):
    selected, order = selected_indices(
        capture["pose"],
        capture["path_cost"],
        capture["parent"],
        capture["permutation"],
        beam_size,
    )
    require(len(selected) == len(records), "retained beam size differs")
    for record, i in zip(records, selected, strict=True):
        for key, value in (
            ("pose_xyz", capture["pose"][i]),
            ("costs", capture["cost"][i]),
            ("path_costs", capture["path_cost"][i]),
            ("loss_parts", capture["parts"][i]),
            ("mask_supported", capture["support"][i]),
        ):
            same(record[key], value, "retained fit " + key)
        require(
            int(record["parent"]) == int(capture["parent"][i])
            and str(record["proposal_kind"]) == str(capture["kind"][i]),
            "retained proposal origin differs",
        )
    return selected, order


class MaskTransportTracker(AcceleratedContinuousPersistentNodeTracker):
    """New branch; numeric base is original isotropic persistent+serial LK.

    vote_provider(native) returns the ordered three immutable VoteEvidence
    objects from a separately bound point stage. It has no candidate/state input.
    Caller input/source sealing remains required. Public bare-state export and
    restore are refused so an anchor sidecar cannot silently disappear.
    """

    def __init__(
        self,
        *args,
        first_native,
        group_counts,
        query_sha256,
        vote_provider,
        input_binding,
        coefficient=3.0,
        **kwargs,
    ):
        require(type(first_native) is int and first_native >= 0, "initial native required")
        require(
            len(group_counts) == len(query_sha256) == 3
            and all(type(n) is int and n >= 0 for n in group_counts),
            "three group counts",
        )
        require(
            all(
                isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)
                for s in query_sha256
            ),
            "fixed query SHA required",
        )
        require(callable(vote_provider), "saved vote provider required")
        require(type(coefficient) in (int, float) and coefficient in (0, 3), "fixed coefficient")
        self._first_native = first_native
        self._group_counts, self._query_sha256 = tuple(group_counts), tuple(query_sha256)
        self._vote_provider, self._coefficient = vote_provider, float(coefficient)
        self._binding = json.loads(json.dumps(input_binding, allow_nan=False))
        self._binding_sha = json_hash(self._binding)
        self._anchors, self._anchor_facts = (), []
        self._anchor_ids = np.empty(0, np.int64)
        self._anchor_history, self._fit_history = [], []
        self._initial_observation = None
        self._transport_frame = self._fit_capture = None
        super().__init__(*args, **kwargs)

    def _context(self):
        require(json_hash(self._binding) == self._binding_sha, "input binding changed")
        result = super()._context()
        result["mask_transport"] = dict(
            schema=SCHEMA,
            coefficient=self._coefficient,
            first_native=self._first_native,
            maximum_frames=200,
            group_counts=list(self._group_counts),
            query_sha256=list(self._query_sha256),
            input_binding=copy.deepcopy(self._binding),
            input_binding_sha256=self._binding_sha,
            target_body_axis="causal_t0_parent_lineage",
            proposal_permutation_relabels_target=False,
        )
        return result

    def _fit(self, proposals, states, rays, targets):
        frame = self._transport_frame
        require(frame is not None, "fit must be owned by one native advance")
        require(
            1 <= len(proposals) <= self.config.max_hypotheses + 6,
            "original proposal budget exceeded",
        )
        parent = np.asarray([p.parent for p in proposals], np.int64)
        packet, target = None, None
        coefficient = 0.0 if not self._history else self._coefficient
        candidate_anchor = np.full(len(proposals), -1, np.int64)
        if self._history:
            candidate_anchor = propagate_anchor_ids(self._anchor_ids, parent)
        if coefficient:
            votes = tuple(self._vote_provider(int(frame.frame_idx)))
            require(len(votes) == 3, "ordered three-view votes required")
            offset = int(frame.frame_idx) - self._first_native
            expected_release = self._first_native + (
                15 if offset < 16 else (offset // 8 + 1) * 8 - 1
            )
            for vi, vote in enumerate(votes):
                require(
                    vote.native == frame.frame_idx
                    and vote.release_horizon == expected_release
                    and vote.group_count == self._group_counts[vi]
                    and vote.query_sha256 == self._query_sha256[vi],
                    "vote native/release/query binding differs",
                )
            pixels = np.stack([targets.pixels[v] for v in VIEWS])
            packet = transport.packet_for_pixels(
                votes,
                self._anchors,
                tuple(frame.masks[v] for v in VIEWS),
                pixels,
            )
            require(
                tuple(packet.anchor_ids) == tuple(range(len(self._anchors))),
                "packet anchor order differs",
            )
            target = packet.target[candidate_anchor].copy()
        result = fitter.fit_with_ownership(
            self,
            proposals,
            states,
            rays,
            targets,
            target,
            coefficient=coefficient,
        )
        pose, cost, parts, support = result
        self._fit_capture = dict(
            native=int(frame.frame_idx),
            parent=parent,
            permutation=np.asarray([p.permutation for p in proposals], np.int64),
            kind=np.asarray([p.kind for p in proposals]),
            pose=pose.copy(),
            cost=cost.copy(),
            path_cost=cost + np.asarray([states[p.parent].cost for p in proposals]),
            parent_cost=np.asarray([states[p.parent].cost for p in proposals]),
            parts=parts.copy(),
            support=support.copy(),
            candidate_anchor=candidate_anchor,
            packet=packet,
            owner=copy.deepcopy(self.last_owner_fit),
            fit_pixels=None if targets is None else np.stack([targets.pixels[v] for v in VIEWS]),
            fit_target=None if targets is None else targets.target.copy(),
            fit_weights=None if targets is None else targets.weights.copy(),
            fit_foreground=None if targets is None else targets.foreground.copy(),
            proposal_pose=np.stack([p.pose for p in proposals]),
            proposal_predicted=np.stack([p.predicted for p in proposals]),
            proposal_plan=np.stack([p.plan for p in proposals]),
        )
        return result

    def _advance_owned(self, frame):
        require(
            len(self._history) < 200 and frame.frame_idx == self._first_native + len(self._history),
            "fixed200 native cursor differs",
        )
        require(json_hash(self._binding) == self._binding_sha, "input binding changed")
        self._transport_frame, self._fit_capture = frame, None
        try:
            super()._advance_owned(frame)
            records = self._history[-1]
            selected, order = check_retained_fit(records, self._fit_capture, self.config.beam_size)
            if len(self._history) == 1:
                self._anchors, self._anchor_facts = initial_anchors(
                    records,
                    frame,
                    self.rig,
                    self._group_counts,
                )
                self._anchor_ids = np.arange(len(records), dtype=np.int64)
                self._initial_observation = dict(
                    detections={v: a.copy() for v, a in frame.detections.items()},
                    masks={v: None if a is None else a.copy() for v, a in frame.masks.items()},
                )
            else:
                self._anchor_ids = propagate_anchor_ids(
                    self._anchor_ids,
                    np.asarray([r["parent"] for r in records], np.int64),
                )
                same(
                    self._anchor_ids,
                    self._fit_capture["candidate_anchor"][selected],
                    "selected anchor lineage",
                )
            self._anchor_history.append(self._anchor_ids.copy())
            self._fit_capture.update(selected=selected, order=order)
            self._fit_history.append(self._fit_capture)
        finally:
            self._transport_frame = self._fit_capture = None

    def export_state(self):
        raise RuntimeError("use export_transport_state: the anchor sidecar is mandatory")

    def restore_state(self, arrays, metadata):
        raise RuntimeError("use restore_transport_state: the anchor sidecar is mandatory")

    def export_transport_state(self):
        """Return core arrays/metadata and side arrays/metadata for joint sealing."""
        core, metadata = ContinuousPersistentNodeTracker.export_state(self)
        content = dict(
            active_anchor_ids=self._anchor_ids,
            anchor_history=self._anchor_history,
            anchor_facts=self._anchor_facts,
            anchors=self._anchors,
            initial_observation=self._initial_observation,
            fits=self._fit_history,
        )
        side_arrays = {}
        tree = _pack(content, side_arrays)
        side_metadata = dict(
            schema=SCHEMA,
            core_array_signatures={k: signature(v) for k, v in core.items()},
            core_metadata_sha256=json_hash(metadata),
            array_signatures={k: signature(v) for k, v in side_arrays.items()},
            tree=tree,
        )
        side_metadata["content_sha256"] = json_hash(side_metadata)
        self._validated_sidecar(core, metadata, side_arrays, side_metadata)
        return core, metadata, side_arrays, copy.deepcopy(side_metadata)

    def _validated_sidecar(self, core, metadata, arrays, side_metadata):
        header = {k: v for k, v in side_metadata.items() if k != "content_sha256"}
        require(
            side_metadata.get("schema") == SCHEMA
            and json_hash(header) == side_metadata["content_sha256"],
            "sidecar content changed",
        )
        require(
            header["core_metadata_sha256"] == json_hash(metadata)
            and header["core_array_signatures"] == {k: signature(v) for k, v in core.items()},
            "sidecar/core binding differs",
        )
        require(
            header["array_signatures"] == {k: signature(v) for k, v in arrays.items()},
            "sidecar arrays changed",
        )
        used = set()
        content = _unpack(header["tree"], arrays, used)
        require(used == set(arrays), "sidecar exact array membership differs")
        count = metadata["processed_frames"]
        require(
            1 <= count <= 200 and len(content["anchor_history"]) == len(content["fits"]) == count,
            "sidecar cursor differs",
        )
        offsets = core["history_offsets"]
        b0 = int(offsets[1])
        require(
            len(content["anchors"]) == len(content["anchor_facts"]) == b0,
            "initial retained anchor count differs",
        )
        ids = np.arange(b0, dtype=np.int64)
        for t in range(count):
            lo, hi = int(offsets[t]), int(offsets[t + 1])
            records = [{k: core["history_" + k][r] for k in RECORD_FIELDS} for r in range(lo, hi)]
            fit = content["fits"][t]
            require(fit["native"] == self._first_native + t, "sidecar fit native differs")
            require(
                all(int(r["frame_idx"]) == self._first_native + t for r in records),
                "retained native differs from fixed cursor",
            )
            parent_costs = (
                core["history_path_costs"][int(offsets[t - 1]) : lo]
                if t
                else np.zeros(1, np.float64)
            )
            require(
                fit["parent"].dtype == np.int64
                and np.all((fit["parent"] >= 0) & (fit["parent"] < len(parent_costs))),
                "saved candidate parent outside earlier beam",
            )
            same(fit["parent_cost"], parent_costs[fit["parent"]], "all candidate parent costs")
            same(fit["path_cost"], fit["cost"] + fit["parent_cost"], "all candidate path costs")
            require(
                all(int(r["hypotheses"]) == len(fit["parent"]) for r in records),
                "saved full proposal count differs",
            )
            selected, order = check_retained_fit(records, fit, self.config.beam_size)
            same(fit["selected"], selected, "saved candidate selection")
            same(fit["order"], order, "saved full candidate order")
            if t:
                same(
                    fit["candidate_anchor"],
                    propagate_anchor_ids(ids, fit["parent"]),
                    "all saved candidate anchors",
                )
                ids = propagate_anchor_ids(ids, core["history_parent"][lo:hi].astype(np.int64))
                same(fit["candidate_anchor"][selected], ids, "saved candidate anchors")
            else:
                same(
                    fit["candidate_anchor"],
                    np.full(len(fit["parent"]), -1, np.int64),
                    "t0 has no earlier anchor",
                )
            same(content["anchor_history"][t], ids, "saved anchor history")
            owner = fit["owner"]
            require(
                type(owner["coefficient"]) in (int, float)
                and owner["coefficient"] == (self._coefficient if t else 0.0),
                "owner coefficient policy differs",
            )
            packet = fit["packet"]
            if not t or self._coefficient == 0:
                require(packet is None and not owner["active"], "off arm cannot hold active packet")
            else:
                require(
                    isinstance(packet, dict)
                    and packet["native"] == fit["native"]
                    and packet["initial_native"] == self._first_native
                    and tuple(packet["query_sha256"]) == self._query_sha256,
                    "saved packet origin differs",
                )
                same(packet["anchor_ids"], np.arange(b0, dtype=np.int64), "packet all anchors")
                same(packet["pixels"], fit["fit_pixels"], "packet actual fit coordinates")
                require(
                    packet["pixels_signature"] == signature(fit["fit_pixels"]),
                    "packet pixel signature differs",
                )
                release = self._first_native + (15 if t < 16 else (t // 8 + 1) * 8 - 1)
                same(
                    packet["release_horizons"],
                    np.full(3, release, np.int64),
                    "packet common first-release horizons",
                )
            same(owner["total_score"], fit["cost"], "owner total cost")
            same(owner["original_loss_parts"], fit["parts"], "owner original parts")
            require(
                np.allclose(
                    owner["base_score"] + owner["coefficient"] * owner["owner_loss"],
                    fit["cost"],
                    rtol=0,
                    atol=1e-10,
                ),
                "owner decomposition differs",
            )
        same(content["active_anchor_ids"], ids, "active anchor IDs")
        for a in range(b0):
            fact = content["anchor_facts"][a]
            require(fact["anchor_id"] == fact["retained_index"] == a, "anchor initial index")
            for k in RECORD_FIELDS:
                same_record(fact["record"][k], core["history_" + k][a], "t0 retained anchor " + k)
            anchor = content["anchors"][a]
            require(
                anchor["anchor_id"] == a and anchor["initial_native"] == self._first_native,
                "initial anchor native differs",
            )
            expected = transport.build_initial_anchor(
                tuple(fact["references"]),
                self._group_counts,
                anchor_id=a,
                initial_native=self._first_native,
            )
            for vi in range(3):
                same(anchor["links"][vi], expected.links[vi], "immutable initial links")
        return content

    def restore_transport_state(self, core, metadata, arrays, side_metadata):
        require(
            not self.initialize_count
            and self._states is None
            and not self._finished
            and not self._failed,
            "transport restore requires a fresh tracker",
        )
        self._decode_state(core, metadata)
        content = self._validated_sidecar(core, metadata, arrays, side_metadata)
        # Recheck t0 geometry against its saved observations, not future records.
        from observations import FrameObservation

        initial_frame = FrameObservation(
            self._first_native,
            content["initial_observation"]["detections"],
            {},
            content["initial_observation"]["masks"],
            {},
        )
        t0 = [f["record"] for f in content["anchor_facts"]]
        anchors, facts = initial_anchors(t0, initial_frame, self.rig, self._group_counts)
        for a, original in enumerate(content["anchors"]):
            for vi in range(3):
                same(anchors[a].links[vi], original["links"][vi], "restored initial geometry links")
        ContinuousPersistentNodeTracker.restore_state(self, core, metadata)
        self._anchors, self._anchor_facts = anchors, facts
        self._anchor_ids = content["active_anchor_ids"]
        self._anchor_history, self._fit_history = content["anchor_history"], content["fits"]
        self._initial_observation = content["initial_observation"]
