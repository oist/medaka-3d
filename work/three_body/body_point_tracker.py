"""Causal body-point support domain over the frozen dynamic gap tracker.

Only target construction, the later-frame fit, and their sealed policy differ.
The original proposal/beam/state/anchor methods remain inherited. The initial
frame calls the frozen gap tracker exactly; neither points nor future poses are
used to initialize or relabel the causal anchor bank. No execution at import.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass

import body_point_fitter as fitter
import body_point_transport as point_transport
import mask_transport_gap_tracker as previous
import numpy as np
from mask_transport_gap_tracker import (
    RECORD_FIELDS,
    VIEWS,
    ContinuousPersistentNodeTracker,
    _pack,
    _unpack,
    check_retained_fit,
    json_hash,
    propagate_anchor_ids,
    same,
    same_record,
    signature,
    transport,
)

require = previous.require
SCHEMA = "body_point_tracker_sidecar_v1"
OBJECTIVE = {
    "schema": "body_point_owner_plus_ray_gap_v1",
    "total": "base + coefficient * (owner_loss + gap_loss)",
    "domain": "reported single-mask points of each admitted initial group",
    "gap": "sqrt(max(per_fish_mahalanobis_sq, 1)) - 1",
    "gap_relative_weight": 1.0,
    "reduction": "sum(point_weight * ray_valid * (owner + gap)) / 3",
    "point_weight": "old effective fit-mask mass m / original single-point count S",
    "invalid_ray_renormalization": False,
    "target_body_axis": "causal_t0_parent_lineage",
    "gap_uses_visibility_weight": False,
    "sigma": "unchanged dynamic renderer; pose-dependent lengths, no detach",
    "invalid_ray": "finite neutral q before sqrt; zero effective weight",
    "target_mass": "sum(point_weight * ray_valid) / 3; uncomputed if inactive",
    "positive_gap_pixel_count": "query indices with effective mass > 0 and q > 1",
    "fit_samples": "unchanged original pixel targets; query rays are separate evidence",
    "initial_frame": "frozen gap fit coefficient zero; no point packet",
    "inactive": "frozen original fit direct dispatch",
}


def packet_dict(packet):
    return asdict(packet) if is_dataclass(packet) else packet


def fit_ray_valid(rig, pixels):
    """Frozen renderer's native-ray validity, without torch or renderer state."""
    require(tuple(rig.views) == tuple(VIEWS), "ordered three-view calibration required")
    result = np.zeros(pixels.shape[:2], bool)
    for vi, view in enumerate(VIEWS):
        xy = np.asarray(pixels[vi], dtype=float)
        finite = np.isfinite(xy).all(axis=1)
        if finite.any():
            origins, directions = rig.camera(view).rays(xy[finite])
            length = np.linalg.norm(directions, axis=1)
            good = np.isfinite(origins).all(axis=1) & np.isfinite(directions).all(axis=1)
            good &= length > 1e-12
            result[vi, np.flatnonzero(finite)[good]] = True
    return result


def validate_body_point_diagnostics(fit, *, coefficient, opt_steps):
    """Reuse the exact 20-field arithmetic on its explicitly new query domain.

    A read-only validation view supplies actual query pixels for the inherited
    count bound and translates only the checked reason label. Saved fit pixels,
    diagnostics, losses, and metadata are never rewritten or summed differently.
    """
    diagnostics = fit["owner"]
    packet = fit["packet"]
    view = dict(fit)
    if diagnostics["active"]:
        require(
            diagnostics["reason"] == "fixed_body_point_owner_plus_gap" and packet is not None,
            "active body-point objective requires its own packet/reason",
        )
        p = packet_dict(packet)
        view["fit_pixels"] = p["point_pixels"]
        view["owner"] = dict(diagnostics, reason="fixed_transport_owner_plus_gap")
        anchors = fit["candidate_anchor"]
        weight = p["point_weight"][anchors]
        valid_weight = weight * p["ray_valid"][None, :, :, None]
        expected_mass = valid_weight.sum(axis=2) / 3
        require(
            np.allclose(diagnostics["target_mass_by_view_body"], expected_mass, atol=1e-10, rtol=0),
            "body-point effective query mass differs",
        )
        require(
            np.all(
                diagnostics["positive_gap_pixel_count"] <= np.count_nonzero(valid_weight, axis=2)
            ),
            "positive gap count exceeds effective query count",
        )
    elif coefficient and packet is not None:
        p = packet_dict(packet)
        require(
            not np.any(
                p["point_weight"][fit["candidate_anchor"]] * p["ray_valid"][None, :, :, None]
            ),
            "inactive body-point fit retained positive effective mass",
        )
    previous.validate_gap_diagnostics(view, coefficient=coefficient, opt_steps=opt_steps)


def decode_sidecar(arrays, metadata):
    """Lossless new-schema decode; semantic restore also validates core state."""
    header = {k: v for k, v in metadata.items() if k != "content_sha256"}
    require(
        header.get("schema") == SCHEMA
        and header.get("objective") == OBJECTIVE
        and json_hash(header) == metadata["content_sha256"],
        "body-point sidecar content/objective changed",
    )
    require(
        header["array_signatures"] == {k: signature(v) for k, v in arrays.items()},
        "body-point sidecar arrays changed",
    )
    used = set()
    content = _unpack(header["tree"], arrays, used)
    require(used == set(arrays), "body-point sidecar exact array membership differs")
    return content


class BodyPointTracker(previous.MaskTransportGapTracker):
    """Same constructor; vote_provider(native) now returns three PointEvidence."""

    def _context(self):
        result = super()._context()
        result["mask_transport"].update(schema=SCHEMA, objective=copy.deepcopy(OBJECTIVE))
        return result

    def _fit(self, proposals, states, rays, targets):
        if not self._history:
            return super()._fit(proposals, states, rays, targets)
        frame = self._transport_frame
        require(frame is not None, "fit must be owned by one native advance")
        require(
            1 <= len(proposals) <= self.config.max_hypotheses + 6,
            "original proposal budget exceeded",
        )
        parent = np.asarray([p.parent for p in proposals], np.int64)
        candidate_anchor = propagate_anchor_ids(self._anchor_ids, parent)
        packet = None
        if self._coefficient:
            evidence = tuple(self._vote_provider(int(frame.frame_idx)))
            require(len(evidence) == 3, "ordered three-view point evidence required")
            offset = int(frame.frame_idx) - self._first_native
            expected_release = self._first_native + (
                15 if offset < 16 else (offset // 8 + 1) * 8 - 1
            )
            for vi, point in enumerate(evidence):
                vote = point.vote
                require(
                    vote.native == frame.frame_idx
                    and vote.release_horizon == expected_release
                    and vote.group_count == self._group_counts[vi]
                    and vote.query_sha256 == self._query_sha256[vi],
                    "body-point native/release/query binding differs",
                )
            pixels = np.stack([targets.pixels[v] for v in VIEWS])
            packet = point_transport.build_packet(
                evidence,
                self._anchors,
                tuple(frame.masks[v] for v in VIEWS),
                pixels,
                # Packet's historical field name denotes y in the old loss:
                # bind the actual target indicator, including its original dtype.
                fit_foreground=targets.target,
                fit_weights=targets.weights,
                fit_ray_valid=fit_ray_valid(self.rig, pixels),
                rig=self.rig,
            )
            require(
                tuple(packet.anchor_ids) == tuple(range(len(self._anchors))),
                "body-point packet anchor order differs",
            )
        result = fitter.fit_with_body_points(
            self,
            proposals,
            states,
            rays,
            targets,
            packet,
            candidate_anchor=candidate_anchor,
            coefficient=self._coefficient,
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
            owner=copy.deepcopy(self.last_gap_fit),
            fit_pixels=None if targets is None else np.stack([targets.pixels[v] for v in VIEWS]),
            fit_target=None if targets is None else targets.target.copy(),
            fit_weights=None if targets is None else targets.weights.copy(),
            fit_foreground=None if targets is None else targets.foreground.copy(),
            proposal_pose=np.stack([p.pose for p in proposals]),
            proposal_predicted=np.stack([p.predicted for p in proposals]),
            proposal_plan=np.stack([p.plan for p in proposals]),
        )
        validate_body_point_diagnostics(
            self._fit_capture, coefficient=self._coefficient, opt_steps=self.config.opt_steps
        )
        return result

    def _validate_packet(self, packet, fit, anchors):
        canonical = point_transport.validate_packet(packet, rig=self.rig)
        p = packet_dict(canonical)
        require(p["schema"] == point_transport.SCHEMA, "body-point packet schema differs")
        require(
            p["native"] == fit["native"]
            and p["initial_native"] == self._first_native
            and tuple(p["query_sha256"]) == self._query_sha256,
            "saved body-point packet origin differs",
        )
        same(p["anchor_ids"], np.arange(len(anchors), dtype=np.int64), "packet all anchors")
        same(p["pixels"], fit["fit_pixels"], "packet actual fit coordinates")
        same(p["fit_weights"], fit["fit_weights"], "packet actual fit weights")
        same(p["fit_foreground"], fit["fit_target"], "packet actual fit target indicator")
        same(
            p["fit_ray_valid"],
            fit_ray_valid(self.rig, fit["fit_pixels"]),
            "packet original fit ray validity",
        )
        pixel_signature = dict(p["pixels_signature"])
        pixel_signature["shape"] = list(pixel_signature["shape"])
        require(
            pixel_signature == signature(fit["fit_pixels"]), "packet fit pixel signature differs"
        )
        t = fit["native"] - self._first_native
        release = self._first_native + (15 if t < 16 else (t // 8 + 1) * 8 - 1)
        same(
            p["release_horizons"],
            np.full(3, release, np.int64),
            "packet common first-release horizons",
        )
        packed_a, packed_b = {}, {}
        require(
            _pack(p["whole_mask_packet"]["anchors"], packed_a) == _pack(anchors, packed_b)
            and {k: signature(v) for k, v in packed_a.items()}
            == {k: signature(v) for k, v in packed_b.items()},
            "packet immutable complete initial anchor bank differs",
        )

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
            objective=copy.deepcopy(OBJECTIVE),
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
        require(header["objective"] == OBJECTIVE, "body-point sidecar objective differs")
        require(
            metadata["context"]["mask_transport"] == self._context()["mask_transport"],
            "body-point objective/context differs",
        )
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
                require(isinstance(packet, dict), "saved body-point packet dict required")
                self._validate_packet(packet, fit, content["anchors"])
            validate_body_point_diagnostics(
                fit, coefficient=self._coefficient if t else 0.0, opt_steps=self.config.opt_steps
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
