"""Fixed registered radial widths after t0, with unchanged causal anchor/state logic.

The same 39 core arrays, 21 final arrays and 19 retained fields are inherited.
t0 delegates exactly to the frozen gap fitter's coefficient-zero path. Later
fits always use the new fixed-width base, including coefficient zero/empty T.
The 20 diagnostic fields retain active=auxiliary active; full iterations after
t0 are mandatory even when auxiliary terms are inactive. No model/data at import.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import fixed_body_width_fitter as fitter
import mask_transport_gap_fitter as initial_fitter
import mask_transport_gap_tracker as gap
import mask_transport_tracker as frozen
import numpy as np
from fixed_body_width_renderer import FixedWidthRenderer
from mask_transport_tracker import (
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
from renderer import RendererConfig

require = frozen.require
SCHEMA = "fixed_body_width_tracker_sidecar_v1"
OBJECTIVE = {
    "schema": "fixed_registered_radial_owner_plus_gap_v1",
    "total": "base + coefficient * (owner_loss + gap_loss)",
    "gap": "sqrt(max(per_fish_mahalanobis_sq, 1)) - 1",
    "gap_relative_weight": 1.0,
    "reduction": "sum(weight * ray_valid * foreground * target * gap) / 3",
    "target_body_axis": "causal_t0_parent_lineage",
    "gap_uses_visibility_weight": False,
    "sigma": "registered absolute radial; current centers/tangents/axial; no width relabel",
    "initial_fit": "frozen gap coefficient_zero direct dispatch",
    "later_fit": "fixed radial base even with coefficient zero or inactive target",
    "invalid_ray": "finite neutral q before sqrt; zero effective weight",
    "target_mass": "sum(weight * ray_valid * foreground * target) / 3; uncomputed if inactive",
}


def decode_sidecar(arrays, metadata):
    """Decode the new objective/schema only; joint restore verifies core state."""
    header = {k: v for k, v in metadata.items() if k != "content_sha256"}
    require(
        metadata.get("schema") == SCHEMA
        and header.get("objective") == OBJECTIVE
        and json_hash(header) == metadata["content_sha256"],
        "fixed-width sidecar content or objective changed",
    )
    require(
        header["array_signatures"] == {k: signature(v) for k, v in arrays.items()},
        "gap sidecar arrays changed",
    )
    used = set()
    content = _unpack(header["tree"], arrays, used)
    require(used == set(arrays), "gap sidecar exact array membership differs")
    return content


def validate_fixed_width_diagnostics(fit, *, coefficient, opt_steps, initial):
    """Keep the old t0 contract; all later base-only fits have measured iterates."""
    require(type(initial) is bool, "explicit initial-frame diagnostic scope required")
    if initial:
        require(coefficient == 0.0, "t0 must use the original zero-coefficient dispatch")
        gap.validate_gap_diagnostics(fit, coefficient=0.0, opt_steps=opt_steps)
        return

    d = fit["owner"]
    fields = {
        "active",
        "reason",
        "coefficient",
        "base_score",
        "owner_loss",
        "gap_loss",
        "total_score",
        "original_loss_parts",
        "iteration_owner_loss",
        "iteration_gap_loss",
        "iteration_owner_gradient_norm",
        "iteration_gap_gradient_norm",
        "iteration_base_score",
        "iteration_total_score",
        "best_iteration",
        "target_mass",
        "target_diagnostics_computed",
        "final_gap_by_view_body",
        "target_mass_by_view_body",
        "positive_gap_pixel_count",
    }
    require(isinstance(d, dict) and set(d) == fields, "fixed-width diagnostic exact fields differ")
    require(
        type(d["active"]) is bool
        and type(d["target_diagnostics_computed"]) is bool
        and type(d["coefficient"]) in (int, float)
        and d["coefficient"] == coefficient
        and coefficient in (0.0, 3.0),
        "fixed-width coefficient/activity policy differs",
    )
    require(type(opt_steps) is int and opt_steps >= 1, "positive original fit steps required")
    b = len(fit["parent"])
    require(b >= 1, "nonempty original proposal pool required")

    def array(name, shape, dtype=np.float64, *, nonnegative=False):
        a = d[name]
        require(
            isinstance(a, np.ndarray)
            and a.shape == shape
            and a.dtype == dtype
            and np.isfinite(a).all(),
            "fixed-width diagnostic array differs: " + name,
        )
        if nonnegative:
            require(np.all(a >= 0), "negative fixed-width diagnostic: " + name)
        return a

    def close(a, c, label):
        require(np.allclose(a, c, atol=1e-10, rtol=0), "fixed-width " + label + " differs")

    for name in ("base_score", "owner_loss", "gap_loss", "total_score"):
        array(name, (b,), nonnegative=name in ("owner_loss", "gap_loss"))
        array(
            "iteration_" + name, (opt_steps + 1, b), nonnegative=name in ("owner_loss", "gap_loss")
        )
    array("original_loss_parts", (b, 5))
    same(d["total_score"], fit["cost"], "fixed-width total cost")
    same(d["original_loss_parts"], fit["parts"], "fixed-width original five parts")
    close(
        d["base_score"] + coefficient * (d["owner_loss"] + d["gap_loss"]),
        fit["cost"],
        "final objective decomposition",
    )
    close(
        d["iteration_base_score"]
        + coefficient * (d["iteration_owner_loss"] + d["iteration_gap_loss"]),
        d["iteration_total_score"],
        "iterate objective decomposition",
    )
    for name in ("iteration_owner_gradient_norm", "iteration_gap_gradient_norm"):
        array(name, (opt_steps, b), nonnegative=True)
    best = array("best_iteration", (b,), np.int64)
    same(best, np.argmin(d["iteration_total_score"], axis=0), "fixed-width first best iteration")
    for name in ("base_score", "owner_loss", "gap_loss", "total_score"):
        close(d[name], d["iteration_" + name][best, np.arange(b)], "best " + name)

    active = d["active"]
    detailed_shape = (b, 3, 3) if active else (0, 3, 3)
    gap_by_body = array("final_gap_by_view_body", detailed_shape, nonnegative=True)
    mass = array("target_mass_by_view_body", detailed_shape, nonnegative=True)
    pixels = array("positive_gap_pixel_count", detailed_shape, np.int64, nonnegative=True)
    total_mass = array("target_mass", (b,) if active else (0,), nonnegative=True)
    require(
        d["target_diagnostics_computed"] is active, "fixed-width target computation flag differs"
    )
    if active:
        require(
            coefficient == 3.0 and d["reason"] == "fixed_radial_owner_plus_gap",
            "active fixed-width owner-plus-gap policy differs",
        )
        require(np.any(total_mass > 0), "active fixed-width fit has no effective target")
        close(gap_by_body.sum((1, 2)), d["gap_loss"], "view/body gap sum")
        close(mass.sum((1, 2)), total_mass, "view/body target mass sum")
        require(
            fit["fit_pixels"] is not None
            and np.all(pixels <= fit["fit_pixels"].shape[1])
            and np.all(gap_by_body[pixels == 0] == 0)
            and np.all(pixels[mass == 0] == 0),
            "fixed-width positive pixel or zero-mass scope differs",
        )
    else:
        expected = (
            {"fixed_radial_coefficient_zero"}
            if coefficient == 0
            else {"fixed_radial_zero_effective_target", "fixed_radial_zero_valid_ray_target"}
        )
        require(d["reason"] in expected, "inactive fixed-width reason differs")
        for name in (
            "owner_loss",
            "gap_loss",
            "iteration_owner_loss",
            "iteration_gap_loss",
            "iteration_owner_gradient_norm",
            "iteration_gap_gradient_norm",
        ):
            require(
                not np.any(d[name]), "inactive fixed-width auxiliary term must be zero: " + name
            )
        same(d["base_score"], fit["cost"], "inactive fixed-width base score")
        same(
            d["iteration_base_score"],
            d["iteration_total_score"],
            "inactive fixed-width base iterates",
        )


class FixedBodyWidthTracker(frozen.MaskTransportTracker):
    """Same public constructor and state bundle; distinctly bound gap objective."""

    def __init__(self, *args, **kwargs):
        # Continuous's constructor calls this class's _context after self.initial
        # exists. The immutable reference is initialized at that first call.
        self._fixed_width_reference_pose = None
        super().__init__(*args, **kwargs)

    def _width_policy(self):
        if self._fixed_width_reference_pose is None:
            reference = np.ascontiguousarray(self.initial, dtype=np.float64)
            self._fixed_width_reference_pose = np.frombuffer(
                reference.tobytes(), dtype=np.float64
            ).reshape(3, 3, 3)
            # Pixels do not participate in R0. One harmless ray per view permits
            # use of the actual renderer constructor/formula, without a copy.
            renderer = FixedWidthRenderer(
                self.rig,
                {v: np.zeros((1, 2), np.float64) for v in VIEWS},
                RendererConfig(width_ratio=self.config.width_ratio),
                device=self.device,
                dtype=self.dtype,
                reference_pose_xyz=self._fixed_width_reference_pose,
            )
            self._fixed_width_signature = renderer.validate_fixed_width()
            radial = renderer.radial_sigma_mm.cpu().numpy()
            self._fixed_width_radial = np.frombuffer(radial.tobytes(), dtype=radial.dtype).reshape(
                radial.shape
            )
            self._fixed_width_policy = dict(
                schema="registered_absolute_radial_tracker_reference_v1",
                reference_pose=signature(self._fixed_width_reference_pose),
                reference_lengths=signature(self.lengths),
                radial_sigma_mm=signature(self._fixed_width_radial),
                renderer_signature=json.loads(json.dumps(self._fixed_width_signature)),
                renderer_config=asdict(renderer.config),
                reference_device=str(self.device),
                reference_dtype=str(self.dtype),
                apply_from_native=self._first_native + 1,
                initial_native=self._first_native,
                initial_fit="frozen coefficient-zero dynamic renderer",
                later_fit="fixed radial base/owner/gap/final image support",
                body_axis="registered seed slot, never proposal permutation",
                cross_device_bitwise_restore_qualified=False,
            )
            self._fixed_width_policy_sha = json_hash(self._fixed_width_policy)
        require(
            json_hash(self._fixed_width_policy) == self._fixed_width_policy_sha,
            "fixed-width reference policy changed",
        )
        same(self.initial, self._fixed_width_reference_pose, "registered initial width reference")
        require(
            signature(self._fixed_width_radial) == self._fixed_width_policy["radial_sigma_mm"]
            and signature(self.lengths) == self._fixed_width_policy["reference_lengths"]
            and signature(self._fixed_width_reference_pose)
            == self._fixed_width_policy["reference_pose"]
            and json.loads(json.dumps(self._fixed_width_signature))
            == self._fixed_width_policy["renderer_signature"]
            and str(self.device) == self._fixed_width_policy["reference_device"]
            and str(self.dtype) == self._fixed_width_policy["reference_dtype"]
            and asdict(RendererConfig(width_ratio=self.config.width_ratio))
            == self._fixed_width_policy["renderer_config"],
            "fixed-width reference storage/configuration changed",
        )
        return copy.deepcopy(self._fixed_width_policy)

    def validate_fixed_width_renderer(self, renderer):
        """Called by the fitter before scoring: actual R0 must match the context."""
        self._width_policy()
        require(type(renderer) is FixedWidthRenderer, "fixed-width fitter renderer type differs")
        require(
            renderer.validate_fixed_width() == self._fixed_width_signature,
            "fixed-width fitted renderer reference differs",
        )
        same(
            renderer.radial_sigma_mm.cpu().numpy(),
            self._fixed_width_radial,
            "fixed-width fitted radial values",
        )
        return self._fixed_width_signature

    def _context(self):
        result = super()._context()
        result["mask_transport"].update(
            schema=SCHEMA, objective=copy.deepcopy(OBJECTIVE), fixed_width=self._width_policy()
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
        self._width_policy()
        fit_function = (
            initial_fitter.fit_with_gap if not self._history else fitter.fit_with_fixed_width
        )
        result = fit_function(
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
            owner=copy.deepcopy(self.last_gap_fit),
            fit_pixels=None if targets is None else np.stack([targets.pixels[v] for v in VIEWS]),
            fit_target=None if targets is None else targets.target.copy(),
            fit_weights=None if targets is None else targets.weights.copy(),
            fit_foreground=None if targets is None else targets.foreground.copy(),
            proposal_pose=np.stack([p.pose for p in proposals]),
            proposal_predicted=np.stack([p.predicted for p in proposals]),
            proposal_plan=np.stack([p.plan for p in proposals]),
        )
        validate_fixed_width_diagnostics(
            self._fit_capture,
            coefficient=coefficient,
            opt_steps=self.config.opt_steps,
            initial=not self._history,
        )
        return result

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
            fixed_width_reference=dict(
                pose=self._fixed_width_reference_pose,
                radial_sigma_mm=self._fixed_width_radial,
                policy=self._width_policy(),
            ),
        )
        side_arrays = {}
        tree = _pack(content, side_arrays)
        side_metadata = dict(
            schema=SCHEMA,
            objective=copy.deepcopy(OBJECTIVE),
            fixed_width=self._width_policy(),
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
        require(header["objective"] == OBJECTIVE, "gap sidecar objective differs")
        require(
            metadata["context"]["mask_transport"] == self._context()["mask_transport"],
            "gap objective/context differs",
        )
        require(header["fixed_width"] == self._width_policy(), "sidecar fixed-width policy differs")
        reference = content["fixed_width_reference"]
        require(
            set(reference) == {"pose", "radial_sigma_mm", "policy"}
            and reference["policy"] == header["fixed_width"],
            "sidecar fixed-width reference fields differ",
        )
        same(reference["pose"], self._fixed_width_reference_pose, "sidecar registered width pose")
        same(
            reference["radial_sigma_mm"],
            self._fixed_width_radial,
            "sidecar registered radial values",
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
            validate_fixed_width_diagnostics(
                fit,
                coefficient=self._coefficient if t else 0.0,
                opt_steps=self.config.opt_steps,
                initial=t == 0,
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
