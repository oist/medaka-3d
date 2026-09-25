"""Read-only proposal/fit/selection recorder for one existing tracker instance.

Use `with CandidateDiagnostics(tracker) as diag`, then surround each original
`tracker.advance(frame)` with `with diag.frame(frame.frame_idx)`. Only frames in
[66656,66706) are retained by default. No subclass, context-type change, extra
fit/render, replacement proposal, or final traceback is introduced. Original
method arguments/return objects are forwarded verbatim. Diagnostic projections
run on copied NumPy inputs; they add runtime, not evidence or biological truth.

`records` returns owned per-frame {frame_idx, arrays, metadata}; arrays are plain
non-object NPZ-compatible values. The driver owns all I/O/provenance and must
bind this source, frozen dependency sources, raw inputs and original checkpoint.
Costs are the original fitter's final retained local score and path accumulation;
loss_parts omit the orientation term, as in the frozen fitter. No initial loss is
recomputed. Per-slot support summaries mix parents/permutations explicitly and
must not be interpreted as same biological-ID alternatives.
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from inspect import signature
from pathlib import Path
from types import MethodType

import numpy as np

LAB = Path(__file__).resolve().parents[2] / "labs/20260909_three_body"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
from node_geometry import RayObservations, supported_nodes  # noqa: E402

from medaka3d.types import VIEWS  # noqa: E402


def require(value, message):
    if not value:
        raise ValueError(message)


def identical(first, second):
    """Scalar/array equality includes dtype, NaN payload and signed zero."""
    a, b = np.asarray(first), np.asarray(second)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def tree_equal(first, second):
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return identical(first, second)
    if is_dataclass(first):
        return type(first) is type(second) and all(
            tree_equal(getattr(first, f.name), getattr(second, f.name)) for f in fields(first)
        )
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(tree_equal(first[k], second[k]) for k in first)
        )
    if isinstance(first, (list, tuple)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(tree_equal(a, b) for a, b in zip(first, second, strict=True))
        )
    return type(first) is type(second) and (
        identical(first, second) if isinstance(first, (float, np.generic)) else first == second
    )


def collisions(sources):
    """Duplicate row for the SAME view/node; cross-node instance grouping is separate."""
    source = np.asarray(sources)
    require(source.shape == (3, 3, 3) and source.dtype.kind in "iu", "invalid source F,V,N array")
    duplicate = np.zeros(source.shape, bool)
    for view in range(3):
        for node in range(3):
            rows = source[:, view, node]
            for fish, row in enumerate(rows):
                duplicate[fish, view, node] = row >= 0 and np.count_nonzero(rows == row) > 1
    return duplicate


def support_snapshot(poses, plans, rays, rig):
    """Frozen float32-ray 2mm/12px gates plus explicit source exclusivity."""
    poses, plans = np.asarray(poses), np.asarray(plans)
    require(
        poses.ndim == 4
        and poses.shape[1:] == (3, 3, 3)
        and plans.shape == poses.shape
        and plans.dtype.kind in "iu",
        "invalid candidate batch",
    )
    require(np.isfinite(poses).all() and np.all(plans >= -1), "nonfinite pose or invalid source")
    require(tuple(rig.views) == tuple(VIEWS), "view order differs")
    copied = RayObservations(
        copy.deepcopy(rays.points), copy.deepcopy(rays.origins), copy.deepcopy(rays.directions)
    )
    data = {
        key: []
        for key in (
            "geometric_sources",
            "geometric_observed",
            "strict_sources",
            "strict_observed",
            "ray_distance_mm",
            "native_pixel_error",
            "plan_source_collision",
            "strict_source_collision",
            "assigned_source",
            "ray_gate_pass",
            "pixel_gate_pass",
        )
    }
    for pose, plan in zip(poses, plans, strict=True):
        for v, view in enumerate(VIEWS):
            held = plan[:, v] >= 0
            require(
                not held.any() or plan[:, v][held].max() < len(copied.points[view]),
                "source row outside current untracked detections",
            )
        xyz, p = pose.copy(), plan.copy()
        sources, observed, distance = supported_nodes(
            xyz, p, copied, rig, maximum_ray_mm=2.0, maximum_pixel_error=12.0
        )
        pixel = np.full((3, 3, 3), np.nan)
        for v, view in enumerate(VIEWS):
            projected = rig.camera(view).project(xyz.copy())
            for fish, node in np.argwhere(p[:, v] >= 0):
                pixel[fish, v, node] = np.linalg.norm(
                    projected[fish, node] - copied.points[view][p[fish, v, node], node]
                )
        duplicate = collisions(sources)
        strict = np.where(duplicate, -1, sources)
        values = dict(
            geometric_sources=sources,
            geometric_observed=observed,
            strict_sources=strict,
            strict_observed=(strict >= 0).sum(axis=1) >= 2,
            ray_distance_mm=distance,
            native_pixel_error=pixel,
            plan_source_collision=collisions(p),
            strict_source_collision=duplicate,
            assigned_source=p >= 0,
            ray_gate_pass=(p >= 0) & np.isfinite(distance) & (distance <= 2.0),
            pixel_gate_pass=(p >= 0) & np.isfinite(pixel) & (pixel <= 12.0),
        )
        for key, value in values.items():
            data[key].append(value.copy())
    return {key: np.stack(value) for key, value in data.items()}


def selected_indices(poses, path_cost, parents, permutations, beam_size):
    """Read-only reproduction of frozen stable ordering and same-parent dedup."""
    order = np.argsort(path_cost, kind="stable")
    selected = []
    for candidate in order:
        if any(
            parents[candidate] == parents[j]
            and identical(permutations[candidate], permutations[j])
            and np.max(np.linalg.norm(poses[candidate] - poses[j], axis=-1)) < 0.5
            for j in selected
        ):
            continue
        selected.append(int(candidate))
        if len(selected) == beam_size:
            break
    return np.asarray(selected, np.int64), order


def stage_facts(generated, fitted, selected):
    """Support-existence facts only: no inferred true fish or causal certification."""
    require(
        generated.shape == fitted.shape and generated.ndim == 3 and generated.shape[1:] == (3, 3),
        "support arrays must be B,F,N",
    )
    chosen = fitted[selected]
    per_slot = []
    for fish in range(3):
        before = bool(generated[:, fish, 1].any())
        after = bool(fitted[:, fish, 1].any())
        kept = bool(chosen[:, fish, 1].any())
        label = (
            "selected strict support exists"
            if kept
            else "fit support not selected"
            if after
            else "generated support lost in fit"
            if before
            else "no generated strict support"
        )
        per_slot.append(
            dict(
                slot=fish,
                generated_chest_support_exists=before,
                fitted_chest_support_exists=after,
                selected_chest_support_exists=kept,
                stage_label=label,
            )
        )
    return dict(
        per_slot=per_slot,
        all_three_chests_supported_scene_exists=dict(
            generated=bool(generated[:, :, 1].all(1).any()),
            fitted=bool(fitted[:, :, 1].all(1).any()),
            selected=bool(chosen[:, :, 1].all(1).any()),
        ),
        all_nine_nodes_supported_scene_exists=dict(
            generated=bool(generated.all((1, 2)).any()),
            fitted=bool(fitted.all((1, 2)).any()),
            selected=bool(chosen.all((1, 2)).any()),
        ),
        scope=(
            "Finite returned pool, mixing saved parents/permutations; "
            "slot index is not an independent biological identity. "
            "Absence here does not exclude intersections discarded before pool return."
        ),
    )


class CandidateDiagnostics:
    """Temporary instance methods, explicit frame boundaries, no hidden state lookup."""

    def __init__(self, tracker, frame_range=(66656, 66706)):
        require(
            len(frame_range) == 2
            and all(type(x) is int for x in frame_range)
            and frame_range[0] < frame_range[1],
            "invalid native diagnostic interval",
        )
        self.tracker, self.frame_range = tracker, tuple(frame_range)
        self._attached = False
        self._active = None
        self._records = []
        self._frame_open = False

    @property
    def records(self):
        return copy.deepcopy(self._records)

    def __enter__(self):
        require(not self._attached, "diagnostics already attached")
        require(
            not any(k in self.tracker.__dict__ for k in ("_proposals", "_fit")),
            "refuse to replace an existing instance override",
        )
        self._context = copy.deepcopy(self.tracker._context())
        self._type = type(self.tracker)
        self._original_proposals = self.tracker._proposals
        self._original_fit = self.tracker._fit
        owner = self

        def proposals(instance, *args, **kwargs):
            return owner._proposals(args, kwargs)

        def fit(instance, *args, **kwargs):
            return owner._fit(args, kwargs)

        self.tracker._proposals = MethodType(proposals, self.tracker)
        self.tracker._fit = MethodType(fit, self.tracker)
        self._attached = True
        require(tree_equal(self.tracker._context(), self._context), "context changed on attach")
        return self

    def __exit__(self, exc_type, exc, tb):
        for name in ("_proposals", "_fit"):
            delattr(self.tracker, name)
        self._attached = False
        require(
            type(self.tracker) is self._type and tree_equal(self.tracker._context(), self._context),
            "tracker type/context changed during diagnostics",
        )
        return False

    @contextmanager
    def frame(self, frame_idx):
        require(self._attached and not self._frame_open, "frame boundary is not available")
        require(
            isinstance(frame_idx, (int, np.integer))
            and not isinstance(frame_idx, (bool, np.bool_)),
            "native frame must be integer",
        )
        native = int(frame_idx)
        self._frame_open = True
        tracked = self.frame_range[0] <= native < self.frame_range[1]
        self._active = dict(frame_idx=native, arrays={}, metadata={}) if tracked else None
        try:
            yield
            if tracked:
                self._complete()
        finally:
            self._active = None
            self._frame_open = False

    def _proposals(self, args, kwargs):
        if self._active is None:
            return self._original_proposals(*args, **kwargs)
        bound = signature(self._original_proposals).bind(*args, **kwargs)
        bound.apply_defaults()
        states, rays, flow = (bound.arguments[k] for k in ("states", "rays", "flow"))
        require("proposal_pose" not in self._active["arrays"], "multiple proposal calls per frame")
        state_copy = copy.deepcopy(states)
        rays_copy = copy.deepcopy(rays)
        flow_copy = copy.deepcopy(flow)
        result = self._original_proposals(*args, **kwargs)
        require(
            tree_equal(states, state_copy)
            and tree_equal(rays, rays_copy)
            and tree_equal(flow, flow_copy),
            "original proposal inputs changed",
        )
        arrays = self._active["arrays"]
        for key in (
            "pose",
            "velocity",
            "missing",
            "review",
            "cost",
            "node_missing",
            "direct_pectoral_age",
        ):
            arrays["parent_" + key] = np.asarray([getattr(s, key) for s in state_copy]).copy()
        arrays.update(
            proposal_pose=np.stack([p.pose for p in result]).copy(),
            proposal_predicted=np.stack([p.predicted for p in result]).copy(),
            proposal_plan=np.stack([p.plan for p in result]).copy(),
            proposal_parent=np.asarray([p.parent for p in result], np.int64),
            proposal_permutation=np.asarray([p.permutation for p in result], np.int64),
        )
        self._active["metadata"]["proposal_kind"] = [p.kind for p in result]
        for key, value in support_snapshot(
            arrays["proposal_pose"], arrays["proposal_plan"], rays_copy, self.tracker.rig
        ).items():
            arrays["generated_" + key] = value
        for v, view in enumerate(VIEWS):
            arrays[view + "_points"] = rays_copy.points[view].copy()
            arrays[view + "_ray_origins"] = rays_copy.origins[v].copy()
            arrays[view + "_ray_directions"] = rays_copy.directions[v].copy()
        return result

    def _fit(self, args, kwargs):
        if self._active is None:
            return self._original_fit(*args, **kwargs)
        bound = signature(self._original_fit).bind(*args, **kwargs)
        proposals, states, rays, targets = (
            bound.arguments[k] for k in ("proposals", "states", "rays", "targets")
        )
        arrays = self._active["arrays"]
        require("proposal_pose" in arrays and "fitted_pose" not in arrays, "fit ordering differs")
        snapshot = copy.deepcopy((proposals, states, rays, targets))
        for key, values in dict(
            pose=[p.pose for p in proposals],
            predicted=[p.predicted for p in proposals],
            plan=[p.plan for p in proposals],
        ).items():
            require(
                identical(np.stack(values), arrays["proposal_" + key]),
                "proposal changed before fit",
            )
        result = self._original_fit(*args, **kwargs)
        require(
            tree_equal((proposals, states, rays, targets), snapshot), "original fit inputs changed"
        )
        poses, local, parts, masks = result
        arrays.update(
            fitted_pose=poses.copy(),
            local_cost=local.copy(),
            loss_parts=parts.copy(),
            mask_supported=masks.copy(),
        )
        arrays["path_cost"] = local + arrays["parent_cost"][arrays["proposal_parent"]]
        for key, value in support_snapshot(
            poses, arrays["proposal_plan"], snapshot[2], self.tracker.rig
        ).items():
            arrays["fitted_" + key] = value
        if targets is not None:
            for key in ("target", "weights", "foreground"):
                arrays["fit_" + key] = getattr(targets, key).copy()
            for view in VIEWS:
                arrays[view + "_fit_pixels"] = targets.pixels[view].copy()
        self._active["metadata"]["fit_targets_present"] = targets is not None
        return result

    def _complete(self):
        arrays = self._active["arrays"]
        native = self._active["frame_idx"]
        require(
            "fitted_pose" in arrays and self.tracker.previous_frame == native,
            "diagnostic frame did not complete original tracking step",
        )
        selected, order = selected_indices(
            arrays["fitted_pose"],
            arrays["path_cost"],
            arrays["proposal_parent"],
            arrays["proposal_permutation"],
            self.tracker.config.beam_size,
        )
        records = self.tracker._history[-1]
        require(len(records) == len(selected), "selected beam length differs")
        matches = []
        for index, record in zip(selected, records, strict=True):
            require(
                record["frame_idx"] == native
                and record["parent"] == arrays["proposal_parent"][index]
                and record["proposal_kind"] == self._active["metadata"]["proposal_kind"][index],
                "selected record label/parent/frame differs",
            )
            for field, key in dict(
                pose_xyz="fitted_pose",
                costs="local_cost",
                path_costs="path_cost",
                loss_parts="loss_parts",
                mask_supported="mask_supported",
                sources="fitted_geometric_sources",
                observed="fitted_geometric_observed",
            ).items():
                require(
                    identical(record[field], arrays[key][index]),
                    "selected record differs: " + field,
                )
            matches.append(index)
        arrays["selected_indices"] = np.asarray(matches, np.int64)
        arrays["stable_path_order"] = order.copy()
        arrays["selected_mask"] = np.isin(np.arange(len(arrays["local_cost"])), selected)
        for key in (
            "identity_review",
            "body_supported",
            "missing",
            "node_missing",
            "direct_pectoral_age",
            "score_margin",
        ):
            arrays["selected_" + key] = np.asarray([r[key] for r in records]).copy()
        arrays["generated_support_lost_in_fit"] = (
            arrays["generated_strict_observed"] & ~arrays["fitted_strict_observed"]
        )
        arrays["fit_support_gained"] = (
            ~arrays["generated_strict_observed"] & arrays["fitted_strict_observed"]
        )
        arrays["fitted_support_unselected"] = (
            arrays["fitted_strict_observed"] & ~arrays["selected_mask"][:, None, None]
        )
        self._active["metadata"].update(
            stage_facts(
                arrays["generated_strict_observed"], arrays["fitted_strict_observed"], selected
            )
        )
        self._active["metadata"].update(
            ray_gate_mm=2.0,
            pixel_gate=12.0,
            node_order=["mouth", "pectoral", "tail"],
            views=list(VIEWS),
            selected_records_byte_exact=True,
            local_cost_scope=(
                "Original retained best-iterate fit score; "
                "path adds the saved parent cumulative cost."
            ),
            original_return_objects_preserved=True,
            independent_ID_truth=False,
        )
        require(
            type(self.tracker) is self._type and tree_equal(self.tracker._context(), self._context),
            "tracker context differs after frame",
        )
        self._records.append(copy.deepcopy(self._active))
