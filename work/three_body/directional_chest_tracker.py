"""Experimental continuous branch using the qualified local chest policy.

All temporal updates remain inherited. Importing an original checkpoint is an
explicit context migration: the original implementation validates it first,
then only the declared implementation/policy context is changed in a copy.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
from continuous_tracker import (
    AcceleratedContinuousPersistentNodeTracker,
    ContinuousPersistentNodeTracker,
)
from directional_device_fitter import DirectionalDeviceFitter
from node_geometry import assignment_plans, node_pools, recovery_poses, supported_nodes
from tracker import Proposal


def chest_proposals(original, states, rays, tracker):
    main = original[0]
    if main.kind != "predict" or main.parent != 0 or main.permutation != (0, 1, 2):
        raise ValueError("original first proposal must be the first parent's prediction")
    parent = states[0]
    _, observed, _ = supported_nodes(main.pose, main.plan, rays, tracker.rig)
    age = parent.node_missing[:, 1]
    eligible = (age >= 4) & ~observed[:, 1]
    enabled = bool(not parent.missing.any() and eligible.any())
    result = []
    diagnostic = dict(
        enabled=enabled,
        eligible_slots=np.flatnonzero(eligible).tolist(),
        generated=0,
        admitted=0,
        excluded_by_budget=0,
    )
    if enabled:
        pools = node_pools(tracker.rig, rays, tracker.lower, tracker.upper)
        extra = recovery_poses(
            main.pose,
            np.where(eligible, age, 0),
            tracker.lengths,
            pools,
            main.plan,
            tracker.lower,
            tracker.upper,
            keep=3,
        )
        for _, pose, _ in extra:
            plan = assignment_plans(
                pose, rays, tracker.config.ray_sigma_mm, tracker.config.missing_cost, keep=1
            )[0][1]
            result.append(Proposal(pose, main.predicted, plan, 0, "chest_recovery", (0, 1, 2)))
        available = max(0, tracker.config.max_hypotheses + 6 - len(original))
        diagnostic.update(
            generated=len(result),
            admitted=min(len(result), available),
            excluded_by_budget=max(0, len(result) - available),
        )
        result = result[:available]
    return result, diagnostic


class DirectionalChestContinuousTracker(ContinuousPersistentNodeTracker):
    def __init__(self, *args, **kwargs):
        self.directional = True
        super().__init__(*args, **kwargs)
        if self.config.mode != "detectron2" or self.device.type not in ("cpu", "cuda"):
            raise ValueError("the registered chest branch requires D2 and CPU/CUDA")
        self.last_diagnostics = None
        # Per-frame logging only; this is not used in a future state transition.
        self.last_chest_proposal_diagnostic = None

    def _context(self):
        context = super()._context()
        context["experimental_chest_policy"] = dict(
            schema="directional_chest_policy_v1",
            directional=self.directional,
            minimum_chest_age=4,
            missing_body_gate_closed=True,
            extra_keep=3,
            total_cap="max_hypotheses+6; preserve every original proposal",
            parallel_axis="current predict single strict source water ray",
            sigma_increment=0.3,
            age_cap=20,
        )
        return context

    def _proposals(self, states, rays, flow=None):
        original = super()._proposals(states, rays, flow)
        extra, diagnostic = chest_proposals(original, states, rays, self)
        self.last_chest_proposal_diagnostic = diagnostic
        return original + extra

    _fit = DirectionalDeviceFitter._fit


def array_signature(arrays):
    return {
        key: dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        )
        for key, value in arrays.items()
    }


def import_original_state(target, original, arrays, metadata):
    """Validate the original, retain every array, and record an explicit migration."""
    if (
        type(target) is not DirectionalChestContinuousTracker
        or type(original) is not AcceleratedContinuousPersistentNodeTracker
    ):
        raise ValueError("explicit original and experimental implementations required")
    if target.initialize_count or original.initialize_count:
        raise ValueError("migration requires two fresh uninitialized objects")
    old_context, new_context = original._context(), target._context()
    if metadata.get("context") != old_context:
        raise ValueError("original metadata does not match its original implementation")
    expected = copy.deepcopy(new_context)
    expected.pop("experimental_chest_policy")
    expected["implementation"] = old_context["implementation"]
    if expected != old_context:
        raise ValueError(
            "migration cannot change calibration, device, seed or existing configuration"
        )
    before = array_signature(arrays)
    original_metadata = copy.deepcopy(metadata)
    original._decode_state(arrays, metadata)
    adapted = copy.deepcopy(metadata)
    adapted["context"] = new_context
    target.restore_state(arrays, adapted)
    restored_arrays, restored_metadata = target.export_state()
    if (
        before != array_signature(arrays)
        or before != array_signature(restored_arrays)
        or metadata != original_metadata
    ):
        raise ValueError("migration changed an original array or metadata")
    if (
        target.initialize_count != 1
        or original.initialize_count != 0
        or target.restore_count != metadata["restore_count"] + 1
        or target.processed_frames != metadata["processed_frames"]
    ):
        raise ValueError("migration changed the logical lifecycle")
    return dict(
        schema="directional_chest_original_state_import_v1",
        status="ok",
        old_context=old_context,
        new_context=new_context,
        original_metadata=json.loads(json.dumps(original_metadata)),
        imported_metadata=restored_metadata,
        arrays=before,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        original_implementation_not_claimed_for_new_tracking=True,
    )
