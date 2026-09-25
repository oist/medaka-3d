"""Declared development sensitivity: only the engineering z minimum becomes -12 mm.

The old -6 mm boundary was development padding, not a measured tank floor.
This separate branch retains the sparse instance scorer and all existing
proposal/support/temporal code. Its enlarged search domain may change both
proposals and fitted trajectories; no physical depth or identity truth is claimed.
No live tracker is reconfigured. Only a fresh constructor changes its own box.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from directional_chest_tracker import array_signature
from fast_tail_recovery_tracker import FastTailRecoveryContinuousTracker
from sparse_instance_rescore_tracker import SparseInstanceRescoreContinuousTracker

SPARSE_TRACKER_SHA = "d041aff3a0d8cbd71341dbe67dd9b5a7a1e2e7aeb165f55e4b08f8fabdb8fb79"
FAST_TAIL_SHA = "fbaa1a81f337e291f487886ace1fbceb7ac9adf95db86b82896904d42cafa3ad"


@dataclass(frozen=True)
class ExpandedDepthPolicy:
    effective_from_native: int = field(default=76360, init=False)
    original_lower_z_mm: float = field(default=-6.0, init=False)
    expanded_lower_z_mm: float = field(default=-12.0, init=False)


DEPTH_POLICY = ExpandedDepthPolicy()


class ExpandedDepthInstanceRescoreContinuousTracker(SparseInstanceRescoreContinuousTracker):
    def __init__(self, *args, sparse_effective_from_native=76360, **kwargs):
        if (
            type(sparse_effective_from_native) is not int
            or sparse_effective_from_native != DEPTH_POLICY.effective_from_native
        ):
            raise ValueError("this depth sensitivity and sparse backend start at native 76360")
        # Frozen constructors first validate the existing tracker/recovery box.
        super().__init__(*args, sparse_effective_from_native=sparse_effective_from_native, **kwargs)
        if (
            self.initialize_count != 0
            or self._states is not None
            or self._history
            or self.previous_frame is not None
            or self._context() != self._initial_context
            or not np.array_equal(self.lower, [0.0, 0.0, DEPTH_POLICY.original_lower_z_mm])
            or not np.array_equal(self.lower, self.recovery_config.lower_mm)
        ):
            raise ValueError("depth configuration requires a fresh validated original box")
        lower = self.lower.copy()
        lower[2] = DEPTH_POLICY.expanded_lower_z_mm
        recovery = replace(self.recovery_config, lower_mm=tuple(lower.tolist()))
        # Constructor-only finalization: both consumers use exactly the same box.
        self.lower, self.recovery_config = lower, recovery
        self._initial_context = self._context()

    def _context(self):
        context = super()._context()
        context["expanded_depth_policy"] = dict(
            schema="expanded_depth_instance_rescore_policy_v1",
            **asdict(DEPTH_POLICY),
            scope="development sensitivity of engineering lower z boundary only",
            rationale_document="labs/20260908_generative_tracking/REACQUISITION_PLAN.md:19",
            sparse_tracker_source_sha256=SPARSE_TRACKER_SHA,
            fast_tail_source_sha256=FAST_TAIL_SHA,
            x_y_bounds_and_upper_z_unchanged=True,
            calibration_and_coordinate_gauge_unchanged=True,
            ray_rms_anatomy_support_thresholds_unchanged=True,
            mask_scientific_loss_and_normalization_unchanged=True,
            proposal_and_temporal_update_code_unchanged=True,
            scientific_search_domain_changed=True,
            proposed_or_fitted_trajectory_equivalence_claimed=False,
            old_history_rewritten=False,
            depth_policy_applied_retroactively_to_prior_history=False,
            runtime_configuration_mutation_allowed=False,
            development_diagnostics_reused=True,
            measured_floor_or_physical_depth_truth_claimed=False,
            permanent_IDs_assigned=False,
            clears_identity_review=False,
        )
        return context

    def export_instance_diagnostics(self):
        # The inherited copy is still logging only; never a temporal input.
        result = super().export_instance_diagnostics()
        result["expanded_depth_policy"] = copy.deepcopy(self._context()["expanded_depth_policy"])
        return result


def import_fast_tail_state(target, original, arrays, metadata):
    """Validate real FastTail codec before explicitly migrating all 39 arrays.

    Native boundary is 76360; lifecycle counts otherwise come from valid source
    metadata. The scientific caller must separately bind the audited source seal
    and its real 26560 retained frames, initialization 1 and restore 4 -> 5.
    """
    if (
        type(target) is not ExpandedDepthInstanceRescoreContinuousTracker
        or type(original) is not FastTailRecoveryContinuousTracker
    ):
        raise ValueError("explicit FastTail and expanded-depth implementations required")
    if target.initialize_count or original.initialize_count:
        raise ValueError("migration requires fresh uninitialized trackers")
    old_context, new_context = original._context(), target._context()
    if metadata.get("context") != old_context:
        raise ValueError("source metadata does not match its FastTail implementation")
    lower_signatures = array_signature(
        dict(old=np.array([0.0, 0.0, -6.0]), new=np.array([0.0, 0.0, -12.0]))
    )
    if (
        target._initial_context != new_context
        or original._initial_context != old_context
        or old_context["lower"] != lower_signatures["old"]
        or old_context["recovery_config"]["lower_mm"] != [0.0, 0.0, -6.0]
        or new_context["lower"] != lower_signatures["new"]
        or new_context["recovery_config"]["lower_mm"] != [0.0, 0.0, -12.0]
    ):
        raise ValueError("only the declared constructor-finalized depth change is permitted")
    expected = copy.deepcopy(new_context)
    for policy in ("instance_rescore_policy", "sparse_distance_backend", "expanded_depth_policy"):
        expected.pop(policy)
    expected["implementation"] = old_context["implementation"]
    expected["lower"] = lower_signatures["old"]
    expected["recovery_config"]["lower_mm"][2] = DEPTH_POLICY.original_lower_z_mm
    if expected != old_context:
        raise ValueError("migration cannot change other bounds, calibration, seed or policies")
    previous = metadata.get("previous_frame")
    if type(previous) is not int or previous + 1 != DEPTH_POLICY.effective_from_native:
        raise ValueError("source previous frame must immediately precede native 76360")
    if len(arrays) != 39:
        raise ValueError("this transition requires all 39 FastTail state arrays")
    before, original_metadata = array_signature(arrays), copy.deepcopy(metadata)
    original._decode_state(arrays, metadata)
    adapted = copy.deepcopy(metadata)
    adapted["context"] = new_context
    target.restore_state(arrays, adapted)
    restored, restored_metadata = target.export_state()
    expected_metadata = copy.deepcopy(adapted)
    expected_metadata["restore_count"] += 1
    if (
        before != array_signature(arrays)
        or before != array_signature(restored)
        or metadata != original_metadata
        or restored_metadata != expected_metadata
        or target.initialize_count != 1
        or original.initialize_count != 0
        or target.restore_count != metadata["restore_count"] + 1
        or target.processed_frames != metadata["processed_frames"]
        or target.previous_frame != previous
    ):
        raise ValueError(
            "depth migration changed historical bytes or undeclared lifecycle metadata"
        )
    return dict(
        schema="expanded_depth_fast_tail_state_import_v1",
        status="ok",
        old_context=old_context,
        new_context=new_context,
        original_metadata=original_metadata,
        imported_metadata=restored_metadata,
        arrays=before,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        processed_frames=target.processed_frames,
        previous_frame=target.previous_frame,
        effective_from_native=DEPTH_POLICY.effective_from_native,
        depth_policy_applied_retroactively_to_prior_history=False,
        review_cleared_by_migration=False,
        companion_diagnostics_required_each_future_frame=True,
    )
