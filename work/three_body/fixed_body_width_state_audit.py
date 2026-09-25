"""CPU semantic validation view for fixed-width GPU checkpoints.

No checkpoint metadata is rewritten and no GPU tensor is constructed. A normal
CPU context computes the registered radial formula; the saved device/R0 numeric
representation is then bound in a separate, non-executable validation view.
This is neither a restore nor CPU/CUDA bitwise restore qualification.
Filesystem membership, successful producer seal and source pins belong to caller.
"""

from __future__ import annotations

import copy

import fixed_body_width_tracker as tracking
import fixed_body_width_tracker_v2 as runtime
import numpy as np
from node_recovery import NodeRecoveryConfig
from observations import FrameObservation
from persistent_node_tracker import PersistentNodeConfig
from tracker import TrackerConfig

require = tracking.require
RADIAL_ATOL = 1e-12


def _readonly(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def _forbidden(*args, **kwargs):
    raise RuntimeError("read-only checkpoint validation view cannot execute or restore a tracker")


class _ReadOnlyValidationView(runtime.FixedBodyWidthTracker):
    """Pure inherited validators only, with every computational entry blocked."""

    __init__ = begin = advance = track = finish = _forbidden
    restore_state = restore_transport_state = export_state = export_transport_state = _forbidden
    _fit = _proposals = _tensor = _advance_owned = validate_fixed_width_renderer = _forbidden

    def _context(self):
        context = super()._context()
        # This class is a validator, not a new scientific implementation.
        context["implementation"] = runtime.FixedBodyWidthTracker.__name__
        return context


def _cpu_tracker(rig, reference_pose, original_context, **options):
    config = copy.deepcopy(original_context["tracker_config"])
    config["device"] = "cpu"
    return runtime.FixedBodyWidthTracker(
        rig,
        reference_pose.copy(),
        config=TrackerConfig(**config),
        recovery_config=NodeRecoveryConfig(**original_context["recovery_config"]),
        persistence_config=PersistentNodeConfig(**original_context["persistence_config"]),
        vote_provider=_forbidden,
        **options,
    )


def _validation_view(cpu, metadata, content):
    """Validate allowed representation differences before binding original metadata."""
    context = metadata["context"]
    saved_policy = context["mask_transport"]["fixed_width"]
    reference = content["fixed_width_reference"]
    require(
        set(reference) == {"pose", "radial_sigma_mm", "policy"},
        "saved fixed-width reference fields differ",
    )
    require(reference["policy"] == saved_policy, "saved reference policy/context differs")
    tracking.same(reference["pose"], cpu._fixed_width_reference_pose, "saved registered seed")
    radial = reference["radial_sigma_mm"]
    require(
        isinstance(radial, np.ndarray)
        and radial.dtype == np.float64
        and radial.shape == cpu._fixed_width_radial.shape
        and np.isfinite(radial).all()
        and np.all(radial > 0)
        and tracking.signature(radial) == saved_policy["radial_sigma_mm"],
        "saved fixed-width radial signature/schema differs",
    )
    require(
        np.allclose(radial, cpu._fixed_width_radial, atol=RADIAL_ATOL, rtol=0),
        "saved radial widths differ from registered CPU formula",
    )
    # Exactly two representation entries may change: device name and radial
    # array SHA (also embedded as the last renderer signature entry).
    expected_width = cpu._width_policy()
    expected_width["reference_device"] = str(
        runtime.resolved_device(context["tracker_config"]["device"])
    )
    expected_width["radial_sigma_mm"] = tracking.signature(radial)
    expected_width["renderer_signature"][-1] = [
        radial.dtype.str,
        list(radial.shape),
        tracking.signature(radial)["sha256"],
    ]
    require(saved_policy == expected_width, "unallowed fixed-width context/reference difference")
    expected_context = cpu._context()
    expected_context["tracker_config"]["device"] = context["tracker_config"]["device"]
    expected_context["mask_transport"]["fixed_width"] = copy.deepcopy(expected_width)
    require(context == expected_context, "checkpoint science/input context differs")
    view = object.__new__(_ReadOnlyValidationView)
    view.__dict__ = copy.copy(cpu.__dict__)
    view.device = runtime.resolved_device(context["tracker_config"]["device"])  # enum only
    view.config = TrackerConfig(**context["tracker_config"])
    view._fixed_width_reference_pose = _readonly(reference["pose"])
    view._fixed_width_radial = _readonly(radial)
    view._fixed_width_signature = _tuples(expected_width["renderer_signature"])
    view._fixed_width_policy = copy.deepcopy(expected_width)
    view._fixed_width_policy_sha = tracking.json_hash(expected_width)
    # This assignment happens only after full expected-context equality above.
    view._initial_context = copy.deepcopy(context)
    require(view._context() == context, "read-only validation view context differs")
    return view


def validate_state(
    core,
    metadata,
    side_arrays,
    side_metadata,
    *,
    rig,
    reference_pose,
    original_context,
    first_native,
    group_counts,
    query_sha256,
    input_binding,
    coefficient,
    expected_frames=200,
):
    """Return (unmodified core, decoded side content, report), with no state restore.

    Caller supplies arrays from a successfully sealed/pinned producer, plus its
    separately bound seed, rig, original science context, query table and plan.
    expected_frames is explicit for small fixtures; production is fixed at 200.
    Saved R0 bytes remain authoritative after the registered CPU formula check
    at fixed atol=1e-12/rtol=0. That tolerance does not change any tracker gate.
    """
    require(
        isinstance(reference_pose, np.ndarray)
        and reference_pose.dtype == np.float64
        and reference_pose.shape == (3, 3, 3)
        and np.isfinite(reference_pose).all(),
        "bound original float64 seed required",
    )
    require(
        type(expected_frames) is int and 1 <= expected_frames <= 200,
        "fixed completed frame count required",
    )
    require(
        len(core) == 39
        and metadata["processed_frames"] == expected_frames
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == 0
        and metadata["previous_frame"] == first_native + expected_frames - 1,
        "fixed-width checkpoint scope/lifecycle differs",
    )
    before = dict(
        core={k: tracking.signature(v) for k, v in core.items()},
        side={k: tracking.signature(v) for k, v in side_arrays.items()},
        metadata=tracking.json_hash(metadata),
        side_metadata=tracking.json_hash(side_metadata),
        reference=tracking.signature(reference_pose),
        original_context=tracking.json_hash(original_context),
        binding=tracking.json_hash(input_binding),
    )
    cpu = _cpu_tracker(
        rig,
        reference_pose,
        original_context,
        first_native=first_native,
        group_counts=group_counts,
        query_sha256=query_sha256,
        input_binding=input_binding,
        coefficient=coefficient,
    )
    require(cpu.initialize_count == cpu.restore_count == 0, "CPU context initialized state")
    science = copy.deepcopy(cpu._context())
    science.pop("mask_transport")
    science["implementation"] = original_context["implementation"]
    science["tracker_config"]["device"] = original_context["tracker_config"]["device"]
    require(science == original_context, "supplied seed/rig differ from original science context")
    require(
        metadata["context"]["tracker_config"]["device"]
        == original_context["tracker_config"]["device"],
        "checkpoint execution device differs from original context",
    )
    content = tracking.decode_sidecar(side_arrays, side_metadata)
    view = _validation_view(cpu, metadata, content)
    history, states, _observation, _diagnostics = view._decode_state(core, metadata)
    checked_content = view._validated_sidecar(core, metadata, side_arrays, side_metadata)
    # Repeat the original joint-restore's t0 geometry check without restoring.
    initial = checked_content["initial_observation"]
    initial_frame = FrameObservation(first_native, initial["detections"], {}, initial["masks"], {})
    records = [fact["record"] for fact in checked_content["anchor_facts"]]
    anchors, _ = tracking.frozen.initial_anchors(records, initial_frame, rig, tuple(group_counts))
    for anchor, saved in zip(anchors, checked_content["anchors"], strict=True):
        for vi in range(3):
            tracking.same(anchor.links[vi], saved["links"][vi], "audited initial geometry links")
    require(
        view.initialize_count == view.restore_count == 0 and view._states is None,
        "validation view executed state initialization",
    )
    require(
        before["core"] == {k: tracking.signature(v) for k, v in core.items()}
        and before["side"] == {k: tracking.signature(v) for k, v in side_arrays.items()}
        and before["metadata"] == tracking.json_hash(metadata)
        and before["side_metadata"] == tracking.json_hash(side_metadata)
        and before["reference"] == tracking.signature(reference_pose)
        and before["original_context"] == tracking.json_hash(original_context)
        and before["binding"] == tracking.json_hash(input_binding),
        "state audit input changed",
    )
    radial = checked_content["fixed_width_reference"]["radial_sigma_mm"]
    report = dict(
        schema="fixed_body_width_read_only_state_audit_v1",
        passed=True,
        native_interval=[first_native, first_native + expected_frames],
        core_arrays=len(core),
        retained_fields=len(tracking.RECORD_FIELDS),
        history_frames=len(history),
        history_rows=int(core["history_offsets"][-1]),
        retained_terminal_states=len(states),
        initial_anchors=len(anchors),
        original_metadata_sha256=before["metadata"],
        sidecar_content_sha256=side_metadata["content_sha256"],
        reference_pose_signature=before["reference"],
        saved_radial_signature=tracking.signature(radial),
        cpu_radial_signature=tracking.signature(cpu._fixed_width_radial),
        radial_formula_atol=RADIAL_ATOL,
        radial_formula_rtol=0.0,
        radial_formula_max_abs=float(np.max(np.abs(radial - cpu._fixed_width_radial))),
        radial_formula_byte_exact=tracking.signature(radial)
        == tracking.signature(cpu._fixed_width_radial),
        saved_device=str(view.device),
        computation_device="cpu",
        metadata_rewritten=False,
        inputs_unchanged=True,
        semantic_core_and_sidecar_verified=True,
        initial_geometry_links_verified=True,
        state_initializations=0,
        restore_calls=0,
        advance_calls=0,
        fit_calls=0,
        GPU_used=False,
        cross_device_restore_qualified=False,
    )
    return core, checked_content, report
