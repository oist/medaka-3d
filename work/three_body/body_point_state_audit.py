"""CPU semantic audit of body-point GPU checkpoints without metadata rewriting.

A normal CPU constructor establishes the independently bound scientific context.
A separate non-executable view binds the original device enum after exact context
comparison. No restore, tracking, rendering, fitting, GPU tensor or model runs.
Successful producer/source/input and filesystem seals are the caller's duty.
"""

from __future__ import annotations

import copy

import body_point_tracker as tracking
import mask_transport_tracker as original_transport
import numpy as np
import torch
from node_recovery import NodeRecoveryConfig
from observations import FrameObservation
from persistent_node_tracker import PersistentNodeConfig
from tracker import TrackerConfig

require = tracking.require


def _forbidden(*args, **kwargs):
    raise RuntimeError("read-only checkpoint validation view cannot execute or restore a tracker")


class _ReadOnlyValidationView(tracking.BodyPointTracker):
    __init__ = begin = advance = track = finish = _forbidden
    restore_state = restore_transport_state = export_state = export_transport_state = _forbidden
    _fit = _proposals = _tensor = _advance_owned = _forbidden

    def _context(self):
        context = super()._context()
        context["implementation"] = tracking.BodyPointTracker.__name__
        return context


def _cpu_tracker(rig, reference_pose, original_context, **options):
    config = copy.deepcopy(original_context["tracker_config"])
    config["device"] = "cpu"
    return tracking.BodyPointTracker(
        rig,
        reference_pose.copy(),
        config=TrackerConfig(**config),
        recovery_config=NodeRecoveryConfig(**original_context["recovery_config"]),
        persistence_config=PersistentNodeConfig(**original_context["persistence_config"]),
        vote_provider=_forbidden,
        **options,
    )


def _validation_view(cpu, metadata):
    context = metadata["context"]
    expected = cpu._context()
    expected["tracker_config"]["device"] = context["tracker_config"]["device"]
    require(context == expected, "checkpoint science/input context differs")
    view = object.__new__(_ReadOnlyValidationView)
    view.__dict__ = copy.copy(cpu.__dict__)
    view.device = torch.device(context["tracker_config"]["device"])  # Enum, no tensor allocation.
    view.config = TrackerConfig(**context["tracker_config"])
    # Binding occurs only after full independently reconstructed context equality.
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
    Saved metadata remains intact. Only the execution-device enum differs from
    the CPU constructor, after exact validation of every other context field.
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
        "body-point checkpoint scope/lifecycle differs",
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
    tracking.decode_sidecar(side_arrays, side_metadata)
    view = _validation_view(cpu, metadata)
    history, states, _observation, _diagnostics = view._decode_state(core, metadata)
    checked_content = view._validated_sidecar(core, metadata, side_arrays, side_metadata)
    # Repeat the original joint-restore's t0 geometry check without restoring.
    initial = checked_content["initial_observation"]
    initial_frame = FrameObservation(first_native, initial["detections"], {}, initial["masks"], {})
    records = [fact["record"] for fact in checked_content["anchor_facts"]]
    anchors, _ = original_transport.initial_anchors(
        records, initial_frame, rig, tuple(group_counts)
    )
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
    report = dict(
        schema="body_point_read_only_state_audit_v1",
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
