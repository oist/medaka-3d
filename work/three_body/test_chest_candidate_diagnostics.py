"""Bounded CPU recorder parity, ownership and stage-selection facts."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
import chest_candidate_diagnostics as m  # noqa: E402

import test_three_body_continuous_tracker as fixture  # noqa: E402


class SpyTracker(fixture.C.ContinuousPersistentNodeTracker):
    """Only test instrumentation; production wrapper never subclasses its tracker."""

    def _proposals(self, states, rays, flow=None):
        result = super()._proposals(states, rays, flow)
        self.returned_proposals = result
        return result

    def _fit(self, proposals, states, rays, targets):
        assert proposals is self.returned_proposals
        result = super()._fit(proposals, states, rays, targets)
        self.returned_fit = result
        return result


@pytest.fixture
def setup():
    rig = fixture.SmallRig(
        tuple(fixture.SmallCamera(axes, np.zeros(2)) for axes in ((0, 1), (0, 2), (1, 2)))
    )
    centers = np.array([[45.0, 48.0, 50.0], [95.0, 100.0, 92.0], [150.0, 145.0, 148.0]])
    pose = (
        centers[:, None] + np.array([[8.0, 6.0, 4.0], [0.0, 0.0, 0.0], [-14.0, -10.0, -6.0]])[None]
    )
    return rig, pose


@pytest.mark.parametrize("mode", ["nodes", "detectron2"])
def test_original_full_loop_array_parity_context_and_method_return_identity(setup, mode):
    rig, pose = setup
    previous = fixture.torch.get_num_threads()
    fixture.torch.set_num_threads(1)
    try:
        plain = SpyTracker(rig, pose, fixture.config(flow_proposals=True, mode=mode))
        traced = SpyTracker(rig, pose, fixture.config(flow_proposals=True, mode=mode))
        frames = fixture.textured_frames(rig, pose)
        original = copy.deepcopy(frames)
        plain.begin()
        traced.begin()
        context = copy.deepcopy(traced._context())
        before_dict = set(traced.__dict__)
        with m.CandidateDiagnostics(traced, frame_range=(601, 603)) as diag:
            assert type(traced) is SpyTracker and traced._context() == context
            for frame in frames:
                plain.advance(frame)
                with diag.frame(frame.frame_idx):
                    # Return object identity reaches advance unchanged, not copied arrays.
                    traced.advance(frame)
                    assert traced._states[0].pose.base is traced.returned_fit[0]
            data = diag.records
            assert [r["frame_idx"] for r in data] == [601, 602]
            for record in data:
                a = record["arrays"]
                assert 1 <= len(a["selected_indices"]) <= 2
                assert record["metadata"]["selected_records_byte_exact"] is True
                assert 1 <= len(a["parent_pose"]) <= 2
                assert all(not v.dtype.hasobject for v in a.values())
            data[0]["arrays"]["proposal_pose"][:] = 0
            assert diag.records[0]["arrays"]["proposal_pose"].any()
        assert "_fit" not in traced.__dict__ and "_proposals" not in traced.__dict__
        assert type(traced) is SpyTracker and traced._context() == context
        state1, meta1 = plain.export_state()
        state2, meta2 = traced.export_state()
        assert len(state1) == 39 and m.tree_equal(state1, state2) and meta1 == meta2
        result1, result2 = plain.finish(), traced.finish()
        assert len(result1) == 21 and m.tree_equal(result1, result2)
        assert m.tree_equal(frames, original)
        assert before_dict <= set(traced.__dict__)
    finally:
        fixture.torch.set_num_threads(previous)


def test_frozen_source_gate_and_exclusivity_are_distinct_and_no_input_mutation(setup):
    rig, pose = setup
    points = {v: rig.camera(v).project(pose) for v in rig.views}
    rays = m.RayObservations.from_points(rig, points)
    plans = np.broadcast_to(np.arange(3, dtype=np.int32)[None, :, None, None], (1, 3, 3, 3)).copy()
    poses = pose[None].copy()
    old = copy.deepcopy((poses, plans, rays))
    good = m.support_snapshot(poses, plans, rays, rig)
    assert good["strict_observed"].all() and good["ray_gate_pass"].all()
    assert not good["plan_source_collision"].any()
    # Two latent bodies can reuse a detection without the geometry function rejecting it.
    poses[0, 1] = pose[0]
    plans[0, 1] = plans[0, 0]
    duplicate = m.support_snapshot(poses, plans, rays, rig)
    assert duplicate["geometric_observed"].all()
    assert not duplicate["strict_observed"][0, :2].any()
    assert duplicate["strict_observed"][0, 2].all()
    assert duplicate["plan_source_collision"][0, :2].all()
    assert m.tree_equal(rays, old[2])
    assert m.tree_equal(good["geometric_sources"], good["strict_sources"])


def test_ray_and_pixel_threshold_failures_remain_diagnostically_separate(setup):
    rig, pose = setup
    points = {v: rig.camera(v).project(pose) for v in rig.views}
    rays = m.RayObservations.from_points(rig, points)
    plan = np.broadcast_to(np.arange(3, dtype=np.int32)[:, None, None], (3, 3, 3)).copy()
    rig.cameras[0].offset[:] = [13.0, 0.0]
    pixel = m.support_snapshot(pose[None], plan[None], rays, rig)
    assert pixel["ray_gate_pass"].all() and not pixel["pixel_gate_pass"][:, :, 0].any()
    assert pixel["strict_observed"].all()  # remaining two views still support
    rig.cameras[0].offset[:] = 0
    displaced = pose.copy()
    displaced[0, 1, 0] += 2.01
    shifted = m.support_snapshot(displaced[None], plan[None], rays, rig)
    assert not shifted["ray_gate_pass"][0, 0, :2, 1].any()
    assert shifted["pixel_gate_pass"][0, 0, :, 1].all()
    assert not shifted["strict_observed"][0, 0, 1]


def test_stage_facts_use_scene_and_per_slot_existence_without_claiming_identity():
    generated = np.zeros((4, 3, 3), bool)
    fitted = np.zeros_like(generated)
    generated[0, 0, 1] = True  # generated, then lost
    fitted[1, 1, 1] = True  # fitted but discarded
    generated[2, 2, 1] = fitted[2, 2, 1] = True  # selected
    result = m.stage_facts(generated, fitted, np.array([2]))
    assert [p["stage_label"] for p in result["per_slot"]] == [
        "generated support lost in fit",
        "fit support not selected",
        "selected strict support exists",
    ]
    assert result["all_three_chests_supported_scene_exists"] == dict(
        generated=False, fitted=False, selected=False
    )
    assert "biological" in result["scope"]
    empty = m.stage_facts(np.zeros_like(generated), np.zeros_like(generated), np.array([0]))
    assert all(p["stage_label"] == "no generated strict support" for p in empty["per_slot"])


def test_stable_selection_preserves_distinct_parent_and_permutation_at_same_geometry(setup):
    _, pose = setup
    poses = np.repeat(pose[None], 5, axis=0)
    poses[4] += 1
    cost = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    parents = np.array([0, 0, 1, 0, 0])
    permutations = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [1, 0, 2], [0, 1, 2]])
    selected, order = m.selected_indices(poses, cost, parents, permutations, 4)
    assert order.tolist() == [0, 1, 2, 3, 4]
    assert selected.tolist() == [0, 2, 3, 4]


def test_exception_restores_instance_methods_and_does_not_record_partial_frame(setup):
    rig, pose = setup
    tracker = SpyTracker(rig, pose, fixture.config())
    tracker.begin()
    original_fit = tracker._fit.__func__
    with pytest.raises(RuntimeError, match="stop fixture"):  # noqa: SIM117
        with m.CandidateDiagnostics(tracker, frame_range=(10, 11)) as diag:
            with diag.frame(10):
                raise RuntimeError("stop fixture")
    assert tracker._fit.__func__ is original_fit and "_fit" not in tracker.__dict__
    assert diag.records == [] and tracker.processed_frames == 0


def test_refuses_existing_instance_override_without_touching_it(setup):
    rig, pose = setup
    tracker = SpyTracker(rig, pose, fixture.config())

    def callback(*a):
        return None

    tracker._fit = callback
    with pytest.raises(ValueError, match="existing instance override"):  # noqa: SIM117
        with m.CandidateDiagnostics(tracker):
            pass
    assert tracker._fit is callback and "_proposals" not in tracker.__dict__


def test_record_to_candidate_mismatch_fails_instead_of_guessing_identity(setup):
    rig, pose = setup
    tracker = SpyTracker(rig, pose, fixture.config(opt_steps=1))
    tracker.begin()
    with m.CandidateDiagnostics(tracker, frame_range=(10, 11)) as diag:  # noqa: SIM117
        with pytest.raises(ValueError, match="selected record differs"):
            with diag.frame(10):
                tracker.advance(fixture.frame(rig, pose, 10))
                tracker._history[-1][0]["sources"][0, 0, 0] = -1  # deliberate fixture corruption
    assert diag.records == []


@pytest.mark.parametrize("native", [True, 10.5])
def test_invalid_native_frame_is_not_silently_cast(setup, native):
    rig, pose = setup
    tracker = SpyTracker(rig, pose, fixture.config())
    with m.CandidateDiagnostics(tracker) as diag:  # noqa: SIM117
        with pytest.raises(ValueError, match="native frame"):
            with diag.frame(native):
                pass


def test_both_wrapped_methods_return_the_exact_original_objects(setup):
    rig, pose = setup
    tracker = SpyTracker(rig, pose, fixture.config(opt_steps=1))
    tracker.begin()
    points = {v: rig.camera(v).project(pose) for v in rig.views}
    rays = m.RayObservations.from_points(rig, points)
    original = copy.deepcopy(tracker._states)
    with pytest.raises(RuntimeError, match="no advance in this fixture"):  # noqa: SIM117
        with m.CandidateDiagnostics(tracker, frame_range=(10, 11)) as diag:
            with diag.frame(10):
                proposals = tracker._proposals(tracker._states, rays)
                assert proposals is tracker.returned_proposals
                fit = tracker._fit(proposals, tracker._states, rays, targets=None)
                assert fit is tracker.returned_fit
                assert m.tree_equal(tracker._states, original)
                raise RuntimeError("no advance in this fixture")
    assert diag.records == []
