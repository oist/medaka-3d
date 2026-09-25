"""New six-arm evaluator boundaries; no payload, renderer, model or GPU calls."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import evaluate_fixed_body_width200 as m

ARM_AXIS_ZERO = {
    "rendered_union": (6, 3, 5),
    "rendered_ownership": (6, 3, 5, 3),
    "rendered_q": (6, 3, 5, 3),
    "union_per_view": (6, 3),
    "union_loss": (6,),
}
ARM_AXIS_ONE = {
    **{f"{term}_per_body_view": (2, 6, 2, 3, 3) for term in ("owner", "gap", "combined")},
    **{f"{term}_per_view": (2, 6, 2, 3) for term in ("owner", "gap", "combined")},
    **{f"{term}_loss": (2, 6, 2) for term in ("owner", "gap", "combined")},
}
COMMON = ("target_mass", "positive_target_mass", "ray_valid", "anchor_ids", "frame_idx")


def score_fixture():
    rng = np.random.default_rng(8169)
    scores = {key: rng.random(shape) for key, shape in (ARM_AXIS_ZERO | ARM_AXIS_ONE).items()}
    scores.update(
        target_mass=rng.random((2, 2, 3, 3)),
        positive_target_mass=np.ones((2, 2, 3, 3), bool),
        ray_valid=np.ones((3, 5), bool),
        anchor_ids=np.array([0, 1], np.int64),
        frame_idx=np.array(m.FIRST, np.int64),
    )
    # An invalid ray's +inf q is retained and compared by bytes, never removed.
    scores["ray_valid"][1, -1] = False
    scores["rendered_q"][:, 1, -1] = np.inf
    original = {}
    for key, value in scores.items():
        if key in ARM_AXIS_ZERO:
            original[key] = value[:4].copy()
        elif key in ARM_AXIS_ONE:
            original[key] = value[:, :4].copy()
        else:
            original[key] = value.copy()
    return scores, original


def test_old_four_control_ignores_only_two_new_arms_and_preserves_inputs():
    scores, original = score_fixture()
    for key in ARM_AXIS_ZERO:
        scores[key][4:] = -17
    for key in ARM_AXIS_ONE:
        scores[key][:, 4:] = -31
    before = {k: v.tobytes() for k, v in scores.items()}
    m.previous_image_control(scores, original)
    assert before == {k: v.tobytes() for k, v in scores.items()}


@pytest.mark.parametrize("key", [*ARM_AXIS_ZERO, *ARM_AXIS_ONE, *COMMON])
def test_each_old_render_score_and_common_evidence_field_is_compared(key):
    scores, original = score_fixture()
    value = scores[key]
    if value.dtype == bool:
        value.flat[0] = ~value.flat[0]
    elif value.dtype.kind == "f":
        value.flat[0] = np.nextafter(value.flat[0], np.inf)
    else:
        value.flat[0] += 1
    with pytest.raises(ValueError, match="differ"):
        m.previous_image_control(scores, original)


def run_fixture():
    binding = dict(
        prior_tracking_sha256=m.producer.prior_gap.PREVIOUS_SHA,
        prior_evaluation_sha256=m.producer.prior_gap.EVALUATION_SHA,
        samples_sha256=m.previous.SAMPLES_SHA,
        point_sha256=m.previous.POINTS_SHA,
        preparation_sha256=m.inputs.PREPARATION_SHA,
        shape_plan_sha256=m.producer.PLAN_SHA,
        cuda_qualification_sha256=m.producer.QUALIFICATION_SHA,
        source_pins=copy.deepcopy(m.producer.SOURCE_PINS),
    )
    plan = dict(
        schema=m.producer.STAGE,
        native_interval=[m.FIRST, m.STOP],
        arms=list(m.producer.ARMS),
        input_binding=binding,
        point_evidence_arm="ct_first",
        gap_relative_coefficient=1.0,
        mask_coefficients=dict(zip(m.producer.ARMS, m.producer.COEFFICIENTS, strict=True)),
        fixed_width_applied_from_native=m.FIRST + 1,
        extra_proposals=0,
        maximum_pool=24,
        opt_steps=14,
    )
    summary = dict(
        schema=m.producer.STAGE,
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        arms={
            arm: dict(
                arm=arm,
                coefficient=coefficient,
                initialize_count=1,
                restore_count=0,
                frames=200,
                fixed_width_fits=199,
            )
            for arm, coefficient in zip(m.producer.ARMS, m.producer.COEFFICIENTS, strict=True)
        },
        maximum_pool=24,
        steps=14,
        initialize_calls=2,
        restore_calls=0,
        advance_calls=400,
        query_refreshes=0,
        model_forwards=0,
        no_external_pose_reset=True,
    )
    initial = {
        arm: dict(
            passed=True,
            native=m.FIRST,
            prior_trajectory_used=False,
            all19_retained_records_exact=True,
            all_initial_anchors_facts_exact=True,
        )
        for arm in m.producer.ARMS
    }
    return plan, summary, initial


def test_two_new_arms_start_after_causal_t0_with_same_seed_qualification():
    args = run_fixture()
    before = copy.deepcopy(args)
    m.validate_run(*args)
    assert args == before


@pytest.mark.parametrize(
    "fault",
    ["qualification", "source", "coefficient", "t0_width", "one_arm", "short", "bank", "future"],
)
def test_new_width_or_causal_control_cannot_drift(fault):
    plan, summary, initial = run_fixture()
    if fault == "qualification":
        plan["input_binding"]["cuda_qualification_sha256"] = "0" * 64
    elif fault == "source":
        plan["input_binding"]["source_pins"]["unregistered.py"] = "0" * 64
    elif fault == "coefficient":
        plan["mask_coefficients"]["static_base"] = 3.0
    elif fault == "t0_width":
        plan["fixed_width_applied_from_native"] = m.FIRST
    elif fault == "one_arm":
        summary["arms"]["static_base"]["fixed_width_fits"] = 0
    elif fault == "short":
        summary["advance_calls"] = 399
    elif fault == "bank":
        initial["static_gap"]["all_initial_anchors_facts_exact"] = False
    else:
        initial["static_base"]["prior_trajectory_used"] = True
    with pytest.raises(ValueError):
        m.validate_run(plan, summary, initial)


def test_width_and_auxiliary_contrasts_do_not_confuse_their_baselines():
    rows = [
        {"union_loss": np.array(row, np.float64)}
        for row in ([10, 50, 80, 8, 7, 6], [10, 50, 80, 10, 12, 14], [10, 50, 80, 12, 10, 9])
    ]
    result = m.paired_scores(rows)
    assert result["width_base"]["mean_delta"] == pytest.approx(-1 / 3)
    assert result["width_base"]["reference"] == "original"
    assert result["fixed_auxiliary"]["mean_delta"] == 0
    assert result["fixed_auxiliary"]["reference"] == "static_base"
    assert result["fixed_auxiliary"]["better_native_frames"] == 2
    assert result["fixed_auxiliary"]["worse_native_frames"] == 1
    assert result["dynamic_auxiliary"]["first"] == "ct_gap"
    assert result["dynamic_auxiliary"]["equal_native_frames"] == 1
    assert result["auxiliary_interaction_mean_delta"] == 0


def support_fixture(monkeypatch):
    detections, rig, rays = object(), object(), object()
    calls = []
    old_rows = {arm: {"old_sentinel": arm} for arm in m.prior_evaluation.ARMS}

    def old_support(candidates, offset, frame, actual_rig):
        assert offset == 1 and frame.detections is detections and actual_rig is rig
        calls.append("old4")
        return dict(old_rows)

    def from_points(actual_rig, actual_detections):
        assert actual_rig is rig and actual_detections is detections
        return rays

    def supported(pose, sources, actual_rays, actual_rig):
        assert actual_rays is rays and actual_rig is rig
        calls.append(int(pose.flat[0]))
        return sources.copy(), np.ones((3, 3), bool), np.zeros((3, 3, 3))

    monkeypatch.setattr(m.prior_evaluation, "support_recheck", old_support)
    monkeypatch.setattr(
        m.old_evaluation, "RayObservations", SimpleNamespace(from_points=from_points)
    )
    monkeypatch.setattr(m.old_evaluation, "supported_nodes", supported)
    candidates = {arm: {} for arm in m.prior_evaluation.ARMS}
    for index, arm in enumerate(m.producer.ARMS):
        candidates[arm] = dict(
            pose_xyz=np.full((2, 3, 3, 3), index + 4, np.float64),
            sources=np.broadcast_to(np.arange(3)[:, None, None], (2, 3, 3, 3)).copy(),
            observed=np.ones((2, 3, 3), bool),
        )
    return candidates, SimpleNamespace(detections=detections), rig, calls, old_rows


def test_support_delegates_old_four_and_checks_each_new_arm(monkeypatch):
    candidates, frame, rig, calls, old_rows = support_fixture(monkeypatch)
    result = m.support_recheck(candidates, 1, frame, rig)
    assert calls == ["old4", 4, 5]
    assert set(result) == set(m.ARMS)
    for arm, row in old_rows.items():
        assert result[arm] == row


@pytest.mark.parametrize("fault", ["changed_support", "shared_source"])
def test_new_arm_cannot_hide_support_loss_or_duplicate_source(monkeypatch, fault):
    candidates, frame, rig, _, _ = support_fixture(monkeypatch)
    if fault == "changed_support":
        candidates["static_gap"]["observed"][1, 2, 2] = False
    else:
        candidates["static_gap"]["sources"][1, 2, 1, 0] = 0
    with pytest.raises(ValueError):
        m.support_recheck(candidates, 1, frame, rig)
