"""Independent gap evaluator lineage and four-arm comparison boundaries."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import evaluate_mask_transport_gap200 as m


def run_fixture():
    binding = dict(
        prior_tracking_sha256=m.producer.PREVIOUS_SHA,
        prior_evaluation_sha256=m.producer.EVALUATION_SHA,
        samples_sha256=m.previous.SAMPLES_SHA,
        point_sha256=m.previous.POINTS_SHA,
        preparation_sha256=m.inputs.PREPARATION_SHA,
        gap_plan_sha256=m.producer.PLAN_SHA,
    )
    plan = dict(
        schema=m.producer.STAGE,
        native_interval=[m.FIRST, m.STOP],
        input_binding=binding,
        arm="ct_gap",
        point_evidence_arm="ct_first",
        objective="base + 3*(owner + ray_gap)",
        gap_relative_coefficient=1.0,
        mask_coefficient=3.0,
        extra_proposals=0,
        maximum_pool=24,
        opt_steps=14,
    )
    summary = dict(
        schema=m.producer.STAGE,
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        prior_tracking_sha256=m.producer.PREVIOUS_SHA,
        prior_evaluation_sha256=m.producer.EVALUATION_SHA,
        arm=dict(arm="ct_gap", initialize_count=1, restore_count=0, frames=200),
        maximum_pool=24,
        steps=14,
        initialize_calls=1,
        restore_calls=0,
        advance_calls=200,
        query_refreshes=0,
        model_forwards=0,
        no_external_pose_reset=True,
        causal_initial_control_passed=True,
    )
    initial = dict(passed=True, native=m.FIRST, prior_trajectory_used=False)
    return plan, summary, initial


def test_complete_new_arm_is_accepted_without_mutation():
    args = run_fixture()
    before = copy.deepcopy(args)
    m.validate_run(*args)
    assert args == before


@pytest.mark.parametrize(
    "fault",
    [
        "parent",
        "samples",
        "objective",
        "weight",
        "pool",
        "short",
        "restore",
        "refresh",
        "t0",
        "future",
    ],
)
def test_rejects_changed_experiment_or_lifecycle(fault):
    plan, summary, initial = run_fixture()
    if fault == "parent":
        plan["input_binding"]["prior_tracking_sha256"] = "f" * 64
    elif fault == "samples":
        plan["input_binding"]["samples_sha256"] = "f" * 64
    elif fault == "objective":
        plan["objective"] = "owner only"
    elif fault == "weight":
        plan["gap_relative_coefficient"] = 0.5
    elif fault == "pool":
        summary["maximum_pool"] = 25
    elif fault == "short":
        summary["advance_calls"] = 199
    elif fault == "restore":
        summary["restore_calls"] = 1
    elif fault == "refresh":
        summary["query_refreshes"] = 1
    elif fault == "t0":
        initial["native"] += 1
    else:
        initial["prior_trajectory_used"] = True
    with pytest.raises(ValueError):
        m.validate_run(plan, summary, initial)


def score_fixture():
    rng = np.random.default_rng(702)
    scores = {}
    for key, shape in {
        "rendered_union": (4, 3, 5),
        "rendered_ownership": (4, 3, 5, 3),
        "union_per_view": (4, 3),
        "union_loss": (4,),
        "owner_per_body_view": (2, 4, 2, 3, 3),
        "owner_per_view": (2, 4, 2, 3),
        "owner_loss": (2, 4, 2),
        "target_mass": (2, 2, 3, 3),
    }.items():
        scores[key] = rng.random(shape)
    scores.update(
        positive_target_mass=scores["target_mass"] > 0,
        ray_valid=np.ones((3, 5), bool),
        anchor_ids=np.arange(2, dtype=np.int64),
        frame_idx=np.array(m.FIRST, np.int64),
    )
    original = {}
    for key, value in scores.items():
        if key.startswith("owner_"):
            original[key] = value[:, :3].copy()
        elif key in ("rendered_union", "rendered_ownership", "union_per_view", "union_loss"):
            original[key] = value[:3].copy()
        else:
            original[key] = value.copy()
    return scores, original


def test_four_arm_control_keeps_both_evidence_and_all_anchors():
    scores, original = score_fixture()
    m.previous_image_control(scores, original)
    scores["owner_loss"][:, 3] += 100
    scores["union_loss"][3] += 100
    m.previous_image_control(scores, original)
    scores["owner_loss"][1, 2, 1] += 1e-12
    with pytest.raises(ValueError):
        m.previous_image_control(scores, original)


@pytest.mark.parametrize("key", ["rendered_ownership", "ray_valid", "anchor_ids", "frame_idx"])
def test_reference_render_or_evidence_change_is_rejected(key):
    scores, original = score_fixture()
    if scores[key].dtype == bool:
        scores[key].flat[0] = False
    else:
        scores[key].flat[0] += 1
    with pytest.raises(ValueError):
        m.previous_image_control(scores, original)


@pytest.fixture(scope="module")
def finished_toy():
    import test_mask_transport_gap_tracker as toy

    _, _, bundle, _, candidate = toy.qualified.__wrapped__()
    return bundle[0], candidate


def test_all21_backtrace_matches_real_frozen_finish_on_synthetic_cpu(finished_toy):
    core, candidate = finished_toy
    selected = m.candidate_backtrace_control(candidate, core)
    assert selected.shape == (2,) and selected[-1] == core["history_offsets"][-2]
    widened = dict(core)
    widened["history_proposal_kind"] = core["history_proposal_kind"].astype("U500")
    np.testing.assert_array_equal(m.candidate_backtrace_control(candidate, widened), selected)


@pytest.mark.parametrize("field", ["pose_xyz", "identity_review", "accepted_xyz"])
def test_changed_scored_candidate_cannot_pass_core_backtrace(finished_toy, field):
    core, original = finished_toy
    candidate = {k: v.copy() for k, v in original.items()}
    if candidate[field].dtype == bool:
        candidate[field].flat[0] = ~candidate[field].flat[0]
    else:
        candidate[field].flat[0] = 12345.0
    with pytest.raises(ValueError):
        m.candidate_backtrace_control(candidate, core)


def test_checkpoint_cannot_supply_its_own_unrelated_evidence_binding(monkeypatch):
    monkeypatch.setattr(m.previous, "read_arrays", lambda *args: {})
    monkeypatch.setattr(
        m.base,
        "bound_json",
        lambda *args: {
            "context": {"mask_transport": {"input_binding": {"point_sha256": "a" * 64}}}
        },
    )
    with pytest.raises(ValueError, match="binding differs"):
        m.validate_gap_state(
            Path("unused"), {}, None, None, None, {"input_binding": {"point_sha256": "b" * 64}}
        )
