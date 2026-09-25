"""Adversarial context and support checks for proposed inspection times."""
import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_plan():
    lab = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
    old = {name: sys.modules.get(name) for name in ("review_events", "review_plan")}
    try:
        for name in old:
            spec = importlib.util.spec_from_file_location(f"_review_plan_audit_{name}",
                                                         lab / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


plan = load_plan()


def result(count):
    return {
        "frame_idx": np.arange(100, 100 + count),
        "pose_xyz": np.ones((count, 3, 3, 3)),
        "observed": np.ones((count, 3, 3), bool),
        "body_supported": np.ones((count, 3), bool),
        "mask_supported": np.zeros((count, 3), bool),
        "identity_review": np.zeros((count, 3), bool),
        "missing": np.zeros((count, 3), int),
        "score_margin": np.full(count, np.inf),
    }


def test_supported_probes_one_frame_outside_fixed_context_are_not_imported():
    data = result(500)
    data["score_margin"][200:204] = .5
    data["observed"][:, :, 0] = False
    data["observed"][[116, 287], :, 0] = True
    scene = plan.build_review_plan(data)["scenes"][0]
    assert [scene["context_start_frame"], scene["context_stop_frame_exclusive"]] == [217, 387]
    assert scene["proposed_anchor_frames"] == [None, None, None]
    assert scene["status"] == "needs_probe_selection"


def test_mask_body_return_cannot_manufacture_observed_mouth_probes():
    data = result(40)
    data["observed"][10:15, 0, :2] = False
    data["body_supported"][10:15, 0] = False
    data["missing"][10:15, 0] = np.arange(1, 6)
    data["observed"][15:, 0, :2] = False
    data["mask_supported"][15:, 0] = True
    data["identity_review"][13:, 0] = True
    report = plan.build_review_plan(data)
    assert report["incomplete_scene_count"] == 1
    assert report["scenes"][0]["proposed_anchor_frames"] == [109, None, None]


def test_past_unresolved_other_identity_is_reported_even_with_complete_new_probes():
    data = result(40)
    data["score_margin"][10:15] = .5
    data["identity_review"][:, 2] = True
    scene = plan.build_review_plan(data)["scenes"][0]
    assert scene["proposed_anchor_frames"] == [109, 115, 139]
    assert scene["entry_contains_unresolved_machine_id"] is True
    assert scene["anchor_identity_review_by_slot"][0] == [False, False, True]
    assert scene["status"] == "proposed_for_human_review"
    assert scene["human_confirmed_links"] == 0
    assert scene["ids_repaired"] is False
