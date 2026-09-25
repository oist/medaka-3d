"""Technical lineage retry: synthetic cases plus actual small-JSON contracts.

No candidate/observation arrays, raw pixels, models, renderers or GPU execution.
"""

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "work/three_body"
sys.path.insert(0, str(WORK))
import render_mask_transport200_v2 as m  # noqa: E402


class Evidence:
    def __init__(self):
        self.added = {}

    def add(self, path, expected):
        # Contract-only recorder: never reopen original SL or any payload.
        self.added[str(path)] = expected


def fixture():
    shared = {
        **{f"view{i}.npz": str(i) * 64 for i in range(3)},
        **{f"calib{i}.json": str(i + 3) * 64 for i in range(4)},
    }
    prep_path = str(m.inputs.PREPARATION / "preparation.provenance.json")
    tracking = {
        "inputs": {
            prep_path: m.inputs.PREPARATION_SHA,
            **{p: digest for p, digest in shared.items() if p.endswith(".json")},
        }
    }
    context = {
        "plan": {"shared_SL_calibration_sha256": shared},
        "seal": {"inputs": dict(shared)},
    }
    return tracking, context


def test_prepared_sl_lineage_does_not_invent_direct_tracking_consumption():
    tracking, context = fixture()
    before = copy.deepcopy((tracking, context))
    evidence = Evidence()
    result = m.bind_geometry_lineage(tracking, context, evidence)
    assert len(result["SL3_sha256"]) == 3
    assert len(result["calibration4_sha256"]) == 4
    assert evidence.added == result["calibration4_sha256"]
    assert all(p not in tracking["inputs"] for p in result["SL3_sha256"])
    assert result["preparation_success_sha256"] == m.inputs.PREPARATION_SHA
    assert (tracking, context) == before


@pytest.mark.parametrize(
    "fault",
    [
        "prep_seal",
        "missing_prep_seal",
        "prepared_sl",
        "prepared_calib",
        "direct_calib",
        "scope",
        "types",
    ],
)
def test_lineage_rejects_wrong_edge_or_shared_scope(fault):
    tracking, context = fixture()
    prep_path = str(m.inputs.PREPARATION / "preparation.provenance.json")
    if fault == "prep_seal":
        tracking["inputs"][prep_path] = "f" * 64
    elif fault == "missing_prep_seal":
        del tracking["inputs"][prep_path]
    elif fault == "prepared_sl":
        context["seal"]["inputs"]["view0.npz"] = "f" * 64
    elif fault == "prepared_calib":
        context["seal"]["inputs"]["calib0.json"] = "f" * 64
    elif fault == "direct_calib":
        tracking["inputs"]["calib0.json"] = "f" * 64
    elif fault == "scope":
        del context["plan"]["shared_SL_calibration_sha256"]["view0.npz"]
    else:
        shared = context["plan"]["shared_SL_calibration_sha256"]
        shared["unexpected.bin"] = shared.pop("view0.npz")
    with pytest.raises(ValueError):
        m.bind_geometry_lineage(tracking, context, Evidence())


def test_actual_small_json_success_edges_without_payload_reads():
    tracking_path = WORK / "mask_transport200_three_arm_tracking_v2/tracking.provenance.json"
    preparation_path = m.inputs.PREPARATION / "preparation.provenance.json"
    plan_path = m.inputs.PREPARATION / "plan.json"
    tracking_bytes = tracking_path.read_bytes()
    preparation_bytes = preparation_path.read_bytes()
    assert hashlib.sha256(tracking_bytes).hexdigest() == (
        "49c8771c175252ccc0cb6fdca565cc7834e4d1c8ff6496846a2ea94126d0feda"
    )
    assert hashlib.sha256(preparation_bytes).hexdigest() == m.inputs.PREPARATION_SHA
    tracking, preparation = json.loads(tracking_bytes), json.loads(preparation_bytes)
    assert tracking["status"] == preparation["status"] == "ok"
    plan_bytes = plan_path.read_bytes()
    plan_member = preparation["outputs"]["plan.json"]
    assert plan_member["path"] == str(plan_path)
    assert hashlib.sha256(plan_bytes).hexdigest() == plan_member["sha256"]
    assert len(plan_bytes) == plan_member["size_bytes"]
    evidence = Evidence()
    result = m.bind_geometry_lineage(
        tracking, {"seal": preparation, "plan": json.loads(plan_bytes)}, evidence
    )
    assert len(result["SL3_sha256"]) == 3
    assert all(path not in tracking["inputs"] for path in result["SL3_sha256"])
    assert evidence.added == result["calibration4_sha256"]
    assert len(evidence.added) == 4


def test_failed_visual_is_retained_exactly_with_no_output_members():
    path = WORK / "mask_transport200_visual_review_v1/review.provenance.json"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == m.FAILED_VISUAL_SHA
    failed = json.loads(data)
    assert failed["stage"] == "mask_transport200_visual_review_v1"
    assert failed["status"] == "failed" and failed["outputs"] == {}
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
    source = (WORK / "render_mask_transport200.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == m.FAILED_VISUAL_SOURCE_SHA


def test_drawing_and_validation_function_asts_remain_frozen():
    def functions(path):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef)
        }

    old = functions(WORK / "render_mask_transport200.py")
    new = functions(WORK / "render_mask_transport200_v2.py")
    unchanged = {
        "validate_completed",
        "validate_reuse",
        "validate_candidate",
        "projection_record",
        "dashed",
        "panel",
        "composite",
        "save_png",
        "verify_video",
    }
    assert unchanged <= old.keys() & new.keys()
    for name in unchanged:
        assert old[name] == new[name], name
