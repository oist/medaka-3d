"""Reject changes to the fixed tail experiment before any fitting occurs."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import fit_tail_packets as tail
import numpy as np
import pytest


def metadata():
    return dict(
        schema=tail.PACKET_STAGE,
        frames=list(tail.FRAMES),
        cpu_fit_frames=list(tail.CPU_FRAMES),
        allow_bounded=False,
        proposal_kind="tail_recovery",
        original_prefix_preserved=True,
        serial_flow=True,
        fitted=False,
        tracked=False,
        persistence_config=dict(
            enabled=True, minimum_missing_frames=4, minimum_pectoral_frames=4
        ),
    )


def test_fixed_scope_is_301_original_contexts_and_four_declared_cpu_frames():
    value = metadata()
    before = copy.deepcopy(value)
    tail.validate_tail_metadata(value)
    assert value == before
    assert (66524, *range(66620, 66920)) == tail.FRAMES
    assert tail.CPU_FRAMES == (66524, 66628, 66703, 66907)


@pytest.mark.parametrize(
    "key,value",
    [
        ("frames", list(range(66619, 66920))),
        ("cpu_fit_frames", [66524, 66628, 66703, 66908]),
        ("allow_bounded", True),
        ("proposal_kind", "bounded_mouth_recovery"),
        ("original_prefix_preserved", False),
        ("serial_flow", False),
        ("fitted", True),
        ("tracked", True),
    ],
)
def test_policy_changes_are_rejected(key, value):
    data = metadata()
    data[key] = value
    with pytest.raises(ValueError, match="packet policy"):
        tail.validate_tail_metadata(data)


@pytest.mark.parametrize(
    "key,value",
    [("enabled", False), ("minimum_missing_frames", 3), ("minimum_pectoral_frames", 5)],
)
def test_age_policy_is_not_relaxed_for_tail_recovery(key, value):
    data = metadata()
    data["persistence_config"][key] = value
    with pytest.raises(ValueError, match="age4"):
        tail.validate_tail_metadata(data)


def test_a_mouth_candidate_cannot_be_smuggled_as_the_new_tail_arm():
    packet = dict(
        proposal_pose_xyz=np.zeros((7, 3, 3, 3)),
        original_count=np.asarray(6),
        proposal_is_added=np.arange(7) >= 6,
        proposal_kind=np.array(["predict"] * 6 + ["bounded_mouth_recovery"]),
    )
    with pytest.raises(ValueError, match="unexpected added candidate kind"):
        tail.fit_inputs(packet, None, SimpleNamespace(max_hypotheses=18))


def test_fit_and_selection_recurrence_are_the_frozen_qualified_driver():
    # Qualification must preserve whole-pool fitting, both beam selections and
    # all 19 record reconstruction steps. Only packet acceptance differs.
    root = Path(tail.__file__).parent
    functions = {}
    for name in ("fit_bounded_mouth_packets.py", "fit_tail_packets.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        functions[name] = {
            f.name: ast.dump(f, include_attributes=False)
            for f in tree.body
            if isinstance(f, ast.FunctionDef)
        }
    for name in ("fitted_frame", "compare_fit_arrays", "member", "bind_input"):
        assert functions["fit_tail_packets.py"][name] == functions[
            "fit_bounded_mouth_packets.py"
        ][name]
