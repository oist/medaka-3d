"""Grouping diagnostics distinguish mixed instances from illegal same-node reuse."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
SPEC = importlib.util.spec_from_file_location(
    "node_instance_diagnostic_test_module", WORK / "diagnose_node_instance_structure.py"
)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def fixture():
    points = {
        view: np.arange(18, dtype=np.float32).reshape(3, 3, 2) + vi * 100
        for vi, view in enumerate(m.p.base.VIEWS)
    }
    scores = {view: np.array([0.8, 0.6, np.nan], np.float32) for view in m.p.base.VIEWS}
    source = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).astype(np.int32).copy()
    observed = np.ones((3, 3), bool)
    return source, points, scores, observed


def test_whole_instance_rows_agree_without_equating_rows_across_views():
    source, points, scores, observed = fixture()
    source[:, 1] = source[[2, 0, 1], 1]
    result = m.source_structure(source, points, scores, observed)
    assert len(result["cells"]) == 9
    assert all(cell["all_three_from_one_row"] for cell in result["cells"])
    assert not any(cell["mixed_rows"] for cell in result["cells"])
    assert result["cross_body_shared_rows"] == []


def test_cross_node_exchange_is_mixed_and_cross_body_shared_but_not_same_node_collision():
    source, points, scores, observed = fixture()
    source[[0, 1], 0, 2] = [1, 0]  # Swap only tails in one view.
    result = m.source_structure(source, points, scores, observed)
    assert sum(cell["mixed_rows"] for cell in result["cells"]) == 2
    assert len(result["cross_body_shared_rows"]) == 2
    top0 = next(
        cell
        for cell in result["cells"]
        if cell["raw_slot"] == 0 and cell["view"] == m.p.base.VIEWS[0]
    )
    assert top0["rows"].tolist() == [0, 0, 1]
    assert top0["nodes"][2]["instance_score"] == float(scores[m.p.base.VIEWS[0]][1])
    np.testing.assert_array_equal(top0["nodes"][2]["point_xy"], points[m.p.base.VIEWS[0]][1, 2])


def test_single_node_missing_view_and_direct_only_denominators_are_explicit():
    source, points, scores, observed = fixture()
    source[0] = -1
    source[0, 0, 1] = 0
    source[1, 0, 2] = 0
    observed[1, 2] = False
    result = m.source_structure(source, points, scores, observed)
    top1 = next(
        cell
        for cell in result["cells"]
        if cell["raw_slot"] == 1 and cell["view"] == m.p.base.VIEWS[0]
    )
    assert top1["mixed_rows"]
    assert not top1["mixed_rows_direct_nodes"]
    assert top1["grouping_testable_direct_nodes"]
    totals = m.aggregate([dict(frame=1, variant="test", structure=result)])["test"]
    assert totals["body_view_cells"] == 9
    assert totals["grouping_testable_cells"] == 6
    assert totals["mixed_row_cells"] == 1
    assert totals["mixed_row_direct_node_cells"] == 0


@pytest.mark.parametrize("bad", ["same_node", "row_range", "sentinel", "missing_xy", "scores"])
def test_invalid_evidence_is_not_reported_as_valid_structure(bad):
    source, points, scores, observed = fixture()
    view = m.p.base.VIEWS[0]
    if bad == "same_node":
        source[1, 0, 0] = 0
    elif bad == "row_range":
        source[1, 0, 0] = 3
    elif bad == "sentinel":
        source[1, 0, 0] = -2
    elif bad == "missing_xy":
        points[view][0, 0, 0] = np.nan
    else:
        scores[view] = np.array([0.9, 0.8])
    with pytest.raises(ValueError):
        m.source_structure(source, points, scores, observed)


def test_grouping_counts_are_invariant_to_frame_local_row_relabelling():
    source, points, scores, observed = fixture()
    source[[0, 1], 0, 2] = [1, 0]
    before = m.source_structure(source, points, scores, observed)
    order = np.array([2, 0, 1])
    inverse = np.argsort(order)
    after = m.source_structure(
        inverse[source].astype(np.int32),
        {v: value[order] for v, value in points.items()},
        {v: value[order] for v, value in scores.items()},
        observed,
    )
    for a, b in zip(before["cells"], after["cells"], strict=True):
        for key in [
            "mixed_rows",
            "all_three_from_one_row",
            "distinct_2d_rows",
            "grouping_testable_direct_nodes",
            "mixed_rows_direct_nodes",
        ]:
            assert a[key] == b[key]
        for x, y in zip(a["nodes"], b["nodes"], strict=True):
            np.testing.assert_array_equal(x["point_xy"], y["point_xy"])
    assert len(before["cross_body_shared_rows"]) == len(after["cross_body_shared_rows"])
