"""Synthetic qualification of semantic query selection; no model or real data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
from semantic_node_queries import VIEWS, select_queries


def fixture():
    a = np.arange(18, dtype=np.float32).reshape(3, 3, 2) + 10
    points = {v: a.copy() + i for i, v in enumerate(VIEWS)}
    sources = np.broadcast_to(np.arange(3, dtype=np.int32)[:, None, None], (3, 3, 3)).copy()
    observed = np.ones((3, 3), bool)
    return points, sources, observed


def test_full_node_labels_exact_pixels_and_owned_results():
    p, s, o = fixture()
    result = select_queries(p, s, o, native_frame=333519)
    assert len(result["slots"]) == 27 and len(result["node_sets"]) == 9
    assert all(r["two_view_query_set_available"] for r in result["node_sets"])
    for view in VIEWS:
        a = result["views"][view]
        assert a["xy"].tobytes() == p[view].reshape(9, 2).tobytes()
        assert a["body_slot"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert a["node_index"].tolist() == [0, 1, 2] * 3
        assert not np.shares_memory(a["xy"], p[view])
        assert not a["duplicate_pixel"].any()
    assert not result["biological_ID_truth"]


def test_partial_nodes_are_absent_not_projected_or_synthesized():
    p, s, o = fixture()
    s[0, :2, 2] = -1
    o[0, 2] = False
    s[1, 0, 1] = -1
    result = select_queries(p, s, o, native_frame=0)
    assert len(result["slots"]) == 27
    assert [len(result["views"][v]["xy"]) for v in VIEWS] == [7, 8, 8]
    assert result["node_sets"][2]["selected_views"] == []
    assert result["node_sets"][4]["selected_views"] == list(VIEWS[1:])
    assert not result["missing_queries_synthesized"]


@pytest.mark.parametrize(
    "pixel,reason", [([np.nan, 2], "nonfinite"), ([-1, 2], "outside"), ([1200, 2], "outside")]
)
def test_unqueryable_supported_pixel_retains_slot_reason(pixel, reason):
    p, s, o = fixture()
    p[VIEWS[0]][0, 0] = pixel
    result = select_queries(p, s, o, native_frame=0)
    assert not result["slots"][0]["selected"]
    assert any(reason in r for r in result["slots"][0]["reasons"])
    assert result["node_sets"][0]["selected_view_count"] == 2


def test_duplicate_pixels_keep_distinct_nodes():
    p, s, o = fixture()
    p[VIEWS[0]][0, 1] = p[VIEWS[0]][0, 0]
    result = select_queries(p, s, o, native_frame=0)
    a = result["views"][VIEWS[0]]
    assert len(a["xy"]) == 9
    assert a["duplicate_pixel"].tolist() == [True, True] + [False] * 7
    assert a["node_index"][:2].tolist() == [0, 1]


def test_all_missing_keeps_empty_arrays_and_full_denominators():
    p = {v: np.empty((0, 3, 2), np.float32) for v in VIEWS}
    result = select_queries(
        p, np.full((3, 3, 3), -1, np.int32), np.zeros((3, 3), bool), native_frame=0
    )
    assert len(result["slots"]) == 27 and len(result["node_sets"]) == 9
    assert all(a["xy"].shape == (0, 2) for a in result["views"].values())
    assert all(not r["two_view_query_set_available"] for r in result["node_sets"])


@pytest.mark.parametrize("fault", ["support", "reuse", "range", "dtype", "frame", "views"])
def test_invalid_correspondence_rejected(fault):
    p, s, o = fixture()
    frame = 0
    if fault == "support":
        o[0, 0] = False
    elif fault == "reuse":
        s[1, 0, 0] = 0
    elif fault == "range":
        s[0, 0, 0] = 3
    elif fault == "dtype":
        p[VIEWS[0]] = p[VIEWS[0]].astype(np.float64)
    elif fault == "frame":
        frame = True
    else:
        p.pop(VIEWS[0])
    with pytest.raises(ValueError):
        select_queries(p, s, o, native_frame=frame)
