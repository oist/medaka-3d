"""The protocol adapter observes baseline association without refining its output."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.stages import Associator
from medaka3d.types import VIEWS, Detection2D, Instance3D


@pytest.fixture
def adapter_module(monkeypatch):
    lab = Path(__file__).resolve().parents[1] / "labs" / "20260916_multiview_node_review"
    # Local module resolution mirrors running this lab as a script, without
    # retaining a generic 'geometry' module in the test process afterwards.
    geometry_spec = importlib.util.spec_from_file_location("geometry", lab / "geometry.py")
    geometry_module = importlib.util.module_from_spec(geometry_spec)
    geometry_spec.loader.exec_module(geometry_module)
    monkeypatch.setitem(sys.modules, "geometry", geometry_module)
    spec = importlib.util.spec_from_file_location(
        "_multiview_node_associator", lab / "associator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Baseline:
    name = "explicit-baseline"

    def __init__(self, instances):
        self.instances = instances
        self.calls = []

    def associate(self, detections, frame):
        self.calls.append((detections, frame))
        return self.instances


class Rig:
    views = VIEWS
    world_units = "mm"


def instances(frame=12):
    return [
        Instance3D(
            frame=frame, xyz=np.array([i, 2, 3.0]), slot=slot, sources={view: row for view in VIEWS}
        )
        for i, (slot, row) in enumerate(((8, 2), (3, 0), (6, 1)))
    ]


def detections(frame=12):
    return {
        view: [
            Detection2D(view, frame, keypoints=np.arange(6).reshape(3, 2) + 100 * vi + 10 * row)
            for row in range(3)
        ]
        for vi, view in enumerate(VIEWS)
    }


def test_same_baseline_list_and_objects_returned_without_identity_claim(
    adapter_module, monkeypatch
):
    original = instances()
    baseline = Baseline(original)
    input_detections = detections()
    captured = []
    result_metrics = {"ray_rms_mm": np.ones((3, 3))}

    def capture(rig, uv):
        captured.append(uv.copy())
        return result_metrics

    monkeypatch.setattr(adapter_module, "geometry_metrics", capture)
    adapter = adapter_module.MultiviewNodeDiagnosticAssociator(baseline, Rig())
    result = adapter.associate(input_detections, 12)
    assert isinstance(adapter, Associator)
    assert result is original
    assert all(result[i] is original[i] for i in range(3))
    assert baseline.calls == [(input_detections, 12)]
    record = adapter.diagnostics[0]
    assert record["metrics"] is result_metrics
    assert record["slots"] == [8, 3, 6]
    assert record["accepted_corrections"] == 0
    assert record["biological_identity_claim"] is False
    assert record["evidence_scope"] == "single_frame_geometric_consistency_only"
    assert not hasattr(adapter, "identify")
    for i, row in enumerate((2, 0, 1)):
        for vi, view in enumerate(VIEWS):
            np.testing.assert_array_equal(captured[0][i, vi], input_detections[view][row].keypoints)


def test_explicit_node_order_source_rows_and_missing_nodes_preserved(adapter_module, monkeypatch):
    original = instances()
    original[0].sources.pop("side1_cam")
    original[1].sources["top_cam"] = -1
    input_detections = detections()
    # Inputs declare tail,mouth,pectoral. No keypoints and one missing mouth
    # must remain missing after standardization, without changing any input.
    input_detections["side2_cam"][0] = Detection2D("side2_cam", 12, keypoints=None)
    input_detections["side2_cam"][1] = Detection2D(
        "side2_cam", 12, keypoints=np.array([[21.0, 22], [np.nan, np.nan], [25, 26]])
    )
    captured = []
    monkeypatch.setattr(
        adapter_module, "geometry_metrics", lambda rig, uv: captured.append(uv.copy()) or {}
    )
    adapter = adapter_module.MultiviewNodeDiagnosticAssociator(
        Baseline(original), Rig(), node_order=("tail", "mouth", "pectoral")
    )
    assert adapter.associate(input_detections, 12) is original
    uv = captured[0]
    assert np.isnan(uv[0, 1]).all()
    assert np.isnan(uv[1, 0]).all()
    assert np.isnan(uv[1, 2]).all()
    assert np.isnan(uv[2, 2, 0]).all()
    np.testing.assert_array_equal(uv[2, 2, 1:], [[25, 26], [21, 22]])
    np.testing.assert_array_equal(
        adapter.diagnostics[0]["source_rows"], [[2, -1, 2], [-1, 0, 0], [1, 1, 1]]
    )


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_other_instance_counts_decline_only_diagnostic(adapter_module, monkeypatch, count):
    original = instances()[:count] if count <= 3 else instances() + instances()[:1]
    monkeypatch.setattr(
        adapter_module, "geometry_metrics", lambda *args: pytest.fail("should not score")
    )
    adapter = adapter_module.MultiviewNodeDiagnosticAssociator(Baseline(original), Rig())
    assert adapter.associate(detections(), 12) is original
    assert adapter.diagnostics[0]["status"] == "declined_diagnostic"
    assert adapter.diagnostics[0]["reason"] == "requires_exactly_three_baseline_instances"
    assert adapter.diagnostics[0]["instance_count"] == count


@pytest.mark.parametrize(
    "row,reason",
    [(99, "source_row_out_of_range"), (0, "baseline_reuses_detection_row_within_view")],
)
def test_invalid_source_rows_do_not_mutate_or_replace_baseline(
    adapter_module, monkeypatch, row, reason
):
    original = instances()
    original[0].sources["top_cam"] = row
    monkeypatch.setattr(
        adapter_module, "geometry_metrics", lambda *args: pytest.fail("should not score")
    )
    adapter = adapter_module.MultiviewNodeDiagnosticAssociator(Baseline(original), Rig())
    assert adapter.associate(detections(), 12) is original
    assert original[0].sources["top_cam"] == row
    assert adapter.diagnostics[0]["reason"] == reason


def test_prior_diagnostic_records_do_not_influence_next_frame(adapter_module, monkeypatch):
    captured = []
    monkeypatch.setattr(
        adapter_module, "geometry_metrics", lambda rig, uv: captured.append(uv.copy()) or {}
    )
    baseline = Baseline(instances())
    adapter = adapter_module.MultiviewNodeDiagnosticAssociator(baseline, Rig())
    adapter.associate(detections(), 12)
    adapter.diagnostics[0]["uv"][:] = 99999
    baseline.instances = instances(20)
    assert adapter.associate(detections(20), 20) is baseline.instances
    np.testing.assert_array_equal(captured[0], captured[1])
    assert adapter.diagnostics[1]["frame"] == 20
