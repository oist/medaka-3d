"""Frozen metric parity, raw row contracts, and score scope for SLEAP experiments."""

from __future__ import annotations

import ast
import gzip
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "labs/20260918_sleap_pipeline_adapter"


def module(name):
    spec = importlib.util.spec_from_file_location("sleap_runner_test_" + name, LAB / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


METRICS = module("frozen_metrics")
ADAPTER = module("adapter")
with patch.dict(sys.modules, {"frozen_metrics": METRICS, "adapter": ADAPTER}):
    RUN = module("run")


def function_ast(source, name):
    return ast.dump(
        next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )


def test_frozen_metric_excerpts_match_original_functions_and_sources():
    frozen = (LAB / "frozen_metrics.py").read_text(encoding="utf-8")
    for source, digest, names in (
        (METRICS.SOURCE_RUN, METRICS.RUN_SOURCE_SHA256, ("measure", "frame_statistics")),
        (METRICS.SOURCE_BASELINE, METRICS.BASELINE_SOURCE_SHA256, ("predecessors",)),
    ):
        path = ROOT / source
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for name in names:
            assert function_ast(frozen, name) == function_ast(
                path.read_text(encoding="utf-8"), name
            )


def test_input_slice_preserves_full_c01_indices_missing_points_and_dtypes():
    frames = np.array([1, 667, 668, 999, 1000, 1001, 1333, 1334], dtype=np.int32)
    points = np.arange(48, dtype=np.float32).reshape(8, 3, 2)
    points[4, 0] = np.nan
    raw = dict(frame_idx=frames, points=points, score=np.arange(8, dtype=np.float32))
    data = RUN.extract_input(raw, dict(start=1000, stop=1002))
    np.testing.assert_array_equal(data["global_row"], np.arange(2, 7))
    np.testing.assert_array_equal(data["frame_idx"], frames[2:7])
    np.testing.assert_array_equal(data["points"], points[2:7])
    np.testing.assert_array_equal(data["core_mask"], [False, False, True, True, False])
    assert data["points"].dtype == np.float32
    assert not data["observed_mask"][2, 0]
    before = RUN.contract_fingerprints(data)
    changed = {key: value.copy() for key, value in data.items()}
    changed["score"][0] += 1
    assert RUN.contract_fingerprints(changed) != before


def test_selected_scores_are_scoped_by_global_row_and_eligible_native_frame():
    data = dict(
        frame_idx=np.array([99, 100, 101, 102]),
        global_row=np.array([1007, 1008, 1009, 1010]),
        core_mask=np.array([False, True, True, False]),
    )
    events = [
        dict(global_row=1007, native_frame=99, score=0.99, confirmed=False),
        dict(global_row=1008, native_frame=100, score=0.9, confirmed=False),
        dict(global_row=1009, native_frame=101, score=0.2, confirmed=True),
        dict(global_row=1010, native_frame=102, score=0.8, confirmed=False),
    ]
    value = RUN.event_summary(events, data, dict(start=100))
    assert value["evidence_core_events"] == 2
    assert value["evidence_eligible_events"] == 1
    assert value["selected_score_observations"] == 1
    assert value["selected_score_median"] == 0.2
    assert value["refusal_reasons"] == {}
    with pytest.raises(AssertionError, match="native-frame"):
        RUN.event_summary([dict(global_row=1009, native_frame=100)], data, dict(start=100))
    with pytest.raises(AssertionError, match="outside"):
        RUN.event_summary([dict(global_row=2, native_frame=101)], data, dict(start=100))


def test_missing_selected_score_does_not_get_replaced_with_candidate_score():
    data = dict(frame_idx=np.array([5]), global_row=np.array([900]), core_mask=np.array([True]))
    events = [
        dict(
            global_row=900,
            native_frame=5,
            score=None,
            confirmed=False,
            candidates=[dict(score=0.99)],
            reason="unmatched",
        )
    ]
    value = RUN.event_summary(events, data, dict(start=4))
    assert value["selected_score_observations"] == 0
    assert value["selected_score_median"] is None
    assert value["refusal_reasons"] == {"unmatched": 1}


def test_confirmed_gap_is_forbidden_but_provisional_and_sp_gaps_are_recordable():
    data = dict(frame_idx=np.array([0, 2]))
    result = SimpleNamespace(labels=np.array([0, 1]), provisional_labels=np.array([0, 0]))
    _, gaps = RUN.validate_result(result, data, "G0")
    np.testing.assert_array_equal(gaps, [0, 0])
    result.labels = np.array([0, 0])
    with pytest.raises(AssertionError, match="gap"):
        RUN.validate_result(result, data, "G0")
    _, gaps = RUN.validate_result(result, data, "SP")
    np.testing.assert_array_equal(gaps, [0, 2])


def test_duplicate_labels_in_one_frame_fail_before_output():
    data = dict(frame_idx=np.array([4, 4]))
    result = SimpleNamespace(labels=np.array([0, 0]), provisional_labels=np.array([0, 1]))
    with pytest.raises(AssertionError, match="one frame"):
        RUN.validate_result(result, data, "G0")


def test_evidence_is_compressed_finite_json_and_deterministic(tmp_path):
    first, second = tmp_path / "first.gz", tmp_path / "second.gz"
    payload = dict(global_row=123456789, score=np.float64(0.5), prediction_error=np.inf)
    RUN.write_evidence(first, payload)
    RUN.write_evidence(second, payload)
    assert first.read_bytes() == second.read_bytes()
    value = json.loads(gzip.decompress(first.read_bytes()))
    assert value == dict(global_row=123456789, score=0.5, prediction_error=None)


def test_prespecified_sensitivities_change_exactly_one_field():
    with patch.dict(sys.modules, {"adapter": ADAPTER}):
        configs = RUN.configurations()
    assert len(RUN.CONDITIONS) == 10
    assert len(configs) == 9
    base = asdict(configs["G0"])
    for name, config in configs.items():
        changed = [key for key, value in asdict(config).items() if value != base[key]]
        assert len(changed) == (0 if name == "G0" else 1)
    assert configs["G_age8"].max_age_frames == 8
    assert configs["G_age33"].max_age_frames == 33


@pytest.mark.requires_artifacts(
    *(
        ROOT / "work/variable_count_tracklets_v1/known_323819" / name
        for method in RUN.REFERENCES
        for name in (f"tracking_{method}_side1_cam.npz", f"metrics_{method}_side1_cam.json")
    )
)
def test_reference_case_metrics_remain_identical_without_tracking_rerun():
    base = ROOT / "work/variable_count_tracklets_v1/known_323819"
    for method in RUN.REFERENCES:
        with np.load(base / f"tracking_{method}_side1_cam.npz", allow_pickle=False) as z:
            stats, _ = METRICS.measure(
                z["frame_idx"], z["points"], z["labels"], z["core_mask"], 323653, 323992, 100, []
            )
        prior = json.loads((base / f"metrics_{method}_side1_cam.json").read_text(encoding="utf-8"))
        for key in ("core_rows", "contiguous_runs", "adjacent_connections", "gap_connections"):
            assert stats[key] == prior[key]
