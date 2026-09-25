"""New window wiring only: synthetic metadata, native labels, unchanged numerics."""

import ast
import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import compare_temporal_known_gap64 as m
import numpy as np
import pytest
from test_compare_temporal_point64 import Backend


def inputs():
    plan = dict(
        schema="temporal_known_gap64_preparation_v1",
        plan_sha256=m.PLAN_SHA,
        frames=list(m.FRAMES),
        views=list(m.VIEWS),
        native_interval=[m.FIRST, m.STOP],
        query_core_sha256=m.CORE_SHA,
        expected_initial_masks=[3, 4, 3],
        expected_window_masks=608,
        query_upper_bound=640,
        temporal_holdout=False,
        known_3D_gap_is_occlusion_truth=False,
    )
    summary = dict(
        status="ok",
        native_frames=64,
        raw_video_delivered_images=192,
        decoded_mask_stacks=192,
        initial_mask_rows=10,
        total_masks=608,
        query_files=3,
        query_count=3,
        model_loaded=False,
        model_forward_calls=0,
        SL_arrays_decoded=False,
        candidate_arrays_read=False,
        ID_or_review_changed=False,
        temporal_holdout=False,
        known_3D_gap_is_occlusion_truth=False,
        new_D2_inference=False,
    )
    counts = {}
    for view, first, total in zip(m.VIEWS, [3, 4, 3], [198, 221, 189], strict=True):
        q, r = divmod(total - first, 63)
        counts[view] = [first] + [q + int(i < r) for i in range(63)]
    natives = [
        dict(
            frame_idx=n,
            view=v,
            native_offset=n - m.FIRST,
            manifest_sha256=m.WINDOW_MANIFEST_SHA,
            mask_count=counts[v][n - m.FIRST],
        )
        for n in m.FRAMES
        for v in m.VIEWS
    ]
    queries = [
        dict(
            view=v,
            frame_idx=m.FIRST,
            mask_count=k,
            query_count=q,
            config=dataclasses.asdict(m.core.CONFIG),
        )
        for v, k, q in zip(m.VIEWS, [3, 4, 3], [0, 2, 1], strict=True)
    ]
    return plan, summary, natives, queries


def test_new_scope_keeps_all_ten_groups_even_zero_query_view():
    values = inputs()
    before = copy.deepcopy(values)
    m.validate_preparation(*values)
    assert values == before


@pytest.mark.parametrize(
    "drift",
    [
        lambda p, s, n, q: p.update(native_interval=[66483, 66547]),
        lambda p, s, n, q: p.update(schema="temporal_point64_preparation_v1"),
        lambda p, s, n, q: n[0].update(native_offset=333519),
        lambda p, s, n, q: n.reverse(),
        lambda p, s, n, q: n[0].update(mask_count=2),
        lambda p, s, n, q: q[0].update(mask_count=4),
        lambda p, s, n, q: q[0].update(frame_idx=0),
        lambda p, s, n, q: (q[0].update(query_count=639), s.update(query_count=642)),
        lambda p, s, n, q: p.update(known_3D_gap_is_occlusion_truth=True),
    ],
)
def test_wrong_window_or_denominator_rejected_without_loading_payloads(drift):
    values = inputs()
    drift(*values)
    with pytest.raises(ValueError):
        m.validate_preparation(*values)


def test_every_numerical_function_and_class_is_frozen_v2_ast():
    assert m.sha(m.IMPLEMENTATION_SOURCE) == m.IMPLEMENTATION_SOURCE_SHA

    def nodes(path):
        return {
            n.name: n
            for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    old, new = nodes(m.IMPLEMENTATION_SOURCE), nodes(m.__file__)
    assert set(new) - set(old) == {"window_binding"}
    assert set(old) - set(new) == {"retry_binding"}
    changed = {"validate_preparation", "load_inputs", "loaded_sources", "retry_binding"}
    for name in old.keys() - changed:
        if name == "run":
            maximum = [
                n
                for n in ast.walk(old[name])
                if isinstance(n, ast.keyword) and n.arg == "maximum_query_count"
            ]
            assert len(maximum) == 1 and maximum[0].value.value == 832
            maximum[0].value.value = 640  # Only the registered-window report bound changes.
        assert ast.dump(old[name], include_attributes=False) == ast.dump(
            new[name], include_attributes=False
        ), name
    assert m.FIRST == 333519 and m.STOP == 333583
    assert tuple((i, i + 16) for i in range(0, 49, 8)) == m.WINDOWS
    assert m.STAGE == "temporal_known_gap64_comparison_v1"


def test_new_absolute_native_call_labels_share_original_relative_horizons():
    q = np.zeros((2, 2), np.float32)
    result, calls = m.run_online(list(range(64)), q, Backend(2), lambda *_: None)
    assert [r["input_native"] for r in calls] == [
        [333519 + lo, 333519 + hi] for lo, hi in m.WINDOWS
    ]
    assert calls[-1]["maximum_seen_native"] == 333582
    assert result["first_horizon"][-1] == 63
    flags = np.ones((64, 2), bool)
    flags[1, 0] = False
    assert m.missing_runs(flags)["first_missing_native"].tolist() == [333520, -1]


def test_window_lineage_uses_implementation_not_prior_results(tmp_path, monkeypatch):
    legacy = dict(
        threshold=0.5,
        current_runtime_contract_imputed=False,
        biological_ID_truth=False,
        unrecorded_runtime_fields=["tf32"],
    )
    p = tmp_path / "legacy_detector_condition.json"
    p.write_bytes(m.json_bytes(legacy))
    monkeypatch.setattr(m, "PREPARATION", tmp_path)
    calls = []
    evidence = SimpleNamespace(
        paths={str(p): m.sha(p)}, add=lambda path, digest: calls.append((path, digest))
    )
    result = m.window_binding(evidence)
    assert calls == [
        (m.IMPLEMENTATION_SOURCE, m.IMPLEMENTATION_SOURCE_SHA),
        (m.IMPLEMENTATION_PLAN, m.IMPLEMENTATION_PLAN_SHA),
    ]
    assert result["legacy_detector_condition"] == legacy
    assert result["legacy_detector_condition_sha256"] == m.sha(p)
    assert result["distinct_registered_window"]
    assert not result["prior_comparison_results_read"]
    assert not result["known_3D_support_gap_is_occlusion_truth"]
    assert "retry" not in str(result)


def test_unsealed_legacy_condition_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PREPARATION", tmp_path)
    with pytest.raises(ValueError, match="condition was not sealed"):
        m.window_binding(SimpleNamespace(paths={}, add=lambda *_: None))


def test_preparation_pin_required_before_execution(monkeypatch):
    monkeypatch.setattr(m, "PREPARATION_SHA", None)
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    with pytest.raises(ValueError, match="frozen preparation"):
        m.execution_gate()
