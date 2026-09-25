"""Synthetic binding/clock checks plus a source-only historical JSON preflight."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import diagnose_semantic_node200_geometry as m


def tracking_metadata():
    policy = dict(no_reseed=True, no_query_replacement=True, first_emission_primary=True)
    plan = dict(
        schema=m.TRACK_STAGE,
        native_interval=[m.FIRST, m.STOP],
        views=list(m.VIEWS),
        windows=[[i, i + 16] for i in range(0, 185, 8)],
        initial_geometry_sha256=m.INITIAL_SHA,
        initial_query_count=27,
        **policy,
    )
    summary = dict(
        schema=m.TRACK_STAGE,
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        views=[
            dict(view=v, query_count=9, model_forward_calls=24, model_initializations=1)
            for v in m.VIEWS
        ],
        query_count=27,
        model_initializations=3,
        model_forward_calls=72,
        all200_query_denominator=5400,
        tracking199_query_denominator=5373,
        **policy,
    )
    return plan, summary


@pytest.mark.parametrize("corrupt", ["unfinished", "scope", "order", "count", "reseed", "window"])
def test_complete_fixed_tracking_contract(corrupt):
    plan, summary = tracking_metadata()
    m.validate_tracking(plan, summary)
    if corrupt == "unfinished":
        summary["status"] = "running"
    elif corrupt == "scope":
        summary["native_interval"] = [m.FIRST, m.STOP - 1]
    elif corrupt == "order":
        summary["views"].reverse()
    elif corrupt == "count":
        summary["views"][1]["query_count"] = 8
    elif corrupt == "reseed":
        plan["no_reseed"] = False
    else:
        plan["windows"][-1][1] -= 1
    with pytest.raises(ValueError):
        m.validate_tracking(plan, summary)


def points():
    query = dict(xy=np.arange(18, dtype=np.float32).reshape(9, 2))
    lk = dict(xy=np.tile(query["xy"], (200, 1, 1)), valid=np.ones((200, 9), bool))
    horizon = m.first_horizons()
    emissions = dict(
        frame_idx=np.arange(m.FIRST, m.STOP, dtype=np.int64),
        first_horizon=horizon,
        first_release_native=m.FIRST + horizon,
        final_horizon=np.full(200, 199, np.int64),
        first_release_index=np.r_[
            np.zeros(16, np.int64), np.repeat(np.arange(1, 24, dtype=np.int64), 8)
        ],
        first_xy=lk["xy"].copy(),
        final_xy=lk["xy"].copy(),
        first_visibility=lk["valid"].copy(),
        final_visibility=lk["valid"].copy(),
    )
    calls = [
        dict(
            call_index=i,
            input_relative=[8 * i, 16 + 8 * i],
            input_native=[m.FIRST + 8 * i, m.FIRST + 16 + 8 * i],
            native_origin=m.FIRST,
            maximum_seen_native=m.FIRST + 15 + 8 * i,
            output_prefix=16 + 8 * i,
            query_count=9,
            query_signature=m.base.signature(query["xy"]),
            forward_called=True,
            status="ok",
        )
        for i in range(24)
    ]
    return query, lk, emissions, calls


@pytest.mark.parametrize("corrupt", ["horizon", "origin", "query", "call", "dtype", "recovery"])
def test_view_release_query_and_lk_absorption(corrupt):
    query, lk, emissions, calls = points()
    # CT can become visible again: this is not LK status or a new query.
    emissions["first_visibility"][20, 0] = False
    m.validate_point_arrays(query, lk, emissions, calls)
    if corrupt == "horizon":
        emissions["first_release_native"][16] -= 1
    elif corrupt == "origin":
        calls[3]["native_origin"] += 1
    elif corrupt == "query":
        calls[-1]["query_signature"]["sha256"] = "0" * 64
    elif corrupt == "call":
        calls[2]["input_relative"] = [15, 31]
    elif corrupt == "dtype":
        emissions["final_xy"] = emissions["final_xy"].astype(np.float64)
    else:
        lk["valid"][20, 0] = False
    with pytest.raises(ValueError):
        m.validate_point_arrays(query, lk, emissions, calls)


def test_initial_query_dtype_and_signature_are_not_coerced():
    expected = dict(xy=np.zeros((9, 2), np.float32), frame_idx=np.asarray(m.FIRST, np.int64))
    row = dict(array_signatures={k: m.base.signature(a) for k, a in expected.items()})
    m.validate_queries(copy.deepcopy(expected), expected, row)
    altered = dict(expected, xy=expected["xy"].astype(np.float64))
    with pytest.raises(ValueError, match="initial qualified"):
        m.validate_queries(altered, expected, row)
    row["array_signatures"]["xy"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="signatures"):
        m.validate_queries(expected, expected, row)


@pytest.mark.parametrize("method", m.METHODS)
@pytest.mark.parametrize("offset", [1, 16, 199])
def test_geometry_uses_actual_reference_previous_native_and_common_release(
    monkeypatch, method, offset
):
    reference = np.arange(200 * 27, dtype=np.float64).reshape(200, 3, 3, 3)
    uv = np.zeros((3, 3, 3, 2), np.float32)
    flags = np.ones((3, 3, 3), bool)
    seen = []

    def helper(pixel, reported, initial, rig, previous, **clock):
        np.testing.assert_array_equal(previous, reference[offset - 1])
        assert pixel is uv and reported is flags and initial is flags
        seen.append(clock)
        return dict(arrays={}, metadata={})

    monkeypatch.setattr(m.geometry, "build_observation", helper)
    result = m.evaluate_frame(method, offset, uv, flags, flags, object(), reference)
    native = m.FIRST + offset
    release = m.STOP - 1 if method == "ct_final" else m.FIRST + int(m.first_horizons()[offset])
    assert seen == [
        dict(
            initial_frame_idx=m.FIRST,
            previous_frame_idx=native - 1,
            frame_idx=native,
            release_horizon=release,
        )
    ]
    assert result["metadata"]["comparison_release_horizon"] == release
    assert result["metadata"]["native_available_at"] == (native if method == "lk" else release)
    assert result["metadata"]["reference_is_truth"] is False


@pytest.mark.parametrize("offset", [0, -1, 200, True])
def test_anchor_has_no_synthesized_predecessor(monkeypatch, offset):
    monkeypatch.setattr(
        m.geometry, "build_observation", lambda *_a, **_k: pytest.fail("helper called")
    )
    with pytest.raises(ValueError, match="later frames"):
        m.evaluate_frame("lk", offset, None, None, None, None, np.zeros((200, 3, 3, 3)))


def test_failed_metadata_before_numeric_load_preserves_failed_seal(monkeypatch, tmp_path):
    destination = tmp_path / "failed_diagnostic"
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "snapshot", lambda *_: None)

    def unavailable(*_):
        raise ValueError("no completed CT200 seal")

    monkeypatch.setattr(m, "metadata_inputs", unavailable)
    monkeypatch.setattr(m, "load_geometry_inputs", lambda *_: pytest.fail("numeric payload opened"))
    with pytest.raises(ValueError, match="completed CT200"):
        m.run(tmp_path / "tracking", "0" * 64, destination)
    record = json.loads((destination / "diagnosis.provenance.json").read_text())
    assert record["stage"] == m.STAGE and record["status"] == "failed"
    assert record["outputs"] == {} and record["error_type"] == "ValueError"
    assert not (destination / "summary.json").exists()


def test_source_only_actual_reference_json_contract(monkeypatch):
    """Real existing small JSON only; no prepared/point/reference numeric arrays."""
    original_open = Path.open

    def small_only(path, *args, **kwargs):
        assert path.suffix == ".json", f"non-JSON read: {path}"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", small_only)
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("np.load used"))
    evidence = m.base.Evidence()
    gs, _ = m.inherited.header(
        m.INITIAL_GEOMETRY,
        "diagnosis.provenance.json",
        "temporal_refractive_geometry_diagnosis_v2",
        m.INITIAL_SHA,
        evidence,
    )
    root, seal, members, metadata = m.reference_metadata(gs, evidence)
    assert root == m.WORK / m.REFERENCE["root"]
    assert seal["status"] == "ok" and metadata["context"]
    assert members[str(root / "candidate.npz")]["sha256"] == m.REFERENCE["candidate"]
    evidence.final_guard()
    assert len(evidence.paths) == 6 and all(Path(p).suffix == ".json" for p in evidence.paths)


def test_declared_but_unimported_dependency_has_no_fabricated_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "tracking"
    source = tmp_path / "unimported.py"
    source.write_text("# declared dependency\n")
    monkeypatch.setattr(m, "WORK", tmp_path)
    digest = m.base.sha(source)
    seal = dict(inputs={str(source): digest})
    plan = dict(source_pins={source.name: digest})
    rows = m.bind_declared_pins(root, seal, {}, plan, dict(source_hashes={}), m.base.Evidence())
    assert rows[0]["snapshot_present"] is False and rows[0]["loaded_and_captured"] is False
    with pytest.raises(ValueError, match="captured source snapshot missing"):
        m.bind_declared_pins(
            root, seal, {}, plan, dict(source_hashes={str(source): digest}), m.base.Evidence()
        )
    seal["inputs"][str(source)] = "0" * 64
    with pytest.raises(ValueError, match="pin input differs"):
        m.bind_declared_pins(root, seal, {}, plan, dict(source_hashes={}), m.base.Evidence())


def test_actual_completed_contract_reads_only_sources_and_small_json(monkeypatch):
    """Complete CT200/prepV2 schema, not a numerical geometry execution."""
    original_open = Path.open

    def restricted(path, *args, **kwargs):
        assert path.suffix in (".json", ".py", ".md"), f"numeric/model/image read: {path}"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", restricted)
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("numeric NPZ read"))
    evidence = m.base.Evidence()
    context = m.metadata_inputs(
        m.WORK / "semantic_node200_tracking_v1",
        "ed92335426ced5fa81aff78af97f62db1898ace778bf488fe2a501296260bb02",
        evidence,
    )
    evidence.final_guard()
    assert context["prep_sha"] == m.PREP_SHA
    assert [r["query_count"] for r in context["rows"]] == [9, 9, 9]
    assert len(context["shared"]) == 7
    selector = next(
        row
        for row in context["declared_pins"]
        if Path(row["source"]).name == "semantic_node_queries.py"
    )
    assert selector["snapshot_present"] is False
    assert selector["loaded_and_captured"] is False
