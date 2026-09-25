"""Resumed seal rejection and lossless remapping with one tiny real beam integration."""

import copy
from pathlib import Path

import numpy as np
import pytest
import run_resumed_nodewise_five_minute_identity_beam as m


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    for module in (m, m.metric, m.frozen, m.previous):
        for key, value in dict(FIRST=100, START=104, STOP=108).items():
            monkeypatch.setattr(module, key, value)
    for module in (m, m.metric, m.frozen, m.common):
        monkeypatch.setattr(module, "WORK", tmp_path)
    monkeypatch.setattr(m.proof, "REPO", tmp_path)
    monkeypatch.setattr(m.common, "REPO", tmp_path)
    n = m.STOP - m.FIRST
    pose = np.zeros((n, 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(3)[None, :, None] * 20
    pose[..., 1] = np.arange(3)[None, None, :] * 2
    raw = dict(
        frame_idx=np.arange(m.FIRST, m.STOP, dtype=np.int64),
        pose_xyz=pose,
        observed=np.ones((n, 3, 3), bool),
        sources=np.broadcast_to(
            np.arange(3, dtype=np.int32)[None, :, None, None], (n, 3, 3, 3)
        ).copy(),
        identity_review=np.broadcast_to([True, False, True], (n, 3)).copy(),
        proposal_kind=np.full(n, "body_row"),
    )
    for value in raw.values():
        value.flags.writeable = False
    return raw


def test_remaps_fish_axis_but_never_source_values_or_frame_axes(tiny):
    mapping = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]])
    result = m.remap_stream(tiny, mapping)
    for frame in range(4):
        for fish in range(3):
            for key in m.FISH_FIELDS:
                assert np.array_equal(
                    result[key][frame, fish], tiny[key][4 + frame, mapping[frame, fish]]
                )
    assert result["identity_review"].tolist() == [
        [True, False, True],
        [True, True, False],
        [False, True, True],
        [True, False, True],
    ]
    assert result["sources"][1, 0].tolist() == [[2, 2, 2]] * 3
    assert np.array_equal(result["proposal_kind"], tiny["proposal_kind"][4:])
    assert not any(value.flags.writeable for value in result.values())


@pytest.mark.parametrize("mapping", [np.zeros((4, 3), int), np.zeros((3, 3), int)])
def test_rejects_nonbijective_or_incomplete_remapping(tiny, mapping):
    with pytest.raises(ValueError, match="bijective"):
        m.remap_stream(tiny, mapping)


def test_tiny_real_solve_freezes_before_search_and_preserves_review(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "source_code", lambda evidence: {})
    monkeypatch.setattr(m.frozen, "qualification", lambda evidence: {"synthetic": True})
    monkeypatch.setattr(m, "comparison_headers", lambda *args: {"reference": {"synthetic": True}})
    monkeypatch.setattr(m, "load_candidate", lambda *args: (tiny, m.common.signature(tiny)))
    monkeypatch.setattr(m.frozen.beam, "verify_sources", lambda: None)
    real_solve = m.frozen.solve_once
    calls = []
    output = tmp_path / "result"

    def checked_solve(part):
        calls.append("solve")
        assert (output / "freeze_before_solve.json").is_file()
        freeze = m.proof.read_json(output / "freeze_before_solve.json")
        assert freeze["solve_calls_started"] == 0
        assert freeze["plan"]["sha256"] == m.proof.sha(output / "plan.json")
        assert part["frame_idx"].tolist() == [104, 105, 106, 107]
        return real_solve(part)

    monkeypatch.setattr(m.frozen, "solve_once", checked_solve)
    seal = m.run("a" * 64, output)
    assert seal["status"] == "ok" and calls == ["solve"]
    summary = m.proof.read_json(output / "summary.json")
    assert summary["solve_calls"] == 1
    assert summary["decoded_fields_unchanged"] and not summary["review_flags_cleared"]
    with np.load(output / "beam_remapped_stream.npz") as stream:
        assert set(stream.files) == set(m.FIELDS)
        assert np.array_equal(stream["identity_review"], tiny["identity_review"][4:])


@pytest.mark.parametrize("damage", ["stage", "horizon", "candidate_edge", "tracking_edge"])
def test_resumed_adapter_rejects_unbound_or_wrong_comparison(tmp_path, monkeypatch, damage):
    root, producer = tmp_path / "comparison", tmp_path / "resumed"
    root.mkdir()
    source = Path(m.resumed.__file__).resolve()
    runner = tmp_path / "resume_nodewise_body_row_five_minutes_099600.py"
    item = dict(
        root=producer,
        seal=dict(inputs={str(runner): m.PRODUCER_SOURCE_SHA}, outputs={}),
        summary=dict(candidate={}, candidate_arrays={k: {} for k in m.FIELDS}),
    )
    comparison = dict(
        schema=m.resumed.STAGE,
        status="ok",
        scientific_interval=[m.START, m.STOP],
        native_frames=m.STOP - m.START,
        arms=["sparse", "nodewise"],
        sparse_tracking_sha256=m.metric.SPARSE_SHA,
        nodewise_tracking_sha256="b" * 64,
        candidate_modified=False,
        support_recomputed=False,
        physical_ID_truth=False,
        temporal_ID_mapping_computed=False,
    )
    seal = dict(
        stage=m.resumed.STAGE,
        status="ok",
        inputs={
            str(source): m.COMPARISON_SHA,
            str(runner): m.PRODUCER_SOURCE_SHA,
            str(producer / "tracking.provenance.json"): "b" * 64,
        },
    )
    for member in ("summary.json", "plan.json", "input_index.json", "candidate.npz"):
        row = dict(path=str(producer / member), sha256="c" * 64)
        item["seal"]["outputs"][member] = row
        seal["inputs"][row["path"]] = row["sha256"]
    if damage == "stage":
        seal["stage"] = m.previous.comparison.STAGE
    elif damage == "horizon":
        comparison["scientific_interval"] = [m.START, m.STOP - 1]
    elif damage == "candidate_edge":
        del seal["inputs"][str(producer / "candidate.npz")]
    elif damage == "tracking_edge":
        del seal["inputs"][str(producer / "tracking.provenance.json")]
    monkeypatch.setattr(m, "COMPARISON_ROOT", root)
    monkeypatch.setattr(m, "PRODUCER_ROOT", producer)
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m.proof, "read_json", lambda path: copy.deepcopy(seal))
    monkeypatch.setattr(m.previous, "document", lambda *args: copy.deepcopy(comparison))
    monkeypatch.setattr(m.resumed, "load_nodewise", lambda *args: copy.deepcopy(item))

    class Evidence:
        def add(self, *args):
            return "a" * 64

    with pytest.raises(ValueError):
        m.comparison_headers("a" * 64, Evidence())
