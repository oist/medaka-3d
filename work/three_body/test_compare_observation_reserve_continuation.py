"""Small adapter checks, with all IO confined to synthetic temporary roots."""

import copy
from pathlib import Path

import compare_observation_reserve_continuation as m
import numpy as np
import pytest

NP_LOAD = np.load
LOAD_HEADERS = m.load_headers


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for module in (m, m.minute, m.metrics):
        monkeypatch.setattr(module, "WORK", tmp_path)
    for name in ("TRACKING", "PARENT", "MILESTONE", "PARENT_CHECKPOINT", "QUALIFIED"):
        monkeypatch.setattr(m, name, tmp_path / name)
    for name in ("TAIL", "NODEWISE", "MILESTONE", "TAIL_COMPARISON", "INPUT_ROOT"):
        monkeypatch.setattr(m.minute, name, tmp_path / ("minute_" + name))
    for name, value in dict(FIRST=100, START=104, STOP=108, PROBES=(104, 105, 106)).items():
        monkeypatch.setattr(m, name, value)
    monkeypatch.setattr(m.minute, "FIRST", 100)
    monkeypatch.setattr(m.minute, "STOP", 108)

    def forbidden(*args, **kwargs):
        pytest.fail("real input or upstream workflow reached")

    for module, names in (
        (m, ("source_code", "load_headers")),
        (m.minute, ("source_code", "load_headers", "run")),
        (m.metrics, ("load_headers", "load_candidate", "compare_arrays", "run")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path), "non-synthetic path reached"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    reads = []

    class Archive:
        def __init__(self, path, *args, **kwargs):
            assert Path(path).resolve().is_relative_to(tmp_path)
            self.name = Path(path).name
            reads.append((self.name, "OPEN"))
            self.inner = NP_LOAD(path, *args, **kwargs)
            self.files = self.inner.files

        def __getitem__(self, key):
            reads.append((self.name, key))
            return self.inner[key]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

    monkeypatch.setattr(np, "load", Archive)
    return reads


def test_context_only_import_and_exact39_signatures():
    old_context = dict(implementation="BodyRowContinuousTracker", bounds=[-6, 200])
    context = dict(
        implementation="ObservationReserveContinuousTracker",
        bounds=[-6, 200],
        observation_reserve_policy=dict(
            effective_from_native=m.START,
            selection_source_sha256=m.SELECTOR_SHA,
            protected_prefix_count=5,
        ),
    )
    metadata = dict(context=old_context, restore_count=4, previous_frame=m.START - 1)
    declarations = {
        f"field{i}": dict(dtype=dict(str="<i8"), shape=[1], bytes_sha256="a" * 64)
        for i in range(39)
    }
    imported = dict(
        schema="observation_reserve_nodewise_state_import_v1",
        status="ok",
        previous_frame=m.START - 1,
        processed_frames=m.START - m.FIRST,
        effective_from_native=m.START,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        logical_initialize_count=1,
        new_begin_calls=0,
        added_restore_count=1,
        old_history_rewritten=False,
        review_cleared_by_migration=False,
        original_metadata=metadata,
        old_context=old_context,
        new_context=context,
        imported_metadata={**metadata, "context": context, "restore_count": 5},
        arrays={
            k: dict(dtype=v["dtype"]["str"], shape=v["shape"], sha256=v["bytes_sha256"])
            for k, v in declarations.items()
        },
    )
    manifest = dict(arrays=declarations)
    m.validate_import(imported, metadata, manifest, context)
    changed = copy.deepcopy(context)
    changed["bounds"][0] -= 1
    with pytest.raises(ValueError, match="only reserve context"):
        m.validate_import(imported, metadata, manifest, changed)
    bad = copy.deepcopy(imported)
    bad["arrays"]["field0"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="original39"):
        m.validate_import(bad, metadata, manifest, context)


def compact_index():
    rows = []
    for frame in range(m.START, m.STOP):
        changed = frame == m.START + 1
        rows.append(
            dict(
                frame_idx=frame,
                candidate_count=8,
                observation_reserve=dict(
                    baseline_indices=[0, 1, 2, 3, 4, 5],
                    selected_indices=[0, 1, 2, 3, 4, 7 if changed else 5],
                    baseline_best_index=0,
                    changed=changed,
                    reservation_count=int(changed),
                    support_evaluation_count=8,
                    eligible_candidate_count=int(changed),
                    reason="reserved_observation_candidate" if changed else "no_eligible_candidate",
                    replacement=dict(removed=5, admitted=7, extra_path_cost=0.5)
                    if changed
                    else None,
                ),
            )
        )
    reported = dict(
        changed_frames=1,
        reasons=dict(reserved_observation_candidate=1, no_eligible_candidate=3),
        current_beam_selection_only=True,
        final_path_reserved_ancestor_count_not_computed=True,
        original_numeric_costs_preserved=True,
    )
    return dict(
        rows=rows,
        process_interval=[m.START, m.STOP],
        batch_size=100,
        diagnostic_families=["instance", "body_row", "observation_reserve"],
    ), reported


def test_compact_reserve_counts_not_misreported_as_final_ancestry():
    index, reported = compact_index()
    result = m.summarize_reserve(index, reported)
    assert result["changed_native_frames"] == [105]
    assert result["final_path_reserved_ancestor_count"] is None
    assert result["batch_payloads_decoded"] == 0
    bad = copy.deepcopy(index)
    bad["rows"][1]["observation_reserve"]["selected_indices"][0] = 6
    with pytest.raises(ValueError, match="invariants"):
        m.summarize_reserve(bad, reported)
    with pytest.raises(ValueError, match="counters"):
        m.summarize_reserve(index, {**reported, "changed_frames": 0})


def candidate():
    n = m.STOP - m.FIRST
    pose = np.zeros((n, 3, 3, 3))
    pose[..., 0] = np.arange(3)[None, :, None] * 20
    pose[..., 1] = np.arange(3)[None, None, :]
    return dict(
        frame_idx=np.arange(m.FIRST, m.STOP, dtype=np.int64),
        pose_xyz=pose,
        observed=np.ones((n, 3, 3), bool),
        sources=np.zeros((n, 3, 3, 3), np.int32),
        identity_review=np.zeros((n, 3), bool),
        proposal_kind=np.full(n, "predict", dtype="U22"),
    )


def test_connected_once_sixfields_nonidentity_and_prefix_report(tmp_path, monkeypatch, isolated):
    first, second = candidate(), candidate()
    inverse = np.array([2, 0, 1])  # A-slot -> B-slot is non-self-inverse P120.
    for key in ("pose_xyz", "observed", "sources", "identity_review"):
        second[key][4:] = second[key][4:, inverse]
    second["pose_xyz"][0] += 3
    refs = {}
    for arm, row in zip(m.ARMS, (first, second), strict=True):
        values = {**row, **{k: np.zeros(1) for k in m.metrics.ALL21 - set(row)}}
        path = tmp_path / f"{arm}.npz"
        np.savez_compressed(path, **values)
        refs[arm] = dict(
            candidate=dict(
                path=str(path), sha256=m.proof.sha(path), size_bytes=path.stat().st_size
            ),
            signatures=m.common.signature(values),
        )
    reserve = m.summarize_reserve(*compact_index())
    monkeypatch.setattr(m, "source_code", lambda evidence: {"synthetic_sources": "qualified"})
    monkeypatch.setattr(
        m,
        "load_headers",
        lambda digest, evidence: (refs, dict(same_parent39_signatures_and_metadata=True), reserve),
    )
    assert m.run("a" * 64, tmp_path / "result")["status"] == "ok"
    assert isolated == [(arm + ".npz", key) for arm in m.ARMS for key in ("OPEN", *m.FIELDS)]
    summary = m.proof.read_json(tmp_path / "result/summary.json")
    assert summary["methods"]["nodewise"]["frames"] == 4
    geometry = summary["geometric_comparison"]
    assert geometry["chest"]["unique_winner_counts"] == [0, 0, 0, 4, 0, 0]
    assert geometry["selected_pre74700_backtrace_difference_frames"]["pose_xyz"] == 1
    assert summary["observation_reserve"]["final_path_reserved_ancestor_count"] is None
    with NP_LOAD(tmp_path / "result/review_samples.npz") as archive:
        assert len(archive.files) == 12
        for arm in m.ARMS:
            np.testing.assert_array_equal(archive[arm + "_frame_idx"], m.PROBES)
    # A declared signature mismatch is refused by the actual reused loader.
    damaged = copy.deepcopy(refs[m.ARMS[0]])
    damaged["signatures"]["pose_xyz"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="selected signatures"):
        m.minute.load_candidate(damaged, m.proof.Evidence())


def test_failed_stage_and_output_guard_never_reach_payload(tmp_path, isolated):
    m.TRACKING.mkdir()
    path = m.TRACKING / "tracking.provenance.json"
    path.write_bytes(m.proof.json_bytes(dict(stage=m.TRACKING_STAGE, status="failed")))
    with pytest.raises(ValueError, match="successful new producer"):
        LOAD_HEADERS(m.proof.sha(path), m.proof.Evidence())
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run("a" * 64, tmp_path)
    assert not isolated
