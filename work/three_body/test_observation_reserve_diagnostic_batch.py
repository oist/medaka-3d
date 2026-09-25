"""Focused three-family persistence contracts using only synthetic diagnostics."""

import copy
from types import SimpleNamespace

import numpy as np
import observation_reserve_diagnostic_batch as m
import pytest
import test_nodewise_diagnostic_batch as memory

# Reuse only the input-blocking fixture and pure memory/assertion helpers.
# None of the frozen writer's 22 tests is collected or repeated here.
isolated_io = memory.isolated_io


def tracker_for(frame, changed=False):
    instance, body = memory.diagnostics(frame)
    count = 8 if changed else 3
    if changed:
        instance["candidate_index"] = np.arange(count, dtype=np.int32)
        body.update(
            incoming_proposal_count=7,
            outgoing_proposal_count=8,
            headroom=1,
            new_proposal_indices=np.array([7], dtype=np.int64),
        )
    baseline = list(range(min(count, 6)))
    selected = [*baseline[:-1], 7] if changed else baseline.copy()
    reserve = dict(
        schema="observation_reserve_frame_diagnostics_v1",
        frame_idx=frame,
        frame_completed=True,
        candidate_index=np.arange(count, dtype=np.int64),
        eligible=np.arange(count) == 7,
        selected_support_reused=True,
        support_evaluation_count=count,
        baseline_best_index=0,
        support_seconds=0.123,
        selection_seconds=0.004,
        parent_pose=np.arange(54, dtype=np.float64).reshape(2, 3, 3, 3),
        parent_velocity=np.arange(18, dtype=np.float64).reshape(2, 3, 3),
        parent_missing=np.arange(6, dtype=np.int64).reshape(2, 3),
        parent_node_missing=np.arange(18, dtype=np.int64).reshape(2, 3, 3),
        parent_direct_pectoral_age=np.arange(6, dtype=np.int64).reshape(2, 3),
        candidate_mask_support=np.zeros((count, 3), bool),
        supported_sources=np.full((count, 3, 3, 3), -1, np.int32),
        observed=np.zeros((count, 3, 3), bool),
        nested=copy.deepcopy(instance["nested"]),
        selection=dict(
            baseline_indices=baseline,
            selected_indices=selected,
            current_best_preserved=True,
            real_costs_modified=False,
            changed=changed,
            reason="reserved_observation_candidate" if changed else "no_eligible_candidate",
            replacement=dict(removed=5, admitted=7, extra_path_cost=2.0) if changed else None,
        ),
    )
    calls = []

    def export(value, family):
        calls.append(family)
        return copy.deepcopy(value)

    tracker = SimpleNamespace(
        previous_frame=frame,
        export_instance_diagnostics=lambda: export(instance, "instance"),
        export_body_row_diagnostics=lambda: export(body, "body_row"),
        export_observation_reserve_diagnostics=lambda: export(reserve, "observation_reserve"),
    )
    return tracker, instance, body, reserve, calls


def test_hundred_auto_flush_chunk_tail_and_all_three_full_exports(tmp_path):
    out = m.proof.Output(tmp_path / "three_families")
    writer = m.ObservationReserveDiagnosticBatch(out)
    expected = []
    for offset in range(101):
        frame = 74700 + offset
        tracker, instance, body, reserve, calls = tracker_for(frame, changed=(offset % 2 == 0))
        expected.append(
            copy.deepcopy(
                dict(frame_idx=frame, instance=instance, body_row=body, observation_reserve=reserve)
            )
        )
        row = writer.append(tracker, frame)
        assert calls == list(m.FAMILIES)
        assert {k: v for k, v in row.items() if not k.startswith("observation_reserve")} == (
            m.legacy.compact_row(instance, body, frame)
        )
        assert row["observation_reserve"]["reservation_count"] == int(offset % 2 == 0)
        assert row["observation_reserve_timing_seconds"] == dict(
            support_seconds=0.123, selection_seconds=0.004
        )
        if offset == 98:
            assert writer.pending_frames == 99 and writer.saved_frames == 0
        if offset == 99:
            assert writer.pending_frames == 0 and writer.saved_frames == 100
        reserve["parent_pose"][:] = -999
        reserve["nested"]["a_second"][:] = -999
        instance["candidate_index"][:] = -1
    assert writer.pending_frames == 1 and writer.totalframes == 101
    writer.flush()
    assert writer.pending_frames == 0 and writer.saved_frames == len(writer.records) == 101
    assert len(writer.batches) == 2 and len(out.records) == 4
    actual = []
    for record in writer.batches:
        document = m.proof.read_json(record["metadata"]["path"])
        assert document["schema"] == m.SCHEMA and document["diagnostic_families"] == list(
            m.FAMILIES
        )
        actual.extend(memory.restore_batch(record))
    memory.assert_exact(actual, expected)
    assert writer.flush() is None
    out.final_guard()


def test_compact_reserve_does_not_alias_pending_selection_or_rewrite_costs(tmp_path):
    writer = m.ObservationReserveDiagnosticBatch(m.proof.Output(tmp_path / "reserve"))
    tracker, instance, body, reserve, _ = tracker_for(74700, changed=True)
    row = writer.append(tracker, 74700)
    assert row["observation_reserve"]["selected_indices"] == [0, 1, 2, 3, 4, 7]
    assert row["observation_reserve"]["eligible_candidate_count"] == 1
    row["observation_reserve"]["selected_indices"][-1] = 6
    row["observation_reserve"]["replacement"]["extra_path_cost"] = -999
    writer.flush()
    saved = memory.restore_batch(writer.batches[0])[0]
    memory.assert_exact(saved["instance"], instance)
    memory.assert_exact(saved["body_row"], body)
    memory.assert_exact(saved["observation_reserve"], reserve)


@pytest.mark.parametrize("fault", ["foreign_native", "candidate_order"])
def test_reserve_family_mismatch_poisoned_before_acceptance(tmp_path, fault):
    writer = m.ObservationReserveDiagnosticBatch(m.proof.Output(tmp_path / "bad_family"))
    tracker, _, _, reserve, _ = tracker_for(74700)
    if fault == "foreign_native":
        reserve["frame_idx"] += 1
    else:
        reserve["candidate_index"] = np.array([0, 2, 1])
    with pytest.raises(ValueError, match="complete reserve"):
        writer.append(tracker, 74700)
    assert writer.totalframes == writer.pending_frames == writer.saved_frames == 0
    with pytest.raises(ValueError, match="previously failed"):
        writer.flush()


def test_duplicate_native_is_rejected_before_any_export(tmp_path):
    writer = m.ObservationReserveDiagnosticBatch(m.proof.Output(tmp_path / "duplicate"))
    writer.append(tracker_for(74700)[0], 74700)
    tracker, _, _, _, calls = tracker_for(74700)
    with pytest.raises(ValueError, match="consecutive"):
        writer.append(tracker, 74700)
    assert calls == [] and writer.totalframes == writer.pending_frames == 1
    with pytest.raises(ValueError, match="previously failed"):
        writer.flush()


def test_failed_third_family_roundtrip_never_commits_or_retries(tmp_path, monkeypatch):
    out = m.proof.Output(tmp_path / "corrupt")
    writer = m.ObservationReserveDiagnosticBatch(out, batch_size=1)
    restore = m.codec.restore_diagnostic

    def corrupt(tree, arrays):
        result = restore(tree, arrays)
        result["frames"][0]["observation_reserve"]["frame_completed"] = False
        return result

    monkeypatch.setattr(m.codec, "restore_diagnostic", corrupt)
    with pytest.raises(ValueError, match="complete reserve"):
        writer.append(tracker_for(74700)[0], 74700)
    assert writer.totalframes == writer.pending_frames == 1
    assert writer.saved_frames == 0 and writer.batches == [] and len(out.records) == 2
    with pytest.raises(ValueError, match="previously failed"):
        writer.flush()
