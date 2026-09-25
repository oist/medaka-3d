"""Synthetic endpoint companion/driver contracts; no scientific inputs or GPU."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_endpoint_mask_priority_continuation as m


def forbidden(*args, **kwargs):
    pytest.fail("scientific input, checkpoint or GPU execution forbidden")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    work = tmp_path / "repo/work/three_body"
    work.mkdir(parents=True)
    monkeypatch.setattr(m, "WORK", work)
    monkeypatch.setattr(m, "REPO", work.parents[1])
    monkeypatch.setattr(m, "PARENT", work / "synthetic_parent")
    for name in ("load_parent_headers", "load_parent_payload"):
        monkeypatch.setattr(m, name, forbidden)
    for owner, names in (
        (m.original, ("verified_inputs", "SourceSequenceReader")),
        (m.prior, ("read_checkpoint", "runtime", "local_code", "smi_sample")),
        (m.sparse, ("SparseInstanceRescoreContinuousTracker",)),
        (m.variant, ("EndpointMaskPriorityTracker",)),
        (m.original.cv2, ("VideoCapture",)),
        (m.torch.cuda, ("synchronize", "set_per_process_memory_fraction")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    load = np.load

    def confined_load(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve())
        return load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", confined_load)
    return work


def endpoint(frame=m.START, *, all_split=False):
    count = 4
    split = np.zeros((count, 3), bool)
    split[0, 0] = split[2, 1] = True
    if all_split:
        split[1, 1] = split[3, 2] = True
    unknown = np.zeros((count, 3, 3), bool)
    unknown[3, 2, 1] = True
    costs = np.array([1.0, 4.0, 2.0, 8.0])
    old, order, fallback = m.variant.candidate_orders(costs, split)
    selected, baseline = order[:3].copy(), old[:3].copy()
    labels = np.full((count, 3, 3), "same_instance", dtype="<U24")
    labels[:, :, 0][split] = "split_instances"
    labels[unknown] = "unavailable"
    return dict(
        schema="endpoint_mask_frame_diagnostics_v1",
        frame_idx=frame,
        frame_completed=True,
        candidate_count=count,
        candidate_split_by_body=split,
        unknown_by_candidate_body_view=unknown,
        selected_candidate_indices=selected,
        baseline_selected_candidate_indices=baseline,
        selected_split_by_body=split[selected].copy(),
        added_identity_review_by_selected_body=split[selected].copy(),
        candidate_path_cost=costs,
        cost_order=old,
        selection_order=order,
        all_candidates_split_fallback=fallback,
        selection_changed_from_baseline=not np.array_equal(selected, baseline),
        selected_original_score_margin=np.array([-3.0, np.inf, 1.0]),
        view_order=list(m.sparse.VIEWS),
        views={
            view: dict(
                classification_by_body=labels[:, :, vi].copy(),
                projection_finite_by_body=None if vi == 1 else np.ones((count, 3), bool),
                unavailable_distance=None,
                diagnostic_nonfinite=np.array([np.nan, np.inf, -0.0]),
            )
            for vi, view in enumerate(m.sparse.VIEWS)
        },
    )


def body(frame):
    return dict(
        schema="body_row_frame_diagnostics_v1",
        frame_idx=frame,
        frame_completed=True,
        incoming_proposal_count=3,
        outgoing_proposal_count=4,
        decoder_candidate_cap=24,
        headroom=21,
        zero_headroom=False,
        generated_scene_count=1,
        added_proposal_count=1,
        excluded_by_capacity=0,
        new_proposal_indices=np.array([3], dtype=np.int64),
        new_proposal_kind="body_row",
    )


def test_endpoint_counts_priority_backup_negative_gap_and_all_split():
    value = endpoint()
    counts = m.endpoint_counts(value)
    assert counts["candidate_count"] == 4
    assert counts["candidates_with_split"] == 2
    assert counts["candidates_with_unknown"] == 1
    assert counts["retained_candidates"] == 3
    assert counts["retained_split_bodies"] == counts["added_review_bodies"] == 1
    assert counts["retained_negative_score_margins"] == 1
    assert counts["retained_beam_changed"]
    assert not counts["preferred_candidate_has_split"]
    assert counts["counts_describe_current_retained_beam_not_final_backtrace"]
    assert counts["classification_counts_by_view"][m.sparse.VIEWS[0]]["split_instances"] == 2
    assert counts["classification_counts_by_view"][m.sparse.VIEWS[1]]["unavailable"] == 1
    assert all(sum(row.values()) == 12 for row in counts["classification_counts_by_view"].values())
    all_split = m.endpoint_counts(endpoint(all_split=True))
    assert all_split["all_candidates_split_fallback"]
    assert all_split["candidates_with_split"] == 4
    assert all_split["retained_split_bodies"] == 3
    assert not all_split["retained_beam_changed"]
    assert all_split["preferred_candidate_has_split"]


@pytest.mark.parametrize(
    "damage",
    [
        "order",
        "split",
        "review",
        "changed",
        "margin_shape",
        "margin_nan",
        "margin_neg_inf",
        "label",
        "view_order",
    ],
)
def test_endpoint_counts_reject_inconsistent_selection_evidence(damage):
    value = endpoint()
    if damage == "order":
        value["selection_order"] = value["cost_order"].copy()
    elif damage == "split":
        value["selected_split_by_body"][:] = False
    elif damage == "review":
        value["added_identity_review_by_selected_body"][0, 2] = True
    elif damage == "changed":
        value["selection_changed_from_baseline"] = False
    elif damage == "margin_shape":
        value["selected_original_score_margin"] = np.array([-3.0])
    elif damage == "margin_nan":
        value["selected_original_score_margin"][0] = np.nan
    elif damage == "margin_neg_inf":
        value["selected_original_score_margin"][0] = -np.inf
    elif damage == "label":
        value["views"][m.sparse.VIEWS[0]]["classification_by_body"][0, 0] = "verified_fish"
    else:
        value["view_order"] = list(reversed(value["view_order"]))
    with pytest.raises(ValueError):
        m.endpoint_counts(value)


def test_three_diagnostic_families_share_native_and_codec_preserves_unknowns(isolated, monkeypatch):
    value = endpoint()
    tracker = SimpleNamespace(
        export_endpoint_mask_diagnostics=lambda: copy.deepcopy(value),
        export_body_row_diagnostics=lambda: body(m.START),
    )
    monkeypatch.setattr(
        m.diagnostic,
        "save_diagnostic",
        lambda out, t, f: dict(frame_idx=f, candidate_count=4),
    )
    out = m.proof.Output(isolated / "synthetic_output")
    saved = m.save_frame_diagnostics(out, tracker, m.START)
    assert saved["body_row"]["admitted"] == 1
    assert saved["endpoint_mask"] == m.endpoint_counts(value)
    assert len(out.records) == 4  # instance family stubbed; two new companion pairs
    row = m.proof.read_json(out.root / f"endpoint_mask_diagnostics/f{m.START}.json")
    with np.load(row["archive"]["path"], allow_pickle=False) as archive:
        restored = m.diagnostic.restore_diagnostic(
            row["tree"], {k: archive[k] for k in archive.files}
        )
    assert restored["frame_idx"] == m.START
    assert restored["views"][m.sparse.VIEWS[1]]["projection_finite_by_body"] is None
    for view in m.sparse.VIEWS:
        assert (
            restored["views"][view]["diagnostic_nonfinite"].tobytes()
            == value["views"][view]["diagnostic_nonfinite"].tobytes()
        )
        np.testing.assert_array_equal(
            restored["views"][view]["classification_by_body"],
            value["views"][view]["classification_by_body"],
        )
    out.final_guard()


@pytest.mark.parametrize(
    "field,new_value",
    [("frame_idx", m.START + 1), ("frame_completed", False), ("candidate_count", 5)],
)
def test_endpoint_companion_rejects_stale_native_incomplete_or_different_bank(field, new_value):
    value = endpoint()
    value[field] = new_value
    tracker = SimpleNamespace(export_endpoint_mask_diagnostics=lambda: value)
    with pytest.raises(ValueError, match="completed fitted candidate bank"):
        m.save_endpoint_diagnostics(SimpleNamespace(write=forbidden), tracker, m.START, 4)


def test_all1660_serial_frames_and_summary_refer_to_current_beams(monkeypatch):
    normal, all_split = m.endpoint_counts(endpoint()), m.endpoint_counts(endpoint(all_split=True))
    tracker = SimpleNamespace(
        initialize_count=1,
        restore_count=7,
        previous_frame=m.START - 1,
        processed_frames=m.START - m.FIRST,
    )
    visited, saved, syncs, guards = [], [], [], []

    def advance(observation):
        visited.append(observation.frame_idx)
        tracker.previous_frame = observation.frame_idx
        tracker.processed_frames += 1

    def save(out, actual, frame):
        assert actual is tracker and actual.previous_frame == frame
        saved.append(frame)
        return dict(frame_idx=frame, endpoint_mask=all_split if frame == m.START + 17 else normal)

    tracker.advance = advance
    monkeypatch.setattr(m, "save_frame_diagnostics", save)
    rows, timings = m.drive(
        tracker,
        iter(SimpleNamespace(frame_idx=f) for f in range(m.START, m.STOP)),
        None,
        synchronize=lambda: syncs.append(1),
        guard=lambda: guards.append(1),
    )
    assert saved == visited == list(range(m.START, m.STOP))
    assert len(saved) == len(syncs) == 1660 and len(guards) == 3320
    assert tracker.initialize_count == 1 and tracker.restore_count == 7
    assert all(value >= 0 for value in timings.values())
    summary = m.endpoint_summary(rows)
    assert summary["frames"] == 1660
    assert summary["all_split_fallback_frames"] == summary["preferred_candidate_split_frames"] == 1
    assert summary["beam_changed_frames"] == 1659
    assert summary["totals"]["candidate_count"] == 6640
    assert summary["totals"]["retained_candidates"] == 4980
    assert summary["totals"]["retained_split_bodies"] == 1662
    assert summary["totals"]["retained_negative_score_margins"] == 1660
    assert summary["classification_counts_by_view"][m.sparse.VIEWS[0]]["split_instances"] == 3322
    assert summary["classification_counts_by_view"][m.sparse.VIEWS[1]]["unavailable"] == 1660
    assert all(
        sum(row.values()) == 19920 for row in summary["classification_counts_by_view"].values()
    )
    assert summary["counts_describe_current_retained_beams_not_final_backtrace"]
    assert summary["final_path_need_not_minimize_original_numeric_cost"]
    assert summary["numerical_objective_unchanged"] and not summary["physical_ID_truth"]
    with pytest.raises(ValueError, match="coverage"):
        m.endpoint_summary(rows[:-1])


def test_source_closure_requires_new_tracker_and_scorer_pins(isolated, monkeypatch):
    assert (
        m.PINS["endpoint_mask_priority_tracker.py"]
        == "fa111b262b48a40f730233be39f464f02e1433bf65f441337b5a473d81c2196c"
    )
    assert (
        m.PINS["endpoint_mask_conflict_score.py"]
        == "8739a48c7fa72fa2f58cb720051fcea4ab81da609612763f42e959cc7ad33d60"
    )
    code = {str(Path(m.__file__).resolve()): "0" * 64}
    code.update({str((isolated / name).resolve()): sha for name, sha in m.PINS.items()})
    registered = {}
    evidence = SimpleNamespace(add=lambda path, digest: registered.__setitem__(path, digest))
    monkeypatch.setattr(m.prior, "local_code", lambda: code.copy())
    monkeypatch.setattr(m.prior, "source_contract", lambda value: None)
    assert m.source_code(evidence) == registered == code
    for name in ("endpoint_mask_priority_tracker.py", "endpoint_mask_conflict_score.py"):
        path = str((isolated / name).resolve())
        correct = code.pop(path)
        with pytest.raises(ValueError, match="frozen dependency"):
            m.source_code(evidence)
        code[path] = correct
