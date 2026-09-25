"""Synthetic tests only; no actual tracking modules or input files are loaded."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import diagnose_sparse_id_aligned_motion as caller
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(caller, "WORK", work)
    monkeypatch.setattr(caller, "COMPARISON", work / "comparison")
    monkeypatch.setattr(caller, "IDENTITY", work / "identity")
    monkeypatch.setattr(caller, "load_headers", lambda *args: pytest.fail("real headers forbidden"))


def arrays(count=4):
    frames = np.arange(100, 100 + count, dtype=np.int64)
    pose = np.zeros((count, 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(3)[None, :, None] * 30 + np.arange(3)[None, None, :]
    observed = np.ones((count, 3, 3), bool)
    return frames, pose, observed


def scatter(canonical, permutation):
    raw = np.empty_like(canonical)
    for t, row in enumerate(permutation):
        for identity, slot in enumerate(row):
            raw[t, slot] = canonical[t, identity]
    return raw


def test_non_self_inverse_whole_body_orientation_and_support():
    frames, canonical, support = arrays()
    permutation = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2]], np.int8)
    support[1, 0, 2] = False
    support[2, 2, 0] = False
    raw_pose, raw_support = scatter(canonical, permutation), scatter(support, permutation)
    pose_before, support_before = raw_pose.copy(), raw_support.copy()
    mapped_pose, mapped_support = caller.reorder_whole_body(raw_pose, raw_support, permutation)
    np.testing.assert_array_equal(mapped_pose, canonical)
    np.testing.assert_array_equal(mapped_support, support)
    np.testing.assert_array_equal(raw_pose, pose_before)
    np.testing.assert_array_equal(raw_support, support_before)
    assert not np.array_equal(raw_pose[1, np.argsort(permutation[1])], canonical[1])
    result = caller.compare_motion(frames, raw_pose, raw_support, permutation)
    assert all(
        row["all_estimated"]["maximum_step_mm"] == 0
        for row in result["saved_beam_ID_aligned"]["by_body_node"]
    )
    assert result["per_node_distance_and_direct_multisets_preserved_off_boundaries"]


def test_direct_support_follows_each_endpoint_identity_and_is_not_cleared():
    frames, canonical, support = arrays(2)
    permutation = np.array([[0, 1, 2], [1, 2, 0]], np.int8)
    canonical[1, 1, 2, 0] += 12
    support[1, 2, 2] = False
    raw_pose, raw_support = scatter(canonical, permutation), scatter(support, permutation)
    result = caller.compare_motion(frames, raw_pose, raw_support, permutation)
    tails = {
        row["body"]: row
        for row in result["saved_beam_ID_aligned"]["by_body_node"]
        if row["node"] == "tail"
    }
    assert tails[0]["both_direct"]["adjacent_steps"] == 1
    assert tails[1]["both_direct"]["steps_gt10mm"] == 1
    assert tails[2]["both_direct"]["adjacent_steps"] == 0
    assert tails[2]["missing_frames"] == 1
    follow = {
        row["body"]: row
        for row in result["raw_direct_large_event_ID_followup"]
        if row["node"] == "tail"
    }
    assert follow[2]["mapped_outcome"] == "not_both_direct"
    assert not follow[2]["mapped_both_direct"]
    assert follow[1]["mapped_outcome"] == "persists_gt10mm"


def test_exact_threshold_and_no_gap_bridging():
    frames, pose, observed = arrays(3)
    pose[1, 0, 0, 0] += 10
    pose[2, 0, 0, 0] += 21
    observed[1, 0, 0] = False
    result, _, _ = caller.motion(frames, pose, observed)
    row = result["by_body_node"][0]
    assert row["all_estimated"]["adjacent_steps"] == 2
    assert row["all_estimated"]["steps_gt10mm"] == 1
    assert row["both_direct"]["adjacent_steps"] == 0
    assert row["both_direct"]["maximum_step_mm"] is None
    assert result["large_events"] == [
        dict(
            mode="all_estimated",
            body=0,
            node="mouth",
            previous_frame=101,
            frame_idx=102,
            distance_mm=11.0,
            previous_direct=False,
            current_direct=True,
        )
    ]


@pytest.mark.parametrize("invalid", ["gap", "duplicate", "float_frame", "nan_pose", "bad_support"])
def test_invalid_geometry_fails_explicitly(invalid):
    frames, pose, observed = arrays()
    if invalid == "gap":
        frames[-1] += 1
    elif invalid == "duplicate":
        frames[-1] -= 1
    elif invalid == "float_frame":
        frames = frames.astype(float)
    elif invalid == "nan_pose":
        pose[0, 0, 0, 0] = np.nan
    else:
        observed = observed.astype(np.int8)
    with pytest.raises(ValueError):
        caller.motion(frames, pose, observed)


@pytest.mark.parametrize("permutation", [[[0, 0, 2]] * 4, [[0, 1, 3]] * 4, [[0.0, 1.0, 2.0]] * 4])
def test_bad_permutation_rejected(permutation):
    _, pose, support = arrays()
    with pytest.raises(ValueError, match="bijective"):
        caller.reorder_whole_body(pose, support, np.array(permutation))


def saved_motion(frames, pose, support, start, stop):
    keep = (frames >= start) & (frames < stop)
    rows = []
    for body in range(3):
        for node, name in enumerate(caller.NODES):
            delta = np.linalg.norm(np.diff(pose[:, body, node], axis=0), axis=-1)
            direct = support[:-1, body, node] & support[1:, body, node] & keep[:-1] & keep[1:]
            values = delta[direct]
            rows.append(
                dict(
                    slot=body,
                    node=name,
                    supported_frames=int(support[keep, body, node].sum()),
                    missing_frames=int((~support[keep, body, node]).sum()),
                    directly_supported_adjacent_steps=int(direct.sum()),
                    directly_supported_adjacent_steps_gt10mm=int((values > 10).sum()),
                    maximum_directly_supported_adjacent_step_mm=float(values.max())
                    if len(values)
                    else None,
                    maximum_latent_adjacent_step_mm=float(delta[keep[1:]].max()),
                )
            )
    return dict(by_slot_node=rows)


def test_initial_boundary_only_in_old_latent_qualification():
    frames, pose, observed = arrays(4)
    pose[0, 0, 0, 0] -= 100
    raw, _, _ = caller.motion(frames[1:], pose[1:], observed[1:])
    saved = saved_motion(frames, pose, observed, 101, 104)
    assert saved["by_slot_node"][0]["maximum_latent_adjacent_step_mm"] == 100
    assert raw["by_body_node"][0]["all_estimated"]["maximum_step_mm"] == 0
    qualification = caller.verify_raw_reference(raw, saved, frames, pose, 101, 104)
    assert qualification["old_latent_maxima_reproduced_in_separate_boundary_scope"]
    bad = copy.deepcopy(saved)
    bad["by_slot_node"][0]["directly_supported_adjacent_steps"] += 1
    with pytest.raises(ValueError, match="raw direct"):
        caller.verify_raw_reference(raw, bad, frames, pose, 101, 104)


def test_constant_nonidentity_permutation_changes_labels_only():
    frames, pose, observed = arrays(4)
    pose[2:, 0, 2, 0] += 20
    observed[2, 1, 1] = False
    permutation = np.tile([1, 2, 0], (4, 1))
    result = caller.compare_motion(frames, pose, observed, permutation)
    assert result["permutation_boundaries"] == []
    assert result["unchanged_permutation_adjacent_pairs"] == 3
    mapped = result["saved_beam_ID_aligned"]["large_events"]
    assert all(event["body"] == 2 for event in mapped)
    sample = caller.focused_samples(result, frames, pose, observed, permutation)
    np.testing.assert_array_equal(sample["frame_idx"], [101, 102])
    np.testing.assert_array_equal(sample["pose_xyz"], pose[1:3])


def test_empty_events_produce_well_shaped_empty_samples():
    frames, pose, observed = arrays()
    permutation = np.tile([0, 1, 2], (4, 1))
    result = caller.compare_motion(frames, pose, observed, permutation)
    samples = caller.focused_samples(result, frames, pose, observed, permutation)
    assert samples["frame_idx"].shape == (0,)
    assert samples["pose_xyz"].shape == (0, 3, 3, 3)
    assert samples["observed"].shape == (0, 3, 3)
    assert samples["permutation_by_id"].shape == (0, 3)


def test_source_only_and_outside_guards_cannot_reach_real_headers(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="disabled"):
        caller.run(caller.WORK / "disabled")
    monkeypatch.setattr(caller, "EXECUTION_READY", True)
    with pytest.raises(ValueError, match="fresh output"):
        caller.run(tmp_path / "outside")


def test_capture_detects_tamper_and_outside(tmp_path):
    path = caller.WORK / "small.json"
    path.write_text("{}")
    digest = caller.sha(path)
    caller.capture(path, {}, digest)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="SHA"):
        caller.capture(path, {}, digest)
    with pytest.raises(ValueError, match="outside"):
        caller.capture(tmp_path / "outside.json", {})


def test_synthetic_run_loads_only_three_plus_two_fields_and_saves_samples(monkeypatch):
    monkeypatch.setattr(caller, "EXECUTION_READY", True)
    monkeypatch.setattr(caller, "FIRST", 100)
    monkeypatch.setattr(caller, "START", 101)
    monkeypatch.setattr(caller, "STOP", 105)
    frames, canonical, support = arrays(5)
    permutation = np.array([[0, 1, 2], [0, 1, 2], [1, 2, 0], [1, 2, 0], [1, 2, 0]], np.int8)
    support[3, 2, 2] = False
    pose, observed = scatter(canonical, permutation), scatter(support, permutation)
    pose[0, 0, 0, 0] -= 150
    raw = dict(frame_idx=frames, pose_xyz=pose, observed=observed)
    reference = dict(arrays={k: caller.signature(v) for k, v in raw.items()})
    mapping = dict(frame_idx=frames[1:], permutation_by_id=permutation[1:])
    candidate_path = caller.COMPARISON / f"candidates/{caller.ARM}.npz"
    candidate_path.parent.mkdir(parents=True)
    mapping_path = caller.IDENTITY / "identity_mapping_and_ambiguity.npz"
    mapping_path.parent.mkdir()
    # Object placeholders must never be decoded (allow_pickle=False).
    np.savez(
        candidate_path, **{k: raw.get(k, np.array([object()])) for k in caller.CANDIDATE_FIELDS}
    )
    np.savez(
        mapping_path, **{k: mapping.get(k, np.array([object()])) for k in caller.MAPPING_FIELDS}
    )

    def seal(path, name):
        return dict(
            outputs={
                name: dict(path=str(path), sha256=caller.sha(path), size_bytes=path.stat().st_size)
            }
        )

    comp = seal(candidate_path, f"candidates/{caller.ARM}.npz")
    ids = seal(mapping_path, mapping_path.name)
    id_summary = dict(
        pruning_occurred=True,
        finite_prefixes_discarded=True,
        gaps_conditional_on_retained_graph=True,
        conditional_tied_frames=1,
        unknown_ID_alternative_cells=0,
        global_optimality_certified=False,
    )
    saved = saved_motion(frames, pose, observed, 101, 105)
    monkeypatch.setattr(
        caller, "load_headers", lambda evidence: (comp, ids, reference, saved, id_summary)
    )
    original_load = np.load
    reads = []

    def guarded_load(path, *args, **kwargs):
        path = Path(path).resolve()
        assert path in {candidate_path.resolve(), mapping_path.resolve()}
        reads.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(caller.np, "load", guarded_load)
    output = caller.WORK / "result"
    result = caller.run(output)
    assert reads == [candidate_path.resolve(), mapping_path.resolve()]
    assert result["native_frames"] == 4 and result["adjacent_frame_pairs"] == 3
    assert [r["frame_idx"] for r in result["permutation_boundaries"]] == [102]
    assert result["raw_direct_large_event_ID_followup"]
    assert not result["saved_beam_ID_aligned"]["large_events"]
    assert result["beam_uncertainty"]["conditional_tied_frames"] == 1
    assert result["review_read_or_cleared"] is False
    written = json.loads((output / "summary.json").read_text())
    assert written == result
    provenance = json.loads((output / "diagnostic.provenance.json").read_text())
    assert set(provenance["outputs"]) == {"summary.json", "event_boundary_samples.npz"}
    for row in provenance["outputs"].values():
        assert caller.sha(Path(row["path"])) == row["sha256"]
    with original_load(output / "event_boundary_samples.npz") as samples:
        assert set(samples.files) == {"frame_idx", "pose_xyz", "observed", "permutation_by_id"}
        np.testing.assert_array_equal(samples["frame_idx"], [101, 102])
        np.testing.assert_array_equal(samples["observed"], observed[1:3])
