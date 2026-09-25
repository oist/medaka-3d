"""Five synthetic tests; every scientific input boundary is isolated or fail-spied."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import compare_sparse_depth_motion_cases as caller
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(caller, "WORK", work)
    monkeypatch.setattr(caller, "SPARSE_ROOT", work / "sparse")
    for module in (
        caller.motion,
        caller.female,
        caller.depth,
        caller.depth.comparison,
        caller.depth.comparison.common,
        caller.proof,
    ):
        for field in ("WORK", "REPO"):
            if hasattr(module, field):
                monkeypatch.setattr(module, field, work)
    monkeypatch.setattr(caller.depth, "COMPARISON_ROOT", work / "comparison")

    def forbidden(*args, **kwargs):
        pytest.fail("real scientific headers/workflow forbidden")

    for module, names in (
        (caller, ("load_headers", "bind_sources")),
        (caller.female, ("identity_headers", "load_mapping", "run", "source_code")),
        (caller.depth, ("comparison_headers", "load_candidate", "run", "source_code")),
        (caller.motion, ("load_headers", "run")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


def samples(frames=caller.FRAMES):
    pose = np.zeros((len(frames), 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(3)[None, :, None] * 30 + np.arange(3)[None, None, :]
    return dict(
        frame_idx=frames.copy(),
        pose_xyz=pose,
        observed=np.ones((len(frames), 3, 3), bool),
        permutation_by_id=np.tile([0, 1, 2], (len(frames), 1)).astype(np.int8),
    )


def test_three_noncontiguous_pairs_are_never_differenced_across_gaps():
    rows = samples()
    # Large offsets between pairs must never become adjacent-step events.
    rows["pose_xyz"][2:4] += 500
    rows["pose_xyz"][4:6] += 1000
    with pytest.raises(ValueError, match="contiguous"):
        caller.motion.motion(rows["frame_idx"], rows["pose_xyz"], rows["observed"])
    result = caller.evaluate_pairs({caller.SPARSE: rows, caller.DEPTH: rows})
    assert [row["native_pair"] for row in result] == [list(pair) for pair in caller.PAIRS]
    for row in result:
        for arm in row["arms"].values():
            np.testing.assert_array_equal(arm["raw"]["distance_mm"], np.zeros((3, 3)))
            assert np.array(arm["saved_beam_ID"]["both_direct"]).all()


def test_each_endpoint_nonselfinverse_P_gathers_all_nodes_and_support():
    rows = samples()
    rows["permutation_by_id"][:] = [
        [0, 1, 2],
        [1, 2, 0],
        [1, 2, 0],
        [2, 0, 1],
        [2, 0, 1],
        [0, 2, 1],
    ]
    canonical = rows["pose_xyz"].copy()
    support = rows["observed"].copy()
    canonical[1, 1, 2, 0] += 12
    support[1, 2, 2] = False
    for t, permutation in enumerate(rows["permutation_by_id"]):
        for identity, slot in enumerate(permutation):
            rows["pose_xyz"][t, slot] = canonical[t, identity]
            rows["observed"][t, slot] = support[t, identity]
    before = {key: value.copy() for key, value in rows.items()}
    result = caller.evaluate_pairs({caller.SPARSE: rows, caller.DEPTH: rows})
    arm = result[0]["arms"][caller.DEPTH]
    assert arm["saved_beam_ID"]["distance_mm"][1][2] == 12
    assert arm["saved_beam_ID"]["gt10mm_both_direct"][1][2]
    assert arm["saved_beam_ID"]["previous_direct"][2][2]
    assert not arm["saved_beam_ID"]["current_direct"][2][2]
    assert not arm["saved_beam_ID"]["both_direct"][2][2]
    assert any(
        event["mapped_outcome"] == "not_both_direct"
        for event in arm["raw_direct_large_event_ID_followup"]
    )
    for key in rows:
        np.testing.assert_array_equal(rows[key], before[key])


def test_native_shape_permutation_and_original_initial_boundary_guards(monkeypatch):
    rows = samples()
    for bad in (rows["frame_idx"][:-1], np.array([80424, 80424, 82719, 82720, 88372, 88373])):
        with pytest.raises(ValueError):
            caller.select_native({**rows, "frame_idx": bad})
    with pytest.raises(ValueError, match="frame axis"):
        caller.select_native({**rows, "observed": rows["observed"][:-1]})
    with pytest.raises(ValueError, match="permutation"):
        caller.check_mapping(rows["frame_idx"], np.tile([0, 0, 2], (6, 1)), full_science=False)
    monkeypatch.setattr(caller, "START", 100)
    monkeypatch.setattr(caller, "STOP", 103)
    full_frames = np.arange(100, 103, dtype=np.int64)
    full_P = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], np.int8)
    caller.check_mapping(full_frames, full_P, full_science=True)
    caller.check_mapping(full_frames[1:], full_P[1:], full_science=False)
    with pytest.raises(ValueError, match="original P012"):
        caller.check_mapping(full_frames, np.tile([1, 2, 0], (3, 1)), full_science=True)
    # Both disabled and invalid-output paths stop before the fail-spied headers.
    with pytest.raises(ValueError, match="preparation"):
        caller.run("x", "y", "z", caller.WORK / "disabled")
    monkeypatch.setattr(caller, "EXECUTION_READY", True)
    with pytest.raises(ValueError, match="workspace"):
        caller.run("x", "y", "z", caller.WORK.parent / "outside")


def test_line_only_flag_normalization_preserves_quotes_and_rejects_other_changes():
    quoted = b'old, new = b"EXECUTION_READY = False", b"EXECUTION_READY = True"\r\n'
    disabled = b"EXECUTION_READY = False\r\n" + quoted
    enabled = b"EXECUTION_READY = True\r\n" + quoted
    expected = hashlib.sha256(disabled).hexdigest()
    assert caller.normalized_flag_digest(disabled) == expected
    assert caller.normalized_flag_digest(enabled) == expected
    assert (
        caller.normalized_flag_digest(enabled.replace(quoted, quoted.replace(b"old,", b"new,")))
        != expected
    )
    with pytest.raises(ValueError, match="one exact"):
        caller.normalized_flag_digest(quoted)
    with pytest.raises(ValueError, match="one exact"):
        caller.normalized_flag_digest(disabled + b"EXECUTION_READY = False\r\n")


def test_synthetic_end_to_end_real_selectors_and_only_minimum_fields(monkeypatch):
    monkeypatch.setattr(caller, "EXECUTION_READY", True)
    monkeypatch.setattr(caller, "FIRST", 80423)
    monkeypatch.setattr(caller, "START", 80424)
    monkeypatch.setattr(caller, "STOP", 88374)
    full = samples(np.arange(caller.FIRST, caller.STOP, dtype=np.int64))
    full["permutation_by_id"][full["frame_idx"] >= 82720] = [1, 2, 0]
    full["permutation_by_id"][full["frame_idx"] >= 88373] = [2, 0, 1]
    full["observed"][full["frame_idx"] == 88373, 1, 2] = False
    sample_frames = np.sort(np.r_[caller.FRAMES, np.arange(80426, 80693)]).astype(np.int64)
    assert len(sample_frames) == 273
    small = caller.select_native(full, sample_frames)
    sparse_path = caller.SPARSE_ROOT / "event_boundary_samples.npz"
    sparse_path.parent.mkdir()
    np.savez(sparse_path, **small)
    depth_path = caller.depth.COMPARISON_ROOT / f"candidates/{caller.DEPTH}.npz"
    depth_path.parent.mkdir(parents=True)
    raw = {key: full[key] for key in ("frame_idx", "pose_xyz", "observed")}
    # Any attempt to decode an unrequested member fails allow_pickle=False.
    np.savez(
        depth_path,
        **{key: raw.get(key, np.array([object()])) for key in caller.motion.CANDIDATE_FIELDS},
    )
    identity_root = caller.WORK / "depth_id"
    identity_root.mkdir()
    mapping_path = identity_root / "identity_mapping_and_ambiguity.npz"
    mapping = {key: full[key][1:] for key in ("frame_idx", "permutation_by_id")}
    np.savez(
        mapping_path,
        **{key: mapping.get(key, np.array([object()])) for key in caller.motion.MAPPING_FIELDS},
    )

    def seal(path, name):
        return dict(
            outputs={
                name: dict(
                    path=str(path.resolve()),
                    sha256=caller.motion.sha(path),
                    size_bytes=path.stat().st_size,
                )
            }
        )

    uncertainty = dict(
        pruning_occurred=True,
        gaps_conditional_on_retained_graph=True,
        conditional_tied_frames=1,
        unknown_ID_alternative_cells=0,
        global_optimality_certified=False,
    )
    headers = dict(
        common_seal=seal(depth_path, f"candidates/{caller.DEPTH}.npz"),
        reference=dict(arrays={key: caller.motion.signature(value) for key, value in raw.items()}),
        identity=dict(
            root=identity_root, seal=seal(mapping_path, mapping_path.name), summary=uncertainty
        ),
        sparse_seal=seal(sparse_path, sparse_path.name),
        sparse_summary=dict(
            focused_samples=dict(
                fields={key: caller.motion.signature(value) for key, value in small.items()}
            ),
            beam_uncertainty=uncertainty,
        ),
        sources={"synthetic_trust_boundary": "stub"},
    )
    # Replace only the already fail-spied header boundary. Real loaders and
    # matching/control flow below it run against three synthetic local archives.
    monkeypatch.setattr(caller, "load_headers", lambda *args: headers)
    original_load = np.load
    allowed = {p.resolve() for p in (sparse_path, depth_path, mapping_path)}
    reads = []

    def guarded_load(path, *args, **kwargs):
        path = Path(path).resolve()
        assert path in allowed
        reads.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(caller.motion.np, "load", guarded_load)
    output = caller.WORK / "result"
    result = caller.run("synthetic_comparison", "synthetic_id", "synthetic_sparse", output)
    assert len(reads) == len(set(reads)) == 3
    assert len(result["pairs"]) == 3
    assert result["pairs"][1]["arms"][caller.DEPTH]["permutation_changed"]
    assert result["pairs"][2]["arms"][caller.DEPTH]["raw"]["current_direct"][1][2] is False
    assert not result["whole_five_minute_rescoring"]
    provenance = json.loads((output / "comparison.provenance.json").read_text())
    assert len(provenance["outputs"]) == 3
    for row in provenance["outputs"].values():
        assert caller.motion.sha(Path(row["path"])) == row["sha256"]
    for arm in (caller.SPARSE, caller.DEPTH):
        with original_load(output / f"selected/{arm}.npz") as saved:
            assert set(saved.files) == caller.SAMPLE_FIELDS
            np.testing.assert_array_equal(saved["frame_idx"], caller.FRAMES)
            np.testing.assert_array_equal(
                saved["pose_xyz"], full["pose_xyz"][caller.FRAMES - caller.FIRST]
            )
