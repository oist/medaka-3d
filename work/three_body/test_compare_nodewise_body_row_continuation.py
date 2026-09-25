"""Synthetic three-arm wiring only; frozen five metric tests are not repeated."""

import copy
from pathlib import Path
from types import SimpleNamespace

import compare_nodewise_body_row_continuation as m
import numpy as np
import pytest
import render_nodewise_body_row_continuation as renderer

NP_LOAD = np.load


@pytest.fixture(autouse=True)
def isolation(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic test reached a real candidate/NPZ input")

    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(renderer, "WORK", tmp_path)
    monkeypatch.setattr(m.old, "WORK", tmp_path)
    monkeypatch.setattr(m.common, "WORK", tmp_path)
    monkeypatch.setattr(m.common, "REPO", tmp_path)
    monkeypatch.setattr(m.old, "CONTROL", tmp_path / "control")
    monkeypatch.setattr(m, "WHOLE", tmp_path / "whole")
    monkeypatch.setattr(m.old, "load_candidate", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    for module, name in ((m, "new_source.py"), (m.old, "old_source.py")):
        path = tmp_path / name
        path.write_text("# synthetic source trust boundary\n")
        monkeypatch.setattr(module, "__file__", str(path))
    monkeypatch.setattr(m, "OLD_SHA", m.proof.sha(Path(m.old.__file__)))


def candidate(offset):
    frames = np.arange(m.FIRST, m.STOP, dtype=np.int64)
    n = len(frames)
    pose = np.zeros((n, 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(3)[None, :, None] * 10 + offset
    observed = np.ones((n, 3, 3), bool)
    observed[:, offset % 3, 2] = False
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).astype(np.int32)
    sources = np.where(observed[:, :, None, :], sources, -1)
    return dict(
        frame_idx=frames,
        pose_xyz=pose,
        observed=observed,
        sources=sources,
        proposal_kind=np.full(n, "predict", dtype="U16"),
    )


def headers():
    control = dict(original_binding={"synthetic_inputs": "same"}, parent_tracking_sha256="a" * 64)
    summary = dict(
        process_interval=[m.START, m.STOP],
        retained_interval=[m.FIRST, m.STOP],
        initialize_count=1,
        restore_count=7,
        external_pose_resets=0,
        new_begin_calls=0,
        finish_calls=1,
        parent_tracking_sha256=control["parent_tracking_sha256"],
        parent_checkpoint_sha256=m.old.PARENT_CHECKPOINT_SHA,
    )
    plan = dict(
        original_binding=control["original_binding"],
        parent_tracking_sha256=control["parent_tracking_sha256"],
        parent_checkpoint_sha256=m.old.PARENT_CHECKPOINT_SHA,
        tracker_context={"cameras": []},
    )
    imported = dict(
        status="ok",
        previous_frame=m.START - 1,
        processed_frames=m.START - m.FIRST,
        effective_from_native=m.START,
        added_restore_count=1,
        new_begin_calls=0,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        original_metadata={"restore_count": 6, "synthetic_parent": "same"},
        arrays={f"field{i}": {"synthetic_signature": i} for i in range(39)},
    )
    return control, summary, plan, imported


@pytest.mark.parametrize("damage", ["checkpoint", "inputs", "restore", "terminal"])
def test_variant_binding_rejects_different_conditions(damage):
    control, summary, plan, imported = headers()
    m.validate_variant(summary, plan, imported, control)
    if damage == "checkpoint":
        plan["parent_checkpoint_sha256"] = "b" * 64
    elif damage == "inputs":
        plan["original_binding"] = {"synthetic_inputs": "different"}
    elif damage == "restore":
        imported["added_restore_count"] = 2
    else:
        summary["retained_interval"][1] += 1
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        m.validate_variant(summary, plan, imported, control)


def test_three_arm_samples_and_projection_keep_body_support_axes():
    candidates = {arm: candidate(i) for i, arm in enumerate(m.ARMS)}
    before = {a: m.common.signature(c) for a, c in candidates.items()}
    metrics, samples, deltas = m.compare_arrays(candidates)
    m.validate_samples(samples)
    assert tuple(metrics) == m.ARMS and len(samples) == 15 and len(deltas) == 3
    rig = SimpleNamespace(camera=lambda view: SimpleNamespace(project=lambda pose: pose[..., :2]))
    projected = renderer.project_samples(samples, 2, rig)
    for i, arm in enumerate(m.ARMS):
        np.testing.assert_array_equal(samples[arm + "_observed"][:, i, 2], False)
        assert metrics[arm]["missing_by_slot_node"][i][2] == m.STOP - m.START
        np.testing.assert_array_equal(
            projected[arm][renderer.drawing.VIEWS[0]][:, 0, 0], np.arange(3) * 10 + i
        )
        assert m.common.signature(candidates[arm]) == before[arm]


@pytest.mark.parametrize("different_parent", [False, True])
def test_synthetic_three_arm_run_and_saved_axes(monkeypatch, tmp_path, different_parent):
    """Full run wiring; frozen all21 loader is an explicit synthetic trust boundary."""
    control, summary, plan, imported = headers()
    roots = (m.old.CONTROL, m.WHOLE, tmp_path / "new")
    loaded, calls = {}, []
    for i, (arm, root) in enumerate(zip(m.ARMS, roots, strict=True)):
        root.mkdir()
        arrays = candidate(i)
        document = copy.deepcopy(control if arm == "control" else summary)
        document["candidate_arrays"] = m.common.signature(arrays)
        seal = dict(
            outputs={
                "candidate.npz": dict(
                    path=str(root / "candidate.npz"), sha256=str(i) * 64, size_bytes=1
                )
            }
        )
        if arm != "control":
            record = copy.deepcopy(imported)
            if different_parent and arm == "nodewise_body_row":
                record["original_metadata"]["synthetic_parent"] = "different"
            for name, value in (("plan.json", plan), ("state_import.json", record)):
                path = root / name
                path.write_bytes(m.proof.json_bytes(value))
                seal["outputs"][name] = dict(
                    path=str(path), sha256=m.proof.sha(path), size_bytes=path.stat().st_size
                )
        loaded[root] = arrays, document, seal

    def synthetic_candidate_loader(root, seal_name, digest, stage, evidence):
        assert root in roots and root.is_relative_to(tmp_path)
        calls.append(root)
        return loaded[root]

    monkeypatch.setattr(m.old, "load_candidate", synthetic_candidate_loader)
    output = tmp_path / "result"
    if different_parent:
        with pytest.raises((ValueError, RuntimeError, AssertionError), match="original39"):
            m.run(roots[2], "f" * 64, output)
        assert m.proof.read_json(output / "comparison.provenance.json")["status"] == "failed"
        assert not (output / "review_samples.npz").exists()
        return
    m.run(roots[2], "f" * 64, output)
    assert len(calls) == len(set(calls)) == 3
    actual = m.proof.read_json(output / "summary.json")
    assert actual["arms"] == list(m.ARMS) and tuple(actual["methods"]) == m.ARMS
    assert (
        not actual["physical_ID_truth"] and not actual["raw_slots_independently_identity_aligned"]
    )
    path = output / "review_samples.npz"
    assert path.is_relative_to(tmp_path)
    with NP_LOAD(path, allow_pickle=False) as archive:
        samples = {k: archive[k] for k in archive.files}
    m.validate_samples(samples)
    assert m.common.signature(samples) == m.proof.read_json(
        output / "review_samples_signatures.json"
    )
    assert m.proof.read_json(output / "comparison.provenance.json")["status"] == "ok"
