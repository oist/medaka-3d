"""Synthetic CPU evidence and original six-beam checkpoint tests; no real data/fit."""

import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import receipted_long_runner_migration as m
from continuous_tracker import RECORD_FIELDS, STATE_FIELDS


@dataclasses.dataclass
class Camera:
    axes: tuple

    def project(self, xyz):
        return np.asarray(xyz)[..., self.axes]


@dataclasses.dataclass
class Rig:
    cameras: tuple
    views: tuple = m.original.base.VIEWS
    world_units: str = "mm"

    def camera(self, view):
        return self.cameras[self.views.index(view)]


def synthetic_state(count=5, restores=7):
    rig = Rig(tuple(Camera(axes) for axes in ((0, 1), (0, 2), (1, 2))))
    pose = np.array(
        [
            [[40.0, 40.0, 40.0], [50.0, 40.0, 40.0], [65.0, 40.0, 40.0]],
            [[90.0, 90.0, 90.0], [100.0, 90.0, 90.0], [115.0, 90.0, 90.0]],
            [[140.0, 140.0, 140.0], [150.0, 140.0, 140.0], [165.0, 140.0, 140.0]],
        ]
    )
    cfg = m.original.TrackerConfig(
        device="cpu", mode="nodes", beam_size=6, max_hypotheses=18, opt_steps=1, flow_proposals=True
    )
    tracker = m.original.AcceleratedContinuousPersistentNodeTracker(rig, pose, cfg)
    held = np.ones((3, 3), bool)
    held[2, 1] = False
    sources = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    sources = np.where(held[:, None], sources, -1)
    records, velocities = [], [np.zeros((3, 3)) for _ in range(6)]
    poses = [pose + np.array([branch * 0.6, 0, 0]) for branch in range(6)]
    for t in range(count):
        for branch in range(6):
            displacement = poses[branch][:, 1] - (pose if t == 0 else poses[branch])[:, 1]
            velocity = 0.5 * velocities[branch] + 0.5 * displacement
            norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
            velocity *= np.minimum(1.0, 2.0 / np.maximum(norm, 1e-12))
            velocities[branch] = np.where(held[:, 1, None], velocity, 0.8 * velocities[branch])
            records.append(
                dict(
                    frame_idx=m.FIRST + t,
                    pose_xyz=poses[branch].copy(),
                    observed=held.copy(),
                    sources=sources.copy(),
                    identity_review=np.array([False, False, True]),
                    mask_supported=np.zeros(3, bool),
                    body_supported=held[:, 1].copy(),
                    missing=np.array([0, 0, t + 1]),
                    costs=float(branch + 1),
                    path_costs=float((t + 1) * (branch + 1)),
                    loss_parts=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                    score_margin=np.inf,
                    parent=branch if t else 0,
                    proposal_kind="predict",
                    hypotheses=18,
                    proposal_flow_nodes=np.zeros((3, 3), bool),
                    proposal_flow_parent=0 if t else -1,
                    node_missing=np.where(held, 0, t + 1),
                    direct_pectoral_age=np.array([t + 1, t + 1, 0]),
                )
            )
    arrays = {"history_offsets": np.arange(0, 6 * (count + 1), 6, dtype=np.int64)}
    arrays.update(
        {"history_" + key: np.asarray([r[key] for r in records]) for key in RECORD_FIELDS}
    )
    for key, value in dict(
        pose=np.stack(poses),
        velocity=np.stack(velocities),
        missing=np.tile([0, 0, count], (6, 1)),
        review=np.tile([False, False, True], (6, 1)),
        cost=np.arange(1, 7, dtype=float) * count,
        node_missing=np.tile(np.where(held, 0, count), (6, 1, 1)),
        direct_pectoral_age=np.tile([count, count, 0], (6, 1)),
    ).items():
        arrays["state_" + key] = value
    presence = {}
    for i, view in enumerate(rig.views):
        for field, value in dict(
            detections=rig.camera(view).project(pose),
            images_bgr=np.full((4, 5, 3), i, np.uint8),
            masks=np.empty((0, 4, 5), bool),
            backgrounds_bgr=np.full((4, 5, 3), i + 10, np.uint8),
        ).items():
            key = "previous_" + field + "_" + view
            arrays[key], presence[key] = value, True
    diag = copy.deepcopy(tracker.node_recovery_diagnostics)
    diag["calls"] = count
    metadata = dict(
        schema="continuous_persistent_node_tracker_v1",
        context=tracker._context(),
        initialize_count=1,
        restore_count=restores,
        processed_frames=count,
        previous_frame=m.FIRST + count - 1,
        record_fields=list(RECORD_FIELDS),
        state_fields=list(STATE_FIELDS),
        previous_observation=presence,
        node_recovery_diagnostics=diag,
    )
    # Use the actual frozen schema constant rather than inventing a format.
    import continuous_tracker

    metadata["schema"] = continuous_tracker.SCHEMA
    tracker._decode_state(arrays, metadata)
    return tracker, arrays, metadata


def bindings(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "REPO", tmp_path)
    legacy = dict(
        schema="long_continuation_fixed_binding_v1",
        code_sha256={str(tmp_path / "original.py"): "a" * 64},
        inputs={"seed": "fixed"},
        package_versions={"torch": "fixed"},
        opencv_threads=1,
        torch_cpu_threads=1,
        cuda_matmul_tf32=False,
    )
    code = {**legacy["code_sha256"], **{str(tmp_path / p): "b" * 64 for p in m.ADDED_IO}}
    for name, value in m.PINNED_IO.items():
        code[str(tmp_path / name)] = value
    return legacy, code


def test_finite_binding_one_migration_then_same_version(monkeypatch, tmp_path):
    legacy, code = bindings(monkeypatch, tmp_path)
    saved = copy.deepcopy(legacy)
    current = m.execution_binding(legacy, code)
    assert m.transition(legacy, current, legacy)["kind"] == "legacy_to_receipted"
    assert m.transition(current, current, legacy)["kind"] == "receipted_to_receipted"
    assert legacy == saved
    current["inputs"]["seed"] = "tampered"
    assert legacy == saved
    with pytest.raises(ValueError, match="finite I/O"):
        m.transition(legacy, current, legacy)


@pytest.mark.parametrize("change", ["old_code", "extra", "missing", "receipt", "bad_digest"])
def test_binding_rejects_unregistered_code_changes(monkeypatch, tmp_path, change):
    legacy, code = bindings(monkeypatch, tmp_path)
    if change == "old_code":
        code[next(iter(legacy["code_sha256"]))] = "f" * 64
    if change == "extra":
        code[str(tmp_path / "other.py")] = "c" * 64
    if change == "missing":
        code.pop(str(tmp_path / m.ADDED_IO[-1]))
    if change == "receipt":
        code[str(tmp_path / m.ADDED_IO[1])] = "d" * 64
    if change == "bad_digest":
        code[str(tmp_path / m.ADDED_IO[-1])] = "x" * 64
    with pytest.raises(ValueError):
        m.execution_binding(legacy, code)


@pytest.mark.parametrize(
    "field,value", [("opencv_threads", 2), ("cuda_matmul_tf32", True), ("torch_cpu_threads", True)]
)
def test_later_binding_runtime_and_type_changes_rejected(monkeypatch, tmp_path, field, value):
    legacy, code = bindings(monkeypatch, tmp_path)
    current = m.execution_binding(legacy, code)
    changed = copy.deepcopy(current)
    changed[field] = value
    with pytest.raises(ValueError, match="exactly"):
        m.transition(changed, current, legacy)


def test_actual_six_beam_39_restore_and_two_codec_bindings(tmp_path):
    tracker, arrays, metadata = synthetic_state()
    before = m.array_signatures(arrays)
    result = m.restore_parent(tracker, arrays, metadata)
    assert len(result["all_checkpoint_arrays_exact"]) == 39
    assert tracker.initialize_count == 1 and tracker.restore_count == 8
    assert result["array_signatures"] == before
    for folder, binding in (("legacy", {"version": 1}), ("new", {"version": 2})):
        m.checkpoint_roundtrip(tmp_path / folder, arrays, metadata, binding)
        with pytest.raises(ValueError):
            m.read_checkpoint(tmp_path / folder, expected_bindings={"version": 3})
    restored, meta = tracker.export_state()
    m.original._same_arrays(restored, arrays)
    assert meta == {**metadata, "restore_count": 8}
    m.compare_backtrace(arrays, tracker.finish())
    assert m.array_signatures(arrays) == before


@pytest.mark.parametrize(
    "change", ["velocity", "age", "review", "parent", "context", "missing_image"]
)
def test_semantic_restore_rejects_corrupt_full_history(change):
    tracker, arrays, metadata = synthetic_state()
    if change == "velocity":
        arrays["state_velocity"][1, 0, 0] += 0.1
    if change == "age":
        arrays["history_node_missing"][-1, 2, 1] = 0
    if change == "review":
        arrays["history_identity_review"][-1, 2] = False
    if change == "parent":
        arrays["history_parent"][-1] = 6
    if change == "context":
        metadata["context"]["world_units"] = "m"
    if change == "missing_image":
        arrays.pop(next(k for k in arrays if "images_bgr" in k))
    with pytest.raises(ValueError):
        m.restore_parent(tracker, arrays, metadata)
    assert tracker.initialize_count == tracker.restore_count == tracker.processed_frames == 0


def test_used_or_subclass_tracker_is_not_a_migration_target():
    tracker, arrays, metadata = synthetic_state()
    m.restore_parent(tracker, arrays, metadata)
    with pytest.raises(ValueError, match="fresh"):
        m.restore_parent(tracker, arrays, metadata)

    class Other(m.original.AcceleratedContinuousPersistentNodeTracker):
        pass

    other = Other(tracker.rig, tracker.initial, tracker.config)
    with pytest.raises(ValueError, match="unchanged original"):
        m.restore_parent(other, arrays, metadata)


def test_prefix_and_all_backtrace_preserve_parent_branches_and_sticky_review():
    _, first, _ = synthetic_state(3)
    tracker, full, meta = synthetic_state(5)
    assert m.compare_prefix(first, full)["parent_rows"] == 18
    m.restore_parent(tracker, full, meta)
    candidate = tracker.finish()
    assert len(m.compare_backtrace(full, candidate)) == 21
    assert np.isnan(candidate["accepted_xyz"][:, 2]).all()
    full["history_pose_xyz"][1, 0, 0, 0] += 0.01  # An unchosen branch must still be retained.
    with pytest.raises(ValueError):
        m.compare_prefix(first, full)


def test_evidence_cache_rejects_conflicting_expected_and_late_same_size_change(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(m, "REPO", tmp_path)
    p = tmp_path / "payload"
    p.write_bytes(b"first")
    evidence = m.Evidence()
    expected = evidence.add(p)
    with pytest.raises(ValueError, match="conflicting"):
        evidence.add(p, "0" * 64)
    p.write_bytes(b"other")
    assert evidence.add(p, expected) == expected  # Lifetime is guarded at commit, not mtime-cached.
    with pytest.raises(ValueError, match="after use"):
        evidence.final_guard()


@pytest.mark.parametrize("start,stop", [(66400, 76360), (345280, 348600)])
def test_registered_phase_boundaries(start, stop):
    m.native_phase(start, stop)


@pytest.mark.parametrize(
    "start,stop", [(66400, 78020), (66401, 76361), (True, 9960), (348600, 348600), (66400, 68060)]
)
def test_phase_skip_partial_and_terminal_restart_rejected(start, stop):
    with pytest.raises(ValueError):
        m.native_phase(start, stop)


def test_validation_only_context_does_not_query_cuda(monkeypatch):
    tracker, arrays, metadata = synthetic_state()
    config = dataclasses.asdict(tracker.config)
    config["device"] = "cuda"
    protocol = dict(
        tracker_config=config,
        node_recovery_config=dataclasses.asdict(tracker.recovery_config),
        persistence_config=dataclasses.asdict(tracker.persistence_config),
    )
    inputs = SimpleNamespace(rig=tracker.rig, initial_pose_xyz=tracker.initial)
    monkeypatch.setattr(m.original.torch.cuda, "is_available", lambda: pytest.fail("CUDA queried"))
    validator = m.validation_tracker(inputs, protocol)
    assert validator.device.type == "cpu" and validator.config.device == "cuda"
    metadata["context"]["tracker_config"]["device"] = "cuda"
    validator._decode_state(arrays, metadata)
    assert validator.initialize_count == validator.restore_count == validator.processed_frames == 0


def parent_files(monkeypatch, tmp_path):
    """An independent serialized producer fixture, with real 39-array codec and all21 NPZ."""
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "CHUNK", 1)
    tracker, arrays, metadata = synthetic_state(5)
    root = tmp_path / "parent"
    root.mkdir()
    source = tmp_path / "original.py"
    source.write_text("# retained fixture code\n")
    original_input = tmp_path / "source-input"
    original_input.write_bytes(b"registered input")
    binding = dict(
        schema="long_continuation_fixed_binding_v1", code_sha256={str(source): m.sha(source)}
    )
    m.write_checkpoint(root / "checkpoint", arrays, metadata, bindings=binding)
    np.savez_compressed(root / "candidate.npz", **m.traced_candidate(arrays))
    chunks = [dict(index=i, start=m.FIRST + i, stop=m.FIRST + i + 1) for i in range(6)]
    (root / "consumed_chunks.json").write_bytes(m._canonical(chunks[:5]))
    (root / "progress.jsonl").write_text("{}\n")
    summary = dict(
        schema="long_continuation_phase_v1",
        technical_completion=True,
        logical_initialization_count=1,
        checkpoint_restore_count=7,
        external_pose_reset_count=0,
        processed_total=5,
        processed_interval=[m.FIRST, m.FIRST + 5],
        consumed_chunk_count=5,
        node_recovery_diagnostics=metadata["node_recovery_diagnostics"],
    )
    (root / "summary.json").write_bytes(m._canonical(summary))
    snapshot = root / "code_snapshot" / source.relative_to(tmp_path)
    snapshot.parent.mkdir()
    snapshot.write_bytes(source.read_bytes())

    def item(path):
        return dict(path=str(path), size=path.stat().st_size, sha256=m.sha(path))

    outputs = {role: item(root / name) for role, name in m.PARENT_ROLES.items()}
    for path in (root / "checkpoint/arrays.npz", root / "checkpoint/metadata.json", snapshot):
        outputs[str(path)] = item(path)
    seal = dict(
        stage=m.original.STAGE,
        status="ok",
        params=dict(
            bindings=binding,
            through_frame=m.FIRST + 5,
            all_input_source_snapshot_output_sha_unchanged=True,
        ),
        outputs=outputs,
        inputs={str(p): item(p) for p in (source, original_input)},
    )
    path = root / "continuation.provenance.json"
    path.write_bytes(m._canonical(seal))
    inputs = SimpleNamespace(
        chunk_roots=tuple(tmp_path / str(i) for i in range(6)),
        verified_hashes={str(original_input): m.sha(original_input)},
        chunk_records=chunks,
    )
    return SimpleNamespace(
        root=root,
        seal=seal,
        path=path,
        binding=binding,
        inputs=inputs,
        arrays=arrays,
        metadata=metadata,
        tracker=tracker,
        item=item,
    )


def test_file_backed_parent_full39_21_source_and_consumed_bindings(monkeypatch, tmp_path):
    data = parent_files(monkeypatch, tmp_path)
    evidence = m.Evidence()
    parent = m.load_parent(data.root, m.sha(data.path), data.binding, data.inputs, evidence)
    data.tracker._decode_state(parent["arrays"], parent["metadata"])
    assert parent["metadata"] == data.metadata
    m.original._same_arrays(parent["arrays"], data.arrays)
    evidence.final_guard()


@pytest.mark.parametrize(
    "change",
    [
        "science_input",
        "candidate",
        "snapshot",
        "summary_bool",
        "consumed",
        "named_role",
        "missing_payload",
        "stage",
    ],
)
def test_serialized_parent_forged_evidence_is_rejected(monkeypatch, tmp_path, change):
    data = parent_files(monkeypatch, tmp_path)
    seal = data.seal
    if change == "science_input":
        key = next(iter(data.inputs.verified_hashes))
        seal["inputs"][key]["sha256"] = "a" * 64
    elif change == "candidate":
        values = m.traced_candidate(data.arrays)
        values["pose_xyz"][0, 0, 0, 0] += 0.1
        np.savez_compressed(data.root / "candidate.npz", **values)
        seal["outputs"]["candidate"] = data.item(data.root / "candidate.npz")
    elif change == "snapshot":
        key = next(k for k in seal["outputs"] if "code_snapshot" in k)
        Path(key).write_text("# different producer\n")
        seal["outputs"][key] = data.item(Path(key))
    elif change == "summary_bool":
        p = data.root / "summary.json"
        summary = m.read(p)
        summary["logical_initialization_count"] = True
        p.write_bytes(m._canonical(summary))
        seal["outputs"]["summary"] = data.item(p)
    elif change == "consumed":
        p = data.root / "consumed_chunks.json"
        chunks = m.read(p)
        chunks[-1]["extra"] = "foreign"
        p.write_bytes(m._canonical(chunks))
        seal["outputs"]["consumed_chunks"] = data.item(p)
    elif change == "named_role":
        seal["outputs"]["summary"] = seal["outputs"]["progress"]
    elif change == "missing_payload":
        seal["outputs"].pop(str(data.root / "checkpoint/arrays.npz"))
    else:
        seal["stage"] = m.STAGE
    data.path.write_bytes(m._canonical(seal))
    with pytest.raises(ValueError):
        m.load_parent(data.root, m.sha(data.path), data.binding, data.inputs, m.Evidence())
