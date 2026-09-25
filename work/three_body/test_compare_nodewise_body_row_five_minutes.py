"""Small synthetic control-flow/geometry tests; every input path is isolated."""

import copy
from pathlib import Path

import compare_nodewise_body_row_five_minutes as m
import numpy as np
import pytest

NP_LOAD = np.load


@pytest.fixture(autouse=True)
def isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "EXECUTION_ENABLED", False)
    for module in (m, m.common, m.old):
        monkeypatch.setattr(module, "WORK", tmp_path)
    for module in (m.common, m.proof):
        monkeypatch.setattr(module, "REPO", tmp_path)
    for key, value in dict(FIRST=100, START=104, STOP=112, CHUNK=2, SPARSE_START=108).items():
        monkeypatch.setattr(m, key, value)
    monkeypatch.setattr(m, "FRAMES", (104, 107, 111))
    for name in ("load_headers", "load_lineage"):
        original = getattr(m, name)

        def guarded(*args, _original=original, **kwargs):
            assert tmp_path == m.WORK and tmp_path == m.common.REPO
            return _original(*args, **kwargs)

        monkeypatch.setattr(m, name, guarded)

    def safe_load(path, *args, **kwargs):
        assert isinstance(path, (str, Path)) and Path(path).resolve().is_relative_to(tmp_path)
        return NP_LOAD(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", safe_load)
    read_bytes, read_text, file_open = Path.read_bytes, Path.read_text, Path.open

    def guard(path):
        assert path.resolve().is_relative_to(tmp_path), "real input or output reached"

    def safe_bytes(path):
        guard(path)
        return read_bytes(path)

    def safe_text(path, *args, **kwargs):
        guard(path)
        return read_text(path, *args, **kwargs)

    def safe_open(path, *args, **kwargs):
        guard(path)
        return file_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", safe_bytes)
    monkeypatch.setattr(Path, "read_text", safe_text)
    monkeypatch.setattr(Path, "open", safe_open)


def candidate():
    frames = np.arange(m.FIRST, m.STOP, dtype=np.int64)
    n = len(frames)
    pose = np.zeros((n, 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(3)[None, :, None] * 30
    pose[..., 1] = np.arange(3)[None, None, :] * 2
    return dict(
        frame_idx=frames,
        pose_xyz=pose,
        observed=np.ones((n, 3, 3), bool),
        sources=np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).astype(np.int32),
        identity_review=np.zeros((n, 3), bool),
        proposal_kind=np.full(n, "predict", dtype="U24"),
    )


def test_nonselfinverse_wholebody_permutation_and_support_ties():
    a = candidate()
    b = copy.deepcopy(a)
    p = np.array([1, 2, 0])
    for key in ("pose_xyz", "observed", "sources", "identity_review"):
        b[key][:, p] = a[key]
    b["observed"][:, p[1], 2] = False
    report, arrays = m.geometric_comparison(a, b)
    index = np.flatnonzero((p == m.PERMUTATIONS).all(1))[0]
    assert np.all(arrays["chest_optimal_permutation_mask"][:, index])
    assert np.all(arrays["all_node_rms_mm"][:, index] == 0)
    assert np.all(arrays["both_direct_node_counts_by_permutation"][:, index] == 8)
    assert report["chest"]["unique_nonidentity_frames"] == len(a["frame_idx"])
    b["pose_xyz"][:] = 0
    a["pose_xyz"][:] = 0
    report, arrays = m.geometric_comparison(a, b)
    assert report["chest"]["tied_frames"] == len(a["frame_idx"])
    assert arrays["all_node_optimal_permutation_mask"].all()


def test_motion_scope_missing_and_node_view_axes():
    c = candidate()
    c["pose_xyz"][3, 0, 2, 0] = 1000  # before science; never an included adjacent endpoint
    c["pose_xyz"][6:, 0, 2, 0] += 10  # exactly10 is not >10
    c["pose_xyz"][8:, 1, 0, 0] += 11
    c["observed"][7:9, 1, 0] = False  # finite latent jump must not become direct
    c["sources"][7:9, 1, :, 0] = -1
    c["sources"][5, 2, 0, 1] = 0  # one conflicting node in top view, not view-number comparison
    report = m.summarize(c, m.START, m.STOP)
    assert report["direct_adjacent_steps_over_10mm"] == []
    assert report["mixed_source_rows_by_slot_camera"][2] == [1, 0, 0]
    assert report["longest_missing_run_by_slot_node"][1][0] == 2
    assert report["all_estimated_max_step_by_slot_node_mm"][1][0] == 11
    assert report["all_estimated_max_step_by_slot_node_mm"][0][2] == 10
    c["proposal_kind"][5] = "body_row"
    with pytest.raises(ValueError, match="source rows"):
        m.summarize(c, m.START, m.STOP)


@pytest.mark.parametrize("damage", ["gap", "axis", "nonfinite", "source_count"])
def test_candidate_rejects_incomplete_or_inconsistent_fields(damage):
    c = candidate()
    m.validate_candidate(c)
    if damage == "gap":
        c["frame_idx"][4] += 1
    elif damage == "axis":
        c["observed"] = c["observed"][:, :, 0]
    elif damage == "nonfinite":
        c["pose_xyz"][0, 0, 0, 0] = np.nan
    else:
        c["sources"][0, 0, :2, 0] = -1
    with pytest.raises(ValueError):
        m.validate_candidate(c)


def test_original_import_preserves_every_metadata_and_array_signature():
    before = dict(context={"implementation": "Original"}, restore_count=3)
    sig = {str(i): {"sha256": str(i)} for i in range(39)}
    context = dict(
        sparse_distance_backend={"effective_from_native": m.START},
        body_row_policy={"effective_from_native": m.START},
        instance_rescore_policy={"effective_from_native": 76360},
    )
    imported = dict(
        schema="nodewise_original_state_import_v1",
        status="ok",
        effective_from_native=m.START,
        previous_frame=m.START - 1,
        processed_frames=m.START - m.FIRST,
        logical_initialize_count=1,
        added_restore_count=1,
        new_begin_calls=0,
        all_arrays_byte_preserved=True,
        original_metadata_unchanged=True,
        instance_policy_effective_from_native=76360,
        review_cleared_by_migration=False,
        mask_split_priority_applied=False,
        original_metadata=copy.deepcopy(before),
        arrays=copy.deepcopy(sig),
        new_context=context,
        imported_metadata=dict(context=context, restore_count=4),
    )
    m.validate_original_import(imported, before, sig, context)
    imported["imported_metadata"]["undeclared"] = True
    with pytest.raises(ValueError, match="undeclared"):
        m.validate_original_import(imported, before, sig, context)


def producer(tmp_path, arm, monkeypatch):
    stage = m.NODEWISE_STAGE if arm == "nodewise" else m.SPARSE_STAGE
    root = tmp_path / stage
    out = m.proof.Output(root)
    new = arm == "nodewise"
    runner = (
        "run_nodewise_body_row_five_minutes.py"
        if new
        else "continue_sparse_instance_five_minutes.py"
    )
    file = tmp_path / runner
    file.write_text("# synthetic successful producer\n")
    digest = m.proof.sha(file)
    monkeypatch.setattr(m, "NODEWISE_RUNNER_SHA" if new else "SPARSE_RUNNER_SHA", digest)
    codes = {str(file): digest}
    resume, restores = (m.START, 4) if new else (m.SPARSE_START, 6)
    arrays = candidate()
    for key in m.ALL21 - set(arrays):
        arrays[key] = np.zeros(1)  # undecoded producer trust boundary, not numeric qualification
    member = out.write("candidate.npz", arrays)
    binding = dict(synthetic_registered_initial_and_inputs=True)
    index = dict(
        original_binding=binding,
        prefix_chunks=[
            dict(index=i, start=f, stop=f + m.CHUNK)
            for i, f in enumerate(range(m.FIRST, m.STOP, m.CHUNK))
        ],
    )
    lifecycle = dict(
        schema=stage,
        scientific_interval=[m.START, m.STOP],
        process_interval=[resume, m.STOP],
        retained_interval=[m.FIRST, m.STOP],
        initialize_count=1,
        restore_count=restores,
        new_begin_calls=0,
        external_pose_resets=0,
    )
    plan = dict(
        **lifecycle,
        source_sha256=codes,
        original_binding=binding,
        checkpoint_binding=dict(original_binding=binding, source_sha256=codes),
    )
    summary = dict(
        **lifecycle,
        status="ok",
        native_frames=m.STOP - m.START,
        finish_calls=1,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        finish_matches_checkpoint_backtrace=True,
        candidate_arrays=m.common.signature(arrays),
        candidate=member,
        checkpoints=[
            dict(
                path=str(root / f"checkpoints/through_{stop:06d}"),
                arrays=39,
                previous_frame=stop - 1,
                processed_frames=stop - m.FIRST,
                initialize_count=1,
                restore_count=restores,
            )
            for stop in range(resume + m.CHUNK, m.STOP + 1, m.CHUNK)
        ],
    )
    if new:
        summary.update(
            source_sha256=codes,
            scientific_native_frames=m.STOP - m.START,
            diagnostic_frames=m.STOP - m.START,
            body_row_and_sparse_backend_effective_from_native=m.START,
            sixth_term_effective_from_native=76360,
            mask_split_priority_applied=False,
        )
    for name, doc in (("summary", summary), ("plan", plan), ("input_index", index)):
        out.write(name + ".json", doc)
    seal = dict(stage=stage, status="ok", inputs=codes, outputs=out.records)
    (root / "tracking.provenance.json").write_bytes(m.proof.json_bytes(seal))
    return root, m.proof.sha(root / "tracking.provenance.json"), digest


def test_synthetic_end_to_end_two_producer_headers_six_field_loads_and_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(m, "NODEWISE_COMPONENTS", {})  # reviewed full5 code/39 trust is synthetic
    monkeypatch.setattr(m, "PINS", {})
    source = tmp_path / "comparator.py"
    source.write_text("# synthetic source provenance\n")
    monkeypatch.setattr(m, "__file__", str(source))
    _, sparse_sha, _ = producer(tmp_path, "sparse", monkeypatch)
    root, digest, runner_sha = producer(tmp_path, "nodewise", monkeypatch)
    monkeypatch.setattr(m, "SPARSE_SHA", sparse_sha)

    def lineage(loaded, evidence):
        assert tuple(loaded) == m.ARMS
        return {"synthetic_original39_lineage_only": True}

    monkeypatch.setattr(m, "load_lineage", lineage)
    with pytest.raises(ValueError, match="disabled"):
        m.run(root, digest, runner_sha, tmp_path / "disabled")
    monkeypatch.setattr(m, "EXECUTION_ENABLED", True)
    seal = m.run(root, digest, runner_sha, tmp_path / "result")
    assert seal["status"] == "ok" and len(seal["outputs"]) == 6
    summary = m.proof.read_json(tmp_path / "result/summary.json")
    assert summary["arms"] == ["sparse", "nodewise"] and summary["native_frames"] == 8
    with np.load(tmp_path / "result/review_samples.npz") as z:
        assert len(z.files) == 12
        assert np.array_equal(z["nodewise_frame_idx"], m.FRAMES)
    evidence = m.proof.Evidence()
    loaded = m.read_producer(root, digest, m.NODEWISE_STAGE, evidence)
    loaded["summary"]["finish_calls"] = 0
    with pytest.raises(ValueError, match="completed"):
        m.validate_completion(loaded, "nodewise", runner_sha, evidence)
