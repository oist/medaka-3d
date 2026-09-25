"""Isolated synthetic loader and one four-frame real frozen solve/writer connection."""

from pathlib import Path

import numpy as np
import pytest
import run_nodewise_body_row_five_minute_identity_beam as m

NP_LOAD = np.load


@pytest.fixture(autouse=True)
def isolation(monkeypatch, tmp_path):
    for module in (m, m.comparison, m.common, m.frozen, m.frozen.comparison):
        monkeypatch.setattr(module, "WORK", tmp_path)
    for module in (m.common, m.proof):
        monkeypatch.setattr(module, "REPO", tmp_path)
    for module in (m, m.comparison, m.frozen):
        for key, value in dict(FIRST=100, START=104, STOP=108).items():
            monkeypatch.setattr(module, key, value)
    monkeypatch.setattr(m.frozen, "QUALIFICATION_ROOT", tmp_path / "never_read_qualification")
    monkeypatch.setattr(m.frozen, "COMPARISON_ROOT", tmp_path / "never_read_old_comparison")

    def forbidden(*args, **kwargs):
        pytest.fail("real legacy workflow or qualification input reached")

    for name in ("source_code", "comparison_headers", "load_candidate", "qualification"):
        monkeypatch.setattr(m.frozen, name, forbidden)
    monkeypatch.setattr(m, "source_code", lambda evidence: {"synthetic_source_boundary": "only"})
    # The actual source inventory is checked separately; the solver's entry pin
    # check must not escape this synthetic-only filesystem trust boundary.
    monkeypatch.setattr(m.frozen.beam, "verify_sources", lambda: None)
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
    reads = []

    class Archive:
        def __init__(self, path, *args, **kwargs):
            guard(Path(path))
            self.inner = NP_LOAD(path, *args, **kwargs)
            self.files = self.inner.files

        def __getitem__(self, key):
            reads.append(key)
            return self.inner[key]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

    monkeypatch.setattr(np, "load", Archive)
    return reads


def fixture_bundle(tmp_path, monkeypatch, *, damage=None):
    runner = tmp_path / "run_nodewise_body_row_five_minutes.py"
    scorer = tmp_path / "compare_nodewise_body_row_five_minutes.py"
    runner.write_text("# synthetic tracking source trust\n")
    scorer.write_text("# synthetic comparison source trust\n")
    monkeypatch.setattr(m.comparison, "NODEWISE_RUNNER_SHA", m.proof.sha(runner))
    monkeypatch.setattr(m, "COMPARISON_SHA", m.proof.sha(scorer))
    producer_root = tmp_path / m.comparison.NODEWISE_STAGE
    out = m.proof.Output(producer_root)
    n = m.STOP - m.FIRST
    pose = np.zeros((n, 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(3)[None, :, None] * 20
    pose[..., 1] = np.arange(3)[None, None, :] * 2
    raw = dict(
        frame_idx=np.arange(m.FIRST, m.STOP, dtype=np.int64),
        pose_xyz=pose,
        observed=np.ones((n, 3, 3), bool),
    )
    for key in m.comparison.ALL21 - set(raw):
        raw[key] = np.zeros(1)  # unused18 signatures are producer claims, not numeric qualification
    signatures = m.common.signature(raw)
    if damage == "native":
        raw["frame_idx"][2] += 1
    elif damage == "signature":
        raw["pose_xyz"][2, 0, 0, 0] += 1
    elif damage == "first_direct":
        raw["observed"][m.START - m.FIRST, 1, 0] = False
        signatures = m.common.signature(raw)
    elif damage == "members":
        raw["extra"] = np.zeros(1)
    candidate = out.write("candidate.npz", raw)
    produced = dict(
        schema=m.comparison.NODEWISE_STAGE,
        status="ok",
        process_interval=[m.START, m.STOP],
        scientific_interval=[m.START, m.STOP],
        retained_interval=[m.FIRST, m.STOP],
        native_frames=m.STOP - m.START,
        initialize_count=1,
        restore_count=4,
        new_begin_calls=0,
        external_pose_resets=0,
        finish_calls=1,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        finish_matches_checkpoint_backtrace=True,
        mask_split_priority_applied=False,
        candidate=candidate,
        candidate_arrays=signatures,
    )
    if damage == "unfinished":
        produced["finish_calls"] = 0
    out.write("summary.json", produced)
    producer = dict(
        stage=m.comparison.NODEWISE_STAGE,
        status="ok",
        inputs={str(runner): m.comparison.NODEWISE_RUNNER_SHA},
        outputs=out.records,
    )
    if damage == "producer_stage":
        producer["stage"] = "different_experiment"
    producer_path = producer_root / "tracking.provenance.json"
    producer_path.write_bytes(m.proof.json_bytes(producer))
    producer_sha = m.proof.sha(producer_path)
    ref = dict(
        root=str(producer_root),
        tracking_seal_sha256=producer_sha,
        candidate=candidate,
        all21_declared_signatures=signatures,
        decoded_field_signatures={k: signatures[k] for k in m.comparison.FIELDS},
    )
    refs = dict(sparse={"synthetic_completed_control": True}, nodewise=ref)
    scope = dict(
        schema=m.comparison.STAGE,
        scientific_interval=[m.START, m.STOP],
        terminal_horizon=m.STOP,
        candidate_references=refs,
    )
    root = tmp_path / m.comparison.STAGE
    comp_out = m.proof.Output(root)
    comp_out.write(
        "summary.json",
        dict(
            **scope,
            status="ok",
            arms=list(m.comparison.ARMS),
            native_frames=m.STOP - m.START,
            candidate_modified=False,
            support_recomputed=False,
            review_cleared=False,
            physical_ID_truth=False,
            temporal_ID_mapping_computed=False,
        ),
    )
    comp_out.write(
        "plan.json",
        dict(
            **scope,
            retained_interval=[m.FIRST, m.STOP],
            decoded_fields=list(m.comparison.FIELDS),
            undecoded15_numerically_verified=False,
            lineage=dict(
                original_tracking_sha256=m.comparison.ORIGINAL_SHA,
                original_checkpoint_sha256=m.comparison.ORIGINAL_CHECKPOINT_SHA,
                new_import_all39_signatures_equal=True,
                same_complete40_source_prefix=True,
                nodewise_backend_and_body_start=m.START,
            ),
        ),
    )
    seal = dict(
        stage=m.comparison.STAGE,
        status="ok",
        outputs=comp_out.records,
        inputs={
            str(producer_path): producer_sha,
            str(runner): m.comparison.NODEWISE_RUNNER_SHA,
            str(scorer): m.COMPARISON_SHA,
            str(Path(candidate["path"])): candidate["sha256"],
        },
    )
    if damage == "comparison_stage":
        seal["stage"] = "legacy_fake_stage"
    (root / "comparison.provenance.json").write_bytes(m.proof.json_bytes(seal))
    return m.proof.sha(root / "comparison.provenance.json")


@pytest.mark.parametrize(
    "damage", ["comparison_stage", "producer_stage", "unfinished", "native", "signature", "members"]
)
def test_invalid_stage_scope_or_candidate_refuses_before_any_solve(
    tmp_path, monkeypatch, isolation, damage
):
    digest = fixture_bundle(tmp_path, monkeypatch, damage=damage)
    evidence = m.proof.Evidence()
    with pytest.raises(ValueError):
        headers = m.comparison_headers(digest, evidence)
        m.load_candidate(headers, evidence)
    if damage in ("comparison_stage", "producer_stage", "unfinished"):
        assert isolation == []
    else:
        assert set(isolation) <= set(m.FIELDS)


def test_one_real_tiny_solve_and_frozen_writer_preserves_scope(tmp_path, monkeypatch, isolation):
    digest = fixture_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        m.frozen, "qualification", lambda evidence: {"synthetic_saved_qualification": True}
    )
    solve, writer = m.frozen.solve_once, m.frozen.write_result
    calls = []

    def counted_solve(part):
        calls.append("solve")
        assert part["frame_idx"].tolist() == [104, 105, 106, 107]
        return solve(part)

    def counted_writer(*args):
        calls.append("write")
        return writer(*args)

    monkeypatch.setattr(m.frozen, "solve_once", counted_solve)
    monkeypatch.setattr(m.frozen, "write_result", counted_writer)
    seal = m.run(digest, tmp_path / "result")
    assert seal["status"] == "ok" and calls == ["solve", "write"]
    assert isolation == list(m.FIELDS)
    summary = m.proof.read_json(tmp_path / "result/summary.json")
    assert summary["solve_calls"] == 1 and summary["beam_width"] == 256
    assert summary["gaps_conditional_on_retained_graph"] is True
    assert summary["undecoded18_numerically_verified"] is False
    assert summary["global_optimality_certified"] is False
    with NP_LOAD(tmp_path / "result/identity_mapping_and_ambiguity.npz") as z:
        assert set(z.files) == set(m.frozen.MAPPING_FIELDS) and len(z.files) == 14
        assert np.array_equal(z["permutation_by_id"][0], [0, 1, 2])
    check = m.proof.read_json(tmp_path / "result/independent_path_ledger_check.json")
    assert check["signed_difference"] == 0 and all(check["terminal_arrays_exact"].values())


@pytest.mark.parametrize("cause", ["first_direct", "capacity"])
def test_no_shift_or_retry_and_no_success_on_failure(tmp_path, monkeypatch, cause):
    digest = fixture_bundle(tmp_path, monkeypatch, damage=cause)
    monkeypatch.setattr(m.frozen, "qualification", lambda evidence: {"synthetic": True})
    calls = []

    def failed(part):
        calls.append("solve")
        raise m.frozen.beam.IdentityBeamCapacityError(2, "total_states", 1000001, 1000000)

    monkeypatch.setattr(m.frozen, "solve_once", failed)
    with pytest.raises((ValueError, m.frozen.beam.IdentityBeamCapacityError)):
        m.run(digest, tmp_path / "failed")
    seal = m.proof.read_json(tmp_path / "failed/identity.provenance.json")
    assert seal["status"] == "failed" and seal["partial_identity_solution_published"] is False
    assert calls == ([] if cause == "first_direct" else ["solve"])
    assert seal["solve_calls"] == len(calls) and not seal["automatic_retry"]
    assert not (tmp_path / "failed/identity_mapping_and_ambiguity.npz").exists()
    if cause == "capacity":
        assert seal["native_frame"] == 106 and not seal["cap_increase"]
